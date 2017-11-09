require 'hdf5'
require 'math'
require 'image'
local utils = require 'misc.utils'

local resAttLoader_kNN = torch.class('resAttLoader_kNN')

--[[
Prepare three input data
  1. Image feature
  2. Guiding sentence feature
  3. true caption (guiding sentence) label sequence and its length
The list opt should contain
  - img_info_file: img information file containing vocabulary, file path, imgId ...
  - cap_h5_file: caption label information
  - img_feat_file, cap_feat_file: pre-computed image or guiding sentence features
  - kNN_cap_json_file: top k consensus captions information
--]]
function resAttLoader_kNN:__init(opt)
  
  print('\n-----------------------Loader for sentence guided attention with precomputed data -------------------------')
  self.on_gpu = utils.getopt(opt, 'on_gpu', true)

  -- load the image info file 
  print('resAttLoader_kNN loading image info file: ', opt.img_info_file)
  self.img_info = utils.read_json(opt.img_info_file)
  self.ix_to_word = self.img_info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)
  print('vocab size is ' .. self.vocab_size)

  -- load NN caption for val/test
  print('resAttLoader_kNN loading kNN caption json file: ', opt.kNN_cap_json_file)
  self.kNN_cap = utils.read_json(opt.kNN_cap_json_file)

  -- load the sequence data (caption)
  print('resAttLoader_kNN loading caption h5 file: ', opt.cap_h5_file)
  self.cap_file = hdf5.open(opt.cap_h5_file, 'r')
  self.label_start_ix = self.cap_file:read('/label_start_ix'):all()
  self.label_end_ix = self.cap_file:read('/label_end_ix'):all()
	self.label_length = self.cap_file:read('/label_length'):all()
  local seq_size = self.cap_file:read('/labels'):dataspaceSize()
  self.seq_length = seq_size[2]
  print('max sequence length in data is ' .. self.seq_length)

  print('resAttLoader_kNN loading base path of precomputed image feat : ', opt.img_feat_file)
  self.img_feat_root = opt.img_feat_file
  print('resAttLoader_kNN loading precomputed caption feat h5 file: ', opt.cap_feat_file)
  self.cap_feat = hdf5.open(opt.cap_feat_file, 'r')
  
  -- separate out indexes for each of the provided splits
  self.split_ix = {}
  self.iterators = {}
  for i,img in pairs(self.img_info.images) do
    local split = img.split
    if not self.split_ix[split] then
      -- initialize new split
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], i)
  end
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end

  -- change from list to torch data for shuffle every epoch on train data
  self.split_ix['train'] = torch.randperm(#self.split_ix['train']):add(10000)
end

function resAttLoader_kNN:resetIterator(split)
  self.iterators[split] = 1
end

function resAttLoader_kNN:getVocabSize()
  return self.vocab_size
end

function resAttLoader_kNN:getVocab()
  return self.ix_to_word
end

function resAttLoader_kNN:getTrainNum()
  return self.split_ix['train']:size(1)
end

function resAttLoader_kNN:getSeqLength()
  return self.seq_length
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Iterators for any split can be reset manually with resetIterator()
--]]
function resAttLoader_kNN:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
  local seq_per_img = utils.getopt(opt, 'seq_per_img', 5) -- number of sequences to return per image

  local isNN = false
  if split == 'valNN' then isNN = true; split = 'val' end
  if split == 'testNN' then isNN = true; split = 'test' end
  if split == 'trainNN' then isNN = true; split = 'train' end

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pre-allocate the memory for batch data
  local label_batch = torch.LongTensor(batch_size * seq_per_img, self.seq_length)
  local label_length_batch = torch.LongTensor(batch_size * seq_per_img)
  local NN_label_batch, NN_label_length_batch
  if isNN == true then
    NN_label_batch = torch.LongTensor(batch_size * seq_per_img, self.seq_length)
    NN_label_length_batch = torch.LongTensor(batch_size * seq_per_img)
  end
  local img_batch_feat = torch.FloatTensor(batch_size, 2048, 14, 14)
  local cap_batch_feat = torch.FloatTensor(batch_size*seq_per_img, 2400)

  local max_index 
  if split == 'train' then max_index = split_ix:size(1)
  else max_index =  #split_ix end
  local wrapped = false
  local infos = {}

  for i=1,batch_size do
    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1           -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next

    ix = split_ix[ri]
    if split == 'train' and wrapped == true and i == batch_size then
      print('======> train data is wrapped!! we shuffle the img idx')
      self.split_ix['train'] = torch.randperm(self.split_ix['train']:size(1)):add(10000)
      print(string.format('imgId of first data : %d', self.img_info.images[ self.split_ix['train'][1] ].id))
    end
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

    -- load the image feature for batch
    local file_path = self.img_info.images[ix].file_path
    file_path = utils.sen_split(file_path, '/')
    file_path = string.gsub(file_path[2], 'jpg', 't7')
    local img_feats = torch.load( self.img_feat_root .. file_path )
    img_batch_feat[i] = img_feats

    -- prepare the GT sequence labels for batch
    local seq, len, cap_feats
    local ix1 = self.label_start_ix[ix]
    local ix2 = self.label_end_ix[ix]
    local ncap = ix2 - ix1 + 1 -- number of captions available for this image
    assert(ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t')
    if ncap < seq_per_img then
      -- we need to subsample (with replacement)
      seq = torch.LongTensor(seq_per_img, self.seq_length)
      len = torch.LongTensor(seq_per_img)
      if isNN == false then cap_feats = torch.FloatTensor(seq_per_img, 2400) end
      for q=1, seq_per_img do
        local ixl = torch.random(ix1,ix2)
        seq[q] = self.cap_file:read('/labels'):partial({ixl, ixl}, {1,self.seq_length})
        len[q] = self.label_length[{ixl}]
        if isNN == false then cap_feats[q] = self.cap_feat:read('/feature'):partial({ixl, ixl}, {1,2400}) end
      end
    else
      -- there is enough data to read a contiguous chunk, but subsample the chunk position
      local ixl = torch.random(ix1, ix2 - seq_per_img + 1)  -- generates integer in the range
      seq = self.cap_file:read('/labels'):partial({ixl, ixl+seq_per_img-1}, {1,self.seq_length})
      len = self.label_length[{ {ixl,ixl+seq_per_img-1} }]
      if isNN == false then cap_feats = self.cap_feat:read('/feature'):partial({ixl, ixl+seq_per_img-1}, {1,2400}) end
    end

    -- prepare the NN sentence labels for batch if needed
    local NN_seq, NN_len
    if isNN == true then
      local NN_imgIdx, capIdx = self.kNN_cap[ix]['NN_imgIdx'], self.kNN_cap[ix]['capIdx']

      NN_seq = torch.LongTensor(seq_per_img, self.seq_length)
      NN_len = torch.LongTensor(seq_per_img)
      cap_feats = torch.FloatTensor(seq_per_img, 2400)

      for q=1,seq_per_img do
        local nn_idx = torch.random(1,#capIdx)
        local capLoc = self.label_start_ix[ NN_imgIdx[nn_idx] ] + capIdx[nn_idx]

        NN_seq[q] = self.cap_file:read('/labels'):partial({capLoc,capLoc}, {1,self.seq_length})
        NN_len[q] = self.label_length[{ {capLoc,capLoc} }]
        cap_feats[q] = self.cap_feat:read('/feature'):partial({capLoc,capLoc}, {1,2400})
      end
    end

    local il = (i-1)*seq_per_img+1
    label_batch[{ {il,il+seq_per_img-1} }] = seq
    label_length_batch[{ {il,il+seq_per_img-1} }] = len
    if isNN == true then
      NN_label_batch[{ {il,il+seq_per_img-1} }] = NN_seq
      NN_label_length_batch[{ {il,il+seq_per_img-1} }] = NN_len
    end
    cap_batch_feat[{ {il,il+seq_per_img-1} }] = cap_feats

    -- and record associated info as well
    local info_struct = {}
    info_struct.id = self.img_info.images[ix].id
    info_struct.file_path = self.img_info.images[ix].file_path
    table.insert(infos, info_struct)
  end

  local data = {}
	if self.on_gpu then 
    cap_batch_feat = cap_batch_feat:cuda()
    img_batch_feat = img_batch_feat:cuda()
  end
  data.img_feat = img_batch_feat
  data.cap_feat = cap_batch_feat
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
	data.label_length = label_length_batch
  if isNN == true then
    data.nn_labels = NN_label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  	data.nn_label_length = NN_label_length_batch
  end
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos  = infos
  return data
end
