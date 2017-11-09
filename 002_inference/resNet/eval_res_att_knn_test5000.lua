require 'torch'
require 'nn'
require 'nngraph'

-- exotics
require 'loadcaffe'
require 'math'
require 'hdf5'

-- local imports
require 'layers.LanguageModel'
require 'layers.guidanceCaptionEncoder'
local textGuidedAtt = require 'layers.textGuidedAtt'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Inference an Image Captioning model with multiple consensus captions')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','','path to model to evaluate')
cmd:option('-output_predictions','prediction_result/res_predictions_test5000.json','path to json for visualization')

-- Loader input
cmd:option('-img_feat_file','data/resnet101_conv_feat_448/','path to precomputed image features')
cmd:option('-cap_feat_file','data/skipthought/cocotalk_trainval_skipthought.h5','path to precomputed skip-thoughts vector for trainval captions')
cmd:option('-img_info_file','data/coco/cocotalk_trainval_img_info.json','path to the json containing the trainval image information for vocabulary')
cmd:option('-cap_h5_file','data/coco/cocotalk_cap_label.h5','path to the h5 file containing the caption label and infomation')
cmd:option('-kNN_cap_json_file','data/coco/10NN_cap_valtrainall_cider.json','path to the json file containing NN caps for test5000')

-- feature dimension
cmd:option('-cap_feat_dim',2400,'dimension of the skipthought feature from caption')
cmd:option('-img_feat_dim',2048,'dimension of the cnn feature from image')

-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

-- misc
cmd:option('-n_knn', 10, 'the number of nn captions')
cmd:option('-sIdx', -1, '')
cmd:option('-eIdx', -1, '')
cmd:option('-seq_per_img', 5, 'the number of captions for each image')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end
--
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
print('Load img info file : ', opt.img_info_file)
local img_info = utils.read_json(opt.img_info_file)
local vocab = img_info.ix_to_word

print('Load kNN caps info file : ', opt.kNN_cap_json_file)
local kNN_cap_info = utils.read_json(opt.kNN_cap_json_file)

print('Load caption label info file : ', opt.cap_h5_file)
local cap_file = hdf5.open(opt.cap_h5_file, 'r')
local label_start_ix = cap_file:read('/label_start_ix'):all()

print('Load img feat file : ', opt.img_feat_file)
local img_feat_path =  opt.img_feat_file

print('Load cap feat file : ', opt.cap_feat_file)
local cap_feat_file =hdf5.open(opt.cap_feat_file, 'r')

-- Set start and end index of images
local sIdx = 5001
local eIdx = 10000
if opt.sIdx ~= -1 then sIdx = opt.sIdx end
if opt.eIdx ~= -1 then eIdx = opt.eIdx end
print(string.format('sIdx (%d)  |  eIdx (%d)', sIdx, eIdx))

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
if string.len(opt.model) > 0 then  -- finetuning the model
  -- load protos from file
  print('initializing weights from ' .. opt.model)
  local loaded_checkpoint = torch.load(opt.model)
  opt.precomputed_feat = loaded_checkpoint.opt.precomputed_feat

  protos = loaded_checkpoint.protos

  ----------------------------------------------------------------------------
  -- Unsanitize gradient for each model
  ----------------------------------------------------------------------------
  -- Attention model
  net_utils.unsanitize_gradients(protos.att)

  -- Image Embedding Model
  net_utils.unsanitize_gradients(protos.imgEmb)

  -- Language Model
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end

  if loaded_checkpoint.opt.sharedE == true then
    for idxNode, node in ipairs(protos.lm.core.forwardnodes) do
      if node.data.annotations.name == 'decoder' then
        node.data.module.gradWeight:set( protos.lm.lookup_table.gradWeight )
        print('gradient sharing done for \'decoder\'')
        break
      end
    end
  end
end
----------------------------------------------------------------------------
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

print('\nloaded options are as follows')
print(opt)

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)

  protos.lm:evaluate(); protos.imgEmb:evaluate(); protos.att:evaluate(); 

  local predictions = {}
  local timer = torch.Timer()

  print('-------------------------------------Evaluation kNN caption---------------------------------------')
  for ii=sIdx,eIdx do

    -- Load img feat
    local num_img  = opt.n_knn / opt.seq_per_img
    local img_feats = torch.FloatTensor(num_img,opt.img_feat_dim,14,14)
    local file_path = img_info.images[ii].file_path
    file_path = utils.sen_split(file_path, '/')
    file_path = string.gsub(file_path[2], 'jpg', 't7')
    local img_feat = torch.load( img_feat_path .. file_path )
    for j=1,num_img do img_feats[j] = img_feat end

    -- Load cap feat
    local cap_feats = torch.FloatTensor(opt.n_knn, opt.cap_feat_dim)
    local knn_imgIdx, knn_capIdx = kNN_cap_info[ii]['NN_imgIdx'], kNN_cap_info[ii]['capIdx']
    for j=1,opt.n_knn do
      local capLoc = label_start_ix[ knn_imgIdx[j] ] + knn_capIdx[j]
      local cap_feat = cap_feat_file:read('/feature'):partial({capLoc,capLoc},{1,opt.cap_feat_dim})
      cap_feats[j] = cap_feat
    end

    -- forward to compute embbeding feature
    local attended_feats, alphas = unpack(protos.att:forward({img_feats:cuda(), cap_feats:cuda()}))
    local emb_feats = protos.imgEmb:forward(attended_feats)

    -- forward the model to also get generated samples for each image
    local seq, probs, beam_seq = protos.lm:sample(emb_feats, {beam_size=opt.beam_size})

    local sents = net_utils.decode_sequence(vocab, seq) -- idx to word

    local entry = {image_id = img_info.images[ii].id, caption = sents}
    table.insert(predictions, entry)

    if verbose and ii % 10 == 0 then
      for i=1,#sents do
        print(string.format('Generated  : %s\n', sents[i]))
      end
      local remaining_time = timer:time().real/(ii-sIdx+1) * (eIdx-ii) / 60.0
      print(string.format('evaluating validation performance... %d/%d (%.3fs) (%.3fm)', ii, eIdx, timer:time().real, remaining_time ))
      print('-----------------------------------------------------------------------------------')
    end

    if ii % 10 == 0 then collectgarbage() end
  end

  return predictions, lang_stats
end

-----------------------------------------------------------------------------------------------------
-- Main loop
-----------------------------------------------------------------------------------------------------
protos.lm:set_vocab(vocab)
local predictions, lang_stats = eval_split(opt.split, {num_images = opt.num_images})

utils.write_json(opt.output_predictions, predictions)
print(string.format('write predictions to %s', opt.output_predictions))
