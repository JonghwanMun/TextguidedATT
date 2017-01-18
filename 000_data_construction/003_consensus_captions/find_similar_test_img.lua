require 'nn'
require 'nngraph'
require 'torch'

require 'hdf5'

local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Find similar images for test data')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-feat_h5_file','','path to the h5 file containing feature')
cmd:option('-test_feat_h5_file','','path to the h5 file containing test feature')
cmd:option('-output_json_path','','path to the output')
cmd:option('-img_info_file','data/coco/coco_trainval_raw.json','path of trainval img information (path, imgID, captions ...)')
cmd:option('-test_img_info_file','data/coco/coco_test_raw.json','path of test img information (path, imgID, captions ...)')

-- NN options
cmd:option('-top_k',60,'number of k-nearest neighbor images')
cmd:option('-sIdx',-1,'start index for test data' )
cmd:option('-eIdx',-1,'end index for test data' )

-- misc
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpu_id', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
print('print options')
print(opt)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpu_id >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu_id + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Function for constructing distance Network
-------------------------------------------------------------------------------
local function distance_net(ref_num)
  -- compute euclidean distance
  inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local test = inputs[1]
  local refs = inputs[2]

  local rep_test = nn.Replicate(ref_num)(test)
  local diff = nn.CSubTable()({refs, rep_test})
  local dists = nn.Square()(diff)
  dists = nn.Sum(2)(dists)
  dists = nn.Sqrt()(dists)

  return nn.gModule(inputs, {dists})
end

-------------------------------------------------------------------------------
-- Main part
-------------------------------------------------------------------------------
local img_info = utils.read_json(opt.img_info_file)
local test_img_info = utils.read_json(opt.test_img_info_file)
local sIdx, eIdx

local feat_file = hdf5.open(opt.feat_h5_file, 'r')
local test_feat_file = hdf5.open(opt.test_feat_h5_file, 'r')

local feat_shape = feat_file:read('/feature'):dataspaceSize()
local test_feat_shape = test_feat_file:read('/feature'):dataspaceSize()

-- basically, find NN for all data of testset
sIdx = 0; 
eIdx = test_feat_shape[1]

if opt.sIdx ~= -1 then sIdx = opt.sIdx end
if opt.eIdx ~= -1 then eIdx = opt.eIdx end
print(string.format('====> set to sIdx (%d) and eIdx (%d)', sIdx,eIdx))

-- the order of trainval imgs :
-- 1~5000     (val)
-- 5001~10000 (test)    <- test imgs of validation
-- 100001~    (train)   <- 10001~92783 imgs are obtained from train2014 data, we compute similar imgs from these imgs
local test_feats = test_feat_file:read('/feature'):partial({sIdx+1,eIdx},{1, test_feat_shape[2]}):float()
local train_feats = feat_file:read('/feature'):partial({10001,92783},{1, feat_shape[2]}):float()

local net = distance_net(train_feats:size(1))
if opt.gpu_id >= 0 then 
  print('Change to cuda mode')
  net:cuda(); 
  test_feats = test_feats:cuda();   train_feats = train_feats:cuda()
end

print('Start to compute distance between test and train imgs')
local timer = torch.Timer()
local dists = torch.FloatTensor(test_feats:size(1), train_feats:size(1))
--for i=1,10 do -- for testing this code works well
for i=1,test_feats:size(1) do
  dists[{i}] = net:forward({test_feats[{i}], train_feats}):float()
  if i%100 == 0 then collectgarbage() end
  if i%100 == 0 then print(string.format('Computing distance %d/%d done (%.3fs)', i, test_feats:size(1), timer:time().real)) end
end
print(string.format('Computing distance done (%.3fs)', timer:time().real))

print('Compute NN imgs')
local topK_NN, idx_NN = torch.topk(dists, opt.top_k, 2, false, true)
idx_NN = idx_NN + 10000  -- because former 10,000 imgs are for val and test

print('Construct NN imgs info to save')
st = os.time()
local nFile = 1
local save_json = {}
--for i=1,10 do    -- for testing this code well works 
for i=1,test_feats:size(1) do
  local ith_feat_NN_info = { imgId=test_img_info[sIdx+i]['id'], NN_idx={}, NN_captions={}, NN_imgIds={}, NN_path={} }
  local NN_imgs_idx = idx_NN[{i}]

  for j=1,NN_imgs_idx:size(1) do
    local NN_img_info = img_info[ NN_imgs_idx[j] ]
    table.insert(ith_feat_NN_info['NN_idx'], NN_imgs_idx[j])             -- idx of NN imgs
    table.insert(ith_feat_NN_info['NN_imgIds'], NN_img_info['id'])       -- imgId of NN imgs
    table.insert(ith_feat_NN_info['NN_path'], NN_img_info['file_path'])  -- file path of NN imgs
    for k,cc in pairs(NN_img_info['captions']) do
      table.insert(ith_feat_NN_info['NN_captions'], cc)
      if k == 5 then break end
    end
  end
  table.insert(save_json, ith_feat_NN_info)
  if i % 10000 == 0 then
    utils.write_json(opt.output_json_path .. string.format('_%d.json', nFile), save_json)
    print(string.format('write NN infos to %s', opt.output_json_path.. string.format('_%d.json', nFile)))
    nFile = nFile + 1
    save_json = {}
  end
end
print(string.format('Constructing NN imgs info done (%.3fs)', os.difftime(os.time(), st)))

-- save file
if nFile == 1 then
  utils.write_json(opt.output_json_path, save_json)
  print(string.format('write NN infos to %s', opt.output_json_path))
else
  utils.write_json(opt.output_json_path .. string.format('_%d.json', nFile), save_json)
  print(string.format('write NN infos to %s', opt.output_json_path.. string.format('_%d.json', nFile)))
end
