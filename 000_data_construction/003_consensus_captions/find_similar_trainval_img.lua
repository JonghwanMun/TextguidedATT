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
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-img_info_file','data/coco/coco_trainval_raw.json','path to the file containing img information (path, imgID, captions ...)')
cmd:option('-feat_h5_file','','path to the h5 file containing feature, the order of imgs should be same with imgs in info file')
cmd:option('-output_json_path','','path to the output')

-- NN options
cmd:option('-top_k',60,'number of k-nearest neighbor images')

-- misc
cmd:option('-sIdx', -1, 'start index for data')
cmd:option('-eIdx', -1, 'end index for data')
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
  inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  test = inputs[1]
  refs = inputs[2]

  rep_test = nn.Replicate(ref_num)(test)
  diff = nn.CSubTable()({refs, rep_test})
  dists = nn.Square()(diff)
  dists = nn.Sum(2)(dists)
  dists = nn.Sqrt()(dists)

  return nn.gModule(inputs, {dists})
end

-------------------------------------------------------------------------------
-- Main part
-------------------------------------------------------------------------------
local img_info = utils.read_json(opt.img_info_file)
print(string.format('the number of img info : %d ', #img_info))

local sIdx = 0
local eIdx = #img_info
local sTrain = 10000
local eTrain = 92783

if opt.sIdx ~= -1 then sIdx = opt.sIdx end
if opt.eIdx ~= -1 then eIdx = opt.eIdx end
print(string.format('sIdx (%d)  |  eIdx (%d)', sIdx, eIdx))

-- the order of imgs :
-- 1~5000     (val)
-- 5001~10000 (test)    <- test imgs of validation
-- 100001~    (train)   <- 10001~92783 imgs are obtained from train2014 data, we compute similar imgs from these imgs
local feat_file = hdf5.open(opt.feat_h5_file, 'r')
local feat_shape = feat_file:read('/feature'):dataspaceSize()
print('Featrue shape : ', feat_shape)
local test_feats = feat_file:read('/feature'):partial({sIdx+1,eIdx},{1, feat_shape[2]}):float()
-local train_feats = feat_file:read('/feature'):partial({sTrain+1,eTrain},{1, feat_shape[2]}):float()
--local train_feats = feat_file:read('/feature'):partial({10001,feat_shape[1]},{1, feat_shape[2]}):float()

local net = distance_net(train_feats:size(1))
if opt.gpu_id >= 0 then 
  print('Change to cuda mode')
  net:cuda(); 
  test_feats = test_feats:cuda();   train_feats = train_feats:cuda()
end

print('Start to compute distance between test and train imgs')
local st = os.time()
local dists = torch.FloatTensor(test_feats:size(1), train_feats:size(1))
--for i=1,10 do -- for testing this code well works
for i=1,test_feats:size(1) do
  dists[{i}] = net:forward({test_feats[{i}], train_feats}):float()
  if i >= sTrain and i <= eTrain then     -- skip same feature 
    local diff = test_feats[{i}] - train_feats[{i-sTrain}]
    if torch.sum(diff) == 0 then dists[{i,i-sTrain}] = 99999.0 end
  end
  if i%100 == 0 then 
    print(string.format('Computing distance %d/%d done (%.3fs)', i, test_feats:size(1), os.difftime(os.time(), st)))
    collectgarbage()
  end
end
print(string.format('Computing distance done (%.3f)', os.difftime(os.time(), st)))

print('Compute NN imgs')
local topK_NN, idx_NN = torch.topk(dists, opt.top_k, 2, false, true)
idx_NN = idx_NN + 10000  -- because former 10,000 imgs are for val and valtest

print('Construct NN imgs info to save')
st = os.time()
local nFile = 1
local save_json = {}
--for i=1,10 do    -- for testing this code well works 
for i=1,test_feats:size(1) do
  local ith_feat_NN_info = { imgId=img_info[i+sIdx]['id'], NN_idx={}, NN_captions={}, NN_imgIds={}, NN_path={} }
  local NN_imgs_idx = idx_NN[{i}]

  for j=1,NN_imgs_idx:size(1) do
    local NN_img_info = img_info[NN_imgs_idx[j]]
    table.insert(ith_feat_NN_info['NN_idx'], NN_imgs_idx[j])             -- idx of NN imgs in img_info_file
    table.insert(ith_feat_NN_info['NN_imgIds'], NN_img_info['id'])       -- imgId of NN imgs
    table.insert(ith_feat_NN_info['NN_path'], NN_img_info['file_path'])  -- file path of NN imgs
    for k,cc in pairs(NN_img_info['captions']) do
      table.insert(ith_feat_NN_info['NN_captions'], cc)                  -- only first 5 captions are selected
      if k == 5 then break end
    end
  end
  table.insert(save_json, ith_feat_NN_info)
  if i % 80 == 0 then
    utils.write_json(opt.output_json_path .. string.format('_%d.json', nFile), save_json)
    print(string.format('write NN infos to %s', opt.output_json_path .. string.format('_%d.json', nFile)))
    nFile = nFile + 1
    save_json = {}
  end
end
print(string.format('Constructing NN imgs info done (%.3f)', os.difftime(os.time(), st)))

-- save file 
if nFile == 1 then
  utils.write_json(opt.output_json_path, save_json)
  print(string.format('write NN infos to %s', opt.output_json_path))
else
  utils.write_json(opt.output_json_path .. string.format('_%d.json', nFile), save_json)
  print(string.format('write NN infos to %s', opt.output_json_path.. string.format('_%d.json', nFile)))
end
