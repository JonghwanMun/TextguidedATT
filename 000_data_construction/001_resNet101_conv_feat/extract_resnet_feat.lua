require 'torch'
require 'nn'

-- exotic things
require 'image'

-- local imports
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local transforms = require 'misc.transforms'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Extract feature using residual network')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-model', 'model/resNet/resnet-101.t7', 'path to the 101-layer Residual Network model.')
cmd:option('-output_path','data/resnet101_conv_feat_448/','path to the output feature.')

-- Img information
cmd:option('-img_root','data/MSCOCO/','root folder containing images')
cmd:option('-cnn_img_size',448,'input image size for the residual network')
cmd:option('-img_trainval_info_file','data/coco/coco_trainval_raw.json','path to train & validation image information')
cmd:option('-img_test_info_file','data/coco/coco_test_raw.json','path to test image information')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
print('print options')
print(opt)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

-- for gpu
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end


-------------------------------------------------------------------------------
-- Load the CNN model (Residual Network)
-------------------------------------------------------------------------------
local protos = {}
protos.net = torch.load(opt.model)
protos.net:remove(11)
protos.net:remove(10)
protos.net:remove(9) -- remove the last three classification layers

-- ship everything to GPU, maybe  &  set to evaluation mode
if opt.gpuid >= 0 then protos.net:cuda() end
protos.net:evaluate()

meanstd = {
  mean = {0.485, 0.456, 0.406},
  std  = {0.229, 0.224, 0.225}
}
local transform = transforms.Compose {
  transforms.ColorNormalize(meanstd),
}

collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- feature extracttion function
-------------------------------------------------------------------------------
function extract_feat(img_info, split)
  
   local timer = torch.Timer()
   local nImg = #img_info

   for ii=1,nImg do
      local file_path = img_info[ii].file_path
      file_path = utils.sen_split(file_path, '/')
      file_path = string.gsub(file_path[2], 'jpg', 't7')
      local save_path = opt.output_path .. file_path

      local remaining_time = (timer:time().real / (ii)) * (nImg-ii) / 60.0
      print( string.format('%d/%d: %s (left %.2fs) for %s',ii, nImg, save_path, remaining_time, split) )

      if pcall(function() torch.load(save_path) end) then
      print('file already exist ==> skip')
   else
      -- load image and scale 
      local img = image.load(opt.img_root .. img_info[ii].file_path, 3, 'float')
      img = image.scale(img, opt.cnn_img_size, opt.cnn_img_size)

      -- Normalize img and view it as input form of batch size 1
      img = transform(img)
      img = img:view(1, table.unpack(img:size():totable()))

      if opt.gpuid >= 0 then img = img:cuda() end
      local output = protos.net:forward(img):squeeze(1)
      torch.save(save_path, output:float())
   end

   if ii % 100 then collectgarbage() end
end
end

-------------------------------------------------------------------------------
-- Main
-------------------------------------------------------------------------------
print('Loading trainval img from ', opt.img_trainval_info_file)
local trainval_info_file = utils.read_json(opt.img_trainval_info_file)
print('Loading test img from ', opt.img_test_info_file)
local test_info_file = utils.read_json(opt.img_test_info_file)

extract_feat(trainval_info_file, 'trainval')    -- trainval
extract_feat(test_info_file, 'test')        -- test
