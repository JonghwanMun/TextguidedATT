require 'torch'
require 'nn'
require 'nngraph'

-- exotic things
require 'hdf5'

-- local imports
require 'layers.LanguageModel'
require 'layers.guidanceCaptionEncoder'
require 'layers.textGuideAtt'
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
cmd:option('-img_info_file','data/coco/cocotalk_trainval_img_info.json','path to the image information')
cmd:option('-label_file','data/coco/cocotalk_cap_label.h5','path to the caption labels')
cmd:option('-output_h5','data/skipthought/cocotalk_trainval_skipthought.h5','path to the h5file containing the caption features')

-- Sentence Embedding model data
cmd:option('-uni_gru_path','model/skipthought/uni_gru_params.t7','path to skip-thoughts vector GRU model')
cmd:option('-uni_gru_word2vec_path','model/skipthought/coco_cap_uni_gru_word2vec.t7','path to skip-thoughts vector word embedding model')

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

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load caption labels and its information 
-------------------------------------------------------------------------------
print( string.format('Load vocabulary from %s', opt.img_info_file) )
print( string.format('Load label and its information from %s', opt.label_file) )
local vocab          = utils.read_json(opt.img_info_file).ix_to_word
local labelInfo      = hdf5.open(opt.label_file, 'r')
local labels         = labelInfo:read('/labels'):all()
local label_length   = labelInfo:read('/label_length'):all()
local label_start_ix = labelInfo:read('/label_start_ix'):all()
local label_end_ix   = labelInfo:read('/label_end_ix'):all()
local nCaptions  = labels:size(1)
local seq_size   = labels:size(2)

-------------------------------------------------------------------------------
-- Load & build the sentence embedding model
-------------------------------------------------------------------------------
print('Load skip-thought vector data')
local uparams = torch.load(opt.uni_gru_path)
local utables = torch.load(opt.uni_gru_word2vec_path)
local seOpt = {}
seOpt.backend = 'nn'   -- cudnn may not work
seOpt.vocab_size = utils.count_keys(vocab)
seOpt.seq_length = seq_size
print('Sentence encoder model option is as follows :')
print(seOpt)
protos = {}
protos.senEncoder = nn.sentenceEncoder(uparams, utables, seOpt)

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

protos.senEncoder:createClones()
protos.senEncoder:evaluate() 

collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Main
-------------------------------------------------------------------------------

local file = hdf5.open(opt.output_h5, 'w')
local feat = torch.FloatTensor(nCaptions,2400)
local timer = torch.Timer()

for i=1,nCaptions do
  local l  = labels[i]:view(1,-1):transpose(1,2)
  local ll = label_length[{{i}}]

	local cap_feat = protos.senEncoder:forward({l, ll})
	feat[{ i,{} }] = cap_feat:float()

	if i%100 == 0 then
    print(cap_feat[{ {1},{1,10} }])
    print( string.format('%d/%d (%.2f%% done) (%.2fs)', i,nCaptions,i*100.0/nCaptions, timer:time().real) )
    collectgarbage() 
	end
end

-- save caption features to h5 file
file:write('feature', feat)
print('write to', opt.output_h5)

-- close the opened hdf5 files
file:close()
labelInfo:close()
