require 'torch'
require 'nn'
require 'nngraph'
require 'math'

-- exotic things
require 'loadcaffe'
require 'image'
local display = require 'display'
display.configure({hostname='141.223.65.127', port=2238})
local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

-- local imports
require 'layers.LanguageModel'
require 'layers.guidanceCaptionEncoder'
require 'misc.resAttLoader_kNN'
require 'misc.resAttLoader_raw'
local textGuidedAtt = require 'layers.textGuidedAtt'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

------------------------ Data input settings ------------------------ 
-- Loader input
cmd:option('-precomputed_feat','true','precomputed skip-thoughts vector and image feature is prepared?')
cmd:option('-img_feat_file','data/resnet101_conv_feat_448/','path to precomputed image features')
cmd:option('-cap_feat_file','data/skipthought/cocotalk_trainval_skipthought.h5','path to precomputed skip-thoughts vector for trainval captions')
cmd:option('-img_info_file','data/coco/cocotalk_trainval_img_info.json','path to the json containing the trainval image information')
cmd:option('-cap_h5_file','data/coco/cocotalk_cap_label.h5','path to the h5 file containing the caption label and infomation')
cmd:option('-kNN_cap_json_file','data/coco/10NN_cap_valtrainall_cider.json','path to the json file containing NN caps for train/val')

-- resNet input
cmd:option('-cnn_model','model/resNet/resnet-101.t7','path to cnn model')

-- Sentence embedding input
cmd:option('-uni_gru_path','model/skipthought/uni_gru_params.t7','path to skip-thoughts vector GRU model')
cmd:option('-uni_gru_word2vec_path','model/skipthought/coco_cap_uni_gru_word2vec.t7','path to skip-thoughts vector word embedding model')

-- Model Finetuning options
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-ft_continue', 1,'whether maintain the epoch and iteration of checkpoint or not')

-- Model settings
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive.')
cmd:option('-cap_feat_dim',2400,'dimension of the skipthought feature from caption')
cmd:option('-conv_feat_dim',2048,'dimension of the cnn feature from image')
cmd:option('-emb_dim',512,'dimension of embedding space of img and cap for attention')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-cnn_img_size', 512, 'input image size for cnn')
cmd:option('-sharedE', true, 'transposed weight sharing in word embedding layer and word prediction layer')

-- Optimization: General
cmd:option('-max_epoch', 100, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-optim_alpha',0.8,'alpha for adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the Attention Model
cmd:option('-reg_alpha_type','entropy','weight of alpha regularization')
cmd:option('-reg_alpha_lambda',0.002,'weight of alpha regularization')

-- Optimization: for the Language Model
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 10, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 3, 'every how many iterations thereafter to drop LR?')
cmd:option('-lr_decay_rate', 0.8, 'every how many iterations thereafter to drop LR by half?')

-- Optimization: for the FCN and sentence embedding
cmd:option('-cnn_finetune_after', -1, 'starting point for finetuning CNN, -1 means no finetuning')
cmd:option('-cnn_learning_rate', 1e-5, 'learning rate for CNN')
cmd:option('-cnn_weight_decay', 5e-4, 'L2 normalization ')

-- Optimization: for the scheduled sampling on input word of LSTM
cmd:option('-scheduled_sampling', 1, 'on/off of scheduled sampling')
cmd:option('-start_scheduled_sampling', 10, 'when starting to sampling for input of LSTM')
cmd:option('-scheduled_sampling_type', 'linear', 'function type of scheduled sampling, linear|i_sigmoid (inverse sigmoid)')
cmd:option('-scheduled_sampling_k', 15, 'decay rate(speed) of scheduled sampling probability for i_sigmoid')
cmd:option('-scheduled_sampling_start_point', 0.75, 'initial rate(speed) of scheduled sampling probability for linear')
cmd:option('-scheduled_sampling_end_point', 0.75, 'final rate(speed) of scheduled sampling probability for linear')
cmd:option('-scheduled_sampling_decay_rate', 0.05, 'decay rate(speed) of scheduled sampling probability for linear')
cmd:option('-scheduled_sampling_decay_every', 1, 'decay rate(speed) of scheduled sampling probability')

-- NN options
cmd:option('-use_NN', 1, '(-1 : train with only GT) and (>0 means using kNN, guidance captions, after the epoch)')
cmd:option('-NN_prob_decay_rate', 0.05, 'decay rate of probability using guidance captions')
cmd:option('-NN_prob_decay_every', 1, 'decay rate per how many epoch')
cmd:option('-NN_prob_start_point', 1.0, 'start probability of using guidance caption')
cmd:option('-NN_prob_end_point', 1.0, 'end probability of using guidance caption')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-checkpoint_path', './trained_model/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 10, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-img_root', 'data/MSCOCO/')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-debug', false, 'Debug mode?')
cmd:option('-every_vis', 300, 'how often visualize attention')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
print('options are as follows :')
print(opt)

-- for gpu
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader 
if opt.precomputed_feat == 'true' then
  print('=====> Using precomputed caption encoded feature!!')
  loader = resAttLoader_kNN{img_info_file=opt.img_info_file, img_feat_file=opt.img_feat_file,
                            cap_h5_file=opt.cap_h5_file, cap_feat_file=opt.cap_feat_file, 
                            kNN_cap_json_file=opt.kNN_cap_json_file, on_gpu=(opt.gpuid>=0)}
else 
  loader = resAttLoader_raw{img_info_file=opt.img_info_file, cap_h5_file=opt.cap_h5_file,
                            kNN_cap_json_file=opt.kNN_cap_json_file, on_gpu=(opt.gpuid>=0),
                            cnn_img_size=448, img_root=opt.img_root}
end

-------------------------------------------------------------------------------
-- Localization of math function
-------------------------------------------------------------------------------
mathceil  = math.ceil; mathfloor = math.floor
mathmax   = math.max; mathmin   = math.min 
mathexp   = math.exp; mathpow   = math.pow 

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local timer = torch.Timer()
local protos = {}
local epoch = 1
local iter = 1
local kk = 14  -- (kk*kk) is spatial dimensions for residual feature map

if string.len(opt.start_from) > 0 then  -- finetuning the model
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  if opt.ft_continue > 0 then
    print('Fintuning continue from before model, meaning epoch and iteration is maintained')
    epoch = loaded_checkpoint.epoch +1
    local every_epoch = mathceil(113286 / opt.batch_size)
    if opt.batch_size ~= loaded_checkpoint.opt.batch_size then
      iter = mathceil(113286 / opt.batch_size) * (epoch-1) + 1
    else
      iter = loaded_checkpoint.iter + 1
    end
  end

  protos = loaded_checkpoint.protos
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually

  ----------------------------------------------------------------------------
  -- Unsanitize gradient for each model
  ----------------------------------------------------------------------------
  -- FCN && Sentence Encoder
  if opt.precomputed_feat ~= 'true' then 
    if protos.fcn ~= nil then
      net_utils.unsanitize_gradients(protos.fcn)
    else
      protos.fcn = torch.load(opt.cnn_model)
      protos.fcn:remove(11)
      protos.fcn:remove(10)
      protos.fcn:remove(9)
    end

    if protos.senEncoder ~= nil then
       protos.guidanceCaptionEncoder = protos.senEncoder
    end

    if protos.guidanceCaptionEncoder ~= nil then
      print('===load model===> No precomputed caption feat, but existing guidanceCaptionEncoder model!!')
      local se_modules = protos.guidanceCaptionEncoder:getModulesList()
      for k,v in pairs(se_modules) do net_utils.unsanitize_gradients(v) end
    else
      print('===load model===> No precomputed caption feat and guidanceCaptionEncoder model!!')
      print('Load skip-thought vector data')
      local uparams = torch.load(opt.uni_gru_path)
      local utables = torch.load(opt.uni_gru_word2vec_path)
      local seOpt = {}
      seOpt.backend = 'nn'   -- cudnn may not work
      seOpt.vocab_size = loader:getVocabSize()
      seOpt.seq_length = loader:getSeqLength()
      print('Sentence encoder model option is as follows :')
      print(seOpt)
      protos.guidanceCaptionEncoder = nn.guidanceCaptionEncoder(uparams, utables, seOpt)
    end
  end

  -- Attention model
  net_utils.unsanitize_gradients(protos.att)

  -- Image Embedding Model
  net_utils.unsanitize_gradients(protos.imgEmb)

  -- Language Model
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end

  if protos.lm.scheduled_sampling ~= opt.scheduled_sampling then protos.lm.scheduled_sampling = opt.scheduled_sampling end
  -- sharing parameter
  if loaded_checkpoint.opt.sharedE == true then
    for idxNode, node in ipairs(protos.lm.core.forwardnodes) do
      if node.data.annotations.name == 'decoder' then
        node.data.module.gradWeight:set( protos.lm.lookup_table.gradWeight )
        print('gradient sharing done for \'decoder\'')
        break
      end
    end
  end
  ----------------------------------------------------------------------------
else -- create protos from scratch
  -- attaching FCN network (here, residual network)
  -- attaching Sentence Embedding network
  if opt.precomputed_feat ~= 'true' then 
    protos.fcn = torch.load(opt.cnn_model)
    protos.fcn:remove(11)
    protos.fcn:remove(10)
    protos.fcn:remove(9)

    print('Load skip-thought vector data')
    local uparams = torch.load(opt.uni_gru_path)
    local utables = torch.load(opt.uni_gru_word2vec_path)
    local seOpt = {}
    seOpt.backend = 'nn'   -- cudnn may not work
    seOpt.vocab_size = loader:getVocabSize()
    seOpt.seq_length = loader:getSeqLength()
    print('Sentence encoder model option is as follows :')
    print(seOpt)
    protos.guidanceCaptionEncoder = nn.guidanceCaptionEncoder(uparams, utables, seOpt)
  end

  -- attaching Attention Model based on caption
  protos.att = textGuidedAtt.create(kk, opt.conv_feat_dim, opt.cap_feat_dim, opt.emb_dim, opt.seq_per_img, opt.backend, opt.debug)

  -- attaching Image Embedding layer
  protos.imgEmb = nn.Sequential()
  protos.imgEmb:add(nn.Linear(opt.conv_feat_dim, opt.input_encoding_size))
  if opt.backend == 'cudnn' then protos.imgEmb:add(cudnn.ReLU(true))
  else protos.imgEmb:add(nn.ReLU(true)) end

  -- Attaching Language Model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.scheduled_sampling = opt.scheduled_sampling
  lmOpt.eta = 1.0
  lmOpt.backend = 'nn'  -- cudnn may not work
  lmOpt.sharedE = opt.sharedE
  print('language model option is as follows :')
  print(lmOpt)
  protos.lm = nn.LanguageModel(lmOpt)

  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
end
--------------------------------------------------------------------------------------------------

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

--------------------------------------------------------------------------------------------------
-- flatten and prepare all model parameters to a single vector. 
local fcn_params, grad_fcn_params 
local sce_params, grad_sce_params 
if opt.precomputed_feat ~= 'true' then 
  fcn_params, grad_fcn_params = protos.fcn:getParameters()
  sce_params, grad_sce_params = protos.guidanceCaptionEncoder:getParameters()
end
local att_params, grad_att_params = protos.att:getParameters()
local imgEmb_params, grad_imgEmb_params = protos.imgEmb:getParameters()
local lm_params, grad_lm_params = protos.lm:getParameters()
if opt.precomputed_feat ~= 'true' then 
  print('total number of parameters in FCN    : ', fcn_params:nElement())
  print('total number of parameters in SE     : ', sce_params:nElement())
end
print('total number of parameters in ATT    : ', att_params:nElement())
print('total number of parameters in IMGEMB : ', imgEmb_params:nElement())
print('total number of parameters in LM     : ', lm_params:nElement())
assert(lm_params:nElement() == grad_lm_params:nElement())

--------------------------------------------------------------------------------------------------
-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files


local thin_se, thin_fcn
if opt.precomputed_feat ~= 'true' then 
  -- sanitize fcn
  thin_fcn = protos.fcn:clone('weight', 'bias')
  net_utils.sanitize_gradients(thin_fcn)

  -- sanitize sentence embedding
  thin_se = protos.guidanceCaptionEncoder:clone()
  thin_se.core:share(protos.guidanceCaptionEncoder.core, 'weight', 'bias')
  thin_se.lookup_table:share(protos.guidanceCaptionEncoder.lookup_table, 'weight', 'bias')
  local se_modules = thin_se:getModulesList()
  for k,v in pairs(se_modules) do net_utils.sanitize_gradients(v) end
end

-- sanitize attention
local thin_att = protos.att:clone('weight','bias')
net_utils.sanitize_gradients(thin_att)

-- sanitize image embedding
local thin_imgEmb = protos.imgEmb:clone('weight', 'bias')
net_utils.sanitize_gradients(thin_imgEmb)

-- sanitize language model
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') 
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

--------------------------------------------------------------------------------------------------
-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
if opt.precomputed_feat ~= 'true' then protos.guidanceCaptionEncoder:createClones() end
protos.lm:createClones()

collectgarbage()

-------------------------------------------------------------------------------
-- Visualizing attention alpha map function
-------------------------------------------------------------------------------
-- Inputs
-- file_path : image file path
-- alpha : attention map (nBatchs*5, nROIs)
-- sent  : guiding sentence
-- idx   : image idx (= batch number)
local function vis_attention(file_path, alpha, sent, idx, seq_per_img, base)

  -- display original image
  local img = cv.imread{opt.img_root .. file_path, cv.IMREAD_COLOR}:float()
  img = cv.cvtColor{img, nil, cv.COLOR_BGR2RGB}
  display.image(image.scale(img:permute(3,1,2), '256x256'), {win=base+0, title='Original image'})

  for ic =0,seq_per_img-1 do
    -- display attention weight of each word
    local map = image.scale(alpha[{ idx+ic,{} }]:view(kk,kk), 256,256, 'simple')
    display.image(map, {win=base+1+ic, title=string.format('%dth sentence', ic+1)} )

    -- plot weight of three words
    local config = { title=string.format('%dth : %s', ic+1,sent[ic+1]), labels={'region', 'weight'}, ylabel='weight', win=base+17+ic }
    local vis_plot = torch.range(1,kk*kk)
    vis_plot = torch.cat(vis_plot, alpha[{idx+ic,{}}], 2)
    display.plot(vis_plot, config)
  end
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.lm:evaluate(); protos.imgEmb:evaluate(); protos.att:evaluate(); 
  if opt.precomputed_feat ~= 'true' then protos.guidanceCaptionEncoder:evaluate(); protos.fcn:evaluate() end

  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local val_loss = {}
  local predictions = {}
  local vocab = loader:getVocab()

  print('\n---------------------- Evaluation ------------------------')
  while true do

    -- get batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split..'NN', seq_per_img = opt.seq_per_img}

    -- In val/test mode, We actually need only one guidance caption for each image in evaluation
    -- but, network requires (seq_per_img) captions for each image
    -- So, redundant calculation conducted...
    local img_feat, cap_feat 
    if opt.precomputed_feat == 'true' then
      img_feat = data.img_feat
      cap_feat = data.cap_feat
      n = n + data.img_feat:size(1)
    else
      img_feat = protos.fcn:forward(data.imgs)
      cap_feat = protos.guidanceCaptionEncoder:forward({data.nn_labels, data.nn_label_length})
      n = n + data.imgs:size(1)
    end
    local attended_feats, alphas = unpack(protos.att:forward({img_feat, cap_feat}))
    local emb_feats = protos.imgEmb:forward(attended_feats)
    local logprobs = protos.lm:forward({emb_feats, data.labels})
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -----------------------------------------------------------------------------
    -- Visualizing Attention
    -----------------------------------------------------------------------------
    if n % opt.every_vis * opt.batch_size == 0 then
      print('*************************** Visualizing Attention *********************************')
      local idx = 1  -- index of image for attention
      local sent = net_utils.decode_sequence(vocab, data.nn_labels[{ {},{idx,idx} }]) -- idx to word
      vis_attention(data.infos[idx]['file_path'], alphas:float(), sent, idx, 1, 0)
    end
    -----------------------------------------------------------------------------

    -- forward the model to also get generated samples for each image
    local seq, probs = protos.lm:sample(emb_feats)
    local sents = net_utils.decode_sequence(vocab, seq) -- idx to word
    print(string.format('evaled  : %s', sents[1]))
    for k=1,#sents/opt.seq_per_img do
      local entry = {image_id = data.infos[k].id, caption = sents[(k-1)*opt.seq_per_img+1]}
      table.insert(predictions, entry)
      if verbose and n%100 == 0 then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = mathmin(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
    print('-----------------------------------------------------------------------------------')
  end
  table.insert(val_loss, loss_sum/loss_evals)

  local lang_stats = {}
  if evalopt.language_eval == 1 then
    table.insert(lang_stats, net_utils.language_eval(predictions, opt.id))
    print('-----------------------------------Score for NN Caption-----------------------------------------')
  end

  return val_loss, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  protos.lm:training(); protos.imgEmb:training(); protos.att:training(); 
  grad_lm_params:zero(); grad_imgEmb_params:zero(); grad_att_params:zero(); 
  if opt.precomputed_feat ~= 'true' then 
    protos.guidanceCaptionEncoder:training(); protos.fcn:training()
    grad_sce_params:zero(); grad_fcn_params:zero()
  end

  local vocab = loader:getVocab()

  -----------------------------------------------------------------------------
  -- Using NN or GT?
  -----------------------------------------------------------------------------
  local train_split = 'train'
  if opt.use_NN >= 0 and epoch > opt.use_NN then
    local NN_prob = opt.NN_prob_start_point - opt.NN_prob_decay_rate
                      + (opt.NN_prob_decay_rate * mathceil((epoch-opt.use_NN) / opt.NN_prob_decay_every)) 
    NN_prob = mathmin(NN_prob, opt.NN_prob_end_point)
    local s_prob = torch.FloatTensor{NN_prob, 1-NN_prob}
    local sampling = torch.multinomial(s_prob, 1, true)
    if sampling[1] == 1 then 
      print(string.format('===> training based on guidance caption (%.2f)', NN_prob))
      train_split = 'trainNN'
    else
      print(string.format('===> training based on GT caption (%.2f)', NN_prob))
    end
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  local dt = timer:time().real
  local data = loader:getBatch{batch_size = opt.batch_size, split = train_split, seq_per_img = opt.seq_per_img}
  local data_load_time = timer:time().real - dt

  local ft = timer:time().real
  local img_feat, cap_feat 
  if opt.precomputed_feat == 'true' then
    img_feat = data.img_feat
    cap_feat = data.cap_feat
  else
    img_feat = protos.fcn:forward(data.imgs)
    if data.nn_labels == nil then
      cap_feat = protos.guidanceCaptionEncoder:forward({data.labels, data.label_length}) -- (nBatch*5, 2400) 
    else
      cap_feat = protos.guidanceCaptionEncoder:forward({data.nn_labels, data.nn_label_length}) -- (nBatch*5, 2400) 
    end
  end
  local attended_feats, alphas = unpack(protos.att:forward({img_feat, cap_feat}))
  local emb_feats = protos.imgEmb:forward(attended_feats)                  -- input to LM
  local logprobs = protos.lm:forward({emb_feats, data.labels})

  local loss = protos.crit:forward(logprobs, data.labels)
  local forward_time = timer:time().real - ft

  -----------------------------------------------------------------------------
  -- Visualizing Attention
  -----------------------------------------------------------------------------
  if iter % opt.every_vis == 0 then
    print('============================> Visualizing Attention in training')
    local vt = timer:time().real
    local idx = 1  -- index of image for attention
    local sent -- idx to word
    if data.nn_labels == nil then sent = net_utils.decode_sequence(vocab, data.labels[{ {},{idx,idx+4} }]) 
    else sent = net_utils.decode_sequence(vocab, data.nn_labels[{ {},{idx,idx+4} }]) end
    vis_attention(data.infos[idx]['file_path'], alphas:float(), sent, idx, opt.seq_per_img, 0)
    print(string.format('time to vis : %.2es', timer:time().real - vt))
  end

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  local bt = timer:time().real
  local grad_alpha
  if opt.reg_alpha_type == 'None' then
    grad_alpha = torch.FloatTensor(opt.batch_size * opt.seq_per_img * 10 * 10):typeAs(alphas)
  elseif opt.reg_alpha_type == 'l2norm' then
    grad_alpha = torch.pow(alphas, 2):sum(2):pow(1/2-1):view(-1):repeatTensor(10*10,1):t():cmul(alphas)  -- l2 regularization
  else
    grad_alpha = torch.log(alphas) + 1   -- entropy regularization term
  end
  grad_alpha = opt.reg_alpha_lambda * grad_alpha

  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  local demb_feats, ddummy = unpack(protos.lm:backward({emb_feats, data.labels}, dlogprobs))
  local dattended_feats = protos.imgEmb:backward(attended_feats, demb_feats)
  local dimg_feat, dcap_feat = unpack(protos.att:backward({img_feat, cap_feat}, {dattended_feats, grad_alpha}))
  if opt.cnn_finetune_after >= 0 and epoch > opt.cnn_finetune_after then
    if opt.precomputed_feat ~= 'true' then 
      local dimages = protos.fcn:backward(data.imgs, dimg_feat)
      if data.nn_labels == nil then local dlables = protos.guidanceCaptionEncoder:backward({data.labels, data.label_length}, dcap_feat)
      else local dlables = protos.guidanceCaptionEncoder:backward({data.nn_labels, data.nn_label_length}, dcap_feat) end
    end
  end
  local backward_time = timer:time().real - bt

  -- clip gradients
  grad_lm_params:clamp(-opt.grad_clip, opt.grad_clip)
  grad_imgEmb_params:clamp(-opt.grad_clip, opt.grad_clip)
  grad_att_params:clamp(-opt.grad_clip, opt.grad_clip)

  if opt.cnn_finetune_after >= 0 and epoch > opt.cnn_finetune_after then
    -- apply L2 regularization
    if opt.precomputed_feat ~= 'true' then 
      if opt.cnn_weight_decay > 0 then
        grad_fcn_params:add(opt.cnn_weight_decay, fcn_params)
      end
      grad_fcn_params:clamp(-opt.grad_clip, opt.grad_clip)
      grad_sce_params:clamp(-opt.grad_clip, opt.grad_clip)
    end
  end
  -----------------------------------------------------------------------------
  print(string.format('Elapsed time : data_load (%.4fs) | forward (%.4fs) | backward (%.4fs)', data_load_time, forward_time, backward_time))
  if loss >= 10 then print(string.format('Oops!! Loss is bigger than 10 (%.3f) (%s)',loss,data.infos[1]['file_path'])) end

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local lm_optim_state, imgEmb_optim_state, att_optim_state, gce_optim_state, fcn_optim_state = {}, {}, {}, {}, {}
local loss_history, val_loss_nn_history, eta_history = {}, {}, {} 
local val_lang_stats_history = {}
local best_score
local every_epoch = mathceil(loader:getTrainNum() / opt.batch_size)
local ek = opt.scheduled_sampling_k
local eta = 1.0

protos.lm:set_vocab(loader:getVocab())
while true do
  print('\n--------------------------------------------------------------------------------')
  print(string.format('epoch %d iter %d', epoch, iter))
  -- update eta (probability) for scheduled sampling of input word on LSTM
  if opt.scheduled_sampling >= 0 then
    if epoch < opt.start_scheduled_sampling then
      protos.lm:update_eta(eta)
    else
      if opt.scheduled_sampling_type == 'i_sigmoid' then
        eta = ek / (ek + mathexp( (epoch-opt.start_scheduled_sampling)/ek ))
      elseif opt.scheduled_sampling_type == 'linear' then
        eta = -opt.scheduled_sampling_decay_rate * mathceil((epoch-opt.start_scheduled_sampling) / opt.scheduled_sampling_decay_every )
                + opt.scheduled_sampling_start_point+opt.scheduled_sampling_decay_rate
      else
        print(string.format('Unrecognized scheduled sampling type (%s)', opt.scheduled_sampling_type))
      end
      eta = mathmax(eta, opt.scheduled_sampling_end_point)
      protos.lm:update_eta(eta)
    end
  end

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then table.insert(loss_history, losses.total_loss) end

  -----------------------------------------------------------------------------
  -- save checkpoint once in a while (or on final iteration)
  if  iter % every_epoch == 0 or iter == opt.max_epoch*every_epoch then
    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats 
    val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use, language_eval = opt.language_eval})
    print('validation loss: ', val_loss)
    print(lang_stats)
    print('eta : ', eta)
    eta_history[epoch] = eta
    val_loss_nn_history[epoch] = val_loss[1]
    if lang_stats then
      val_lang_stats_history[epoch] = lang_stats[1]
    end

    -- write a (thin) json report
    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. tostring(epoch))
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.epoch = epoch
    checkpoint.eta_history = eta_history 
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_nn_history = val_loss_nn_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    local save_protos = {}
    if opt.precomputed_feat ~= 'true' then 
      save_protos.fcn        = thin_fcn
      save_protos.guidanceCaptionEncoder = thin_se 
    end
    save_protos.att        = thin_att
    save_protos.imgEmb     = thin_imgEmb
    save_protos.lm         = thin_lm     -- these are shared clones, and point to correct param storage
    checkpoint.protos = save_protos
    -- also include the vocabulary mapping so that we can use the checkpoint 
    -- alone to run on arbitrary images without the data loader
    checkpoint.vocab = loader:getVocab()
    torch.save(checkpoint_path .. '.t7', checkpoint)
    print('wrote checkpoint to ' .. checkpoint_path .. '.t7')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats[1]['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        torch.save(checkpoint_path .. 'best_score.t7', checkpoint)
        print('wrote best score checkpoint to ' .. checkpoint_path .. 'best_score.t7')
      end
    end
    if iter ~= 0 then epoch = epoch + 1 end
  end
  --------------------------------------------------------------------------------------------------------------

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = mathpow(opt.lr_decay_rate, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    if opt.cnn_finetune_after >= 0 and epoch > opt.cnn_finetune_after then
      cnn_learning_rate = cnn_learning_rate * decay_factor
    end
  end
  print(string.format('Loss : %.2f\t | eta : %.3f\t | lr(lm) : %.2e\t | lr(cnn) : %.2e', losses.total_loss, eta, learning_rate, cnn_learning_rate))

  -- perform a parameter update
  adam(lm_params, grad_lm_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, lm_optim_state)
  adam(imgEmb_params, grad_imgEmb_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, imgEmb_optim_state)
  adam(att_params, grad_att_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, att_optim_state)
  if opt.cnn_finetune_after >= 0 and epoch > opt.cnn_finetune_after then
    if opt.precomputed_feat ~= 'true' then
      adam(fcn_params, grad_fcn_params, cnn_learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, fcn_optim_state)
      adam(sce_params, grad_sce_params, cnn_learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, gce_optim_state)
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_epoch> 0 and epoch >= opt.max_epoch then break end -- stopping criterion
end
