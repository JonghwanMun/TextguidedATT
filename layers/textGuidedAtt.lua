require 'nn'
require 'nngraph'
require 'layers.Linear_wo_bias'

local textGuidedAtt = {}

function textGuidedAtt.create(k, ann_size, cap_size, emb_size, seq_per_img, backend, debug)

  print('\n----------------Attention LSTM parameter------------------------')
  print(string.format('k           : %d',k))
  print(string.format('ann_size    : %d',ann_size))
  print(string.format('cap_size    : %d',cap_size))
  print(string.format('emb_size    : %d',emb_size))
  print(string.format('seq_per_img : %d',seq_per_img))
  print(string.format('backend     : %s',backend))
  print(string.format('debug       : %s',debug))
  print('')

	if backend == 'cudnn' then
		require 'cudnn'
		backend = cudnn
	else
		backend = nn
	end

  -- there will be 2 outputs - attented_feat, alphas
  local outputs = {}

  -- there will be 2 inputs - img_feat(ann), cap_feat(key)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  --(nBatch,ann_size,k,k)-> (nBatch,k,k,ann_size)-> (nBatch,kk,ann_size)  (k is width & height of img feature map)
  local anns = nn.View(-1,ann_size):setNumInputDims(3)( nn.Transpose({2,3},{3,4})(inputs[1]) )
  local key  = inputs[2]         -- (nBatch*5, cap_size) this is key for attention
  local kk   = k*k
 
  -- compute context vector (attention model)
  local eAnns  = nn.View(kk,ann_size)(nn.Contiguous()(nn.Replicate(seq_per_img,2)(anns)))   -- (nBatch,5,kk,ann_size) -> (nBatch*5, kk, ann_size)
  local fAnns  = nn.View(ann_size)(eAnns)                                                   -- (nBatch*5,kk,ann_size) -> (nBatch*5*kk, ann_size)
  local pAnns  = nn.Linear(ann_size, emb_size)(fAnns):annotate{name='emb_ann_'}             -- (nBatch*5*kk, emb_size)
  local pKey   = nn.Linear_wo_bias(cap_size, emb_size)(key):annotate{name='emb_key_'}       -- (nBatch*5, emb_size)
  local eKey   = nn.View(emb_size)(nn.Contiguous()(nn.Replicate(kk,2)(pKey)))               -- (nBatch*5*kk, emb_size)
  local emb    = backend.Tanh()(nn.CAddTable()({pAnns, eKey}))                              -- (nBatch*5*kk, emb_size)
  local softmax_emb = backend.SoftMax()( nn.View(kk)(nn.Linear(emb_size,1)(emb)) )          -- (nBatch*5*kk, 1) -> (nBatch*5, kk)
  local alphas = nn.View(-1)(softmax_emb)                                                   -- (nBatch*5*kk)
  local ctxs   = nn.CMulTable()({fAnns, nn.View(ann_size)(nn.Contiguous()(nn.Replicate(ann_size,2)(alphas)))})  -- (nBatch*5*kk, ann_size)
  ctxs = nn.View(kk, ann_size)(ctxs)                                                        -- (nBatch*5,kk,ann_size)
  local ctx    = nn.Sum(2)(ctxs)                                                            -- (nBatch*5, ann_size)

  if debug then
    print('FCN Attention model is debug mode')
    table.insert(outputs, anns); table.insert(outputs,eAnns); table.insert(outputs,fAnns); table.insert(outputs,pAnns)
    table.insert(outputs,pKey); table.insert(outputs,eKey)
    table.insert(outputs,emb); table.insert(outputs,softmax_emb)
    table.insert(outputs,alphas); table.insert(outputs,ctxs); table.insert(outputs,ctx)
  else
    print('FCN Attention model is not debug mode')
    table.insert(outputs, ctx)
    table.insert(outputs, softmax_emb)
  end

  return nn.gModule(inputs, outputs)
end

return textGuidedAtt
