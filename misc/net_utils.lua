require 'image'
require 'math'
local utils = require 'misc.utils'
local net_utils = {}

function net_utils.compute_norm_param_grad(layer_params, layer_grad)
  local norm_grad = 0
  local norm_param = 0
  for k, v in pairs(layer_params) do
  	local norm_loc = v:norm()
  	norm_param = norm_param + norm_loc * norm_loc
  end
  for k, v in pairs(layer_grad) do
  	local norm_loc = v:norm()
  	norm_grad = norm_grad + norm_loc * norm_loc
  end
  return torch.sqrt(norm_param), torch.sqrt(norm_grad)
end

-- take a raw CNN from Caffe and perform surgery.
-- Input cnn: FCN (mil)
function net_utils.build_vgg_tune(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local tune_start = utils.getopt(opt, 'tune_start', 11)  -- 11 (from conv3)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_notune = nn.Sequential()
  for i = 1, tune_start-1 do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_notune:add(layer)
  end
	
  local cnn_tune = nn.Sequential()
  for i = tune_start, layer_num do
    local layer = cnn:get(i)
    cnn_tune:add(layer)
	end

  return cnn_notune, cnn_tune
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  if seq:dim() == 1 then seq:resize(seq:size(1), 1) end
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  local out_len = torch.LongTensor(N)
  for i=1,N do
    local sent_finish = false
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word or word == 0 then 
        out_len[i] = j
        sent_finish = true
        break 
      end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
    if sent_finish == false then out_len[i] = D end
  end
  return out, out_len
end

function net_utils.decode_sequence_with_prob(ix_to_word, seq, logProb)
  local D,N = seq:size(1), seq:size(2)
  local out_cap = {}
  local out_cap_prob = {}

  for i=1,N do
    local txt_cap = ''
    local txt_cap_prob = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local prob = logProb[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then 
				txt_cap = txt_cap .. ' ' 
				txt_cap_prob = txt_cap_prob .. ' '
			end
      txt_cap = txt_cap .. word
      txt_cap_prob = txt_cap_prob .. word .. '(' .. string.format('%.2f',prob) .. ')'
    end
    table.insert(out_cap, txt_cap)
    table.insert(out_cap_prob, txt_cap_prob)
  end
  return out_cap, out_cap_prob
end

function net_utils.decode_one_sequence(ix_to_word, seq, logProb)
  local out_cap = {}
  local out_cap_prob = {}
  local D = seq:size(1)

  for i=1,D do
    local ix = seq[i]
    local word = ix_to_word[tostring(ix)]
    table.insert(out_cap, word)
    table.insert(out_cap_prob, logProb[i])
  end

  return out_cap, out_cap_prob
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh ' .. id) 
  local result_struct = utils.read_json('coco-caption/' .. id .. '_out.json') 
  return result_struct
end

return net_utils
