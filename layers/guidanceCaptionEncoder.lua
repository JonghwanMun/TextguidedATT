local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local GRU = require 'layers.skipthoughts_GRU'

-------------------------------------------------------------------------------
-- Guidance Caption Encoder Model
-- This is based on skip-thought vector
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.guidanceCaptionEncoder','nn.Module')

function layer:__init(uparams, utables, opt)
  parent.__init(self)

  -- options for GRU core network
	self.backend = utils.getopt(opt, 'backend', 'nn')
  self.seq_length = utils.getopt(opt, 'seq_length')     -- TODO, where from?
	self.rnn_size = uparams.Ux:size(1)
  -- options for word embedding Model
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.word_dim = utables:size(2)

  -- create the core lstm network. note +1 for both the START and END tokens
  self.core = GRU.create(uparams, self.backend)
  --self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.word_dim)
  self.lookup_table = nn.LookupTable(9567 + 1, self.word_dim)
  self.lookup_table.weight:copy(utables)
  self:_createInitState(1) -- will be lazily resized later during forward passes

	print('\n----------Guidance Caption Encoder initialized')
	print(string.format('Guidance Caption Encoder backend    : %s (sentence vector size)',self.backend))
	print(string.format('Guidance Caption Encoder rnn size   : %d ',self.rnn_size))
	print(string.format('Guidance Caption Encoder word dim   : %d ',self.word_dim))
	print(string.format('Guidance Caption Encoder seq len    : %d ',self.seq_length))
	print(string.format('Guidance Caption Encoder vocab size : %d ',self.vocab_size))
	print('')
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the GRU
  if self.init_state then
    if self.init_state:size(1) ~= batch_size then
      self.init_state:resize(batch_size, self.rnn_size):zero() -- expand the memory
    end
  else
    self.init_state = torch.zeros(batch_size, self.rnn_size)
  end
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the Guidance Caption Encoder Model')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,self.seq_length do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
		collectgarbage()
  end
end

function layer:getRnnSize()
  return self.rnn_size
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end

--[[
input is a tuple of:
1. torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length
2. torch.LongTensor of size Nx1, label length for each sentence
--]]
function layer:updateOutput(input)
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  local seq = input[1]
  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
	local label_length = input[2]
	local len_min = torch.min(label_length)
  self.output:resize(batch_size, self.rnn_size):zero()
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency

  for t=1,self.seq_length do
    local can_skip = false
    local xt 
    local it = seq[t]:clone()
    if torch.sum(it) == 0 then
      can_skip = true
    end

    if not can_skip then
      -- seq may contain zeros as null tokens, make sure we take them out to end token
      it[torch.eq(it,0)] = self.vocab_size+1

      self.lookup_tables_inputs[t] = it       -- save input seq for gradient calculation
      xt = self.lookup_tables[t]:forward(it)  -- embedding words 
      self.inputs[t] = {xt,self.state[t-1]}   -- word, prev_h
      -- forward the network
      local out = self.clones[t]:forward(self.inputs[t])
      self.state[t] = out 
			-- save sentence vector for ended sentence
			if t >= len_min then
				for k = 1, batch_size do
					if label_length[k] == t then self.output[k] = out[k] end
				end
			end
      self.tmax = t
    end
  end

  return self.output
end

--[[
gradOutput is an (batch_size, rnn_size) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
	local seq = input[1]
	local batch_size = input[1]:size(2)
	local label_length = input[2]
	local min_len = torch.min(label_length)

	self.gradInput:resizeAs(seq:typeAs(self.gradInput)):zero()

  -- go backwards and lets compute gradients
  local dstate = {[self.tmax+1] = self.init_state} -- this works when init_state is all zeros
	local doutput = torch.Tensor(batch_size, self.rnn_size):typeAs(gradOutput)
  for t=self.tmax,1,-1 do
		doutput:copy(dstate[t+1])
    if t >= min_len then
      for k = 1, batch_size do
        if label_length[k] == t then doutput[k] = gradOutput[k] end
      end
    end
    local dcore = self.clones[t]:backward(self.inputs[t], doutput)   -- dcore = {dx, dprev_h}
    local dwords = self.lookup_tables[t]:backward(self.lookup_tables_inputs[t], dcore[1])

    dstate[t] = dcore[2]
    self.gradInput[t] = dwords
  end

  return self.gradInput
end
