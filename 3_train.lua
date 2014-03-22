require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> 4_train.lua'
print '==> defining some tools'

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

if model then
   parameters,gradParameters = model:getParameters()
end


print '==> configuring optimizer'

optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   learningRateDecay = 5e-7
}
optimMethod = optim.sgd

print '==> defining training procedure'
function train()
   for i=1,#dropouts do
      dropouts[i].train = true
   end
   -- epoch tracker
   epoch = epoch or 1
   -- local vars
   local time = sys.clock()
   local batchSize = opt.batchSize
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   local tMSE = 0
   for t = 1,epochSize,batchSize do
      -- disp progress
      if opt.progressBar then xlua.progress(t, epochSize) end

      -- create mini batch
      local inputs, targets = getBatch(batchSize)

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
	 -- get new parameters
	 if x ~= parameters then
	    parameters:copy(x)
	 end

	 -- reset gradients
	 gradParameters:zero()

	 -- f is the average of all criterions
	 local f = 0

	 -- evaluate function for complete mini batch
	 for i = 1,inputs:size(1) do
	    -- estimate f
	    local output = model:forward(inputs[i])
	    local err = criterion:forward(output, targets[i])
	    f = f + err
	    tMSE = tMSE + err

	    -- estimate df/dW
	    local df_do = criterion:backward(output, targets[i])
	    model:backward(inputs[i], df_do)
	 end

	 -- normalize gradients and f(X)
	 gradParameters:div(inputs:size(1))
	 f = f/inputs:size(1)

	 -- return f and df/dX
	 return f,gradParameters
      end

      -- optimize on current mini-batch
      optim.sgd(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / epochSize
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   tMSE = tMSE / (epochSize)
   local rMSE = math.sqrt(tMSE)
   print('epoch: ' .. epoch .. ' + rMSE (train set) : ', rMSE)
   trainLogger:add{['% rMSE (train set)'] = rMSE}

end