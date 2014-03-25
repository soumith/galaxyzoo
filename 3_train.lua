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
      local inputs, _, targets = getBatch(batchSize)

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
	 -- estimate f
	 local outputs = model:forward(inputs)
	 outputs = outputs:float()
	 for kk = 1,outputs:size(1) do
	    outputs[kk] = normalizedToOriginal(outputs[kk])
	    originalToNormalized(outputs[kk]) -- to test assertions
	 end
	 local errs = criterion:forward(outputs, targets)
	 f = f + errs
	 -- sum individual RMSE
	 for i=1,outputs:size(1) do
	    tMSE = tMSE + math.sqrt((outputs[i]-targets[i]):pow(2):sum()/outputs:size(2))
	 end
	 -- estimate df/dW
	 local df_do = criterion:backward(outputs, targets)
	 model:backward(inputs, df_do:cuda())

	 -- normalize gradients and f(X)
	 gradParameters:div(batchSize)
	 f = f/batchSize

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

   local rMSE = tMSE / (epochSize)
   print('epoch: ' .. epoch .. ' + rMSE (train set) : ', rMSE)
   print('')
   print('')
   trainLogger:add{['% rMSE (train set)'] = rMSE}

end