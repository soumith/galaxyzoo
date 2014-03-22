require 'torch'
require 'xlua'
require 'optim'

print '==> 5_test.lua'
print '==> defining test procedure'
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- test function
function test()
   epoch = epoch or 1
   for i=1,#dropouts do
      dropouts[i].train = false
   end
   -- local vars
   local time = sys.clock()
   -- test over test data
   print('==> testing on test set:')
   local tMSE = 0
   local tMSEu = 0
   for t = 1,nTesting do
      if opt.progressBar then xlua.progress(t, nTesting) end
      -- test sample
      local input, target, targetU = getTest(t)
      local output = model:forward(input)
      local outputU = normalizedToOriginal(output)
      local err = criterion:forward(output, target)
      tMSE = tMSE + err
      tMSEu = tMSEu + (outputU - targetU):pow(2):sum()/outputU:size(1)
   end
   tMSE = tMSE / nTesting
   tMSEu = tMSEu / nTesting
   local rMSE = math.sqrt(tMSE)
   local rMSEu = math.sqrt(tMSEu)
   -- timing
   time = sys.clock() - time
   time = time / nTesting
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   
   print('epoch: ' .. epoch .. ' + RMSE (test set) : ' .. rMSE .. ' + RMSEu : ' .. rMSEu)
   testLogger:add{['RMSE (test set)'] = rMSE, ['RMSEu (test set)'] = rMSEu}

   -- save/log current net
   local filename = paths.concat(opt.save, 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)
end
