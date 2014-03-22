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
   for t = 1,nTesting do
      if opt.progressBar then xlua.progress(t, nTesting) end
      -- test sample
      local input, target = getTest(t)
      local output = model:forward(input)
      local err = criterion:forward(output, target)
      tMSE = tMSE + err
   end
   tMSE = tMSE / nTesting
   local rMSE = math.sqrt(tMSE)
   -- timing
   time = sys.clock() - time
   time = time / nTesting
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   
   print('epoch: ' .. epoch .. ' + RMSE (test set) : ' .. rMSE)
   testLogger:add{['RMSE (test set)'] = rMSE}

   -- save/log current net
   local filename = paths.concat(opt.save, 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)
end
