require 'torch'
require 'xlua'
require 'optim'
require 'image'

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
      local input, target, targetU = getTest(t)
      local output = model:forward(input)
      output = output:mean(1)[1]:float()
      local output = normalizedToOriginal(output)
      originalToNormalized(output) -- to test assertions
      local err = criterion:forward(output, target)
      tMSE = tMSE + math.sqrt(err)
   end
   local rMSE = tMSE / nTesting
   -- timing
   time = sys.clock() - time
   time = time / nTesting
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   
   print('epoch: ' .. epoch .. ' + RMSE (test set) : ' .. rMSE )
   testLogger:add{['RMSE (test set)'] = rMSE}

   -- save/log current net
   local filename = paths.concat(opt.save, 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('<trainer> saving network to '..filename)
   print('')
   print('')
   local weight_l1 = model.modules[1].modules[1].weight:transpose(4,3):transpose(3,2):transpose(2,1):float()
   local filters_l1 = {}
   for i=1,weight_l1:size(1) do
      for j=1,weight_l1:size(2) do
	 table.insert(filters_l1, weight_l1[i][j])
      end
   end
   image.save('results/l1_' .. epoch .. '.jpg', image.toDisplayTensor{input=filters_l1,
								 padding=3})
   image.save('results/l1color_' .. epoch .. '.jpg', image.toDisplayTensor{input=weight_l1,
								 padding=3})
   torch.save(filename, model)
end
