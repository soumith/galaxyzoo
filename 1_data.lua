require 'csvigo'
require 'xlua'
require 'paths'
require 'nn'
require 'image'

os.execute('mkdir -p cache')
dofile('1_datafunctions.lua')
print '==> 1_data.lua'
print('==> Loading data')
if not paths.filep('cache/data.t7') then
   local csvdata = csvigo.load{path='data/training_solutions_rev1.csv', verbose=false}
   nSamples = #csvdata.GalaxyID
   data = torch.Tensor(nSamples, 38)
   for i=1,nSamples do
      data[i][1]  = tonumber(csvdata['GalaxyID'][i])

      data[i][2]  = tonumber(csvdata['Class1.1'][i])
      data[i][3]  = tonumber(csvdata['Class1.2'][i])
      data[i][4]  = tonumber(csvdata['Class1.3'][i])

      data[i][5]  = tonumber(csvdata['Class2.1'][i])
      data[i][6]  = tonumber(csvdata['Class2.2'][i])

      data[i][7]  = tonumber(csvdata['Class3.1'][i])
      data[i][8]  = tonumber(csvdata['Class3.2'][i])

      data[i][9]  = tonumber(csvdata['Class4.1'][i])
      data[i][10] = tonumber(csvdata['Class4.2'][i])

      data[i][11] = tonumber(csvdata['Class5.1'][i])
      data[i][12] = tonumber(csvdata['Class5.2'][i])
      data[i][13] = tonumber(csvdata['Class5.3'][i])
      data[i][14] = tonumber(csvdata['Class5.4'][i])

      data[i][15] = tonumber(csvdata['Class6.1'][i])
      data[i][16] = tonumber(csvdata['Class6.2'][i])

      data[i][17] = tonumber(csvdata['Class7.1'][i])
      data[i][18] = tonumber(csvdata['Class7.2'][i])
      data[i][19] = tonumber(csvdata['Class7.3'][i])

      data[i][20] = tonumber(csvdata['Class8.1'][i])
      data[i][21] = tonumber(csvdata['Class8.2'][i])
      data[i][22] = tonumber(csvdata['Class8.3'][i])
      data[i][23] = tonumber(csvdata['Class8.4'][i])
      data[i][24] = tonumber(csvdata['Class8.5'][i])
      data[i][25] = tonumber(csvdata['Class8.6'][i])
      data[i][26] = tonumber(csvdata['Class8.7'][i])

      data[i][27] = tonumber(csvdata['Class9.1'][i])
      data[i][28] = tonumber(csvdata['Class9.2'][i])
      data[i][29] = tonumber(csvdata['Class9.3'][i])

      data[i][30] = tonumber(csvdata['Class10.1'][i])
      data[i][31] = tonumber(csvdata['Class10.2'][i])
      data[i][32] = tonumber(csvdata['Class10.3'][i])

      data[i][33] = tonumber(csvdata['Class11.1'][i])
      data[i][34] = tonumber(csvdata['Class11.2'][i])
      data[i][35] = tonumber(csvdata['Class11.3'][i])
      data[i][36] = tonumber(csvdata['Class11.4'][i])
      data[i][37] = tonumber(csvdata['Class11.5'][i])
      data[i][38] = tonumber(csvdata['Class11.6'][i])
   end   
   torch.save('cache/data.t7', data)
else
   print('Loading from cache')
   data = torch.load('cache/data.t7')
   nSamples = data:size(1)
end
normalizedData = torch.Tensor(nSamples, 38)
for i=1,nSamples do
   local input  = data[i]
   local output = originalToNormalized(input)
   local inputSanity = normalizedToOriginal(output)
   assert((input - inputSanity):abs():gt(1e-4):sum() == 0, 'Failed normalization sanity')
   normalizedData[i] = output
end
-- split into training/testing 90/10
nTraining = math.floor(nSamples * 0.90)
nTesting = nSamples - nTraining

local randIndices = torch.randperm(nSamples)
local trIndices = randIndices[{{1,nTraining}}]
local tsIndices = randIndices[{{nTraining+1,nSamples}}]

trainData = torch.Tensor(nTraining, 38)
testData = torch.Tensor(nTesting, 38)
unnormalizedTestData = torch.Tensor(nTesting, 38)
unnormalizedTrainData = torch.Tensor(nTraining, 38)
for i=1,nTraining do
   trainData[i] = normalizedData[trIndices[i]]
   unnormalizedTrainData[i] = data[trIndices[i]]
end

for i=1,nTesting do
   testData[i] = normalizedData[tsIndices[i]]
   unnormalizedTestData[i] = data[tsIndices[i]]
end
collectgarbage()
--=========
print('Number of Samples: ' .. nSamples)
print('Training samples: ' .. nTraining)
print('Testing samples: ' .. nTesting)

function getSample()
   local i = math.floor(torch.uniform(1, nTraining+0.5))
   local filename = paths.concat(dataroot, tostring(trainData[i][1]) .. '.jpg')
   local im = image.load(filename, 3)
   im = jitter(im)
   im:add(-im:mean())
   im:div(im:std())
   local gt = trainData[i][{{2, 38}}]
   local gtu = unnormalizedTrainData[i][{{2,38}}]
   return im, gt, gtu
end

function getBatch(n)
   local img, gt, gtu
   img = torch.Tensor(n, sampleSize[1], sampleSize[2], sampleSize[3])
   gt = torch.Tensor(n, 37)
   gtu = torch.Tensor(n, 37)
   for i=1,n do
      img[i], gt[i], gtu[i] = getSample()
   end
   return img, gt, gtu
end

function getTest(i, lightTesting)
   local filename = paths.concat(dataroot, tostring(testData[i][1]) .. '.jpg')
   local im = image.load(filename, 3)
   im = expandTestSample(im, lightTesting)
   local gt = testData[i][{{2, 38}}]
   local gtu = unnormalizedTestData[i][{{2,38}}]   
   return im, gt, gtu
end

-- sanity check of test variation generator
if opt and opt.dataTest then
   local lena = expandTestSample(image.scale(image.lena(), loadSize[2], loadSize[3]))
   image.display{image=lena, nrow=16}
   print(#getTest(1))
   local testImage = getTest(1)
   print(#testImage)
   image.display{image=testImage[1], legend='original-image after mean'}
   image.display{image=testImage, nrow=16}
   local a,b = getBatch(128)
   image.display{a:float(), nrow=16}
   print(#a)
   print(#b)
end

