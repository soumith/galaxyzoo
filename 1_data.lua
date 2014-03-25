require 'csv'
require 'xlua'
require 'paths'
require 'torchffi'
require 'cutorch'
require 'nn'
require 'image'
local gm = require 'graphicsmagick'

dofile('1_datafunctions.lua')
print '==> 1_data.lua'
print('==> Loading data')
if not paths.filep('cache/data.t7') then
   local csvdata = csv.load{path='data/training_solutions_rev1.csv', verbose=false}
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
nTraining = math.floor(nSamples * 0.99)
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

--=========
print('Number of Samples: ' .. nSamples)
print('Training samples: ' .. nTraining)
print('Testing samples: ' .. nTesting)




local meanImageFname = 'mean_' .. sampleSize[1] .. 'x' .. sampleSize[2] .. 'x' .. sampleSize[3] .. '.t7'
if paths.filep(meanImageFname) then
   print('Loading mean-image from cache')
   meanImage = torch.load(meanImageFname)
else
   print('Calculating mean-image')
   meanImage = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3])
   for i=1,nTraining do
      xlua.progress(i, nTraining)
      local filename = paths.concat(dataroot, tostring(trainData[i][1]) .. '.jpg')
      local im = gm.Image()
      im:load(filename, sampleSize[2], sampleSize[3])
      im:size(sampleSize[2], sampleSize[3])
      im = im:toTensor('float', 'RGB', 'DHW', true)
      meanImage:add(im)
   end
   meanImage:div(nTraining)
   torch.save(meanImageFname, meanImage)
end

local norm = nn.SpatialContrastiveNormalization(3, image.gaussian1D{size=13})

function getSample()
   local i = math.floor(torch.uniform(1, nTraining+0.5))
   local filename = paths.concat(dataroot, tostring(trainData[i][1]) .. '.jpg')
   local im = gm.Image()
   im:load(filename, loadSize[2], loadSize[3])
   im:size(loadSize[2], loadSize[3])
   im = jitter(im)
   im = im:toTensor('float', 'RGB', 'DHW', true)
   im:add(-meanImage)
   -- im = norm:forward(im)   
   local gt = trainData[i][{{2, 38}}]
   local gtu = unnormalizedTrainData[i][{{2,38}}]
   return im, gt, gtu
end

function getBatch(n)
   local img, gt, gtu
   if bmode == 'BDHW' then
      img = torch.Tensor(n, sampleSize[1], sampleSize[2], sampleSize[3])
   else      
      img = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3], n)
   end
   gt = torch.Tensor(n, 37)
   gtu = torch.Tensor(n, 37)
   for i=1,n do
      if bmode == 'BDHW' then
	 img[i], gt[i], gtu[i] = getSample()
      else
	 img[{{},{},{},i}], gt[i], gtu[i] = getSample()
      end
   end
   img = img:cuda()
   -- gt = gt:cuda()
   -- gtu = gtu:cuda()
   return img, gt, gtu
end

local transposer = nn.Transpose({1,4},{1,3},{1,2})
local rtransposer = nn.Transpose({4,1},{4,2},{4,3})
   -- have 128 deterministic outputs for each input
   --[[
      original
       - rotate   0
       - rotate  90
       - rotate -90
       - rotate 180
         - rotate further 0
         - rotate further 45
           - translate 0
           - translate -10px (x)   0px (y)
           - translate +10px (x)   0px (y)
           - translate   0px (x) -10px (y)
           - translate   0px (x) +10px (y)
           - translate -10px (x) -10px (y)
           - translate +10px (x) -10px (y)
           - translate +10px (x) +10px (y)
      vflip
       - rotate   0
       - rotate  90
       - rotate -90
       - rotate 180
         - rotate further 0
         - rotate further 45
           - translate 0
           - translate -10px (x)   0px (y)
           - translate +10px (x)   0px (y)
           - translate   0px (x) -10px (y)
           - translate   0px (x) +10px (y)
           - translate -10px (x) -10px (y)
           - translate +10px (x) -10px (y)
           - translate +10px (x) +10px (y)
      Total number: 2 * 4 * 2 * 8 = 128
   ]]--
local function test_t(im, o)
   local x1 = math.ceil((loadSize[2] - sampleSize[2])/2)
   local size = sampleSize[2]
   local t = math.floor(loadSize[2] * 0.04)
   o[1] = image.crop(im, x1, x1, x1+size, x1+size) -- center patch
   o[2] = image.crop(im, x1-t, x1, x1-t+size, x1+size)
   o[3] = image.crop(im, x1+t, x1, x1+t+size, x1+size)
   o[4] = image.crop(im, x1, x1-t, x1+size, x1-t+size)
   o[5] = image.crop(im, x1, x1+t, x1+size, x1+t+size)
   o[6] = image.crop(im, x1-t, x1-t, x1-t+size, x1-t+size)
   o[7] = image.crop(im, x1+t, x1-t, x1+t+size, x1-t+size)
   o[8] = image.crop(im, x1+t, x1+t, x1+t+size, x1+t+size)
end
local function test_rt(im, o)
   -- rotate further 0
   test_t(im, o[{{1,8},{},{},{}}])
   -- rotate further 45
   local im2 =image.rotate(im, math.pi/4)   
   test_t(im2, o[{{9,16},{},{},{}}])
end

local function test_rrt(im, o)
   -- rotate 0
   test_rt(im, o[{{1,16},{},{},{}}])
   if not lightTesting then
      -- rotate -90
      local minus90 = torch.Tensor(im:size())
      for i=1,3 do
	 minus90[i] = im[i]:t()
      end
      test_rt(minus90, o[{{17,32},{},{},{}}])
      -- rotate 90
      local plus90 = image.hflip(image.vflip(minus90))
      test_rt(plus90, o[{{33,48},{},{},{}}])
      -- rotate 180
      local plus180 = image.hflip(image.vflip(im))
      test_rt(plus180, o[{{49,64},{},{},{}}])
   end
end
function expandTestSample(im)
   -- produce the 128 combos, given an input image (3D tensor)
   local o
   if lightTesting then
      o = torch.Tensor(16, sampleSize[1], sampleSize[2], sampleSize[3])
      test_rrt(im, o[{{1,16},{},{},{}}])
   else
      o = torch.Tensor(128, sampleSize[1], sampleSize[2], sampleSize[3])
      -- original
      test_rrt(im, o[{{1,64},{},{},{}}])
      -- vflip
      test_rrt(image.vflip(im), o[{{65,128},{},{},{}}])
   end
   for i=1,o:size(1) do
      o[i]:add(-meanImage)
      -- o[i] = norm:forward(o[i])
   end
   if bmode == 'BDHW' then
      return o
   else
      return transposer:forward(o)
   end
end

collectgarbage()
function getTest(i)
   local filename = paths.concat(dataroot, tostring(testData[i][1]) .. '.jpg')
   local im = gm.Image()
   im:load(filename, loadSize[2], loadSize[3])
   im:size(loadSize[2], loadSize[3])
   im = im:toTensor('float', 'RGB', 'DHW', true)
   im = expandTestSample(im)
   im = im:cuda()
   local gt = testData[i][{{2, 38}}]
   local gtu = unnormalizedTestData[i][{{2,38}}]   
   return im, gt, gtu
end

-- sanity check of test variation generator
if opt.dataTest then
   local lena = expandTestSample(image.scale(image.lena(), loadSize[2], loadSize[3]))
   image.display{image=rtransposer:forward(lena), nrow=16}
   image.display{image=rtransposer:forward(getTest(1):float()), nrow=16}
   local a,b = getBatch(128)
   image.display{image=rtransposer:forward(a:float()), nrow=16}
   print(#a)
   print(#b)
end

