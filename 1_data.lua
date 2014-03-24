require 'csv'
require 'xlua'
require 'paths'
require 'torchffi'
require 'cutorch'
require 'nn'
require 'image'

local gm = require 'graphicsmagick'

print('==> Loading data')
if not paths.filep('data.t7') then
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
   torch.save('data.t7', data)
else
   print('Loading from cache')
   data = torch.load('data.t7')
   nSamples = data:size(1)
end
print('Number of Samples: ' .. nSamples)
function originalToNormalized(s)
   o = s:clone()

   -- class 1 (already normalized)
   local sum = s[2] + s[3] + s[4]
   assert(math.abs(sum - 1.0) <= 1e-4,  'Class1')
   
   -- class 2
   sum = s[5] + s[6]
   if sum ~= 0 then
      o[5] = s[5] / sum
      o[6] = s[6] / sum
   end
   assert(math.abs(sum - s[3]) <= 1e-4,  'Class2')

   -- class 3
   sum = s[7] + s[8]
   if sum ~= 0 then
      o[7] = s[7] / sum
      o[8] = s[8] / sum
   end
   assert(math.abs(sum - s[6]) <= 1e-4,  'Class3,' .. sum .. ',' .. s[6])

   -- class 4
   sum = s[9] + s[10]
   if sum ~= 0 then
      o[9]  = s[9] / sum
      o[10] = s[10] / sum
   end
   assert(math.abs(sum - (s[7] + s[8])) <= 1e-4,  'Class4,' 
	     .. sum .. ',' .. s[7] .. ',' .. s[8])

   -- class 5
   sum = s[11] + s[12] + s[13] + s[14]
   if sum ~= 0 then
      o[11] = s[11] / sum
      o[12] = s[12] / sum
      o[13] = s[13] / sum
      o[14] = s[14] / sum
   end
   assert(math.abs(sum - (s[10] + s[33]+ s[34]+ s[35]
			     + s[36]+ s[37]+ s[38])) <= 1e-4,  'Class5')
   
   -- class 6 (already normalized)
   sum = s[15] + s[16]
   assert(math.abs(sum - 1.0) <= 1e-4,  'Class6')

   -- class 7
   sum = s[17] + s[18] + s[19]
   if sum ~= 0 then
      o[17]  = s[17] / sum
      o[18] = s[18] / sum
      o[19] = s[19] / sum
   end
   assert(math.abs(sum - s[2]) <= 1e-4,  'Class7')

   -- class 8
   sum = s[20] + s[21] + s[22] + s[23] 
      + s[24] + s[25] + s[26]
   if sum ~= 0 then
      o[20]  = s[20] / sum
      o[21]  = s[21] / sum
      o[22]  = s[22] / sum
      o[23]  = s[23] / sum
      o[24]  = s[24] / sum
      o[25]  = s[25] / sum
      o[26]  = s[26] / sum
   end
   assert(math.abs(sum - s[15]) <= 1e-4,  'Class8')

   -- class 9
   sum = s[27] + s[28] + s[29]
   if sum ~= 0 then
      o[27]  = s[27] / sum
      o[28] = s[28] / sum
      o[29] = s[29] / sum
   end
   assert(math.abs(sum - s[5]) <= 1e-4,  'Class9')

   -- class 10
   sum = s[30] + s[31] + s[32]
   if sum ~= 0 then
      o[30]  = s[30] / sum
      o[31] = s[31] / sum
      o[32] = s[32] / sum
   end
   assert(math.abs(sum - s[9]) <= 1e-4,  'Class10')

   -- class 11
   sum = s[33] + s[34] + s[35] 
      + s[36] + s[37] + s[38]
   if sum ~= 0 then
      o[33]  = s[33] / sum
      o[34] = s[34] / sum
      o[35] = s[35] / sum
      o[36] = s[36] / sum
      o[37] = s[37] / sum
      o[38] = s[38] / sum
   end
   assert(math.abs(sum - (s[30]+s[31]+s[32])) <= 1e-4,  'Class11')
   return o;
end

function normalizedToOriginal(s)
   local prune = false
   if s:size(1) == 37 then
      local t = torch.zeros(38)
      t[{{2,38}}] = s
      s = t
      prune = true
   end
   -- sample is a tensor of size 38.
   local o = s:clone()
   -- class 2
   o[5] = o[5] * s[3]
   o[6] = o[6] * s[3]
   -- class 3
   o[7] = o[7] * o[6]
   o[8] = o[8] * o[6]
   -- class 4
   o[9] = o[9] * (o[7] + o[8])
   o[10] = o[10] * (o[7] + o[8])
   -- class 10
   o[30] = o[30] * o[9]
   o[31] = o[31] * o[9]
   o[32] = o[32] * o[9]
   -- class 11
   o[33] = o[33] * (o[30]+o[31]+o[32])
   o[34] = o[34] * (o[30]+o[31]+o[32])
   o[35] = o[35] * (o[30]+o[31]+o[32])
   o[36] = o[36] * (o[30]+o[31]+o[32])
   o[37] = o[37] * (o[30]+o[31]+o[32])
   o[38] = o[38] * (o[30]+o[31]+o[32])
   -- class 5
   o[11] = o[11] * (o[10] + o[33] + o[34] + o[35] + o[36] + o[37] + o[38])
   o[12] = o[12] * (o[10] + o[33] + o[34] + o[35] + o[36] + o[37] + o[38])
   o[13] = o[13] * (o[10] + o[33] + o[34] + o[35] + o[36] + o[37] + o[38])
   o[14] = o[14] * (o[10] + o[33] + o[34] + o[35] + o[36] + o[37] + o[38])
   -- class 6 (already normalized)
   -- class 7
   o[17] = o[17] * o[2]
   o[18] = o[18] * o[2]
   o[19] = o[19] * o[2]
   -- class 8
   o[20] = o[20] * o[15]
   o[21] = o[21] * o[15]
   o[22] = o[22] * o[15]
   o[23] = o[23] * o[15]
   o[24] = o[24] * o[15]
   o[25] = o[25] * o[15]
   o[26] = o[26] * o[15]
   -- class 9
   o[27] = o[27] * o[5]
   o[28] = o[28] * o[5]
   o[29] = o[29] * o[5]
   if prune then
      o = o[{{2,38}}]
   end
   return o
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
print('Training samples: ' .. nTraining)
print('Testing samples: ' .. nTesting)

local randIndices = torch.randperm(nSamples)
local trIndices = randIndices[{{1,nTraining}}]
local tsIndices = randIndices[{{nTraining+1,nSamples}}]

trainData = torch.Tensor(nTraining, 38)
testData = torch.Tensor(nTesting, 38)
unnormalizedTestData = torch.Tensor(nTesting, 38)
for i=1,nTraining do
   trainData[i] = normalizedData[trIndices[i]]
end

for i=1,nTesting do
   testData[i] = normalizedData[tsIndices[i]]
   unnormalizedTestData[i] = data[tsIndices[i]]
end

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

function jitter(s)
   local d = torch.rand(7)
   -- vflip
   if d[1] > 0.5 then
      s = image.vflip(s)
   end
   -- hflip
   if d[2] > 0.5 then
      s = image.hflip(s)
   end
   -- rotation
   if d[3] > 0.5 then
      s = image.rotate(s, math.pi * d[4])
   end
   -- crop a 223x223 random patch
   local startX = math.ceil(d[6] * (loadSize[2] - sampleSize[2] - 1))
   local startY = math.ceil(d[7] * (loadSize[3] - sampleSize[3] - 1))
   local endX = startX + sampleSize[2]
   local endY = startY + sampleSize[3]
   s = image.crop(s, startX, startY, endX, endY)
   return s
end
function getSample()
   local i = math.floor(torch.uniform(1, nTraining+0.5))
   local filename = paths.concat(dataroot, tostring(trainData[i][1]) .. '.jpg')
   local im = gm.Image()
   im:load(filename, loadSize[2], loadSize[3])
   im:size(loadSize[2], loadSize[3])
   im = im:toTensor('float', 'RGB', 'DHW', true)
   im = jitter(im)
   im:add(-meanImage)
   -- im = norm:forward(im)   
   local gt = trainData[i][{{2, 38}}]
   return im, gt
end

function getBatch(n)
   local img, gt
   img = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3], n)
   gt = torch.Tensor(n, 37)
   for i=1,n do
      img[{{},{},{},i}], gt[i] = getSample()
   end
   img = img:cuda()
   gt = gt:cuda()
   return img, gt
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
   o[1] = image.crop(im, 17, 17, 17+223, 17+223) -- center patch
   o[2] = image.crop(im, 7, 17, 7+223, 17+223)
   o[3] = image.crop(im, 27, 17, 27+223, 17+223)
   o[4] = image.crop(im, 17, 7, 17+223, 7+223)
   o[5] = image.crop(im, 17, 27, 17+223, 27+223)
   o[6] = image.crop(im, 7, 7, 7+223, 7+223)
   o[7] = image.crop(im, 27, 7, 27+223, 7+223)
   o[8] = image.crop(im, 27, 27, 27+223, 27+223)
end
local function test_rt(im, o)
   -- rotate further 0
   test_t(im, o[{{1,8},{},{},{}}])
   -- rotate further 45
   test_t(image.rotate(im, math.pi/4), o[{{9,16},{},{},{}}])
end

local function test_rrt(im, o)
   -- rotate 0
   test_rt(im, o[{{1,16},{},{},{}}])
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
function expandTestSample(im)
   -- produce the 128 combos, given an input image (3D tensor)
   local o = torch.Tensor(128, sampleSize[1], sampleSize[2], sampleSize[3])
   -- original
   test_rrt(im, o[{{1,64},{},{},{}}])
   -- vflip
   test_rrt(image.vflip(im), o[{{65,128},{},{},{}}])
   print(#o)
   print(#meanImage)
   for i=1,o:size(1) do
      o[i]:add(-meanImage)
      -- o[i] = norm:forward(o[i])
   end
   return transposer:forward(o)
end

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
   image.display{image=rtransposer:forward(lena), nrow=32}
   image.display{image=rtransposer:forward(getTest(1):float()), nrow=32}
   local a,b = getBatch(128)
   image.display{image=rtransposer:forward(getBatch(128):float()), nrow=32}
   print(#a)
   print(#b)
end

