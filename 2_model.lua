require 'torch'
require 'nn'
require 'nnx'
require 'cunn'

print '==> 2_model.lua'
print '==> defining CNN model'
-- features size
fSize = {3, 96, 256, 256, 256}
featuresOut = fSize[5] * 3 * 3

-- classifier size
classifierHidden = {512}
dropout_p = 0.5

features = nn.Sequential()
features:add(nn.SpatialConvolutionMM(fSize[1], fSize[2], 9, 9, 2, 2)) -- (111 - 9 + 2)/2 = 52
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 26
features:add(nn.SpatialConvolutionMM(fSize[2], fSize[3], 5, 5)) -- 22
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 11
features:add(nn.SpatialConvolutionMM(fSize[3], fSize[4], 4, 4)) -- 8
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialConvolutionMM(fSize[4], fSize[5], 3, 3)) -- 6
features:add(nn.Threshold(0,1e-6))
features:add(nn.SpatialMaxPooling(2,2,2,2)) -- 3
features:add(nn.View(featuresOut))

dropouts = {}
for i=1,22 do
   table.insert(dropouts, nn.Dropout(dropout_p))
end

branch = {}
branch[1] = nn.Sequential()
if opt.dropout then
   branch[1]:add(dropouts[1])
end
branch[1]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[1]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[1]:add(dropouts[2])
end
branch[1]:add(nn.Linear(classifierHidden[1], 3))
branch[1]:add(nn.SoftMax())

branch[2] = nn.Sequential()
if opt.dropout then
   branch[2]:add(dropouts[3])
end
branch[2]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[2]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[2]:add(dropouts[4])
end
branch[2]:add(nn.Linear(classifierHidden[1], 2))
branch[2]:add(nn.SoftMax())

branch[3] = nn.Sequential()
if opt.dropout then
   branch[3]:add(dropouts[5])
end
branch[3]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[3]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[3]:add(dropouts[6])
end
branch[3]:add(nn.Linear(classifierHidden[1], 2))
branch[3]:add(nn.SoftMax())

branch[4] = nn.Sequential()
if opt.dropout then
   branch[4]:add(dropouts[7])
end
branch[4]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[4]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[4]:add(dropouts[8])
end
branch[4]:add(nn.Linear(classifierHidden[1], 2))
branch[4]:add(nn.SoftMax())

branch[5] = nn.Sequential()
if opt.dropout then
   branch[5]:add(dropouts[9])
end
branch[5]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[5]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[5]:add(dropouts[10])
end
branch[5]:add(nn.Linear(classifierHidden[1], 4))
branch[5]:add(nn.SoftMax())

branch[6] = nn.Sequential()
if opt.dropout then
   branch[6]:add(dropouts[11])
end
branch[6]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[6]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[6]:add(dropouts[12])
end
branch[6]:add(nn.Linear(classifierHidden[1], 2))
branch[6]:add(nn.SoftMax())

branch[7] = nn.Sequential()
if opt.dropout then
   branch[7]:add(dropouts[13])
end
branch[7]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[7]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[7]:add(dropouts[14])
end
branch[7]:add(nn.Linear(classifierHidden[1], 3))
branch[7]:add(nn.SoftMax())

branch[8] = nn.Sequential()
if opt.dropout then
   branch[8]:add(dropouts[15])
end
branch[8]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[8]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[8]:add(dropouts[16])
end
branch[8]:add(nn.Linear(classifierHidden[1], 7))
branch[8]:add(nn.SoftMax())

branch[9] = nn.Sequential()
if opt.dropout then
   branch[9]:add(dropouts[17])
end
branch[9]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[9]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[9]:add(dropouts[18])
end
branch[9]:add(nn.Linear(classifierHidden[1], 3))
branch[9]:add(nn.SoftMax())

branch[10] = nn.Sequential()
if opt.dropout then
   branch[10]:add(dropouts[19])
end
branch[10]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[10]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[10]:add(dropouts[20])
end
branch[10]:add(nn.Linear(classifierHidden[1], 3))
branch[10]:add(nn.SoftMax())

branch[11] = nn.Sequential()
if opt.dropout then
   branch[11]:add(dropouts[21])
end
branch[11]:add(nn.Linear(featuresOut, classifierHidden[1]))
branch[11]:add(nn.Threshold(0, 1e-6))
if opt.dropout then
   branch[11]:add(dropouts[22])
end
branch[11]:add(nn.Linear(classifierHidden[1], 6))
branch[11]:add(nn.SoftMax())


dgraph = nn.Concat(2)
dgraph:add(branch[1])
dgraph:add(branch[2])
dgraph:add(branch[3])
dgraph:add(branch[4])
dgraph:add(branch[5])
dgraph:add(branch[6])
dgraph:add(branch[7])
dgraph:add(branch[8])
dgraph:add(branch[9])
dgraph:add(branch[10])
dgraph:add(branch[11])

model = nn.Sequential()
model:add(features)
model:add(dgraph)

-- the output of "model" will be a 37-dimensional vector


criterion = nn.MSECriterion()


-- model = model:cuda()
-- criterion = criterion:cuda()


-- local inp = torch.rand(sampleSize[1], sampleSize[2], sampleSize[3], 128):cuda()
-- local o = inp
-- for i=1,#(features.modules) do
--    o = features.modules[i]:forward(o)
--    print(#o)
-- end
-- model:forward(inp)
