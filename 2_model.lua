require 'torch'
require 'nn'
require 'nnx'

-- features size

featuresOut = 16 * 18 * 18

-- classifier size
classifierHidden = {100, 100}
dropout_p = 0.5

features = nn.Sequential()
features:add(nn.SpatialConvolution(3, 16, 7, 7))
features:add(nn.Reshape(featuresOut))

dropouts = {}
for i=1,22 do
   table.insert(dropouts, nn.Dropout(dropout_p))
end

branch1 = nn.Sequential()
branch1:add(dropouts[1])
branch1:add(nn.Linear(featuresOut, classifierHidden[1]))
branch1:add(nn.Threshold(0, 1e-6))
branch1:add(dropouts[2])
branch1:add(nn.Linear(classifierHidden[1], 3))
branch1:add(nn.SoftMax())

branch2 = nn.Sequential()
branch2:add(dropouts[3])
branch2:add(nn.Linear(featuresOut, classifierHidden[1]))
branch2:add(nn.Threshold(0, 1e-6))
branch2:add(dropouts[4])
branch2:add(nn.Linear(classifierHidden[1], 2))
branch2:add(nn.SoftMax())

branch3 = nn.Sequential()
branch3:add(dropouts[5])
branch3:add(nn.Linear(featuresOut, classifierHidden[1]))
branch3:add(nn.Threshold(0, 1e-6))
branch3:add(dropouts[6])
branch3:add(nn.Linear(classifierHidden[1], 2))
branch3:add(nn.SoftMax())

branch4 = nn.Sequential()
branch4:add(dropouts[7])
branch4:add(nn.Linear(featuresOut, classifierHidden[1]))
branch4:add(nn.Threshold(0, 1e-6))
branch4:add(dropouts[8])
branch4:add(nn.Linear(classifierHidden[1], 2))
branch4:add(nn.SoftMax())

branch5 = nn.Sequential()
branch5:add(dropouts[9])
branch5:add(nn.Linear(featuresOut, classifierHidden[1]))
branch5:add(nn.Threshold(0, 1e-6))
branch5:add(dropouts[10])
branch5:add(nn.Linear(classifierHidden[1], 4))
branch5:add(nn.SoftMax())

branch6 = nn.Sequential()
branch6:add(dropouts[11])
branch6:add(nn.Linear(featuresOut, classifierHidden[1]))
branch6:add(nn.Threshold(0, 1e-6))
branch6:add(dropouts[12])
branch6:add(nn.Linear(classifierHidden[1], 2))
branch6:add(nn.SoftMax())

branch7 = nn.Sequential()
branch7:add(dropouts[13])
branch7:add(nn.Linear(featuresOut, classifierHidden[1]))
branch7:add(nn.Threshold(0, 1e-6))
branch7:add(dropouts[14])
branch7:add(nn.Linear(classifierHidden[1], 3))
branch7:add(nn.SoftMax())

branch8 = nn.Sequential()
branch8:add(dropouts[15])
branch8:add(nn.Linear(featuresOut, classifierHidden[1]))
branch8:add(nn.Threshold(0, 1e-6))
branch8:add(dropouts[16])
branch8:add(nn.Linear(classifierHidden[1], 7))
branch8:add(nn.SoftMax())

branch9 = nn.Sequential()
branch9:add(dropouts[17])
branch9:add(nn.Linear(featuresOut, classifierHidden[1]))
branch9:add(nn.Threshold(0, 1e-6))
branch9:add(dropouts[18])
branch9:add(nn.Linear(classifierHidden[1], 3))
branch9:add(nn.SoftMax())

branch10 = nn.Sequential()
branch10:add(dropouts[19])
branch10:add(nn.Linear(featuresOut, classifierHidden[1]))
branch10:add(nn.Threshold(0, 1e-6))
branch10:add(dropouts[20])
branch10:add(nn.Linear(classifierHidden[1], 3))
branch10:add(nn.SoftMax())

branch11 = nn.Sequential()
branch11:add(dropouts[21])
branch11:add(nn.Linear(featuresOut, classifierHidden[1]))
branch11:add(nn.Threshold(0, 1e-6))
branch11:add(dropouts[22])
branch11:add(nn.Linear(classifierHidden[1], 6))
branch11:add(nn.SoftMax())


dgraph = nn.Concat(1)
dgraph:add(branch1)
dgraph:add(branch2)
dgraph:add(branch3)
dgraph:add(branch4)
dgraph:add(branch5)
dgraph:add(branch6)
dgraph:add(branch7)
dgraph:add(branch8)
dgraph:add(branch9)
dgraph:add(branch10)
dgraph:add(branch11)

model = nn.Sequential()
model:add(features)
model:add(dgraph)

-- the output of "model" will be a 37-dimensional vector


criterion = nn.MSECriterion()



-- model:forward(torch.rand(3, 24, 24))