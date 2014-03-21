require 'torch'
require 'nn'

-- features size

featuresOut = 2048

-- classifier size
classifierHidden = {100, 100}

features = nn.Sequential()



branch1 = nn.Sequential()
branch1:add(nn.Linear(featuresOut, classifierHidden[1]))
branch1:add(nn.Threshold(0, 1e-6))
branch1:add(nn.Linear(classifierHidden[1], 3))
branch1:add(nn.Softmax())

branch2 = nn.Sequential()
branch2:add(nn.Linear(featuresOut, classifierHidden[1]))
branch2:add(nn.Threshold(0, 1e-6))
branch2:add(nn.Linear(classifierHidden[1], 2))
branch2:add(nn.Softmax())

branch3 = nn.Sequential()
branch3:add(nn.Linear(featuresOut, classifierHidden[1]))
branch3:add(nn.Threshold(0, 1e-6))
branch3:add(nn.Linear(classifierHidden[1], 2))
branch3:add(nn.Softmax())

branch4 = nn.Sequential()
branch4:add(nn.Linear(featuresOut, classifierHidden[1]))
branch4:add(nn.Threshold(0, 1e-6))
branch4:add(nn.Linear(classifierHidden[1], 2))
branch4:add(nn.Softmax())

branch1 = nn.Sequential()
branch5:add(nn.Linear(featuresOut, classifierHidden[1]))
branch5:add(nn.Threshold(0, 1e-6))
branch5:add(nn.Linear(classifierHidden[1], 4))
branch5:add(nn.Softmax())

branch6 = nn.Sequential()
branch6:add(nn.Linear(featuresOut, classifierHidden[1]))
branch6:add(nn.Threshold(0, 1e-6))
branch6:add(nn.Linear(classifierHidden[1], 2))
branch6:add(nn.Softmax())

branch7 = nn.Sequential()
branch7:add(nn.Linear(featuresOut, classifierHidden[1]))
branch7:add(nn.Threshold(0, 1e-6))
branch7:add(nn.Linear(classifierHidden[1], 3))
branch7:add(nn.Softmax())

branch8 = nn.Sequential()
branch8:add(nn.Linear(featuresOut, classifierHidden[1]))
branch8:add(nn.Threshold(0, 1e-6))
branch8:add(nn.Linear(classifierHidden[1], 7))
branch8:add(nn.Softmax())

branch9 = nn.Sequential()
branch9:add(nn.Linear(featuresOut, classifierHidden[1]))
branch9:add(nn.Threshold(0, 1e-6))
branch9:add(nn.Linear(classifierHidden[1], 3))
branch9:add(nn.Softmax())

branch10 = nn.Sequential()
branch10:add(nn.Linear(featuresOut, classifierHidden[1]))
branch10:add(nn.Threshold(0, 1e-6))
branch10:add(nn.Linear(classifierHidden[1], 3))
branch10:add(nn.Softmax())

branch11 = nn.Sequential()
branch11:add(nn.Linear(featuresOut, classifierHidden[1]))
branch11:add(nn.Threshold(0, 1e-6))
branch11:add(nn.Linear(classifierHidden[1], 6))
branch11:add(nn.Softmax())


dgraph = nn.Concat(1)
dgraph:add(class1)
dgraph:add(class2)
dgraph:add(class3)
dgraph:add(class4)
dgraph:add(class5)
dgraph:add(class6)
dgraph:add(class7)
dgraph:add(class8)
dgraph:add(class9)
dgraph:add(class10)
dgraph:add(class11)

model = nn.Sequential()
model:add(features)
model:add(dgraph)

-- the output of "model" will be a 37-dimensional vector


criterion = nn.MSECriterion()