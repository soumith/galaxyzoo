require 'csv'
require 'xlua'
require 'paths'
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

-- split into training/testing 80/20
nTraining = math.floor(nSamples * 0.8)
nTesting = nSamples - nTraining
print('Training samples: ' .. nTraining)
print('Testing samples: ' .. nTesting)

local randIndices = torch.randperm(nSamples)
local trIndices = randIndices[{{1,nTraining}}]
local tsIndices = randIndices[{{nTraining+1,nSamples}}]

trainData = torch.Tensor(nTraining, 38)
testData = torch.Tensor(nTesting, 38)
for i=1,nTraining do
   trainData[i] = data[trIndices[i]]
end

for i=1,nTesting do
   testData[i] = data[tsIndices[i]]
end
