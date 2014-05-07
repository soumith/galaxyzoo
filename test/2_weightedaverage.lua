-- takes in several folders as input and averages the values (mean or median)
require 'env'
require 'torch'
require 'paths'
require 'csvigo'
torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('GalaxyZoo Test script')
cmd:text()
cmd:text('Options:')
cmd:option('-input', 'testresults', 'folder pointing to where all the test outputs are')
cmd:option('-save', 'kagglesubmissions',   'subdirectory to save results in')
cmd:option('-model', 'none', 'path to model')
cmd:text()
opt = cmd:parse(arg or {})

os.execute('mkdir -p ' .. opt.save)

local dirs = {}
for f in paths.files(opt.input) do
   local bf = paths.concat(opt.input, f)
   if paths.dirp(bf) and f ~= '.' and f ~= '..' then
      table.insert(dirs, bf)      
   end
end
print(dirs)
weights = {.4, .1, .4, .1}
print(weights)

nSamples = 79975
nBatch = 16
local out = torch.Tensor(nSamples, 38)

local index = 1
-- for each file in first folder,
for f in paths.files(dirs[1]) do
   if not paths.dirp(paths.concat(dirs[1], f)) then
      local o = torch.Tensor(38)
      o[1] = tonumber(f)
      for i=1,#dirs do
	 -- load the same filename from all folders
	 local fname = paths.concat(dirs[i], f)
	 local ot = torch.load(fname)
	 local startidx = (i-1)*nBatch + 1
	 -- take the mean, weight it
	 ot = ot:mean(1)[1]:mul(weights[i])
	 -- add it
	 o[{{2, 38}}] = o[{{2, 38}}] + ot
      end
      assert(tostring(o[1]) == f, 'file name got corrupted')
      out[index] = o
      index = index + 1
   end
end

local csd = {}
csd['GalaxyID'] = {}
csd['Class1.1'] = {}
csd['Class1.2'] = {}
csd['Class1.3'] = {}
csd['Class2.1'] = {}
csd['Class2.2'] = {}
csd['Class3.1'] = {}
csd['Class3.2'] = {}
csd['Class4.1'] = {}
csd['Class4.2'] = {}
csd['Class5.1'] = {}
csd['Class5.2'] = {}
csd['Class5.3'] = {}
csd['Class5.4'] = {}
csd['Class6.1'] = {}
csd['Class6.2'] = {}
csd['Class7.1'] = {}
csd['Class7.2'] = {}
csd['Class7.3'] = {}
csd['Class8.1'] = {}
csd['Class8.2'] = {}
csd['Class8.3'] = {}
csd['Class8.4'] = {}
csd['Class8.5'] = {}
csd['Class8.6'] = {}
csd['Class8.7'] = {}
csd['Class9.1'] = {}
csd['Class9.2'] = {}
csd['Class9.3'] = {}
csd['Class10.1'] = {}
csd['Class10.2'] = {}
csd['Class10.3'] = {}
csd['Class11.1'] = {}
csd['Class11.2'] = {}
csd['Class11.3'] = {}
csd['Class11.4'] = {}
csd['Class11.5'] = {}
csd['Class11.6'] = {}


for i=1,nSamples do
   table.insert(csd['GalaxyID'], out[i][1])
   table.insert(csd['Class1.1'], out[i][2]) 
   table.insert(csd['Class1.2'], out[i][3]) 
   table.insert(csd['Class1.3'], out[i][4]) 
   table.insert(csd['Class2.1'], out[i][5]) 
   table.insert(csd['Class2.2'], out[i][6]) 
   table.insert(csd['Class3.1'], out[i][7]) 
   table.insert(csd['Class3.2'], out[i][8]) 
   table.insert(csd['Class4.1'], out[i][9]) 
   table.insert(csd['Class4.2'], out[i][10]) 
   table.insert(csd['Class5.1'], out[i][11]) 
   table.insert(csd['Class5.2'], out[i][12]) 
   table.insert(csd['Class5.3'], out[i][13]) 
   table.insert(csd['Class5.4'], out[i][14]) 
   table.insert(csd['Class6.1'], out[i][15]) 
   table.insert(csd['Class6.2'], out[i][16]) 
   table.insert(csd['Class7.1'], out[i][17]) 
   table.insert(csd['Class7.2'], out[i][18]) 
   table.insert(csd['Class7.3'], out[i][19]) 
   table.insert(csd['Class8.1'], out[i][20]) 
   table.insert(csd['Class8.2'], out[i][21]) 
   table.insert(csd['Class8.3'], out[i][22]) 
   table.insert(csd['Class8.4'], out[i][23]) 
   table.insert(csd['Class8.5'], out[i][24]) 
   table.insert(csd['Class8.6'], out[i][25]) 
   table.insert(csd['Class8.7'], out[i][26]) 
   table.insert(csd['Class9.1'], out[i][27]) 
   table.insert(csd['Class9.2'], out[i][28]) 
   table.insert(csd['Class9.3'], out[i][29]) 
   table.insert(csd['Class10.1'], out[i][30]) 
   table.insert(csd['Class10.2'], out[i][31]) 
   table.insert(csd['Class10.3'], out[i][32]) 
   table.insert(csd['Class11.1'], out[i][33]) 
   table.insert(csd['Class11.2'], out[i][34]) 
   table.insert(csd['Class11.3'], out[i][35]) 
   table.insert(csd['Class11.4'], out[i][36]) 
   table.insert(csd['Class11.5'], out[i][37]) 
   table.insert(csd['Class11.6'], out[i][38]) 
end

-- print(csd)
csvigo.save{path=paths.concat(opt.save, 'submission.csv'), data = csd}
