require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'paths'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('GalaxyZoo Test script')
cmd:text()
cmd:text('Options:')
cmd:option('-gpuid', 1, 'gpu id')
cmd:option('-save', 'testresults',   'subdirectory to save results in')
cmd:option('-id', 0, 'Give an id to tag these results with')
cmd:option('-model', 'none', 'path to model')
cmd:option('-data', 'data/test', 'root dir of test images')
cmd:text()
opt = cmd:parse(arg or {})
if opt.id == 0 then
   dok.error('Give valid id')
end
opt.save = paths.concat(opt.save, opt.id)

dofile('../1_datafunctions.lua')

cutorch.setDevice(opt.gpuid)

-- load model
model = torch.load(opt.model)
os.execute('mkdir -p ' .. opt.save)

bmode = 'DHWB' -- depth x height x width x batch
for f in paths.files(opt.data) do
   -- for each image in dir,
   if paths.filep(f) and string.sub(f, #f-3) == '.jpg' then
      local basename = string.sub(f, 1, #f-4)
      local f = paths.concat(opt.data, f)
      local im = image.load(f, 3)
      im = expandTestSample(im, false)  -- generate 128 test samples
      im = im:cuda()
      local out = model:forward(im) -- fprop through model
      out = out:float()
      for i=1,out:size(1) do
	 out[i] = normalizedToOriginal(out[i]) -- normalize the output
      end
      torch.save(paths.concat(opt.save, basename), out) -- save it to output file
   end
end

 

  

