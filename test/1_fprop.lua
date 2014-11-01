require 'torch'
require 'nn'
require 'nnx'
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
cmd:option('-id', '0', 'Give an id to tag these results with')
cmd:option('-model', 'none', 'path to model')
cmd:option('-data', 'data/images_test_rev1_128', 'root dir of test images')
cmd:option('-dryrun', false, 'fill output with ones.')
cmd:text()
opt = cmd:parse(arg or {})
if opt.id == '0' then
   dok.error('Give valid id')
end
opt.save = paths.concat(opt.save, opt.id)

sampleSize = {3, 111, 111}
loadSize = {3, 128, 128}
dofile('1_datafunctions.lua')

-- load model
if not opt.dryrun then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpuid)
   model = torch.load(opt.model)
end
os.execute('mkdir -p ' .. opt.save)

index = 1
for f in paths.files(opt.data) do
   -- for each image in dir,
   local basename = string.sub(f, 1, #f-4)
   local f = paths.concat(opt.data, f)
   if paths.filep(f) and string.sub(f, #f-3) == '.jpg' then
      local out
      if opt.dryrun then
	 out = torch.ones(512, 37)
      else
	 xlua.progress(index, 79935); index = index + 1;
	 local im = image.load(f, 3)
	 im = expandTestSample(im, false)  -- generate 512 test samples
	 im = im:cuda()	 
	 out = model:forward(im) -- fprop through model
	 out = out:float()
	 for i=1,out:size(1) do
	    out[i] = normalizedToOriginal(out[i]) -- normalize the output
	 end
      end
      torch.save(paths.concat(opt.save, basename), out) -- save it to output file	 
   end
end

 

  

