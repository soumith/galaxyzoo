require 'torch'
require 'cutorch'

torch.setdefaulttensortype('torch.FloatTensor')
cmd = torch.CmdLine()
cmd:text()
cmd:text('GalaxyZoo Training script')
cmd:text()
cmd:text('Options:')
cmd:option('-seed',            1,           'fixed input seed for repeatable experiments')
cmd:option('-threads',         1,           'number of threads')
cmd:option('-gpuid',           1,           'gpu id')
cmd:option('-save',            'results',   'subdirectory to save/log experiments in')
cmd:option('-learningRate',    5e-2,        'learning rate at t=0')
cmd:option('-momentum',          0.6,        'momentum')
cmd:option('-weightDecay',       1e-5,        'weight decay')
cmd:option('-batchSize',       32,           'mini-batch size (1 = pure stochastic)')
cmd:option('-progressBar',     true,       'Display a progress bar')
cmd:option('-dataTest',     false,       'visual sanity checks for data loading')
cmd:option('-dropout',     true,       'do dropout with 0.5 probability')
cmd:option('-retrain',     "none",       'provide path to model to retrain with')
cmd:text()
opt = cmd:parse(arg or {})


-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuid)


-- loadSize   = {3, 256, 256}
-- sampleSize = {3, 223, 223}
-- dataroot = 'data/images_training_rev1_256'
loadSize   = {3, 128, 128}
sampleSize = {3, 111, 111}
dataroot = 'data/images_training_rev1_128'

lightTesting = true

epochSize = opt.batchSize * 1000

dofile('1_data.lua')
if not opt.dataTest then
   dofile('2_model.lua')
   dofile('3_train.lua')
   dofile('4_test.lua')

   epoch = 0
   test()
   while true do
      epoch = epoch + 1
      collectgarbage()
      train()
      collectgarbage()
      test()
      if epoch == 50 then lightTesting = false; end
   end
end
