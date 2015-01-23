torch.setnumthreads(8)
require 'cutorch'
require 'cunn'
require 'image'
--require 'gfx.js'
require 'xlua'
require 'optim'
dofile('/home/jz1672/gpu_lock.lua')

dofile 'util.lua'
dofile './Modules/init.lua'


cutorch.setDevice(2)
torch.setdefaulttensortype('torch.FloatTensor')

inputW = 32
inputH = 32
inputSize = inputW*inputW
outputSize = inputW*inputW
regSize = 10
tranSize = 3
genSize = 20
-- TODO Tentative
--capsule = torch.Tensor({{40,40}, {40,40}, {40,40}, {40,40}, {40,40}, {40,40}, {40,40}, {40,40}})
capsule = torch.rand(30)

flagDebug = false
if flagDebug then
   print("============ Hey man I am in DEBUG mode =============")
else
   print("============ Let's get started! ============")
end
