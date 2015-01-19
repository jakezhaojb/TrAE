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


cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

inputSize = 32*32
outputSize = 32*32
regSize = 40
tranSize = 3
genSize = 40
-- TODO Tentative
capsule = torch.Tensor({{40,40}, {40,40}, {40,40}, {40,40}, {40,40}, {40,40}, {40,40}, {40,40}})
