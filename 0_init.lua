require 'nn'
require 'image'
require 'gfx.js'
require 'xlua'
require 'optim'

dofile 'util.lua'
dofile './Modules/init.lua'

inputSize = 32*32
outputSize = 32*32
regSize = 40
tranSize = 3
genSize = 40
-- TODO Tentative
capsule = torch.Tensor({{40,40}, {40,40}, {40,40}})
