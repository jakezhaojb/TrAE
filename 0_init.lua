require 'nn'
require 'image'
require 'gfx.js'
dofile 'util.lua'

inputSize = 32*32
outputSize = 32*32
regSize = 40
tranSize = 3
genSize = 40
-- TODO Tentative
capsule = torch.Tensor({{40,40}, {40,40}, {40,40}})
delx = 1
dely = -1
