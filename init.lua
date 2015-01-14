require 'nn'
require 'image'
require 'gfx.js'

inputSize = 28*28
regSize = 40
tranSize = 3
genSize = 40
-- TODO Tentative
capsule = torch.Tensor({{40,40}, {40,40}, {40,40}})
delx = 1
dely = -1
