encoder = nn.Sequential()

local stage1 = nn.ConcatTable()
-- TODO identical capsule settings?
for i=1, (#capsule)[1] do
   stage1:add(nn.Linear(inputSize, regSize))
end
encoder:add(stage1)

-- stage2 p, dx, dy
local stage2 = nn.ParallelTable()
for i=1,(#capsule)[1] do
   local tmp = nn.Sequential()
   local split = nn.ConcatTable()
   -- for p
   local splitProb = nn.Sequential()
   splitProb:add(nn.Linear(regSize, 1))
   splitProb:add(nn.Linear(1, genSize))
   splitProb:get(2).accGradParameters = function() end -- keep fixed
   splitProb:get(2).weight:fill(1)
   splitProb:get(2).bias:zero()
   split:add(splitProb)
   -- for transform
   local splitTrans = nn.Sequential()
   splitTrans:add(nn.Linear(regSize, tranSize-1))
   --TODO Customize the transformation adopted.
   local transformLayer = nn.Add(2)
   local delxy = torch.Tensor({delx, dely})
   transformLayer.bias:copy(delxy)
   transformLayer.accGradParameters = function() end -- keep fixed
   splitTrans:add(transformLayer)
   splitTrans:add(nn.Linear(tranSize-1, genSize))
   split:add(splitTrans)
   -- combine
   tmp:add(split)
   tmp:add(nn.CMulTable())
   stage2:add(tmp)
end
encoder:add(stage2)
encoder:add(nn.CAddTable())
-- geneLayer to output; TODO to confirm..
encoder:add(nn.Linear(genSize, outputSize))

-- loss
criterion = nn.MSECriterion()
criterion.sizeAverage = false
