encoder = nn.Sequential()

local stage1 = nn.ConcatTable()
-- TODO identical capsule settings?
for i=1, (#capsule)[1] do
   local split = nn.ParallelTable()
   split:add(nn.Linear(inputSize, regSize))
   split:add(nn.Identity())
   stage1:add(split)
end
encoder:add(stage1)

-- stage2 p, dx, dy
local stage2 = nn.ParallelTable()
for i=1,(#capsule)[1] do
   local tmp = nn.Sequential()
   local trSplit = nn.ParallelTable() -- struct forward transformation
   local splitUp = nn.ConcatTable()
   -- for p
   local splitProb = nn.Sequential()
   splitProb:add(nn.Linear(regSize, 1))
   splitProb:add(nn.Linear(1, genSize))
   splitProb:get(2).accGradParameters = function() end -- keep fixed
   splitProb:get(2).weight:fill(1)
   splitProb:get(2).bias:zero()
   splitUp:add(splitProb)
   -- for transform
   splitUp:add(nn.Linear(regSize, tranSize-1))
   trSplit:add(splitUp)
   trSplit:add(nn.Identity()) -- convey transformation
   local forwd = function(x) return {x[1][1], {x[1][2], x[2]}} end
   local backwd = function(x) return {{x[1], x[2][1]}, x[2][2]} end
   tmp:add(trSplit)
   tmp:add(nn.ReshapeTable(forwd, backwd))
   local postTran = nn.ParallelTable()
   postTran:add(nn.Identity())
   local postTranDown = nn.Sequential()

   --TODO Customize the transformation adopted.
   postTranDown:add(nn.CAddTable())
   postTranDown:add(nn.Linear(tranSize-1, genSize))
   postTran:add(postTranDown)
   tmp:add(postTran)

   -- add into main line
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

encoder:cuda()
criterion:cuda()
