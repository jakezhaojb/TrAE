dofile('init.lua')

encoder = nn.Sequential()

stage1 = nn.ConcatTable()
-- TODO identical capsule settings?
for i=1, (#capsule)[1] do
   stage1:add(nn.Linear(inputSize, regSize))
end
encoder:add(stage1)

-- stage2 p, dx, dy
stage2 = nn.ParallelTable()
for i=1,(#capsule)[1] do
   tmp = nn.Sequential()
   tmp:add(nn.Linear(regSize, tranSize))
   split = nn.ConcatTable()
   -- for p
   Isplit1 = torch.zeros(tranSize)
   Isplit1[1] = 1
   split:add(nn.Linear(tranSize,1))
   split:get(1).weight = Isplit1:reshape(tranSize,1)
   -- for x and y
   Isplit2 = torch.eye(tranSize)[{{}, {2,tranSize}}]
   split2 = nn.Sequential()
   split2:add(nn.Linear(tranSize,tranSize-1))
   split2:get(1).weight = Isplit2
   delxy = torch.Tensor({delx, dely})
   split2:add(nn.Add(delxy))
   split2:add(nn.Linear(tranSize, genSize))
   split:add(split2)
   -- combine
   tmp:add(split)
   tmp:add(nn.CMulTable())
   stage2:add(tmp)
end
encoder:add(stage2)
encoder:add(nn.CAddTable())
