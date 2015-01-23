assert(flagDebug == true)


encoder = nn.Sequential()

encoder:add(nn.SelectTable(1))

encoder:add(nn.Linear(inputSize, 256))
encoder:add(nn.Sigmoid())
encoder:add(nn.Linear(256, 64))
encoder:add(nn.Sigmoid())
encoder:add(nn.Linear(64, 16))
encoder:add(nn.Sigmoid())
encoder:add(nn.Linear(16, inputSize))
encoder:add(nn.Sigmoid())

criterion = nn.MSECriterion()
criterion.sizeAverage = false

encoder:cuda()
criterion:cuda()

