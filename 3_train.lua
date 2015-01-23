-- SGD training
batchSize = 256

parameters, gradParameters = encoder:getParameters()

-- SGD doesn't work. Way too ill-conditioned.
sgdOptimState = {
   learningRate = 1e-3,
   weightDecay = 1e-6,
   momentum = 0.9,
   learningRateDecay = 1e-4
}

function train()
   epoch = epoch or 1
   local time = sys.clock()
   -- set training model
   encoder:training()

   shufTrData, idx = randpermTable(trData) --TODO
   shufTrTgtData, _ = randpermTable(trTgtData, idx)

   print('==> training')
   print('epoch #' .. epoch .. '[ batchSize = ' .. batchSize .. ']')
   
   for t = 1,#shufTrData,batchSize do
      -- create a mini-batch
      local inputs = {}
      local targets = {}
      for i = t, math.min(t+batchSize-1, trSize) do
         local input = deepcopy(shufTrData[i])
         local target = deepcopy(shufTrTgtData[i])
         table.insert(inputs, input)
         table.insert(targets, target)
      end
      
      -- closure for loss and gradients
      local feval = function(x)
         -- ensure using the newest parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         gradParameters:zero()

         f = 0
         for i = 1, #inputs do
            local output = encoder:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            f = f + err
            -- gradients
            local df_do = criterion:backward(output, targets[i])
            encoder:backward(inputs[i], df_do)
         end
         
         gradParameters:div(#inputs)
         f = f/#inputs
         print("loss: ".. f) 

         return f, gradParameters
      end -- end of feval
   
   optim.sgd(feval, parameters, sgdOptimState)

   xlua.progress(math.min(t+batchSize, trSize), #shufTrData)
   end -- for loop on each epoch

   -- finish one epoch
   time = sys.clock() - time
   time = time / trSize
   print("\n==>time to learn 1 sample = " .. (time * 1000) .. 'ms')
   print("loss: ".. f) 

   -- TODO save the net.
   epoch = epoch + 1

end -- end of train()
