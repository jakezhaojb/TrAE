-- SGD training
assert(TrImgs ~= nil)

batchSize = 128

function train()
   epoch = epoch or 1
   local time = sys.clock()
   -- set training model
   encoder:training()

   shuf = torch.randpermTable(TrImgs)

   print('==> training')
   print('epoch #' .. epoch .. '[ batchSize = ' .. batchSize .. ']')
   
   for t = 1,#shuf,batchSize do
      xlua.progress(t, #shuf)
      -- create a mini-batch
      input = {}
      target = {}

      
   end
   



end
