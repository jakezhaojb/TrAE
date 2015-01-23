-- 1. load the saved net
-- TODO gfx.image() how to save
dofile('0_init.lua')

filename = './save/model.net.Jan.20.13_45_14.2015'

encoder = torch.load(filename)

visualizeWeight = function(nnModule)
   local w = nnModule.weight:clone()
   local nImg = w:size(1)
   local nSize = w:size(2)
   local nSizeSqrt = math.floor(math.sqrt(nSize))
   w = w[{ {}, {1,nSizeSqrt^2} }]
   w = w:reshape(nImg, nSizeSqrt, nSizeSqrt)
   for i = 1, nImg do
      w[i] = w[i]:div(w[i]:norm())
   end
   local zoomOpt = 1 --default
   if nSizeSqrt < 40 then
      zoomOpt = math.ceil(40/nSizeSqrt)
   end
   if w:size(1) > 49 then
      gfx.image(w[{ {1,49}, {}, {}}], {zoom=zoomOpt, legend=''})
   else
      gfx.image(w, {zoom=zoomOpt, legend=''})
   end

   return w
end


visualizeOutput = function(enc)
   dofile("1_data.lua")
   local trDataFrac = {}
   local outputData = {}
   for i = 1,10 do 
      trDataFrac[i] = {}
      trDataFrac[i][1] = (trData[i][1]):float()
      trDataFrac[i][2] = (trData[i][2]):float()
      outputData[i] = enc:forward(trDataFrac[i]):reshape(inputH, inputW)
   end
   inImgs = torch.Tensor():resize(10, inputH, inputW):type('torch.FloatTensor')
   outImgs = torch.Tensor():resize(10, inputH, inputW):type('torch.FloatTensor')
   for i = 1,10 do
      inImgs[i] = trDataFrac[i][1]:resizeAs(inImgs[i])
      outImgs[i] = outputData[i][1]:resizeAs(outImgs[i])
   end
   gfx.image(inImgs, {legend=''})
   gfx.image(outImgs, {legend=''})
   
   return inImgs, outImgs
end


visualizeInput = function(num)
   dofile("1_data.lua")
   assert(num < 100)
   local tr = torch.Tensor(num, inputH, inputW)
   local tgt = torch.Tensor(num, inputH, inputW)
   for i=1,3 do
      local a, b, c, _
      a, b = randpermTable(trData)
      c, _ = randpermTable(trTgtData, b)
   end
   for i = 1, num do
      tr[i] = trData[i][1]:reshape(inputH, inputW)
      tgt[i] = trTgtData[i]:reshape(inputH, inputW)
   end
   gfx.image(tr, {legend=""})
   gfx.image(tgt, {legend=""})

   return tr, tgt
end


-- TODO visualize all the recognition and generation weights here.
--wtG = visualizeWeight(encoder:get(4))
--wtR = visualizeWeight(encoder:get(1):get(1):get(1):get(1))

--_, __ = visualizeOutput(encoder)

_, __ = visualizeInput(81)
