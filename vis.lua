-- 1. load the saved net
-- TODO gfx.image() how to save
dofile('0_init.lua')

filename = './save/model.net.Jan.23.15_45_19.2015'

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
   encoder:double()
   tgtImgs = torch.Tensor():resize(49, inputH, inputW):type('torch.FloatTensor')
   outImgs = torch.Tensor():resize(49, inputH, inputW):type('torch.FloatTensor')
   for i = 1,49 do
      tgtImgs[i] = trTgtData[i]:reshape(inputH, inputW)
      outImgs[i] = enc:forward(trData[i]):reshape(inputH, inputW)
      local err = tgtImgs[{i, {}}] - outImgs[{i, {}}]
      local errNorm = err:norm() / (tgtImgs[{i, {}}]:norm())
      print("Normalized error: " .. errNorm)
   end
   gfx.image(tgtImgs, {legend=''})
   gfx.image(outImgs, {legend=''})
   
   return tgtImgs, outImgs
end


visualizeInput = function(num)
   dofile("1_data.lua")
   assert(num < 100)
   local tr = torch.Tensor(num, inputH, inputW)
   local tgt = torch.Tensor(num, inputH, inputW)
   local a, b, c, _
   for i=1,3 do -- Test randpermTable
      a, b = randpermTable(trData)
      c, _ = randpermTable(trTgtData, b)
   end
   for i = 1, num do
      tr[i] = a[i][1]:reshape(inputH, inputW)
      tgt[i] = c[i]:reshape(inputH, inputW)
   end
   gfx.image(tr, {legend=""})
   gfx.image(tgt, {legend=""})

   return tr, tgt
end


-- TODO visualize all the recognition and generation weights here.
--wtG = visualizeWeight(encoder:get(1))
--wtR = visualizeWeight(encoder:get(1):get(1):get(1):get(1))

_, __ = visualizeOutput(encoder)

--_, __ = visualizeInput(81)
