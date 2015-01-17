-- 1. load the saved net
-- TODO gfx.image() how to save

filename = ''

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
   gfx.image(w, {zoom=zoomOpt, legend=''})
end

-- TODO visualize all the recognition and generation weights here.
