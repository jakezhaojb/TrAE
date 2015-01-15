require 'torch'

randint = function(x, n)
   local res = {}
   local len = #x
   for i = 1,n do
      local randidx = torch.random(1, len)
      res[i] = x[randidx]
   end
   return res
end
