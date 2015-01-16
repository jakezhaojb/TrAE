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

-- TODO verify
deepcopy = function(orig)
   local orig_type = type(orig)
   local copy
   if orig_type == 'table' then
       copy = {}
       for orig_key, orig_value in next, orig, nil do
           copy[deepcopy(orig_key)] = deepcopy(orig_value)
       end
       setmetatable(copy, deepcopy(getmetatable(orig)))
   else -- number, string, boolean, etc
       copy = orig
   end
   return copy
end

randpermTable = function(x)
   y = {}
   len = table.getn(x)
   idx = torch.randperm(len)
   for i = 1,(#idx)[1] do
      y[i] = deepcopy(x[idx[i]])
   end
   return y
   

end
