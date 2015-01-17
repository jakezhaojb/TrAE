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

randpermTable = function(x, idx)
   y = {}
   len = table.getn(x)
   if idx == nil then
      idx = torch.randperm(len)
   end
   for i = 1,(#idx)[1] do
      y[i] = deepcopy(x[idx[i]])
   end
   return y, idx
end

normalize = function(x)
   normalized_x = x:clone()
   normalized_x:div(255.0)
   return normalized_x
end
