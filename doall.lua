dofile("0_init.lua")

dofile("1_data.lua")

dofile("2_trae.lua")

dofile("3_train.lua")

--[[
-- LBFGS
for i = 1, 10 do
   train(2)
end
-- derive time for model name
current = os.date()
pos = current:find(" ") -- remove Sat
current = string.sub(current, pos)
name = current:gsub("%s+", "")
name = 'model.net' .. name
-- save the model
filename = paths.concat('save', name)
--safeguard
if paths.filep(filename) then
   filename = paths.concat(filename, '.1')
end
os.execute("mkdir -p " .. sys.dirname(filename))
print("==> saving model to" .. filename)
torch.save(filename, encoder)
--]]

-- SGD
for i = 1, 20 do
   train(1)
end
-- derive time for model name
current = os.date()
pos = current:find(" ") -- remove Sat
current = string.sub(current, pos)
name = current:gsub("%s+", ".")
name = name:gsub(":", "_")
name = 'model.net' .. name
-- save the model
filename = paths.concat('save', name)
--safeguard
if paths.filep(filename) then
   filename = paths.concat(filename, '.1')
end
os.execute("mkdir -p " .. sys.dirname(filename))
print("==> saving model to" .. filename)
torch.save(filename, encoder:float())

-- LBFGS
for i = 1, 10 do
   train(2)
end

-- derive time for model name
current = os.date()
pos = current:find(" ") -- remove Sat
current = string.sub(current, pos)
name = current:gsub("%s+", ".")
name = name:gsub(":", "_")
name = 'model.net' .. name
-- save the model
filename = paths.concat('save', name)
--safeguard
if paths.filep(filename) then
   filename = paths.concat(filename, '.1')
end
os.execute("mkdir -p " .. sys.dirname(filename))
print("==> saving model to" .. filename)
torch.save(filename, encoder:float())

