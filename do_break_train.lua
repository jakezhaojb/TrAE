dofile("0_init.lua")

dofile("1_data.lua")

dofile("2_trae.lua")
encoder = nil

filename = ''
if paths.filep(filename) == false then
   print('no model file found.')
   sys.exit()
end
encoder = torch.load(filename):cuda()

dofile("3_train.lua")
assert(type(train) == 'function')

for i = 1, 5 do
   train()
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

