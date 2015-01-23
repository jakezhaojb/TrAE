dofile("0_init.lua")

dofile("1_data.lua")

if flagDebug then
   dofile("2_ae.lua")
else
   dofile("2_trae.lua")
end

dofile("3_train.lua")


-- SGD
for i = 1, 15 do
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

if flagDebug then
   encoder:get(1).output = encoder:get(1).output:type('torch.FloatTensor')
end

torch.save(filename, encoder:float())
