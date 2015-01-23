train_file = 'mnist/train_32x32.t7'
test_file = 'mnist/test_32x32.t7'

assert(paths.filep(train_file))
assert(paths.filep(test_file))

print '==> loading dataset'

loaded = torch.load(train_file, 'ascii')
--loadedImgs = loaded.data:transpose(3,4)
loadedImgs = loaded.data:clone()

-- TODO size branches
-- cmd:()
-- Now tentative
trSize = loadedImgs:size(1)
imgs = torch.Tensor(trSize,(#loadedImgs)[2],(#loadedImgs)[3],(#loadedImgs)[4])
idx = torch.randperm((#imgs)[1])[{ {1,trSize} }]
for i = 1,trSize do
   imgs[{ i,{},{},{} }] = loadedImgs:select(1,idx[i]):clone()
end
loadedImgs = nil
collectgarbage()

-- TODO rgb2gray?
print '==> Data preprocessing'
if imgs:size(2) == 3 then
   -- rgb2grey
      rgb2grey = function(img)
         local im = img:clone()
         im = torch.squeeze(im):type('torch.DoubleTensor')
         local imGrey = im[{1,{},{}}] * .2126 + im[{2,{},{}}] * .7152 + im[{3,{},{}}] * .0722
         return imGrey
      end
   imgsGrey = torch.zeros((#imgs)[1], (#imgs)[3], (#imgs)[4])
   for i = 1,(#imgs)[1] do
      imgsGrey[i] = rgb2grey(imgs[{ i,{},{},{} }])
   end
   imgs = imgsGrey:clone()
   imgsGrey = nil
elseif imgs:size(2) == 1 then
   imgs = torch.squeeze(imgs)
else
   sys.exit()
end
collectgarbage()

-- Shifting
-- TODO customize to other transformations
xTb = {-2,-1,0,1,2}
yTb = {-2,-1,0,1,2}
   -- closure
   local getShiftImgs = function(imgs, xTb, yTb)
      local shiftImgs = {}
      -- form a multinomial picking up the shifting
      local xLongTb = randint(xTb, (#imgs)[1])
      local yLongTb = randint(yTb, (#imgs)[1])
      for i = 1, (#imgs)[1] do
         shiftImgs[i] = {}
         shiftImgs[i].delx = xLongTb[i]
         shiftImgs[i].dely = yLongTb[i]
         local imgElem = imgs[i]:clone()
         local xShift = torch.zeros((#imgs)[3], (#imgs)[2])
         local yShift = torch.zeros((#imgs)[2], (#imgs)[3])
         -- column manipulation, right multiplication, shift x
         if shiftImgs[i].delx > 0 then
            local tranSize = (#imgs)[2] - shiftImgs[i].delx
            xShift[{ {1, tranSize}, {shiftImgs[i].delx+1, (#imgs)[2]} }] = torch.eye(tranSize)
         elseif shiftImgs[i].delx < 0 then
            local tranSize = (#imgs)[3] + shiftImgs[i].delx
            xShift[{ {-shiftImgs[i].delx+1, (#imgs)[3]}, {1, tranSize} }] = torch.eye(tranSize)
         else -- == 0
            xShift = torch.eye((#imgs)[3], (#imgs)[2])
         end
         -- row manipulation, left multiplication, shift y
         if shiftImgs[i].dely > 0 then
            local tranSize = (#imgs)[3] - shiftImgs[i].dely
            yShift[{ {shiftImgs[i].dely+1, (#imgs)[3]}, {1, tranSize} }] = torch.eye(tranSize)
         elseif shiftImgs[i].delx < 0 then
            local tranSize = (#imgs)[2] + shiftImgs[i].dely
            yShift[{ {1, tranSize}, {-shiftImgs[i].dely+1, (#imgs)[2]} }] = torch.eye(tranSize)
         else -- == 0
            yShift = torch.eye((#imgs)[2], (#imgs)[3])
         end
         shiftImgs[i].TrX = normalize(yShift * imgs[i] * xShift)
         shiftImgs[i].X = normalize(imgs[i])
         -- pad boundary by mean value
         --shiftImgs[i].TrX = shiftImgs[i].TrX + shiftImgs[i].TrX:eq(0):type('torch.FloatTensor'):mul(shiftImgs[i].X:mean())
      end
      return shiftImgs
   end

print '==> form transformed images'
local TrImgs = getShiftImgs(imgs, xTb, yTb)

-- More preparations here
trData = {}
for i = 1,#TrImgs do
   trData[i] = {}
   trData[i][1] = (TrImgs[i].X:reshape(inputSize)):cuda()
   trData[i][2] = (torch.Tensor({ TrImgs[i].delx, TrImgs[i].dely })):cuda()
end
trTgtData = {}
for i = 1,#TrImgs do
   trTgtData[i] = (TrImgs[i].TrX:reshape(inputSize)):cuda()
end
