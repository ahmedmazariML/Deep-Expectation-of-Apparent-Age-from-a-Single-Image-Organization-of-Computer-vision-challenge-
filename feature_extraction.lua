require 'torch'
require 'image'
require 'nn'
csvfile = require "src/simplecsv"

input = csvfile.read('output/cleaned.csv')

output = nil
output = {}

path = "input/wiki_crop/"


net = torch.load('input/VGG_FACE.t7')
net:evaluate()

net.modules[35] = nil
net.modules[36] = nil
net.modules[37] = nil
net.modules[38] = nil
net.modules[39] = nil
net.modules[40] = nil
collectgarbage()
net:evaluate()


--10001-20000
for i = 2, 3 do

    im = image.load(path .. input[i][2], 3, 'float')

    im_scaled = image.scale(im, "224x224")
    im_scaled = im_scaled * 255

    mean = {129.1863,104.7624,93.5940}
    im_bgr = im_scaled:index(1, torch.LongTensor{3,2,1})

    for h = 1, 3 do
        im_bgr[h]:add(-mean[h])
    end
    features=net(im_bgr)

    output[#output + 1] = torch.totable(features)

end

csvfile.write('output/output.csv', output)