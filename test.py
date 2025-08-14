import torch
from lib.models.ego_tokenhmr import Model

x = torch.randn((1, 128, 9)).cuda()
model = Model().cuda()
y = model(x)
for k, v in y.items():
    print(k, v.shape)