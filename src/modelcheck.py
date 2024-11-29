import torch
from model import InpaintGenerator

model = torch.load('./src/pretrain/G0000000.pt')
inpainter = InpaintGenerator()
inpainter.load_state_dict(model)
print(model)