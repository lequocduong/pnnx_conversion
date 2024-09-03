#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
import torchvision.models as models
import pnnx
from convnext import ConvNeXtV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = f'convnext-tiny-llie.pth'
# pnnx_file_path = f'convnext-tiny-llie.pt'
pnnx_file_path = f'output/convnext.pt'

generator = nn.DataParallel(ConvNeXtV2(depths=[1, 1, 2, 1], dims=[96//6, 192//6, 384//6, 768//6]))
generator.load_state_dict(torch.load(ckpt_path,map_location=torch.device(DEVICE)))
generator.eval()

model = generator.module
model.eval()
model.to(DEVICE)
# model.to("cuda")
dummy_input = torch.randn(1, 3, 1088, 1920).to(DEVICE)

# model = models.resnet18(pretrained=True)
# model = model.eval()
# dummy_input = torch.randn(1, 3, 224, 224)

opt_model = pnnx.export(model, 
                        ptpath= pnnx_file_path, 
                        inputs= dummy_input,
                        check_trace=False,
                       )

# mod = torch.jit.trace(model, dummy_input)
# mod.save(pnnx_file_path)

print(f"Model has been converted to PNNX and saved at {pnnx_file_path}")