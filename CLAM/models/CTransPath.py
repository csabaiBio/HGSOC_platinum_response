import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from .ctran import ctranspath


def CTransPath():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load('/mnt/ncshare/ozkilim/BRCA/TransPath/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)

    return model, transform


if __name__ == "__main__":
    model, transform = CTransPath()
    
    print(model)