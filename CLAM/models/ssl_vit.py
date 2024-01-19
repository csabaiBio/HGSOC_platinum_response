import torch
from timm.models.vision_transformer import VisionTransformer
import torchvision

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        _ = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url,
                                               progress=progress)
        )

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.70322989, 0.53606487, 0.66096631],
            std=[0.21716536, 0.26081574, 0.20723464]
        )
    ])
    
    return model, transform


if __name__ == "__main__":
    # initialize ViT-S/16 trunk using DINO pre-trained weight
    model = vit_small(pretrained=True, progress=False,
                      key="DINO_p16", patch_size=16)
