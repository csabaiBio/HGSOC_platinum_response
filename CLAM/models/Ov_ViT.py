from torchvision import transforms as T
from models.dino_vit import vit_small, load_pretrained_weights

def Ov_vit_small():
    transform = T.Compose([
        # T.Resize(256, interpolation=3),
        # T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.7477, 0.5728, 0.6827), (0.1968, 0.2565, 0.1995)),
    ])
    
    model = vit_small(patch_size=16, drop_path_rate=0.1)
    # load checkpoint
    load_pretrained_weights(model, '/mnt/ncshare/ozkilim/BRCA/Ov_SSL_checkpoints/checkpoint0010.pth', 'teacher')
    
    return model, transform

if __name__ == "__main__":
    model, transform = Ov_vit_small()
    
    print(model)