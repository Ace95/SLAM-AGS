import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import Model, ResNet18Classifier
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--split", type=str, required=True, help="Train or Test")
parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
parser.add_argument("--dim",type=int,required=True, help="Encoder output dimension")
parser.add_argument("--model", type=str, help="Path to pretrained model")
parser.add_argument("--pre", type=str, help="[weakly, self, wcs]")
args = parser.parse_args()

def load_encoder(checkpoint_path, device, pre):
    
    if pre=="weakly":
        model = ResNet18Classifier(num_classes=2, weights=None)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)

        model = nn.Sequential(*list(model.backbone.children())[:-1])
        model = model.to(device)
        model.eval()

    else:
        model = Model(base_model='resnet18', out_dim=args.dim)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

    return model

def extract_embeddings(bags_root, output_root, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_encoder(checkpoint_path, device,pre=args.pre)

    mean = (0.6009, 0.5513, 0.7394)
    std = (0.1982, 0.2448, 0.1076)
    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        normalize,
    ])

    os.makedirs(output_root, exist_ok=True)

    for bag_name in sorted(os.listdir(bags_root)):
        bag_path = os.path.join(bags_root, bag_name)
        if not os.path.isdir(bag_path):
            continue

        embeddings = []
        names = []
        for img_name in sorted(os.listdir(bag_path)):
            img_path = os.path.join(bag_path, img_name)
            if not (img_name.endswith(".png") or img_name.endswith(".jpg")):
                continue

            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(x).cpu().numpy()  # expected (1, 512)
                if args.pre == "sil":
                    feat = feat.squeeze()

            embeddings.append(feat)
            names.append(img_name)

        embeddings = np.concatenate(embeddings, axis=0)  # (num_patches, 512)
        np.save(os.path.join(output_root, f"{bag_name}_features.npy"),
                {"feature": embeddings,
                 "file_name": names})
        print(f"Saved {bag_name}: {embeddings.shape}")

    
extract_embeddings(
        bags_root=args.data_dir,           
        output_root="output/path/embeddings", 
        checkpoint_path=args.model)
