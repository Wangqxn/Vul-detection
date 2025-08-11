# executable_feature_extraction.py
import os
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from vit_model import vit_base_patch16_224_in21k as create_model
import torch.nn as nn

class BinaryDataset(Dataset):
    def __init__(self, file_paths, save_folder=None):
        self.file_paths = file_paths
        self.save_folder = save_folder
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Changed to 224x224 for ViT compatibility
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        if self.save_folder and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx]
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        image_side = int(math.sqrt(file_size))
        with open(file_path, 'rb') as file:
            binary_data = file.read()
        num_pixels = image_side * image_side
        if len(binary_data) < num_pixels:
            binary_data += b'\x00' * (num_pixels - len(binary_data))
        elif len(binary_data) > num_pixels:
            binary_data = binary_data[:num_pixels]
        image = Image.frombytes('L', (image_side, image_side), binary_data)
        if self.save_folder:
            image_save_path = os.path.join(self.save_folder, filename + '.png')
            image.save(image_save_path)
        image_tensor = self.transform(image)
        return image_tensor, label, filename


class ViTAllTokens(nn.Module):
    def __init__(self):
        super().__init__()
        # Create base ViT model
        self.vit = create_model(num_classes=0, has_logits=False)  # Remove classification head

        # Modify to keep all tokens
        self.vit.head = nn.Identity()  # Remove final classification layer
        self.vit.pre_logits = nn.Identity()  # Remove pre-logits layer

    def forward(self, x):
        # Convert grayscale to 3 channels (ViT expects 3 channels)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # Get patch embeddings
        x = self.vit.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # Forward through transformer blocks
        x = self.vit.blocks(x)
        x = self.vit.norm(x)  # [B, num_patches+1, embed_dim]

        # Project to target dimension (240)
        if x.shape[-1] != 240:
            proj = nn.Linear(x.shape[-1], 240).to(x.device)
            x = proj(x)

        return x


def binary_to_grayscale_and_infer(binary_folder, output_folder):
    """
    Convert binary files to grayscale images and extract features using ViT

    Args:
        binary_folder (str): Path to directory containing binary files
        output_folder (str): Path to save converted grayscale images

    Returns:
        tuple: (image tokens numpy array, list of filenames)
    """
    # Prepare file paths and labels
    file_paths = []
    labels = []
    filenames = []
    for filename in os.listdir(binary_folder):
        if filename.endswith('.exe'):
            file_path = os.path.join(binary_folder, filename)
            mark = int(filename.split('_')[0])
            file_paths.append((file_path, mark))
            labels.append(mark)
            filenames.append(filename)

    # Create dataset and dataloader
    dataset = BinaryDataset(file_paths, save_folder=output_folder)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Initialize ViT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = ViTAllTokens().to(device)
    vit_model.eval()

    # Extract image tokens
    all_image_tokens = []
    with torch.no_grad():
        for inputs, _, names in data_loader:
            inputs = inputs.to(device)
            outputs = vit_model(inputs)  # Output shape: (batch_size, 197, 240)
            all_image_tokens.append(outputs.cpu().numpy())

    image_tokens = np.concatenate(all_image_tokens, axis=0)
    return image_tokens, filenames