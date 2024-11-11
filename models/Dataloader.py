import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
import numpy as np

class ImageTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, tokenizer=None, seq_len=256):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_paths = self._get_all_paths()

    def _get_all_paths(self):
        # Collect all paths to "image.jpg" and "caption.txt" in the nested directory structure
        data_paths = []
        for subdir_0 in os.listdir(self.root_dir):
            subdir_0_path = os.path.join(self.root_dir, subdir_0)
            if os.path.isdir(subdir_0_path):
                for subdir_1 in os.listdir(subdir_0_path):
                    subdir_1_path = os.path.join(subdir_0_path, subdir_1)
                    if os.path.isdir(subdir_1_path):
                        image_path = os.path.join(subdir_1_path, 'image.jpg')
                        caption_path = os.path.join(subdir_1_path, 'caption.txt')
                        if os.path.exists(image_path) and os.path.exists(caption_path):
                            data_paths.append((image_path, caption_path))
        return data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, caption_path = self.data_paths[idx]

        # Load the image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load the corresponding text
        with open(caption_path, 'r') as f:
            text = f.read().strip()

        # Tokenize the text
        tokenized_text = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Extract tokenized tensor
        text_tokens = tokenized_text["input_ids"].squeeze()

        return text_tokens, image

# Instantiate the tokenizer, dataset, and DataLoader
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the image transformation to normalize images to range [0, 1] and convert to tensor
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Set up the dataset and DataLoader
dataset = ImageTextDataset(root_dir='Dataset', transform=transform, tokenizer=tokenizer)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
