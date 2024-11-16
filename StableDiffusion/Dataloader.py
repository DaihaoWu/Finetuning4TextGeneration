import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class ImageTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, tokenizer=None, seq_len=256):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_paths = self._get_all_paths()

    def _get_all_paths(self, root_path=None):
        # Default to using self.root_dir if root_path is not provided
        root_path = Path(root_path or self.root_dir)
        data_paths = []

        def search_directory(directory):
            # Recursively search directories for "image.jpg" and "caption.txt"
            image_path = directory / 'image.jpg'
            caption_path = directory / 'caption.txt'

            # Check if both files exist in the current directory
            if image_path.exists() and caption_path.exists():
                data_paths.append((str(image_path), str(caption_path)))

            # Recursively search subdirectories
            for subdir in directory.iterdir():
                if subdir.is_dir():
                    search_directory(subdir)

        # Start the search from the root_path
        search_directory(root_path)
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
        if self.tokenizer:
            tokenized_text = self.tokenizer(
                text,
                max_length=self.seq_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            # Extract tokenized tensor
            text_tokens = tokenized_text["input_ids"].squeeze()

            return {
            "text_tokens": text_tokens,  # Tokenized text
            "text": text,  # Original text
            "image": image  # Processed image
            }
        else:
            return {
            "text": text,  # Original text
            "image": image  # Processed image
            }


if __name__ == '__main__':
    from transformers import BertTokenizer
    # Instantiate the tokenizer, dataset, and DataLoader
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define the image transformation to normalize images to range [0, 1] and convert to tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Set up the dataset and DataLoader
    # "D:/Finetuning_DALLE/Finetuning4TextGeneration/Datasets/0/00000"      Will's dataset path
    # "/w/340/michaelyuan/Finetuning4TextGeneration/Datasets/MARIO-10M/0/00000"    Michael's dataset path
    root_dir = "/w/340/michaelyuan/Finetuning4TextGeneration/Datasets/MARIO-10M/0/00000"    # Relative dir like "/Datasets/..." don't work here, not sure why
    dataset = ImageTextDataset(root_dir=root_dir, transform=transform, tokenizer=None)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in data_loader:
        print(batch)
        break
