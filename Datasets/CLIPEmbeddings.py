import webdataset as wds
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
from tqdm import tqdm

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# WebDataset path
dataset_path = "./MSCOCO/{00001..00059}.tar"    # should be {00000..00059}

# Directory to save embeddings
image_emb_dir = "./MSCOCO/image_embeddings"
# text_emb_dir = "text_embeddings"
os.makedirs(image_emb_dir, exist_ok=True)
# os.makedirs(text_emb_dir, exist_ok=True)

# Process each shard
current_shard_num = None
image_embeddings = []
text_embeddings = []

dataset = wds.WebDataset(dataset_path).decode("pil").to_tuple("jpg", "txt", "__key__")

for image, text, key in tqdm(dataset, desc="Processing Shards"):
    # Extract shard number and index from key
    shard_num, index = key[:5], key[5:]
    print(key, shard_num, index)
    index = int(index)

    # Initialize new shard if necessary
    if current_shard_num != shard_num:
        # Save previous shard's embeddings
        if current_shard_num is not None:
            np.save(f"{image_emb_dir}/img_emb_{current_shard_num}.npy", np.array(image_embeddings))
            # np.save(f"{text_emb_dir}/text_emb_{current_shard_num}.npy", np.array(text_embeddings))

        # Reset for the new shard
        current_shard_num = shard_num
        image_embeddings = [None] * 10000  # Assuming 10,000 possible images per shard
        text_embeddings = [None] * 10000

    # Preprocess inputs
    # inputs = processor(images=image, text=text, return_tensors="pt", padding=True)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # Generate embeddings
        image_embedding = model.get_image_features(**inputs).cpu().numpy().squeeze()
        # text_embedding = model.get_text_features(**inputs).cpu().numpy().squeeze()

    # Store embeddings at the correct index
    image_embeddings[index] = image_embedding
    # text_embeddings[index] = text_embedding

# Save last shard
if current_shard_num is not None:
    np.save(f"{image_emb_dir}/img_emb_{current_shard_num}.npy", np.array(image_embeddings))
    # np.save(f"{text_emb_dir}/text_emb_{current_shard_num}.npy", np.array(text_embeddings))

print("All embeddings saved.")