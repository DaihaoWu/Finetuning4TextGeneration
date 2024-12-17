import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dalle2_pytorch import DALLE2, Unet, Decoder, CLIP, DecoderTrainer, OpenAIClipAdapter
from dalle2_pytorch.dataloaders import ImageEmbeddingDataset, create_image_embedding_dataloader

# Initialize CLIP
# openai pretrained clip - defaults to ViT-B/32
clip = OpenAIClipAdapter()

# Two-stage U-Net setup
unet1 = Unet(
    dim=128,
    image_embed_dim=512,
    # text_embed_dim=512,
    cond_dim=128,
    channels=3,
    dim_mults=(1, 2, 4, 8),
    # cond_on_text_encodings=False,
).cuda()

unet2 = Unet(
    dim=16,
    image_embed_dim=512,
    cond_dim=128,
    channels=3,
    dim_mults=(1, 2, 4, 8, 16),
).cuda()

# Decoder setup
decoder = Decoder(
    unet=(unet1, unet2),
    image_sizes=(128, 256),
    clip=clip,
    timesteps=1000
).cuda()

# Decoder trainer setup
decoder_trainer = DecoderTrainer(
    decoder,
    lr=3e-4,
    wd=1e-2,
    ema_beta=0.99,
    ema_update_after_step=1000,
    ema_update_every=10,
)

# Define the path for saving the model
model_path = "./PretrainedModels/decoder2_50k_checkpoint_313.pth"
# Load the saved state dictionary into the model
decoder.load_state_dict(torch.load(model_path))
decoder.eval()  # Set to evaluation mode if not training

# Sampling from the trained model (after sufficient training)
# mock_image_embed = torch.randn(9, 512).cuda()  # Mock image embeddings, replace with actual embeddings
image_embed = torch.from_numpy(np.load("/w/340/michaelyuan/Finetuning4TextGeneration/Datasets/MSCOCO/image_embeddings/img_emb_00000.npy"))[:9]
print(image_embed.shape)
generated_images = decoder_trainer.sample(image_embed=image_embed)
print(generated_images.shape)  # Expected output shape: (32, 3, 256, 256)

import matplotlib.pyplot as plt
import torchvision.transforms as T
import os

# Assuming `generated_images` is a tensor with shape (32, 3, 256, 256)
# Move to CPU if needed
if generated_images.is_cuda:
    generated_images = generated_images.cpu()

# Normalize pixel values to [0, 1] if needed
generated_images = (generated_images - generated_images.min()) / (generated_images.max() - generated_images.min())

# Select a subset of images to display (e.g., first 9 images)
num_images_to_show = 9
images_to_show = generated_images[:num_images_to_show]

# Create a grid for visualization
fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # 3x3 grid
axes = axes.flatten()

model_id = os.path.splitext(os.path.basename(model_path))[0]
# Display each image
for img, ax in zip(images_to_show, axes):
    # Permute dimensions from (C, H, W) to (H, W, C) for plotting
    img = T.ToPILImage()(img)
    ax.imshow(img)
    ax.axis("off")  # Hide axes

plt.tight_layout()
plt.savefig(f"./plots/{model_id}_samples.jpg")
plt.show()