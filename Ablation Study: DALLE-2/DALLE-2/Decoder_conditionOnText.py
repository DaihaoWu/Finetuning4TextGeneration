import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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

# Define preprocessing function
def img_preproc(img):
    """
    Preprocessing function for images. Converts the image into a torch tensor.
    You can add more transformations (like normalization) here.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        # Optionally, you can add normalization here as well
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
    ])
    return transform(img)   

# Dataset and DataLoader setup
dataset = ImageEmbeddingDataset(
    urls="Datasets/MSCOCO/{00000..00000}.tar",
    img_embedding_folder_url="Datasets/MSCOCO/image_embeddings",
    index_width=4,
    shuffle_shards=True,
    resample=False,
    img_preproc = img_preproc,
)

dataloader = DataLoader(
    dataset,
    batch_size=32,  # Adjust based on your memory and needs
    # shuffle=True,   # Shuffle the dataset
    num_workers=1,  # Number of workers for parallel data loading
    pin_memory=True,  # Set to True for faster data transfer to GPU (if using GPU)
)

#Training loop
print("Start training:")
for epoch in range(1):  # Example epoch count, adjust as necessary
    for img, emb in dataloader:
        # Transfer batch to GPU
        img = img.cuda()
        img_emb = emb["img"].cuda()  # Assuming text embeddings are in emb["img"], update key if different

        # Train each U-Net in the decoder
        for unet_number in (1, 2):
            # Compute loss
            loss = decoder_trainer(
                image=img,
                image_embed = img_emb,
                unet_number=unet_number,
                max_batch_size=4
            )

            # Update the specific U-Net with EMA
            decoder_trainer.update(unet_number)

        print(f'Epoch [{epoch+1}] - Loss: {loss:.4f}')

# Sampling from the trained model (after sufficient training)
mock_image_embed = torch.randn(32, 512).cuda()  # Mock image embeddings, replace with actual embeddings
generated_images = decoder_trainer.sample(image_embed=mock_image_embed)
print(generated_images.shape)  # Expected output shape: (32, 3, 256, 256)