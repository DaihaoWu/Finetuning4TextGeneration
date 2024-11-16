import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
from lora import LoRA
from datasets import load_dataset
import wandb

from Dataloader import ImageTextDataset


# Initialize wandb
wandb.init(
    project="stable-diffusion-lora",
    config={
        "model_name": "CompVis/stable-diffusion-v1-4",
        "learning_rate": 5e-5,
        "epochs": 3,
        "batch_size": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
)

# Load Stable Diffusion model
model_name = wandb.config.model_name
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare LoRA module
lora_config = {
    "r": wandb.config.lora_rank,  # Rank of the adaptation matrix
    "alpha": wandb.config.lora_alpha,  # Scaling factor
    "target_modules": ["cross_attention"],  # Modules to apply LoRA
}

lora = LoRA(pipe.text_encoder, config=lora_config)

# Prepare Dataset
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
root_dir = "/w/340/michaelyuan/Finetuning4TextGeneration/Datasets/MARIO-10M/0/00000"    # Relative dir like "/Datasets/..." don't work here, not sure why
dataset = ImageTextDataset(root_dir=root_dir, transform=transform, tokenizer=None)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimizer and Training Configurations
optimizer = torch.optim.AdamW(lora.parameters(), lr=wandb.config.learning_rate)
num_epochs = wandb.config.epochs
batch_size = wandb.config.batch_size

# Training Loop
for epoch in range(num_epochs):
    for step, batch in enumerate(dataset["train"].shuffle().batch(batch_size)):
        # Get inputs
        texts = batch["text"]
        images = batch["image"]  # Replace with actual image loading logic

        # Forward pass
        latents = pipe.vae.encode(images.to("cuda")).latent_dist.sample()
        latents = latents * 0.18215  # Scale as required by SD

        # Compute loss
        loss = pipe.text_encoder(texts, latents).loss
        loss.backward()

        # Update LoRA parameters
        optimizer.step()
        optimizer.zero_grad()

        # Log metrics to wandb
        wandb.log({"epoch": epoch + 1, "step": step + 1, "loss": loss.item()})

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {loss.item()}")

# Save the fine-tuned LoRA module
save_path = "path_to_save_lora_weights"
lora.save_pretrained(save_path)

# Log the final model path to wandb
wandb.log({"model_save_path": save_path})

# Finish wandb session
wandb.finish()