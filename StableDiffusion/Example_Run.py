import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of a cat holding a sign saying hello world"

def concatenate_images(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    new_image = Image.new('RGB', (max_width * 2, max_height * 2))
    for i, im in enumerate(images):
        x_offset = (i % 2) * max_width
        y_offset = (i // 2) * max_height
        new_image.paste(im, (x_offset, y_offset))
    return new_image

images = []

for _ in range(4):
    image = pipe(prompt).images[0]  
    images.append(image)
final_image = concatenate_images(images)
final_image.save("D:\Finetuning_DALLE\Finetuning4TextGeneration\StableDiffusionBaselinePics\cat_hello_world.png")
