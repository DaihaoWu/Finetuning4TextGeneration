from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# input the path to the model
model_path = "sd-texted-model-lora"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)

prompt = "a photo of a cat holding a sign saying hello world"

pipe.to("cuda")
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
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    images.append(image)
    output_dir = f"{model_path}\InferenceImages"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
final_image = concatenate_images(images)

# save the final image and change the name of the file here
final_image.save(f"{model_path}\InferenceImages\combined_image.png")