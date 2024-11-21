import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Paths
dataset_root = "MARIO-10M"
output_dir = "enhanced_mario_10m"
metadata_file = os.path.join(output_dir, "metadata.jsonl")

# Initialize storage for metadata
metadata_entries = []

# Function to convert JPG to PNG with the specified naming format
def mask_and_convert(image_path, charseg_path, output_folder, relative_path):
    # Derive the new name based on the relative path
    parts = relative_path.split(os.sep)  # Split by directory separators
    new_name = "".join(parts) + ".png"  # Combine parts into a single name
    png_path = os.path.join(output_folder, new_name)

    # Load image as RGB
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Load the mask
    mask = np.load(charseg_path).astype(bool)  # Convert to a boolean array if not already
    mask = Image.fromarray(mask).resize(image.size, Image.NEAREST)
    mask = np.array(mask)
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")  # Display in grayscale
    plt.title("Character Segmentation Mask")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, "sample_mask.jpg"))
    plt.show()

    # Apply the mask: keep pixels in the mask, set the rest to white
    masked_image_array = np.where(mask[..., None], image_array, 255)  # Extend mask to 3 channels (RGB)

    # Convert the masked array back to an image
    masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
    
    # Save the image as PNG
    masked_image.save(png_path, "PNG")
    return new_name

# Specify range of data to convert
start_subfolder = "0/00000"
end_subfolder = "0/00001"  # Change as needed

# Traverse the dataset structure
for root, dirs, files in os.walk(dataset_root):
    # Filter directories based on the specified range
    relative_path = os.path.relpath(root, dataset_root)
    if start_subfolder <= relative_path <= end_subfolder:
        if "image.jpg" in files:
            # Paths to image and metadata
            image_path = os.path.join(root, "image.jpg")
            caption_path = os.path.join(root, "caption.txt")
            ocr_path = os.path.join(root, "ocr.txt")
            charseg_path = os.path.join(root, "charseg.npy")

            # visualize charseg.npy
            img = Image.open

            # Convert image to PNG with the specified naming format
            os.makedirs(output_dir, exist_ok=True)
            file_name = mask_and_convert(image_path, charseg_path, output_dir, relative_path)

            # Load metadata
            caption = ""
            ocr = ""
            if os.path.exists(caption_path):
                with open(caption_path, "r") as f:
                    caption = f.read().strip()
            if os.path.exists(ocr_path):
                with open(ocr_path, "r") as f:
                    ocr = f.read().strip()

            # Add to metadata entries
            metadata_entries.append({
                "file_name": file_name,
                "caption": caption,
                "ocr": ocr
            })

# Save metadata to JSONL file
with open(metadata_file, "w") as f:
    for entry in metadata_entries:
        json.dump(entry, f)
        f.write("\n")

print(f"Processed images and metadata saved to {output_dir} and {metadata_file}")