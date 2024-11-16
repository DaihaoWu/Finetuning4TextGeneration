import os
import json
from PIL import Image

# Paths
dataset_root = "MARIO-10M"
output_dir = "processed_mario_10m"
metadata_file = os.path.join(output_dir, "metadata.jsonl")

# Initialize storage for metadata
metadata_entries = []

# Function to convert JPG to PNG with the specified naming format
def convert_to_png(image_path, output_folder, relative_path):
    # Derive the new name based on the relative path
    parts = relative_path.split(os.sep)  # Split by directory separators
    new_name = "".join(parts) + ".png"  # Combine parts into a single name
    png_path = os.path.join(output_folder, new_name)
    
    # Load the image
    img = Image.open(image_path).convert("RGB")
    
    # Save the image as PNG
    img.save(png_path, "PNG")
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

            # Convert image to PNG with the specified naming format
            os.makedirs(output_dir, exist_ok=True)
            file_name = convert_to_png(image_path, output_dir, relative_path)

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