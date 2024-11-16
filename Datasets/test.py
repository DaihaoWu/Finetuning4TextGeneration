from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="processed_mario_10m", split="train")
print(dataset[0])