This file specify how the datasets for training the model is acquired

MARIO-10M:
    Obtained directly from Huggingface. Check the original tutorial: https://huggingface.co/datasets/JingyeChen22/TextDiffuser-MARIO-10M 

MSCOCO: 600k image/text pairs
    1. Use img2datset to generate a webdataset. Tutorial can be seen here: https://github.com/rom1504/img2dataset/blob/main/dataset_examples/mscoco.md
        url for webdataset: https://github.com/webdataset/webdataset
    2. ! This does not work! Use clip-retrieval to generate clip embeddings for the dataset. Checkout CLIP inference section in this https://github.com/rom1504/clip-retrieval 
    instead
    2. use Datasets/CLIPEmbeddings.py to generate .npy clip embedding files for image