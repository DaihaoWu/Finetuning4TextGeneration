## Datasets
MARIO-10M:
    Obtained directly from Huggingface. Check the original tutorial: https://huggingface.co/datasets/JingyeChen22/TextDiffuser-MARIO-10M 

MARIOEval:
    a comprehensive text rendering benchmark containing 10k prompts at [link](https://drive.google.com/file/d/1_tnWtOqC6S4_D4z8bqcBQ9xKPlsoPB0B/view)

## Prepare the dataset
MARIO-10M dataset require another process step before it can be use to fine-tuning stable diffussion, run the following command: (Modified the path in the script to safe in your desire directory)
``` bash
    python Datasets_converter.py
```

To generate our text-enhanced dataset, run: (Modified the path in the script to safe in your desire directory)
``` bash
    python Datasets_enhancement.py
```
