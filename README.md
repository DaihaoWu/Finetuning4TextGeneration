# Finetuning4TextGeneration

server directory: ~/w/340



GPU for HPC: 
'''bash
srun --partition=gpunodes -c 1 --mem=2G --gres=gpu:1 -t 60 --pty bash --login

Inference Baseline Images:
open folder "StableDiffusion" ---> run Example_Run.py to generate baseline images

Inference Lora-Finetuned Images:
1. Follow the instruction on "https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README.md" to install the dependencies
2. open the "diffuser_docs" folder ---> use "inference.py" to generate images

If failed, then try below:

copy and paste all the files inside the "diffuser_docs" folder to "diffusers\examples\text_to_image" and try running there.