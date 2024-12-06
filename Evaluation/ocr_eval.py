# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import os
import re
import copy

gts = {
    # 'ChineseDrawText': [],
    # 'DrawBenchText': [],
    # 'DrawTextCreative': [],
    # 'LAIONEval4000': [],
    # 'OpenLibraryEval500': [],
    'TMDBEval500': [],
}

results = {
    'stablediffusion': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    'lora_stablediffusion': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    'enhanced_lora_stablediffusion': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    # 'textdiffuser': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    # 'controlnet': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    # 'deepfloyd': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
}

def get_key_words(text: str):
    words = []
    text = text
    matches = re.findall(r"'(.*?)'", text) # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())
   
    return words


# load gt
files = os.listdir('../Datasets/MARIOEval/')    # /path/to/MARIOEval
for file in files:
    lines = open(os.path.join('../Datasets/MARIOEval/', file, f'{file}.txt')).readlines()
    for line in lines:
        line = line.strip().lower()
        try:
            gts[file].append(get_key_words(line))
        except:
            pass
print(gts['TMDBEval500'][:10])


def get_p_r_acc(method, pred, gt):

    pred = [p.strip().lower() for p in pred] 
    gt = [g.strip().lower() for g in gt]

    pred_orig = copy.deepcopy(pred)
    gt_orig = copy.deepcopy(gt)

    pred_length = len(pred)
    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            pred_orig.remove(p) 
            gt_orig.remove(p)

    p = (pred_length - len(pred_orig)) / (pred_length + 1e-8)
    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)
   
    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    if ''.join(pred_sorted) == ''.join(gt_sorted):
        acc = 1
    else:
        acc = 0

    return p, r, acc

def splite_filename(file):
    filename = file.strip()

    # Split the filename into parts
    parts = filename.split('_')

    # Extract the last three parts as dataset, prompt_index, and image_index
    dataset = parts[-3]
    prompt_index = parts[-2]
    image_index = parts[-1].split('.')[0]  # Remove the .txt extension if present

    # Everything before the dataset is part of the method
    method = '_'.join(parts[:-3])

    return method, dataset, prompt_index, image_index


ocr_path = "/w/340/michaelyuan/Finetuning4TextGeneration/MaskTextSpotterV3/ocr_result"
files = os.listdir(ocr_path)
print(len(files))

for file in files:
    # method, dataset, prompt_index, image_index = file.strip().split('_')    # file looks like method_dataset_prompindex_imageindex.txt
    method, dataset, prompt_index, image_index = splite_filename(file)

    ocrs = open(os.path.join(ocr_path, file)).readlines()
    if not ocrs:
        continue
    p, r, acc = get_p_r_acc(method, ocrs, gts[dataset][int(prompt_index)])
    results[method]['cnt'] += 1
    results[method]['p'] += p
    results[method]['r'] += r
    results[method]['acc'] += acc

for method in results.keys():
    results[method]['p'] /= results[method]['cnt']
    results[method]['r'] /= results[method]['cnt']
    results[method]['f'] = 2 * results[method]['p'] * results[method]['r'] / (results[method]['p'] + results[method]['r'] + 1e-8)
    results[method]['acc'] /= results[method]['cnt']
    
print(results)
