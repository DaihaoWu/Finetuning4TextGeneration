import json
import torch
import random
import numpy as np
from dalle2_pytorch import DALLE2, Unet, Decoder, CLIP, DecoderTrainer, OpenAIClipAdapter, train_configs
from dalle2_pytorch.tokenizer import tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def conditioned_on_text(config):
    try:
        return config.decoder.unets[0].cond_on_text_encodings
    except AttributeError:
        pass
    
    try:
        return config.decoder.condition_on_text_encodings
    except AttributeError:
        pass
    
    return False

decoder_text_conditioned = False
clip_config = None
def load_decoder(decoder_state_dict_path, config_file_path):
  config = train_configs.TrainDecoderConfig.from_json_path(config_file_path)
  global decoder_text_conditioned
  decoder_text_conditioned = conditioned_on_text(config)
  global clip_config
  clip_config = config.decoder.clip
  config.decoder.clip = None
  print("Decoder conditioned on text", decoder_text_conditioned)
  decoder = config.decoder.create().to(device)
  decoder_state_dict = torch.load(decoder_state_dict_path, map_location='cpu')
  decoder.load_state_dict(decoder_state_dict, strict=False)
  del decoder_state_dict
  decoder.eval()
  return decoder

model_path = "/w/340/michaelyuan/Finetuning4TextGeneration/PretrainedModels/decoder_hf/decoder.pth"
config_path = "/w/340/michaelyuan/Finetuning4TextGeneration/PretrainedModels/decoder_hf/decoder_config.json"
decoder = load_decoder(model_path, config_path)