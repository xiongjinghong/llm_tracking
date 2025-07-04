from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

class VLMConfig(PretrainedConfig):
  model_type = "vlm_model"
  def __init__(self,llm_model_path = './Qwen2.5-0.5B-Instruct',
               vision_model_path = './siglip-base-patch16-224',
               freeze_vision_model = True,
               image_pad_num = 49,
              **kwargs):
      self.vision_model_path = vision_model_path
      self.llm_model_path = llm_model_path
      self.freeze_vision_model = freeze_vision_model
      self.image_pad_num = image_pad_num
      super().__init__(**kwargs)
      
      
      
class VLM(PreTrainedModel):
  config_class = VLMConfig
  def __init__(self, config):
      super().__init__(config)
      self.config = config
      self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
      self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
      self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
      self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
      self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
      self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
      if self.config.freeze_vision_model:
          for param in self.vision_model.parameters():
              param.requires_grad = False
      for param in self.llm_model.parameters():
          param.requires_grad = False
      
  def forward(self, input_ids, labels, pixel_values, attention_mask=None):
      text_embeds = self.llm_model.get_input_embeddings()(input_ids)

      image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
      # text_embeds.shape torch.Size([8, 109, 896])
      # image_embeds.shape torch.Size([8, 196, 768])
      b, s, d = image_embeds.shape
      image_embeds = image_embeds.view(b, -1, d*4)  # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
      image_features = self.linear2(F.silu(self.linear1(image_embeds)))
      
      text_embeds = text_embeds.to(image_features.dtype)
      
      inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
      outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
      logits = outputs[0]
      loss = None
      if labels is not None:
          loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
          loss = loss_fct(
              logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
          )
      return CausalLMOutputWithPast(loss=loss, logits=logits)
      
  def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
      
      num_images, num_image_patches, embed_dim = image_features.shape
      # 2, 49, 896
      batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])

      #98 * 896
      #torch.Size([98, 896])

      inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
      
      return inputs_embeds
    

    
    
