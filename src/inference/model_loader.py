# -*- coding: utf-8 -*-
"""Model yükleme fonksiyonları"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def load_peft_model(peft_model_id: str, load_in_4bit: bool = True):
    """PEFT modelini yükle"""
    config = PeftConfig.from_pretrained(peft_model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        load_in_4bit=load_in_4bit,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device


def load_base_model(model_id: str, load_in_8bit: bool = True):
    """Base model yükleme"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer
