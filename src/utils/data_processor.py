# -*- coding: utf-8 -*-
"""Veri işleme fonksiyonları"""

from datasets import load_dataset, disable_caching
from functools import partial
import copy
from typing import Dict, List

disable_caching()


def load_and_prepare_dataset(dataset_name: str, size: int, prompt_template: str, answer_template: str):
    """Dataset yükleme ve hazırlama"""
    dataset = load_dataset(dataset_name, split='train')
    small_dataset = dataset.select([i for i in range(size)])
    
    def _add_text(rec):
        instruction = rec["soru"]
        response = rec["uzun_cevap"]
        
        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")
        if not response:
            raise ValueError(f"Expected a response in: {rec}")
        
        rec["prompt"] = prompt_template.format(soru=instruction)
        rec["answer"] = answer_template.format(uzun_cevap=response)
        rec["text"] = rec["prompt"] + rec["answer"]
        return rec
    
    return small_dataset.map(_add_text)


def preprocess_batch(batch: Dict[str, List], tokenizer, max_length: int = 512):
    """Batch preprocessing"""
    model_inputs = tokenizer(
        batch["text"],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
    return model_inputs


def prepare_training_data(dataset, tokenizer):
    """Eğitim verisi hazırlama"""
    _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer)
    
    encoded_dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["soru", "uzun_cevap", "prompt", "answer"],
    )
    
    processed_dataset = encoded_dataset.filter(
        lambda rec: len(rec["input_ids"]) <= tokenizer.model_max_length
    )
    
    return processed_dataset.train_test_split(test_size=0.10, seed=0)
