# -*- coding: utf-8 -*-
"""Llama Model Eğitim Scripti"""

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils.data_processor import load_and_prepare_dataset, prepare_training_data
from src.utils.config import (
    LLAMA_MODEL_ID, DATASET_NAME, DATASET_SIZE,
    LORA_CONFIG, TRAINING_CONFIG
)
from src.inference.model_loader import load_base_model


def train_llama_model(output_dir: str = "./outputs_llama", repo_name: str = None):
    """Llama modelini eğit"""
    
    # Prompt templates
    prompt_template = """<|begin_of_text|> İlaç etkileşimlerini değerlendirerek, potansiyel sağlık risklerini açıklayın. {soru} \n """
    answer_template = """{uzun_cevap}"""
    
    # Dataset hazırlama
    print("Dataset yükleniyor...")
    dataset = load_and_prepare_dataset(
        DATASET_NAME, DATASET_SIZE, prompt_template, answer_template
    )
    
    # Model yükleme
    print("Model yükleniyor...")
    model, tokenizer = load_base_model(LLAMA_MODEL_ID)
    
    # Veri hazırlama
    print("Veri işleniyor...")
    split_dataset = prepare_training_data(dataset, tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        model=model, tokenizer=tokenizer, padding='max_length', pad_to_multiple_of=8
    )
    
    # LoRA konfigürasyonu
    print("LoRA uygulanıyor...")
    lora_config = LoraConfig(
        r=LORA_CONFIG["llama"]["r"],
        lora_alpha=LORA_CONFIG["llama"]["alpha"],
        lora_dropout=LORA_CONFIG["llama"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj", "lm_head"],
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    cfg = TRAINING_CONFIG["llama"]
    training_args = TrainingArguments(
        fp16=True,
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=0.001,
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        do_eval=True,
        report_to="none",
        logging_dir="./logs",
        logging_steps=20,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )
    
    # Eğitim
    print("Eğitim başlıyor...")
    model.config.use_cache = False
    trainer.train()
    
    # Model kaydetme
    if repo_name:
        print(f"Model {repo_name} repo'suna yükleniyor...")
        trainer.model.push_to_hub(repo_name)
    
    return model, tokenizer


if __name__ == "__main__":
    train_llama_model()
