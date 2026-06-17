# -*- coding: utf-8 -*-
"""Proje konfigürasyon ayarları"""

# Model IDs
# Temel (base) modeller
# NOT: "mistral" adaptörü aslında Trendyol-LLM-7b (Mistral-7B mimarisi tabanlı,
# Türkçe sohbet modeli) üzerine eğitilmiştir; adapter_config.json bunu doğrular.
LLAMA_MODEL_ID = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
MISTRAL_MODEL_ID = "Trendyol/Trendyol-LLM-7b-chat-v0.1"
# Fine-tuned LoRA adaptörleri (Hugging Face)
PEFT_LLAMA_MODEL_ID = "magahcicek/turkish-llama8b-drug-interaction-lora"
PEFT_MISTRAL_MODEL_ID = "magahcicek/mistral7b-drug-interaction-lora"

# Dataset
DATASET_NAME = "magahcicek/turkish-drug-interaction-qa"
DATASET_SIZE = 260

# LoRA Configuration
LORA_CONFIG = {
    "llama": {
        "r": 64,
        "alpha": 128,
        "dropout": 0.05
    },
    "mistral": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05
    }
}

# Training Arguments
TRAINING_CONFIG = {
    "llama": {
        "epochs": 1,
        "max_steps": 40,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "save_steps": 20,
        "eval_steps": 20
    },
    "mistral": {
        "epochs": 1,
        "max_steps": 20,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "save_steps": 25,
        "eval_steps": 25
    }
}

# İlaç ve Hastalık Listeleri
ILAC_LISTESI = [
    "aspirin", "parol", "klopidogrel", "metformin", "ağrı kesici",
    "grip ilacı", "kan sulandırıcı", "antibiyotik", "depresyon",
    "potasyum", "miğren", "anestezi", "sakinleştirici", "barbitürat",
    "warfarin", "antidepresan", "alerji", "hormon", "potasyum takviyesi",
    "ateş düşürücü", "kas gevşetici", "mide koruyucu", "sinir ilacı",
    "iltihap kurutucu", "idrar söktürücü", "ürik asit azaltıcı", "adrenalin"
]

HASTALIK_LISTESI = [
    "hipertansiyon", "diyabet", "kalp", "astım", "şeker", "tansiyon",
    "depresyon", "kanser", "karaciğer", "koah", "böbrek", "obezite",
    "romatizma", "kansızlık", "zatürre", "nezle", "grip", "gut"
]
