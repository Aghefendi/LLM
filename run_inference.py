#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Komut satırından model inference"""

from src.inference.model_loader import load_peft_model
from src.inference.evaluator import IlacHastalikEvaluator
from src.utils.config import PEFT_MISTRAL_MODEL_ID


def main():
    """Ana fonksiyon"""
    print("Model yükleniyor...")
    model, tokenizer, device = load_peft_model(PEFT_MISTRAL_MODEL_ID)
    evaluator = IlacHastalikEvaluator(model, tokenizer, device, model_type="mistral")
    
    print("\n=== İlaç-Hastalık Etkileşim Değerlendirici ===\n")
    
    while True:
        ilac = input("Kullandığınız ilaç veya hastalığınızı giriniz (çıkmak için 'q'): ")
        if ilac.lower() == 'q':
            break
            
        istenilen = input("Kullanmak istediğiniz ilacı giriniz: ")
        
        print("\nDeğerlendirme yapılıyor...\n")
        cevap = evaluator.get_response(ilac, istenilen)
        print(f"\nSonuç:\n{cevap}\n")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
