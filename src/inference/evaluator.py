# -*- coding: utf-8 -*-
"""İlaç-Hastalık Etkileşim Değerlendirici"""

import torch
from typing import Optional
from src.utils.config import ILAC_LISTESI, HASTALIK_LISTESI


class IlacHastalikEvaluator:
    """İlaç ve hastalık etkileşimlerini değerlendiren sınıf"""
    
    def __init__(self, model, tokenizer, device, model_type: str = "mistral"):
        """
        Args:
            model: Yüklenmiş model
            tokenizer: Tokenizer
            device: Cihaz (cuda/cpu)
            model_type: Model tipi ("llama" veya "mistral")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type
        
    def generate_prompt(self, ilac: str, istenilen: str) -> Optional[str]:
        """Prompt oluştur"""
        if self.model_type == "llama":
            prompt_ilac = f"<|begin_of_text|> İlaç etkileşimlerini değerlendirerek, potansiyel sağlık risklerini açıklayın.\n {ilac} ve {istenilen} birlikte kullanımı hangi kronik hastalıklara neden olur? \n"
            prompt_hastalik = f"<|begin_of_text|> İlaç etkileşimlerini değerlendirerek, potansiyel sağlık risklerini açıklayın.\n {ilac} hastasıyım ve {istenilen} birlikte kullanımı hangi kronik hastalıklara neden olur? \n"
        else:  # mistral
            prompt_ilac = f"[INST] İlaç etkileşimlerini değerlendirerek, potansiyel sağlık risklerini açıklayın.\n {ilac} ve {istenilen} birlikte kullanımı hangi kronik hastalıklara neden olur? \n[/INST]"
            prompt_hastalik = f"[INST] İlaç etkileşimlerini değerlendirerek, potansiyel sağlık risklerini açıklayın.\n {ilac} hastasıyım ve {istenilen} birlikte kullanımı hangi kronik hastalıklara neden olur? \n[/INST]"
        
        if self.is_ilac(ilac):
            return prompt_ilac
        elif self.is_hastalik(ilac):
            return prompt_hastalik
        else:
            return None
    
    def is_ilac(self, text: str) -> bool:
        """İlaç kontrolü"""
        return any(ilac.lower() in text.lower() for ilac in ILAC_LISTESI)
    
    def is_hastalik(self, text: str) -> bool:
        """Hastalık kontrolü"""
        return any(hastalik.lower() in text.lower() for hastalik in HASTALIK_LISTESI)
    
    def get_response(self, ilac: str, istenilen: str, max_new_tokens: int = 256) -> str:
        """Model yanıtı al"""
        prompt = self.generate_prompt(ilac, istenilen)
        
        if not prompt:
            return "Geçersiz giriş. Lütfen geçerli bir ilaç veya hastalık girin."
        
        batch = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Model tipine göre generation parametreleri
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        if self.model_type == "llama":
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            gen_kwargs.update({
                "num_return_sequences": 1,
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 2,
                "eos_token_id": terminators
            })
        
        output_tokens = self.model.generate(**batch, **gen_kwargs)
        generated_answer = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Prompt'u temizle
        return generated_answer.replace(prompt, '').strip()
