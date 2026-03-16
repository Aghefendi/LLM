# -*- coding: utf-8 -*-
"""Gradio Web Arayüzü"""

import gradio as gr
from src.inference.model_loader import load_peft_model
from src.inference.evaluator import IlacHastalikEvaluator
from src.utils.config import PEFT_MISTRAL_MODEL_ID, ILAC_LISTESI, HASTALIK_LISTESI


# Model yükleme
print("Model yükleniyor...")
model, tokenizer, device = load_peft_model(PEFT_MISTRAL_MODEL_ID)
evaluator = IlacHastalikEvaluator(model, tokenizer, device, model_type="mistral")

# Geçerli girişler
valid_entries = ILAC_LISTESI + [f"{h} hastası" for h in HASTALIK_LISTESI]
valid_drugs = ILAC_LISTESI


def evaluate_ilac_hastalik(ilac: str, istenilen: str) -> str:
    """İlaç-hastalık değerlendirme fonksiyonu"""
    
    # Giriş doğrulaması
    if not ilac or not istenilen:
        return "Lütfen hem ilaç hem de hastalık bilgilerini giriniz."
    
    if len(ilac) > 50 or len(istenilen) > 50:
        return "Girdiğiniz bilgiler çok uzun. Lütfen daha kısa bir açıklama yapınız."
    
    if ilac.lower() not in [v.lower() for v in valid_entries]:
        return "Girdiğiniz ilaç veya hastalık listede bulunmuyor. Lütfen geçerli bir değer giriniz."
    
    if istenilen.lower() not in [v.lower() for v in valid_drugs]:
        return "Girdiğiniz ikinci değer desteklenen ilaç listesinde bulunmuyor. Lütfen geçerli bir ilaç giriniz."
    
    # Model değerlendirmesi
    return evaluator.get_response(ilac, istenilen)


# Gradio Arayüzü
interface = gr.Interface(
    fn=evaluate_ilac_hastalik,
    inputs=[
        gr.Textbox(
            label="Kullandığınız ilaç veya hastalık durumunuzu giriniz?",
            placeholder="Örneğin: aspirin"
        ),
        gr.Textbox(
            label="Kullanmak istediğiniz ilaç",
            placeholder="Örneğin: grip ilacı"
        )
    ],
    outputs=gr.Textbox(label="Değerlendirme Sonucu", lines=10),
    title="İlaç ve Hastalık Değerlendirici",
    description="İlaç etkileşimlerini ve potansiyel sağlık risklerini değerlendiren bir araç. Yalnızca desteklenen ilaç ve hastalık isimlerini kabul eder.",
    examples=[
        ["miğren", "antidepresan"],
        ["tansiyon hastası", "grip ilacı"],
        ["tansiyon hastası", "potasyum"],
        ["idrar söktürücü", "tansiyon"],
        ["böbrek hastası", "idrar söktürücü"],
        ["kan sulandırıcı", "ağrı kesici"],
        ["astım hastası", "antibiyotik"],
        ["diyabet hastası", "ağrı kesici"]
    ]
)


if __name__ == "__main__":
    interface.launch(debug=True)
