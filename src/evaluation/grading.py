# -*- coding: utf-8 -*-
"""Serbest metin model çıktılarını etiketlere çeviren ve metrik hesaplayan modül.

Saf Python; ağır bağımlılık (torch/transformers) gerektirmez, böylece GPU
olmadan da test edilebilir. Model çıktısı serbest Türkçe metin olduğundan,
"etkileşim var/yok" kararı basit bir anahtar-kelime sezgiseliyle çıkarılır.
Bu, önerilen "basit doğruluk oranı" yaklaşımıdır.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

# Çıktıda "risk/etkileşim var" sinyali veren ifadeler.
# NOT: Yalın "etkilesim" kelimesi RİSK sinyali DEĞİLDİR; çünkü olumsuz
# cümlelerde de geçer ("anlamlı etkileşim beklenmez"). Onun yerine açık
# risk/yan-etki kökleri ve "etkileşim var/mevcut" gibi olumlu kalıplar aranır.
RISK_INDICATORS = [
    "risk", "artar", "artabilir", "artir", "artis", "yukselt", "yuksel",
    "tehlike", "kanama", "yan etki", "advers", "zararl", "onerilmez",
    "kacinil", "dikkat", "olumsuz", "komplikasyon", "toksik", "hiperkalemi",
    "bronkospazm", "aritmi", "solunum", "baskil", "serotonin sendromu",
    "nefrotoksik", "hepatotoksi", "kotules", "bozab", "bozar", "boz ",
    "azalt", "sedasyon", "hasar", "zorlas", "tetikle",
    "yol acab", "neden olab", "neden olur",
    "etkilesime", "ciddi etkilesim", "onemli etkilesim", "etkilesim var",
    "etkilesim mevcut", "etkilesim soz konusu",
]

# Çıktıda "etkileşim yok / güvenli" sinyali veren ifadeler
SAFE_INDICATORS = [
    "etkilesim yok", "etkilesim bulunmamak", "anlamli etkilesim yok",
    "anlamli bir etkilesim", "guvenli", "guvenle", "sorun olusturmaz",
    "birlikte kullanilabil", "risk tasimaz", "zararsiz", "beklenmez",
    "sakinca yok", "sakincasi yok",
]


def _normalize(text: str) -> str:
    """Türkçe karakterleri sadeleştirip küçük harfe çevirir (eşleştirme için)."""
    text = text.casefold()
    # ı/İ ve diğer aksanları ASCII'ye indir
    replacements = {"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c", "â": "a", "î": "i", "û": "u"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def predict_interaction(output_text: str) -> Optional[bool]:
    """Model çıktısından 'etkileşim var mı?' kararını çıkarır.

    Dönüş:
        True  -> çıktı bir etkileşim/risk tarif ediyor
        False -> çıktı güvenli/etkileşim yok diyor
        None  -> karar verilemiyor (boş veya belirsiz çıktı)
    """
    if not output_text or not output_text.strip():
        return None

    norm = _normalize(output_text)
    has_safe = any(ind in norm for ind in SAFE_INDICATORS)
    has_risk = any(ind in norm for ind in RISK_INDICATORS)

    if has_safe and not has_risk:
        return False
    if has_risk:
        return True
    if has_safe:
        return False
    return None


def mechanism_recall(output_text: str, expected_keywords: List[str]) -> Optional[float]:
    """Beklenen mekanizma anahtar kelimelerinin çıktıda bulunma oranı.

    Beklenen kelime yoksa None döner (recall tanımsız).
    """
    if not expected_keywords:
        return None
    norm = _normalize(output_text or "")
    hits = sum(1 for kw in expected_keywords if _normalize(kw) in norm)
    return hits / len(expected_keywords)


def compute_metrics(records: List[Dict]) -> Dict:
    """Tahmin edilmiş kayıtlardan metrikleri hesaplar.

    Her kayıt: {"etkilesim_var": bool, "tahmin": Optional[bool],
                "mekanizma": [...], "cikti": str}
    """
    total = len(records)
    scored = [r for r in records if r.get("tahmin") is not None]
    abstained = total - len(scored)

    tp = sum(1 for r in scored if r["etkilesim_var"] and r["tahmin"])
    tn = sum(1 for r in scored if not r["etkilesim_var"] and not r["tahmin"])
    fp = sum(1 for r in scored if not r["etkilesim_var"] and r["tahmin"])
    fn = sum(1 for r in scored if r["etkilesim_var"] and not r["tahmin"])

    correct = tp + tn
    accuracy = correct / len(scored) if scored else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Mekanizma kapsamı (yalnızca pozitif örnekler)
    recalls = [
        mechanism_recall(r.get("cikti", ""), r.get("mekanizma", []))
        for r in records if r["etkilesim_var"]
    ]
    recalls = [x for x in recalls if x is not None]
    mech_coverage = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "toplam": total,
        "puanlanan": len(scored),
        "cekimser": abstained,
        "dogru": correct,
        "dogruluk": accuracy,
        "kesinlik": precision,
        "duyarlilik": recall,
        "f1": f1,
        "mekanizma_kapsami": mech_coverage,
        "karisiklik_matrisi": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def format_report(metrics: Dict, model_id: str = "?") -> str:
    """Metrikleri okunabilir bir Markdown raporuna çevirir."""
    cm = metrics["karisiklik_matrisi"]
    lines = [
        "# Değerlendirme Raporu",
        "",
        f"- **Model:** `{model_id}`",
        f"- **Test seti boyutu:** {metrics['toplam']} örnek "
        f"(puanlanan: {metrics['puanlanan']}, çekimser: {metrics['cekimser']})",
        f"- **Kaynak:** TİTCK KÜB §4.5 (klasik, belgelenmiş etkileşimler)",
        "",
        "## Etkileşim Tespiti (var/yok)",
        "",
        "| Metrik | Değer |",
        "|--------|-------|",
        f"| Doğruluk (accuracy) | **{metrics['dogruluk']:.1%}** |",
        f"| Kesinlik (precision) | {metrics['kesinlik']:.1%} |",
        f"| Duyarlılık (recall) | {metrics['duyarlilik']:.1%} |",
        f"| F1 | {metrics['f1']:.1%} |",
        f"| Mekanizma kapsamı | {metrics['mekanizma_kapsami']:.1%} |",
        "",
        "## Karışıklık Matrisi",
        "",
        "| | Tahmin: Var | Tahmin: Yok |",
        "|---|---|---|",
        f"| **Gerçek: Var** | {cm['TP']} (TP) | {cm['FN']} (FN) |",
        f"| **Gerçek: Yok** | {cm['FP']} (FP) | {cm['TN']} (TN) |",
        "",
        "> Not: Etkileşim var/yok kararı, serbest metin çıktıdan anahtar-kelime",
        "> sezgiseliyle çıkarılır. Bu, basit ve şeffaf bir doğruluk ölçütüdür;",
        "> klinik geçerlilik iddiası taşımaz.",
    ]
    return "\n".join(lines)
