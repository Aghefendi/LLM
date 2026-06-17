# -*- coding: utf-8 -*-
"""Test setinin şema ve kalite kontrolü.

Eğitim/değerlendirme verisi kullanılmadan önce doğrulanmalıdır (fine-tuning
en iyi uygulaması). Bu betik şema, tekrar ve liste-uyumu kontrolleri yapar.

Kullanım:
    python -m src.evaluation.validate_test_set
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"

REQUIRED_FIELDS = {
    "id", "girdi_1", "girdi_1_tip", "girdi_2",
    "etkilesim_var", "siddet", "mekanizma", "referans_ozet", "kaynak",
}
VALID_TIP = {"ilac", "hastalik"}
VALID_SIDDET = {"major", "orta", "hafif", "yok"}


def validate():
    errors = []
    warnings = []

    if not TEST_SET.exists():
        print(f"HATA: Test seti bulunamadı: {TEST_SET}")
        print("Önce çalıştırın: python data/eval/build_test_set.py")
        return 1

    rows = [json.loads(line) for line in TEST_SET.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Modelin tanıdığı listelerle uyum kontrolü
    sys.path.insert(0, str(ROOT))
    from src.utils.config import ILAC_LISTESI, HASTALIK_LISTESI

    def bilinen_mi(text, liste):
        return any(item.lower() in text.lower() for item in liste)

    ids = Counter()
    pairs = Counter()
    for i, r in enumerate(rows):
        eksik = REQUIRED_FIELDS - set(r.keys())
        if eksik:
            errors.append(f"#{i} ({r.get('id')}): eksik alanlar {eksik}")
            continue

        ids[r["id"]] += 1
        pairs[(r["girdi_1"], r["girdi_2"])] += 1

        if r["girdi_1_tip"] not in VALID_TIP:
            errors.append(f"{r['id']}: geçersiz girdi_1_tip '{r['girdi_1_tip']}'")
        if r["siddet"] not in VALID_SIDDET:
            errors.append(f"{r['id']}: geçersiz siddet '{r['siddet']}'")
        if not isinstance(r["etkilesim_var"], bool):
            errors.append(f"{r['id']}: etkilesim_var bool olmalı")
        if r["etkilesim_var"] and r["siddet"] == "yok":
            errors.append(f"{r['id']}: etkilesim_var=True iken siddet 'yok' olamaz")
        if not r["etkilesim_var"] and r["siddet"] != "yok":
            errors.append(f"{r['id']}: etkilesim_var=False iken siddet 'yok' olmalı")

        # girdi_1 modelin tanıdığı bir terim mi? (uçtan uca çalışabilirlik)
        liste = ILAC_LISTESI if r["girdi_1_tip"] == "ilac" else HASTALIK_LISTESI
        if not bilinen_mi(r["girdi_1"], liste):
            warnings.append(f"{r['id']}: girdi_1 '{r['girdi_1']}' model listesinde bulunamadı")

    for _id, c in ids.items():
        if c > 1:
            errors.append(f"Tekrarlı id: {_id} ({c} kez)")
    for pair, c in pairs.items():
        if c > 1:
            warnings.append(f"Tekrarlı çift: {pair} ({c} kez)")

    pos = sum(1 for r in rows if r.get("etkilesim_var"))
    neg = len(rows) - pos

    print(f"Test seti: {TEST_SET}")
    print(f"Örnek sayısı: {len(rows)} (etkileşim var: {pos}, yok: {neg})")
    print(f"Benzersiz çift: {len(pairs)}")
    if warnings:
        print(f"\n{len(warnings)} UYARI:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\n{len(errors)} HATA:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n✓ Tüm kontroller geçti.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
