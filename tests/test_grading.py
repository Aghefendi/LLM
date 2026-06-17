# -*- coding: utf-8 -*-
"""Grading sezgiselinin birim testleri (saf Python, GPU/model gerektirmez).

Çalıştırma:
    python -m unittest discover -s tests
"""

import json
import unittest
from pathlib import Path

from src.evaluation.grading import (
    predict_interaction,
    mechanism_recall,
    compute_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "data" / "eval" / "test_set.jsonl"


class TestPredictInteraction(unittest.TestCase):
    def test_risk_output_detected_as_interaction(self):
        out = "Bu iki ilacın birlikte kullanımı kanama riskini artırır."
        self.assertIs(predict_interaction(out), True)

    def test_safe_output_detected_as_no_interaction(self):
        out = "Bu iki ilaç arasında anlamlı bir etkileşim beklenmez, güvenle kullanılır."
        self.assertIs(predict_interaction(out), False)

    def test_negation_not_misread_as_risk(self):
        # "etkileşim beklenmez" RİSK sayılmamalı (regresyon testi)
        out = "Anlamlı etkileşim beklenmez."
        self.assertIs(predict_interaction(out), False)

    def test_empty_output_abstains(self):
        self.assertIsNone(predict_interaction(""))
        self.assertIsNone(predict_interaction("   "))

    def test_turkish_case_folding(self):
        # Büyük İ / ı normalize edilmeli
        out = "İLAÇ ETKİLEŞİMİ KANAMA RİSKİNİ ARTIRIR"
        self.assertIs(predict_interaction(out), True)


class TestMechanismRecall(unittest.TestCase):
    def test_full_recall(self):
        out = "kanama ve antikoagülan etkisi"
        self.assertEqual(mechanism_recall(out, ["kanama", "antikoagülan"]), 1.0)

    def test_partial_recall(self):
        out = "sadece kanama var"
        self.assertEqual(mechanism_recall(out, ["kanama", "hiperkalemi"]), 0.5)

    def test_no_expected_keywords_returns_none(self):
        self.assertIsNone(mechanism_recall("herhangi bir metin", []))


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        records = [
            {"etkilesim_var": True, "tahmin": True, "mekanizma": ["kanama"], "cikti": "kanama riski"},
            {"etkilesim_var": False, "tahmin": False, "mekanizma": [], "cikti": "güvenli"},
        ]
        m = compute_metrics(records)
        self.assertEqual(m["dogruluk"], 1.0)
        self.assertEqual(m["karisiklik_matrisi"], {"TP": 1, "TN": 1, "FP": 0, "FN": 0})

    def test_abstention_excluded_from_accuracy(self):
        records = [
            {"etkilesim_var": True, "tahmin": True, "mekanizma": [], "cikti": ""},
            {"etkilesim_var": True, "tahmin": None, "mekanizma": [], "cikti": ""},
        ]
        m = compute_metrics(records)
        self.assertEqual(m["puanlanan"], 1)
        self.assertEqual(m["cekimser"], 1)
        self.assertEqual(m["dogruluk"], 1.0)


class TestGoldSetAlignment(unittest.TestCase):
    """Grader, gold referans özetleri üzerinde ~mükemmel skor vermeli."""

    @unittest.skipUnless(TEST_SET.exists(), "test_set.jsonl yok (build_test_set.py çalıştırın)")
    def test_reference_summaries_score_high(self):
        rows = [json.loads(l) for l in TEST_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            r["cikti"] = r["referans_ozet"]
            r["tahmin"] = predict_interaction(r["cikti"])
        m = compute_metrics(rows)
        self.assertGreaterEqual(m["dogruluk"], 0.95, "Grader gold etiketlerle hizalı değil")
        self.assertEqual(m["cekimser"], 0, "Bazı gold özetler çekimser kaldı")


if __name__ == "__main__":
    unittest.main()
