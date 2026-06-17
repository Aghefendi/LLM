# -*- coding: utf-8 -*-
"""Test seti üzerinde modeli çalıştırıp doğruluk oranı hesaplayan değerlendirme aracı.

Akış iki aşamaya ayrılmıştır; böylece pahalı 'generate' adımı (GPU + model)
ile ucuz 'grade' adımı (saf Python) ayrı ayrı çalıştırılabilir:

    generate : Modeli test setindeki her örnek için çalıştırır -> predictions.jsonl
    grade    : predictions.jsonl + test_set.jsonl -> metrikler + report.md
    run      : ikisini arka arkaya yapar (varsayılan)

Örnekler:
    # Tam değerlendirme (GPU gerektirir)
    python -m src.evaluation.run_eval --model mistral

    # Yalnızca üretim, sonra ayrı makinede puanlama
    python -m src.evaluation.run_eval --mode generate --model mistral
    python -m src.evaluation.run_eval --mode grade
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.grading import (
    predict_interaction,
    compute_metrics,
    format_report,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "eval"
TEST_SET = EVAL_DIR / "test_set.jsonl"
PREDICTIONS = EVAL_DIR / "predictions.jsonl"
REPORT = EVAL_DIR / "report.md"


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(model_type: str, limit: int = 0) -> List[Dict]:
    """Modeli yükleyip test setindeki her örnek için çıktı üretir."""
    # Ağır importlar yalnızca gerektiğinde yüklenir
    from src.inference.model_loader import load_peft_model
    from src.inference.evaluator import IlacHastalikEvaluator
    from src.utils.config import PEFT_MISTRAL_MODEL_ID, PEFT_LLAMA_MODEL_ID

    model_id = PEFT_LLAMA_MODEL_ID if model_type == "llama" else PEFT_MISTRAL_MODEL_ID
    print(f"Model yükleniyor: {model_id}")
    model, tokenizer, device = load_peft_model(model_id)
    evaluator = IlacHastalikEvaluator(model, tokenizer, device, model_type=model_type)

    test = load_jsonl(TEST_SET)
    if limit:
        test = test[:limit]

    rows = []
    for i, ex in enumerate(test, 1):
        print(f"[{i}/{len(test)}] {ex['girdi_1']} + {ex['girdi_2']}")
        cikti = evaluator.get_response(ex["girdi_1"], ex["girdi_2"])
        rows.append({**ex, "model": model_id, "cikti": cikti})
    return rows


def grade(rows: List[Dict], model_id: str = "?") -> Dict:
    """Üretilmiş çıktıları puanlar, raporu yazar ve metrikleri döner."""
    for r in rows:
        r["tahmin"] = predict_interaction(r.get("cikti", ""))
    metrics = compute_metrics(rows)
    report = format_report(metrics, model_id=model_id)
    REPORT.write_text(report + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="İlaç etkileşim modeli değerlendirmesi")
    parser.add_argument("--mode", choices=["generate", "grade", "run"], default="run")
    parser.add_argument("--model", choices=["mistral", "llama"], default="mistral")
    parser.add_argument("--limit", type=int, default=0, help="Yalnızca ilk N örnek (test için)")
    args = parser.parse_args()

    if args.mode in ("generate", "run"):
        rows = generate(args.model, limit=args.limit)
        write_jsonl(PREDICTIONS, rows)
        print(f"Çıktılar yazıldı: {PREDICTIONS}")
    else:
        rows = load_jsonl(PREDICTIONS)

    if args.mode in ("grade", "run"):
        model_id = rows[0].get("model", args.model) if rows else args.model
        metrics = grade(rows, model_id=model_id)
        print("\n" + format_report(metrics, model_id=model_id))
        print(f"\nRapor yazıldı: {REPORT}")
        print(f"\n>>> DOĞRULUK: {metrics['dogruluk']:.1%} "
              f"({metrics['dogru']}/{metrics['puanlanan']})")


if __name__ == "__main__":
    main()
