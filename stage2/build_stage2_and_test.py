#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Stage-2 alignment and Test datasets for MemAdapter.

Datasets (public, non-dialogue):
- WikiMultiHopQA (qangaroo/wikihop)
- NarrativeQA (summary QA)
- MuSiQue (official test)

Splits (seed=42, no overlap):
- WikiMultiHopQA: 1,200 (Stage-2), 10,000 (Test)
- NarrativeQA: 1,300 from train (Stage-2), official test (Test)
- MuSiQue: official test (Test)

Outputs are written under stage2/data and a manifest is recorded in stage2/manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset

SEED = 42
WIKIMULTIHOP_STAGE2 = 1200
WIKIMULTIHOP_TEST = 10_000
NARRATIVEQA_STAGE2 = 1300


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: Iterable[Dict], path: Path) -> int:
    ensure_dir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------
# WikiMultiHopQA (qangaroo/wikihop)
# --------------------------
def build_wikimultihop(output_dir: Path) -> Dict:
    print(">> Loading WikiMultiHopQA (qangaroo/wikihop)…")
    ds = load_dataset("qangaroo", "wikihop")["train"].shuffle(seed=SEED)

    stage2_ds = ds.select(range(WIKIMULTIHOP_STAGE2))
    test_ds = ds.select(range(WIKIMULTIHOP_STAGE2, WIKIMULTIHOP_STAGE2 + WIKIMULTIHOP_TEST))

    def to_rec(example, idx_prefix: str) -> Dict:
        docs = example.get("supports") or []
        context_parts = [f"### Document {i+1}: {doc}" for i, doc in enumerate(docs)]
        context = "\n".join(context_parts)
        question = example.get("query") or example.get("question") or ""
        answer = example.get("answer") or ""
        rec_id = example.get("id") or f"{idx_prefix}_{example['idx'] if 'idx' in example else ''}".strip("_")
        return {
            "id": rec_id,
            "context": context,
            "question": question,
            "answer": answer,
            "_source": "wikimultihop",
        }

    out_stage2 = output_dir / "wikimultihop" / "stage2_alignment.jsonl"
    out_test = output_dir / "wikimultihop" / "test.jsonl"

    n_stage2 = write_jsonl((to_rec(ex, "wikim_stage2") for ex in stage2_ds), out_stage2)
    n_test = write_jsonl((to_rec(ex, "wikim_test") for ex in test_ds), out_test)

    print(f"   Saved WikiMultiHopQA Stage-2 -> {out_stage2} ({n_stage2})")
    print(f"   Saved WikiMultiHopQA Test    -> {out_test} ({n_test})")

    return {
        "source": "qangaroo/wikihop",
        "seed": SEED,
        "stage2_count": n_stage2,
        "test_count": n_test,
        "stage2_indices": [0, WIKIMULTIHOP_STAGE2 - 1],
        "test_indices": [WIKIMULTIHOP_STAGE2, WIKIMULTIHOP_STAGE2 + WIKIMULTIHOP_TEST - 1],
    }


# --------------------------
# NarrativeQA (summary QA)
# --------------------------
def _get_narrativeqa_context(doc: Dict) -> str:
    if not doc:
        return ""
    if isinstance(doc, dict):
        # Prefer summary if present; fallback to full text
        if doc.get("summary"):
            return doc["summary"]
        if doc.get("text"):
            return doc["text"]
        # Some variants may nest summary under "summary.text"
        if isinstance(doc.get("summary"), dict) and doc["summary"].get("text"):
            return doc["summary"]["text"]
    return str(doc)


def build_narrativeqa(output_dir: Path) -> Dict:
    print(">> Loading NarrativeQA (summary QA, parquet from deepmind/narrativeqa)…")
    # Use parquet files directly to avoid legacy script issues.
    train_ds = load_dataset(
        "parquet",
        data_files={"train": "hf://datasets/deepmind/narrativeqa@main/data/train-*.parquet"},
        split="train",
    ).shuffle(seed=SEED)
    test_ds = load_dataset(
        "parquet",
        data_files={"test": "hf://datasets/deepmind/narrativeqa@main/data/test-*.parquet"},
        split="test",
    )

    stage2_ds = train_ds.select(range(NARRATIVEQA_STAGE2))

    def to_rec(example, idx_prefix: str) -> Dict:
        context = _get_narrativeqa_context(example.get("document"))
        question_field = example.get("question") or {}
        question = question_field.get("text") if isinstance(question_field, dict) else str(question_field)

        answers = example.get("answers") or []
        if isinstance(answers, list) and answers:
            first_answer = answers[0]
            if isinstance(first_answer, dict):
                answer = first_answer.get("text") or ""
            else:
                answer = str(first_answer)
        else:
            answer = ""

        rec_id = (
            example.get("document", {}).get("id")
            or example.get("id")
            or f"{idx_prefix}_{example['idx'] if 'idx' in example else ''}".strip("_")
        )
        return {
            "id": rec_id,
            "context": context,
            "question": question,
            "answer": answer,
            "_source": "narrativeqa",
        }

    out_stage2 = output_dir / "narrativeqa" / "stage2_alignment.jsonl"
    out_test = output_dir / "narrativeqa" / "test.jsonl"

    n_stage2 = write_jsonl((to_rec(ex, "nar_stage2") for ex in stage2_ds), out_stage2)
    n_test = write_jsonl((to_rec(ex, "nar_test") for ex in test_ds), out_test)

    print(f"   Saved NarrativeQA Stage-2 -> {out_stage2} ({n_stage2})")
    print(f"   Saved NarrativeQA Test    -> {out_test} ({n_test})")

    return {
        "source": "narrativeqa",
        "seed": SEED,
        "stage2_count": n_stage2,
        "test_count": n_test,
        "stage2_indices": [0, NARRATIVEQA_STAGE2 - 1],
        "test_indices": "official_test_split",
    }


# --------------------------
# MuSiQue (official test)
# --------------------------
def _load_musique_test():
    """
    Try a few common config names for the MuSiQue dataset.
    """
    # Public HF repo provides train/dev; dev is used as held-out test.
    try:
        ds = load_dataset(
            "json",
            data_files={"test": "hf://datasets/dgslibisey/MuSiQue@main/musique_ans_v1.0_dev.jsonl"},
        )
        return ds["test"], "dgslibisey/MuSiQue:dev_as_test"
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not load MuSiQue test split. Error: {e}")


def build_musique(output_dir: Path) -> Dict:
    print(">> Loading MuSiQue (official test)…")
    test_ds, source_used = _load_musique_test()

    def to_rec(example, idx_prefix: str) -> Dict:
        # MuSiQue paragraphs might be list[str] or list[dict{title,text}]
        paras = example.get("paragraphs") or example.get("context") or []
        context_parts = []
        if isinstance(paras, list):
            for i, p in enumerate(paras):
                if isinstance(p, dict):
                    text = p.get("text") or p.get("paragraph_text") or ""
                    title = p.get("title")
                    header = f"### Paragraph {i+1}" + (f": {title}" if title else "")
                    context_parts.append(f"{header}\n{text}")
                else:
                    context_parts.append(f"### Paragraph {i+1}: {p}")
        else:
            context_parts.append(str(paras))
        context = "\n\n".join(context_parts)

        question = example.get("question") or ""
        # MuSiQue may store answer under "answer" or "answers"
        answer_field = example.get("answer")
        if isinstance(answer_field, list):
            answer = answer_field[0] if answer_field else ""
        elif answer_field:
            answer = answer_field
        else:
            answers = example.get("answers")
            answer = answers[0] if isinstance(answers, list) and answers else ""

        rec_id = example.get("id") or example.get("qid") or f"{idx_prefix}_{example['idx'] if 'idx' in example else ''}".strip("_")
        return {
            "id": rec_id,
            "context": context,
            "question": question,
            "answer": answer,
            "_source": "musique",
        }

    out_test = output_dir / "musique" / "test.jsonl"
    n_test = write_jsonl((to_rec(ex, "mus_test") for ex in test_ds), out_test)
    print(f"   Saved MuSiQue Test -> {out_test} ({n_test}) [source={source_used}]")

    return {
        "source": source_used,
        "seed": SEED,
        "test_count": n_test,
        "test_indices": "official_test_split",
    }


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Build Stage-2 and Test datasets for MemAdapter.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root that contains stage2/ (defaults to this script's parent).",
    )
    args = parser.parse_args()

    output_dir = (args.root / "data").resolve()
    manifest_dir = (args.root / "manifests").resolve()
    ensure_dir(output_dir)
    ensure_dir(manifest_dir)

    manifest = {"seed": SEED, "datasets": {}}

    manifest["datasets"]["wikimultihop"] = build_wikimultihop(output_dir)
    manifest["datasets"]["narrativeqa"] = build_narrativeqa(output_dir)
    manifest["datasets"]["musique"] = build_musique(output_dir)

    manifest_path = manifest_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f">> Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
