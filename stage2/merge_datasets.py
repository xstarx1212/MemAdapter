#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 Stage-2 alignment 和 Test 数据集为两个统一文件。

输出：
- stage2_alignment_merged.jsonl: 所有 Stage-2 alignment 数据（WikiMultiHop + NarrativeQA）
- test_merged.jsonl: 所有 Test 数据（WikiMultiHop + NarrativeQA + MuSiQue）

格式统一：
- id: 字符串
- context: 字符串（统一格式）
- question: 字符串
- answer: 字符串
- dataset_source: 字符串（"wikimultihop", "narrativeqa", "musique"）
"""

import json
from pathlib import Path
from typing import Dict, Any


def normalize_context(context: Any) -> str:
    """将 context 统一转换为字符串格式"""
    if isinstance(context, str):
        return context
    elif isinstance(context, dict):
        # NarrativeQA 格式：{"text": "...", "tokens": [...]}
        if "text" in context:
            return context["text"]
        elif "summary" in context:
            return context["summary"] if isinstance(context["summary"], str) else str(context["summary"])
        else:
            return str(context)
    else:
        return str(context)


def normalize_field(field: Any) -> str:
    """将字段统一转换为字符串"""
    if isinstance(field, str):
        return field
    elif isinstance(field, dict):
        # 如果是字典，尝试提取 text 字段
        if "text" in field:
            return field["text"]
        else:
            return str(field)
    elif isinstance(field, list):
        # 如果是列表，取第一个元素
        if field:
            return normalize_field(field[0])
        else:
            return ""
    else:
        return str(field) if field is not None else ""


def normalize_record(record: Dict[str, Any], dataset_source: str) -> Dict[str, str]:
    """标准化记录格式"""
    normalized = {
        "id": normalize_field(record.get("id", "")),
        "context": normalize_context(record.get("context", "")),
        "question": normalize_field(record.get("question", "")),
        "answer": normalize_field(record.get("answer", "")),
        "dataset_source": dataset_source,  # 明确标注数据集来源
    }
    return normalized


def load_jsonl(file_path: Path) -> list:
    """加载 JSONL 文件"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list, output_path: Path):
    """写入 JSONL 文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "data" / "merged"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Stage-2 Alignment 数据 ==========
    print(">> 合并 Stage-2 Alignment 数据...")
    stage2_records = []
    
    # WikiMultiHop Stage-2
    wikimultihop_stage2_path = data_dir / "wikimultihop" / "stage2_alignment.jsonl"
    if wikimultihop_stage2_path.exists():
        records = load_jsonl(wikimultihop_stage2_path)
        for rec in records:
            normalized = normalize_record(rec, "wikimultihop")
            stage2_records.append(normalized)
        print(f"   WikiMultiHop Stage-2: {len(records)} 条")
    else:
        print(f"   ⚠️  文件不存在: {wikimultihop_stage2_path}")
    
    # NarrativeQA Stage-2
    narrativeqa_stage2_path = data_dir / "narrativeqa" / "stage2_alignment.jsonl"
    if narrativeqa_stage2_path.exists():
        records = load_jsonl(narrativeqa_stage2_path)
        for rec in records:
            normalized = normalize_record(rec, "narrativeqa")
            stage2_records.append(normalized)
        print(f"   NarrativeQA Stage-2: {len(records)} 条")
    else:
        print(f"   ⚠️  文件不存在: {narrativeqa_stage2_path}")
    
    # 写入合并文件
    stage2_output = output_dir / "stage2_alignment_merged.jsonl"
    write_jsonl(stage2_records, stage2_output)
    print(f"   ✅ Stage-2 Alignment 合并完成: {len(stage2_records)} 条 -> {stage2_output}")
    
    # ========== Test 数据 ==========
    print("\n>> 合并 Test 数据...")
    test_records = []
    
    # WikiMultiHop Test
    wikimultihop_test_path = data_dir / "wikimultihop" / "test.jsonl"
    if wikimultihop_test_path.exists():
        records = load_jsonl(wikimultihop_test_path)
        for rec in records:
            normalized = normalize_record(rec, "wikimultihop")
            test_records.append(normalized)
        print(f"   WikiMultiHop Test: {len(records)} 条")
    else:
        print(f"   ⚠️  文件不存在: {wikimultihop_test_path}")
    
    # NarrativeQA Test
    narrativeqa_test_path = data_dir / "narrativeqa" / "test.jsonl"
    if narrativeqa_test_path.exists():
        records = load_jsonl(narrativeqa_test_path)
        for rec in records:
            normalized = normalize_record(rec, "narrativeqa")
            test_records.append(normalized)
        print(f"   NarrativeQA Test: {len(records)} 条")
    else:
        print(f"   ⚠️  文件不存在: {narrativeqa_test_path}")
    
    # MuSiQue Test
    musique_test_path = data_dir / "musique" / "test.jsonl"
    if musique_test_path.exists():
        records = load_jsonl(musique_test_path)
        for rec in records:
            normalized = normalize_record(rec, "musique")
            test_records.append(normalized)
        print(f"   MuSiQue Test: {len(records)} 条")
    else:
        print(f"   ⚠️  文件不存在: {musique_test_path}")
    
    # 写入合并文件
    test_output = output_dir / "test_merged.jsonl"
    write_jsonl(test_records, test_output)
    print(f"   ✅ Test 合并完成: {len(test_records)} 条 -> {test_output}")
    
    # ========== 统计信息 ==========
    print("\n>> 数据集统计:")
    print(f"   Stage-2 Alignment: {len(stage2_records)} 条")
    stage2_sources = {}
    for rec in stage2_records:
        source = rec["dataset_source"]
        stage2_sources[source] = stage2_sources.get(source, 0) + 1
    for source, count in stage2_sources.items():
        print(f"     - {source}: {count} 条")
    
    print(f"\n   Test: {len(test_records)} 条")
    test_sources = {}
    for rec in test_records:
        source = rec["dataset_source"]
        test_sources[source] = test_sources.get(source, 0) + 1
    for source, count in test_sources.items():
        print(f"     - {source}: {count} 条")
    
    print("\n✅ 所有数据集合并完成！")


if __name__ == "__main__":
    main()
