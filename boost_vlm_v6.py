"""
boost_vlm_v6: boost_best_vlm_v5 기반 추가 파인튜닝
- 라벨 정정 후 취약 6개 타깃 재보강
- loss: MultipleNegativesRankingLoss
- config: lr=2e-6, epochs=3, warmup_steps=20, weight_decay=0.05
"""
import gc
import json
import os
import random
import time

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, free = line.split(",")
            gpus.append((int(idx.strip()), int(free.strip())))
        best_gpu = max(gpus, key=lambda item: item[1])[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(f"GPU {best_gpu} 선택")
except Exception:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from training_data import training_pairs
from vlm_training_data import vlm_pairs
from vlm_training_data_v2 import vlm_pairs_v2
from vlm_training_data_v3 import vlm_pairs_v3
from vlm_training_data_v6 import hard_negative_pairs_v6, vlm_pairs_v6


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.path.join(BASE_DIR, "boost_best_vlm_v5")
TEMP_DIR = os.path.join(BASE_DIR, "boost_best_vlm_v6_tmp")
OUT_DIR = os.path.join(BASE_DIR, "boost_best_vlm_v6")
META_JSON = os.path.join(BASE_DIR, "boost_vlm_v6_train_meta.json")

TARGET_CHUNK_IDS = {
    "차41-1",
    "차3-1",
    "차4-1",
    "차31-2",
    "차11-2",
    "차15-1",
}

TARGET_REPEAT = 3
HARD_NEGATIVE_REPEAT = 3
PAIR_BATCH_SIZE = 6
TRIPLET_BATCH_SIZE = 4


def load_chunks():
    with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    return {chunk["id"]: chunk["content"] for chunk in chunks}


def filter_non_target_pairs(pairs):
    return [pair for pair in pairs if pair["positive"] not in TARGET_CHUNK_IDS]


def build_pair_examples(pairs, chunk_dict, repeat=1):
    examples = []
    for pair in pairs:
        content = chunk_dict.get(pair["positive"], "")
        if not content:
            continue
        for _ in range(repeat):
            examples.append(InputExample(texts=[pair["query"], content]))
    return examples


def build_hard_negative_examples(pairs, chunk_dict, repeat=1):
    examples = []
    for pair in pairs:
        positive = chunk_dict.get(pair["positive"], "")
        negative = chunk_dict.get(pair["negative"], "")
        if not positive or not negative:
            continue
        for _ in range(repeat):
            examples.append(InputExample(texts=[pair["query"], positive, negative]))
    return examples


def main():
    if not os.path.isdir(BASE_MODEL):
        raise FileNotFoundError(f"베이스 모델 디렉토리를 찾을 수 없습니다: {BASE_MODEL}")

    print("\n" + "=" * 100)
    print("  boost_vlm_v6: boost_best_vlm_v5 기반 라벨 정정 후 추가 파인튜닝")
    print("  loss=MultipleNegativesRankingLoss | lr=2e-6 | epochs=3 | warmup=20 | wd=0.05")
    print("=" * 100)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
    gc.collect()

    chunk_dict = load_chunks()

    base_keyword_pairs = filter_non_target_pairs(training_pairs)
    base_vlm_pairs = filter_non_target_pairs(vlm_pairs + vlm_pairs_v2 + vlm_pairs_v3)

    pair_examples = []
    pair_examples.extend(build_pair_examples(base_keyword_pairs, chunk_dict, repeat=1))
    pair_examples.extend(build_pair_examples(base_vlm_pairs, chunk_dict, repeat=1))
    pair_examples.extend(build_pair_examples(vlm_pairs_v6, chunk_dict, repeat=TARGET_REPEAT))

    hard_negative_examples = build_hard_negative_examples(
        hard_negative_pairs_v6,
        chunk_dict,
        repeat=HARD_NEGATIVE_REPEAT,
    )

    print(f"  base keyword pairs: {len(base_keyword_pairs)}")
    print(f"  base vlm pairs: {len(base_vlm_pairs)}")
    print(f"  target v6 pairs: {len(vlm_pairs_v6)} x{TARGET_REPEAT}")
    print(f"  hard negative pairs: {len(hard_negative_pairs_v6)} x{HARD_NEGATIVE_REPEAT}")
    print(f"  total pair examples: {len(pair_examples)}")
    print(f"  total hard negative examples: {len(hard_negative_examples)}")

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 512

    train_objectives = []
    if pair_examples:
        pair_loader = DataLoader(pair_examples, shuffle=True, batch_size=PAIR_BATCH_SIZE)
        train_objectives.append((pair_loader, losses.MultipleNegativesRankingLoss(model)))
    if hard_negative_examples:
        hard_negative_loader = DataLoader(
            hard_negative_examples,
            shuffle=True,
            batch_size=TRIPLET_BATCH_SIZE,
        )
        train_objectives.append(
            (hard_negative_loader, losses.MultipleNegativesRankingLoss(model))
        )

    start_time = time.time()
    model.fit(
        train_objectives=train_objectives,
        epochs=3,
        warmup_steps=20,
        output_path=TEMP_DIR,
        show_progress_bar=True,
        optimizer_params={"lr": 2e-6},
        weight_decay=0.05,
    )
    train_time = time.time() - start_time

    model.save(OUT_DIR)

    meta = {
        "model": "boost_best_vlm_v6",
        "base_model": "boost_best_vlm_v5",
        "config": {
            "lr": 2e-6,
            "epochs": 3,
            "warmup_steps": 20,
            "weight_decay": 0.05,
            "loss": "MultipleNegativesRankingLoss",
            "pair_batch_size": PAIR_BATCH_SIZE,
            "hard_negative_batch_size": TRIPLET_BATCH_SIZE,
        },
        "data_counts": {
            "base_keyword_pairs": len(base_keyword_pairs),
            "base_vlm_pairs": len(base_vlm_pairs),
            "target_pairs": len(vlm_pairs_v6),
            "target_repeat": TARGET_REPEAT,
            "hard_negative_pairs": len(hard_negative_pairs_v6),
            "hard_negative_repeat": HARD_NEGATIVE_REPEAT,
            "pair_examples": len(pair_examples),
            "hard_negative_examples": len(hard_negative_examples),
        },
        "train_time_sec": round(train_time, 2),
    }

    with open(META_JSON, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print(f"\n  학습 완료: {OUT_DIR}")
    print(f"  메타 저장: {META_JSON}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
