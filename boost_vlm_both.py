"""
boost_vlm_both: boost_best, boost_v3b_best 두 모델 모두 VLM 장문 서술형 추가 학습
- VLM 스타일 장문 서술 데이터 + 기존 키워드 데이터 혼합 학습
- 평가: VLM 스타일 30개 (키워드 형식 쿼리 절대 사용 금지)
- 손실함수: CachedMNRL + AnglE + TripletLoss
"""
import json, torch, random, numpy as np, os, gc, sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GPU 자동 선택
try:
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, free = line.split(",")
            gpus.append((int(idx.strip()), int(free.strip())))
        best_gpu = max(gpus, key=lambda x: x[1])[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(f"GPU {best_gpu} 선택 (free={max(gpus, key=lambda x: x[1])[1]}MB)")
except Exception:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from training_data import training_pairs
from vlm_training_data import vlm_pairs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_dict = {c["id"]: c["content"] for c in chunks}
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

# VLM 스타일 테스트 30개
from compare_boost_v3 import GROUP_A, GROUP_B, GROUP_C
VLM_TEST_ITEMS = GROUP_A + GROUP_B + GROUP_C


def evaluate_vlm(model):
    """VLM 스타일 30개 테스트 (긴 문장 쿼리만 사용)"""
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    top1_ok, top3_ok, top5_ok = 0, 0, 0
    scores = []
    group_results = {"A": [], "B": [], "C": []}
    details = []

    for item in VLM_TEST_ITEMS:
        q_emb = model.encode([item["query"]], convert_to_tensor=True, show_progress_bar=False)
        sims = cos_sim(q_emb, chunk_emb)[0]
        sorted_idx = sims.argsort(descending=True)
        top_ids = [chunk_ids[j] for j in sorted_idx[:5]]
        expected = item["expected"]
        exp_idx = chunk_ids.index(expected)
        s = sims[exp_idx].item()
        scores.append(s)
        rank = (top_ids.index(expected) + 1) if expected in top_ids else 999
        if rank == 1: top1_ok += 1
        if rank <= 3: top3_ok += 1
        if rank <= 5: top5_ok += 1

        group = item["name"][0]
        group_results[group].append({"rank": rank, "score": s})
        details.append({
            "name": item["name"],
            "expected": expected,
            "top1": top_ids[0],
            "rank": rank,
            "score": s,
            "hit": rank == 1,
        })

    n = len(VLM_TEST_ITEMS)
    group_stats = {}
    for g, items in group_results.items():
        gt1 = sum(1 for x in items if x["rank"] == 1)
        gavg = sum(x["score"] for x in items) / len(items)
        group_stats[g] = {"top1": gt1, "total": len(items), "avg": gavg}

    return {
        "top1": top1_ok, "top3": top3_ok, "top5": top5_ok, "total": n,
        "avg": sum(scores) / n, "scores": scores,
        "group_stats": group_stats, "details": details,
    }


def mine_hard_negatives(model, queries, pos_ids, top_k=2):
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    hard_negs = []
    for i in range(len(queries)):
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        sorted_indices = all_sim.argsort(descending=True)
        pos_idx = chunk_ids.index(pos_ids[i])
        neg_ids = []
        for idx in sorted_indices:
            idx = idx.item()
            if idx != pos_idx and len(neg_ids) < top_k:
                neg_ids.append(chunk_ids[idx])
        hard_negs.append(neg_ids)
    return hard_negs


def train_and_evaluate(base_model_path, base_name, out_dir, temp_dir, configs):
    """한 모델에 대해 VLM 추가 학습 + 평가"""
    print(f"\n{'=' * 100}")
    print(f"  {base_name} VLM 추가 학습")
    print(f"{'=' * 100}")

    model = SentenceTransformer(base_model_path)
    model.max_seq_length = 512

    # 베이스 성능 측정 (VLM 쿼리만)
    base_result = evaluate_vlm(model)
    print(f"[Base] VLM Top1: {base_result['top1']}/{base_result['total']} | "
          f"Top3: {base_result['top3']}/{base_result['total']} | Avg: {base_result['avg']:.4f}")
    for g, gs in base_result["group_stats"].items():
        print(f"  그룹{g}: Top1={gs['top1']}/{gs['total']} Avg={gs['avg']:.4f}")

    # Hard negative mining
    all_q = [p["query"] for p in training_pairs] + [p["query"] for p in vlm_pairs]
    all_p = [p["positive"] for p in training_pairs] + [p["positive"] for p in vlm_pairs]
    print(f"\n학습 데이터: 키워드 {len(training_pairs)}쌍 + VLM {len(vlm_pairs)}쌍 = 총 {len(all_q)}쌍")

    mined = mine_hard_negatives(model, all_q, all_p)
    del model; gc.collect(); torch.cuda.empty_cache()

    best_top1 = 0
    best_avg = 0.0
    best_combined = 0.0
    best_cfg_idx = -1
    best_result = None

    for ci, cfg in enumerate(configs):
        print(f"\n{'─' * 80}")
        print(f"[Config {ci+1}/{len(configs)}] {cfg['label']}")
        print(f"{'─' * 80}")

        gc.collect(); torch.cuda.empty_cache()

        random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])

        model = SentenceTransformer(base_model_path)
        model.max_seq_length = 512

        vlm_repeat = cfg["vlm_repeat"]

        # Pair examples (CachedMNRL)
        pair_examples = []
        for p in training_pairs:
            c = chunk_dict.get(p["positive"], "")
            if c:
                pair_examples.append(InputExample(texts=[p["query"], c]))
        for p in vlm_pairs:
            c = chunk_dict.get(p["positive"], "")
            if c:
                for _ in range(vlm_repeat):
                    pair_examples.append(InputExample(texts=[p["query"], c]))

        # Triplet examples
        triplet_examples = []
        for i, (q, pid) in enumerate(zip(all_q, all_p)):
            pc = chunk_dict.get(pid, "")
            if pc and i < len(mined):
                for nid in mined[i]:
                    nc = chunk_dict.get(nid, "")
                    if nc:
                        triplet_examples.append(InputExample(texts=[q, pc, nc]))

        # Scored pairs (AnglE)
        scored_pairs = []
        for p in training_pairs:
            c = chunk_dict.get(p["positive"], "")
            if c:
                scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
        for p in vlm_pairs:
            c = chunk_dict.get(p["positive"], "")
            if c:
                scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
        for i, (q, pid) in enumerate(zip(all_q, all_p)):
            if i < len(mined) and mined[i]:
                nc = chunk_dict.get(mined[i][0], "")
                if nc:
                    scored_pairs.append(InputExample(texts=[q, nc], label=0.0))

        print(f"  Pairs: {len(pair_examples)} | Triplets: {len(triplet_examples)} | Scored: {len(scored_pairs)}")

        train_objectives = []
        dl1 = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
        train_objectives.append((dl1, losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_bs"])))
        dl2 = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
        train_objectives.append((dl2, losses.AnglELoss(model)))
        dl3 = DataLoader(triplet_examples, shuffle=True, batch_size=cfg["bs"])
        train_objectives.append((dl3, losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

        model.fit(
            train_objectives=train_objectives,
            epochs=cfg["epochs"],
            warmup_steps=cfg["warmup"],
            output_path=temp_dir,
            show_progress_bar=False,
            optimizer_params={"lr": cfg["lr"]},
            weight_decay=0.01,
        )

        result = evaluate_vlm(model)
        # Combined: VLM Top1 비중 높게
        combined = (result["top1"] / result["total"]) * 0.55 + result["avg"] * 0.45

        print(f"  [VLM30] Top1: {result['top1']}/{result['total']} ({result['top1']/result['total']*100:.1f}%) | "
              f"Top3: {result['top3']}/{result['total']} | Avg: {result['avg']:.4f} | Combined: {combined:.4f}")
        for g, gs in result["group_stats"].items():
            base_gs = base_result["group_stats"][g]
            d_top1 = gs["top1"] - base_gs["top1"]
            d_avg = gs["avg"] - base_gs["avg"]
            print(f"    그룹{g}: Top1={gs['top1']}/{gs['total']}({d_top1:+d}) Avg={gs['avg']:.4f}({d_avg:+.4f})")

        if combined > best_combined:
            best_combined = combined
            best_top1 = result["top1"]
            best_avg = result["avg"]
            best_cfg_idx = ci
            best_result = result
            model.save(out_dir)
            print(f"  >>> New best! Combined={combined:.4f}")

        with open(os.path.join(BASE_DIR, f"{base_name}_vlm_r{ci+1}_result.json"), "w") as f:
            json.dump({
                "vlm_top1": result["top1"], "vlm_top3": result["top3"],
                "vlm_avg": result["avg"], "vlm_scores": result["scores"],
                "combined": combined, "config": cfg["label"],
                "group_stats": {g: {"top1": gs["top1"], "total": gs["total"], "avg": gs["avg"]}
                                for g, gs in result["group_stats"].items()},
            }, f, indent=2, ensure_ascii=False)

        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    if best_cfg_idx >= 0:
        print(f"Best Config: [{best_cfg_idx+1}] {configs[best_cfg_idx]['label']}")
        print(f"  VLM Top1: {best_top1}/{base_result['total']} ({best_top1/base_result['total']*100:.1f}%)")
        print(f"  VLM Avg:  {best_avg:.4f}")
        print(f"  Combined: {best_combined:.4f}")
    else:
        print("  개선된 모델 없음")

    return base_result, best_result, best_cfg_idx


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

configs = [
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 8,  "warmup": 25, "seed": 42,  "vlm_repeat": 4, "label": "lr=3e-6 ep=8 vr=4"},
    {"lr": 5e-6, "bs": 4, "mini_bs": 32, "epochs": 6,  "warmup": 20, "seed": 42,  "vlm_repeat": 5, "label": "lr=5e-6 ep=6 vr=5"},
    {"lr": 2e-6, "bs": 4, "mini_bs": 32, "epochs": 12, "warmup": 35, "seed": 42,  "vlm_repeat": 5, "label": "lr=2e-6 ep=12 vr=5"},
    {"lr": 4e-6, "bs": 4, "mini_bs": 32, "epochs": 8,  "warmup": 25, "seed": 77,  "vlm_repeat": 4, "label": "lr=4e-6 ep=8 seed=77 vr=4"},
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 10, "warmup": 30, "seed": 42,  "vlm_repeat": 6, "label": "lr=3e-6 ep=10 vr=6"},
    {"lr": 1e-6, "bs": 4, "mini_bs": 32, "epochs": 15, "warmup": 40, "seed": 42,  "vlm_repeat": 5, "label": "lr=1e-6 ep=15 vr=5"},
]

print("\n" + "=" * 100)
print("  boost_best & boost_v3b_best VLM 추가 학습")
print("  평가: VLM 스타일 30개 쿼리 (키워드 형식 절대 미사용)")
print("=" * 100)

all_results = {}

# 1) boost_best 학습
base_1, best_1, cfg_1 = train_and_evaluate(
    base_model_path=os.path.join(BASE_DIR, "boost_best"),
    base_name="boost_best",
    out_dir=os.path.join(BASE_DIR, "boost_best_vlm"),
    temp_dir=os.path.join(BASE_DIR, "boost_best_vlm_tmp"),
    configs=configs,
)
all_results["boost_best"] = {"base": base_1, "best": best_1, "cfg_idx": cfg_1}

# 2) boost_v3b_best 학습
base_2, best_2, cfg_2 = train_and_evaluate(
    base_model_path=os.path.join(BASE_DIR, "boost_v3b_best"),
    base_name="boost_v3b_best",
    out_dir=os.path.join(BASE_DIR, "boost_v3b_best_vlm"),
    temp_dir=os.path.join(BASE_DIR, "boost_v3b_best_vlm_tmp"),
    configs=configs,
)
all_results["boost_v3b_best"] = {"base": base_2, "best": best_2, "cfg_idx": cfg_2}

# ══════════════════════════════════════════════════════════════
# 최종 비교 표
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("  최종 비교 결과 (VLM 쿼리 스타일 30개)")
print(f"{'=' * 100}\n")

models = ["boost_best", "boost_v3b_best"]
phases = ["Before (기존)", "After (VLM 학습 후)"]

# 종합표
col_w = 22
print(f"{'지표':<30}", end="")
for m in models:
    print(f"  {m + ' Before':>{col_w}}  {m + ' After':>{col_w}}", end="")
print()
print("-" * (30 + (col_w + 2) * 4))

# VLM Top1
print(f"{'VLM Top1 (/30)':<30}", end="")
for m in models:
    r = all_results[m]
    before = r["base"]["top1"]
    after = r["best"]["top1"] if r["best"] else before
    d = after - before
    print(f"  {before:>{col_w}}  {after:>{col_w-6}}({d:+d})", end="")
print()

# VLM Top3
print(f"{'VLM Top3 (/30)':<30}", end="")
for m in models:
    r = all_results[m]
    before = r["base"]["top3"]
    after = r["best"]["top3"] if r["best"] else before
    d = after - before
    print(f"  {before:>{col_w}}  {after:>{col_w-6}}({d:+d})", end="")
print()

# VLM Avg
print(f"{'VLM Avg Score':<30}", end="")
for m in models:
    r = all_results[m]
    before = r["base"]["avg"]
    after = r["best"]["avg"] if r["best"] else before
    d = after - before
    print(f"  {before:>{col_w}.4f}  {after:>{col_w-8}.4f}({d:+.4f})", end="")
print()
print("-" * (30 + (col_w + 2) * 4))

# 그룹별
for g in ["A", "B", "C"]:
    g_label = {"A": "A (학습 多)", "B": "B (학습 少)", "C": "C (극소/복합)"}[g]
    print(f"{'그룹' + g_label + ' Top1':<30}", end="")
    for m in models:
        r = all_results[m]
        bs = r["base"]["group_stats"][g]
        asgs = r["best"]["group_stats"][g] if r["best"] else bs
        d = asgs["top1"] - bs["top1"]
        print(f"  {bs['top1']:>{col_w-3}}/{bs['total']}  {asgs['top1']:>{col_w-9}}/{asgs['total']}({d:+d})", end="")
    print()
    print(f"{'그룹' + g_label + ' Avg':<30}", end="")
    for m in models:
        r = all_results[m]
        bs = r["base"]["group_stats"][g]
        asgs = r["best"]["group_stats"][g] if r["best"] else bs
        d = asgs["avg"] - bs["avg"]
        print(f"  {bs['avg']:>{col_w}.4f}  {asgs['avg']:>{col_w-8}.4f}({d:+.4f})", end="")
    print()

print("-" * (30 + (col_w + 2) * 4))

# 과적합 분석
print(f"\n{'과적합 분석':<30}", end="")
for m in models:
    print(f"  {'Before':>{col_w}}  {'After':>{col_w}}", end="")
print()
print("-" * (30 + (col_w + 2) * 4))

print(f"{'A-C Top1 Gap':<30}", end="")
for m in models:
    r = all_results[m]
    for phase_r in [r["base"], r["best"] if r["best"] else r["base"]]:
        a_top1_rate = phase_r["group_stats"]["A"]["top1"] / phase_r["group_stats"]["A"]["total"]
        c_top1_rate = phase_r["group_stats"]["C"]["top1"] / phase_r["group_stats"]["C"]["total"]
        gap = (a_top1_rate - c_top1_rate) * 100
        print(f"  {gap:>+{col_w-1}.1f}%p", end="")
print()

print(f"{'A-C Avg Gap':<30}", end="")
for m in models:
    r = all_results[m]
    for phase_r in [r["base"], r["best"] if r["best"] else r["base"]]:
        a_avg = phase_r["group_stats"]["A"]["avg"]
        c_avg = phase_r["group_stats"]["C"]["avg"]
        print(f"  {a_avg - c_avg:>+{col_w}.4f}", end="")
print()

print(f"\n{'=' * 100}")
print("  완료!")
print(f"{'=' * 100}")
