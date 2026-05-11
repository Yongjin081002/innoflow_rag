"""
boost_v3b_best VLM 파인튜닝 (과적합 방지 강화)
- v3b_best 모델만 대상
- VLM 장문 서술형 학습 데이터 사용
- 과적합 방지: weight_decay, VLM repeat 제한, A-C gap 모니터링
- 평가: VLM 30개 (A/B/C 그룹) + 키워드 10개
- 라운드별 결과 JSON 저장 + 최종 비교표 출력

Usage:
  python3 boost_v3b_vlm_tune.py
"""
import json, torch, random, numpy as np, os, gc, sys, time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GPU 자동 선택 (가장 여유 있는 GPU)
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

# VLM 스타일 테스트 30개 (A/B/C 그룹)
from compare_boost_v3 import GROUP_A, GROUP_B, GROUP_C
VLM_TEST_ITEMS = GROUP_A + GROUP_B + GROUP_C

# 키워드 테스트 10개
TEST_QUERIES = [
    "신호위반 직진 충돌", "비신호교차로 직진 vs 좌회전", "추돌 사고 과실",
    "야간 교차로 충돌", "중앙선 침범 충돌", "끼어들기 충돌",
    "유턴 중 충돌", "고속도로 추돌 사고", "주차장 출차 중 충돌", "횡단보도 보행자 충돌"
]
TEST_POS = [
    "차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
    "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"
]


def evaluate_keyword(model, chunk_emb=None):
    """키워드 10개 테스트"""
    if chunk_emb is None:
        chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(TEST_QUERIES, convert_to_tensor=True, show_progress_bar=False)
    scores, top1_ok = [], 0
    for i in range(10):
        pos_idx = chunk_ids.index(TEST_POS[i])
        s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(s)
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        sorted_idx = all_sim.argsort(descending=True)
        if chunk_ids[sorted_idx[0]] == TEST_POS[i]:
            top1_ok += 1
    return top1_ok, sum(scores) / len(scores)


def evaluate_vlm(model, chunk_emb=None):
    """VLM 30개 테스트 (A/B/C 그룹별)"""
    if chunk_emb is None:
        chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    top1_ok, top3_ok = 0, 0
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

        group = item["name"][0]
        group_results[group].append({"rank": rank, "score": s})
        details.append({
            "name": item["name"],
            "expected": expected,
            "top1": top_ids[0],
            "rank": rank,
            "score": s,
        })

    n = len(VLM_TEST_ITEMS)
    group_stats = {}
    for g, items in group_results.items():
        gt1 = sum(1 for x in items if x["rank"] == 1)
        gavg = sum(x["score"] for x in items) / len(items) if items else 0
        group_stats[g] = {"top1": gt1, "total": len(items), "avg": gavg}

    return {
        "top1": top1_ok, "top3": top3_ok, "total": n,
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


def compute_combined_score(vlm_result, keyword_top1=None):
    """과적합 페널티 적용 combined score"""
    vt1 = vlm_result["top1"]
    vn = vlm_result["total"]
    vavg = vlm_result["avg"]

    kw_factor = (keyword_top1 / 10) if keyword_top1 is not None else 1.0
    base_score = (vt1 / vn) * 0.55 + vavg * 0.35 + kw_factor * 0.10

    # 과적합 페널티: A-C gap
    gs = vlm_result["group_stats"]
    a_rate = gs["A"]["top1"] / gs["A"]["total"] if gs["A"]["total"] > 0 else 0
    c_rate = gs["C"]["top1"] / gs["C"]["total"] if gs["C"]["total"] > 0 else 0
    gap = max(0, a_rate - c_rate)
    penalty = gap * 0.10

    return base_score - penalty


# ══════════════════════════════════════════════════════════════
# 학습 설정 (과적합 방지 강화)
# - VLM repeat 1~2로 제한 (기존 3 대비 축소)
# - weight_decay 0.05~0.10
# - epoch 3~6 (짧게)
# - 다양한 seed로 안정성 확인
# ══════════════════════════════════════════════════════════════
configs = [
    {"lr": 3e-6, "epochs": 5, "warmup": 15, "seed": 42, "vlm_repeat": 1, "wd": 0.05, "label": "R1: lr=3e-6 ep=5 vr=1 wd=0.05"},
    {"lr": 5e-6, "epochs": 3, "warmup": 10, "seed": 42, "vlm_repeat": 1, "wd": 0.08, "label": "R2: lr=5e-6 ep=3 vr=1 wd=0.08"},
    {"lr": 2e-6, "epochs": 6, "warmup": 20, "seed": 42, "vlm_repeat": 2, "wd": 0.05, "label": "R3: lr=2e-6 ep=6 vr=2 wd=0.05"},
    {"lr": 3e-6, "epochs": 4, "warmup": 15, "seed": 77, "vlm_repeat": 1, "wd": 0.10, "label": "R4: lr=3e-6 ep=4 vr=1 wd=0.10 seed=77"},
    {"lr": 4e-6, "epochs": 4, "warmup": 15, "seed": 42, "vlm_repeat": 2, "wd": 0.05, "label": "R5: lr=4e-6 ep=4 vr=2 wd=0.05"},
    {"lr": 1e-6, "epochs": 8, "warmup": 25, "seed": 42, "vlm_repeat": 1, "wd": 0.05, "label": "R6: lr=1e-6 ep=8 vr=1 wd=0.05"},
]

BASE_MODEL = os.path.join(BASE_DIR, "boost_v3b_best")
OUT_DIR = os.path.join(BASE_DIR, "boost_v3b_vlm_tuned")
TEMP_DIR = os.path.join(BASE_DIR, "boost_v3b_vlm_tmp")

print("\n" + "=" * 100)
print("  boost_v3b_best VLM 파인튜닝 (과적합 방지 강화)")
print("  VLM repeat 제한, weight_decay 강화, A-C gap 모니터링")
print("=" * 100)

# ── 베이스 모델 성능 측정 ──
print("\n[Base 모델 평가]")
model = SentenceTransformer(BASE_MODEL)
model.max_seq_length = 384
chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
base_kw_top1, base_kw_avg = evaluate_keyword(model, chunk_emb)
base_vlm = evaluate_vlm(model, chunk_emb)

print(f"  [키워드10] Top1: {base_kw_top1}/10 | Avg: {base_kw_avg:.4f}")
print(f"  [VLM30]   Top1: {base_vlm['top1']}/{base_vlm['total']} | "
      f"Top3: {base_vlm['top3']}/{base_vlm['total']} | Avg: {base_vlm['avg']:.4f}")
for g, gs in base_vlm["group_stats"].items():
    print(f"    그룹{g}: Top1={gs['top1']}/{gs['total']} Avg={gs['avg']:.4f}")

a_rate = base_vlm["group_stats"]["A"]["top1"] / base_vlm["group_stats"]["A"]["total"]
c_rate = base_vlm["group_stats"]["C"]["top1"] / base_vlm["group_stats"]["C"]["total"]
print(f"  [과적합] A-C gap: {(a_rate - c_rate) * 100:.1f}%p")

# ── Hard negative mining (베이스 모델 기준) ──
all_q = [p["query"] for p in training_pairs] + [p["query"] for p in vlm_pairs]
all_p = [p["positive"] for p in training_pairs] + [p["positive"] for p in vlm_pairs]
print(f"\n  학습 데이터: 키워드 {len(training_pairs)}쌍 + VLM {len(vlm_pairs)}쌍 = 총 {len(all_q)}쌍")
print("  Hard negative mining...")
mined = mine_hard_negatives(model, all_q, all_p)
del model, chunk_emb; gc.collect(); torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# 라운드별 학습 + 평가
# ══════════════════════════════════════════════════════════════
best_combined = -1.0
best_cfg_idx = -1
best_result = None
best_kw = None
config_results = []

for ci, cfg in enumerate(configs):
    result_path = os.path.join(BASE_DIR, f"boost_v3b_vlm_r{ci+1}_result.json")

    # Resume: 기존 결과 있으면 로드
    if os.path.exists(result_path):
        with open(result_path, "r") as rf:
            cached = json.load(rf)
        config_results.append(cached)
        print(f"\n[{cfg['label']}] 기존 결과 로드 (combined={cached['combined']:.4f})")
        if cached.get("kw_top1", 0) >= 7 and cached["combined"] > best_combined:
            best_combined = cached["combined"]
            best_cfg_idx = ci
        continue

    print(f"\n{'─' * 100}")
    print(f"[{cfg['label']}]")
    print(f"{'─' * 100}")

    gc.collect(); torch.cuda.empty_cache()

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 384

    vlm_repeat = cfg["vlm_repeat"]

    # ── Pair examples (CachedMNRL) ──
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

    # ── Triplet examples (hard negative) ──
    triplet_examples = []
    for i, (q, pid) in enumerate(zip(all_q, all_p)):
        pc = chunk_dict.get(pid, "")
        if pc and i < len(mined):
            for nid in mined[i]:
                nc = chunk_dict.get(nid, "")
                if nc:
                    triplet_examples.append(InputExample(texts=[q, pc, nc]))

    # ── Scored pairs (AnglE) ──
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

    # ── Loss 설정 (OOM 방지: bs=2, mini_bs=16) ──
    train_objectives = []

    dl1 = DataLoader(pair_examples, shuffle=True, batch_size=2)
    mini_bs = 8 if len(pair_examples) > 650 else 16  # OOM 방지
    train_objectives.append((dl1, losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_bs)))

    dl2 = DataLoader(scored_pairs, shuffle=True, batch_size=2)
    train_objectives.append((dl2, losses.AnglELoss(model)))

    dl3 = DataLoader(triplet_examples, shuffle=True, batch_size=2)
    train_objectives.append((dl3, losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

    # ── 학습 ──
    t0 = time.time()
    model.fit(
        train_objectives=train_objectives,
        epochs=cfg["epochs"],
        warmup_steps=cfg["warmup"],
        output_path=TEMP_DIR,
        show_progress_bar=False,
        optimizer_params={"lr": cfg["lr"]},
        weight_decay=cfg["wd"],
    )
    train_time = time.time() - t0

    # ── 평가 ──
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    kw_t1, kw_a = evaluate_keyword(model, chunk_emb)
    vlm_r = evaluate_vlm(model, chunk_emb)
    combined = compute_combined_score(vlm_r, kw_t1)

    print(f"  학습 시간: {train_time:.0f}초")
    print(f"  [키워드10] Top1: {kw_t1}/10 (base: {base_kw_top1}) | Avg: {kw_a:.4f} (base: {base_kw_avg:.4f})")
    print(f"  [VLM30]   Top1: {vlm_r['top1']}/{vlm_r['total']} (base: {base_vlm['top1']}) | "
          f"Top3: {vlm_r['top3']}/{vlm_r['total']} (base: {base_vlm['top3']}) | "
          f"Avg: {vlm_r['avg']:.4f} (base: {base_vlm['avg']:.4f})")

    for g in ["A", "B", "C"]:
        gs = vlm_r["group_stats"][g]
        bgs = base_vlm["group_stats"][g]
        d_top1 = gs["top1"] - bgs["top1"]
        d_avg = gs["avg"] - bgs["avg"]
        print(f"    그룹{g}: Top1={gs['top1']}/{gs['total']}({d_top1:+d}) Avg={gs['avg']:.4f}({d_avg:+.4f})")

    a_rate = vlm_r["group_stats"]["A"]["top1"] / vlm_r["group_stats"]["A"]["total"]
    c_rate = vlm_r["group_stats"]["C"]["top1"] / vlm_r["group_stats"]["C"]["total"]
    ac_gap = (a_rate - c_rate) * 100
    print(f"  [과적합] A-C gap: {ac_gap:.1f}%p | Combined(w/penalty): {combined:.4f}")

    # 과적합 경고
    if ac_gap > 30:
        print(f"  *** 경고: A-C gap {ac_gap:.1f}%p > 30%p → 과적합 의심! ***")
    if kw_t1 < 7:
        print(f"  *** 경고: 키워드 Top1 {kw_t1}/10 < 7 → 키워드 성능 저하! ***")

    cr = {
        "config": cfg["label"],
        "kw_top1": kw_t1, "kw_avg": kw_a,
        "vlm_top1": vlm_r["top1"], "vlm_top3": vlm_r["top3"],
        "vlm_avg": vlm_r["avg"], "combined": combined,
        "ac_gap_pct": ac_gap, "train_time_sec": train_time,
        "group_stats": {g: {"top1": gs["top1"], "total": gs["total"], "avg": gs["avg"]}
                        for g, gs in vlm_r["group_stats"].items()},
        "details": vlm_r["details"],
    }
    config_results.append(cr)

    # Best 모델 저장 (키워드 Top1 >= 7 유지 조건)
    if kw_t1 >= 7 and combined > best_combined:
        best_combined = combined
        best_cfg_idx = ci
        best_result = vlm_r
        best_kw = {"top1": kw_t1, "avg": kw_a}
        model.save(OUT_DIR)
        print(f"  >>> New best! Combined={combined:.4f} → 저장: {OUT_DIR}")

    # 라운드 결과 JSON 저장
    with open(result_path, "w") as f:
        json.dump(cr, f, indent=2, ensure_ascii=False)

    del model, chunk_emb; gc.collect(); torch.cuda.empty_cache()

    # ── ROUND_DONE 마커 (외부 모니터링용) ──
    print(f"\n===ROUND_DONE {ci+1}/{len(configs)}===")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════
# 최종 비교표
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print("  최종 비교 결과: boost_v3b_best VLM 파인튜닝")
print(f"{'=' * 120}\n")

# ── Config별 요약표 ──
print(f"{'Config':<40} {'KW_T1':>5} {'KW_Avg':>7} {'VLM_T1':>6} {'VLM_T3':>6} {'VLM_Avg':>8} {'AC_Gap':>7} {'Combined':>9}")
print(f"{'─'*40} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*9}")
print(f"{'[Base] v3b_best':<40} {base_kw_top1:>5} {base_kw_avg:>7.4f} {base_vlm['top1']:>6} {base_vlm['top3']:>6} {base_vlm['avg']:>8.4f} {(a_rate-c_rate)*100:>6.1f}% {'—':>9}")

for ci, cr in enumerate(config_results):
    best_mark = " *" if ci == best_cfg_idx else ""
    print(f"{cr['config']:<40} {cr['kw_top1']:>5} {cr['kw_avg']:>7.4f} {cr['vlm_top1']:>6} {cr['vlm_top3']:>6} {cr['vlm_avg']:>8.4f} {cr['ac_gap_pct']:>6.1f}% {cr['combined']:>9.4f}{best_mark}")

# ── Markdown 표 ──
print(f"\n### boost_v3b_best VLM 파인튜닝 결과")
print()
print("| Config | KW Top1 | KW Avg | VLM Top1 | VLM Top3 | VLM Avg | A-C Gap | Combined |")
print("|--------|---------|--------|----------|----------|---------|---------|----------|")
print(f"| **Base** | {base_kw_top1}/10 | {base_kw_avg:.4f} | {base_vlm['top1']}/30 | {base_vlm['top3']}/30 | {base_vlm['avg']:.4f} | — | — |")
for ci, cr in enumerate(config_results):
    best_mark = " **best**" if ci == best_cfg_idx else ""
    print(f"| {cr['config']} | {cr['kw_top1']}/10 | {cr['kw_avg']:.4f} | {cr['vlm_top1']}/30 | {cr['vlm_top3']}/30 | {cr['vlm_avg']:.4f} | {cr['ac_gap_pct']:.1f}%p | {cr['combined']:.4f}{best_mark} |")
print()

# ── 그룹별 비교 ──
print("### 그룹별 Top1 비교 (과적합 검증)")
print()
print("| Config | 그룹A Top1 | 그룹B Top1 | 그룹C Top1 | A-C Gap |")
print("|--------|-----------|-----------|-----------|---------|")
bs = base_vlm["group_stats"]
print(f"| **Base** | {bs['A']['top1']}/{bs['A']['total']} | {bs['B']['top1']}/{bs['B']['total']} | {bs['C']['top1']}/{bs['C']['total']} | — |")
for ci, cr in enumerate(config_results):
    gs = cr["group_stats"]
    best_mark = " **best**" if ci == best_cfg_idx else ""
    print(f"| {cr['config']} | {gs['A']['top1']}/{gs['A']['total']} | {gs['B']['top1']}/{gs['B']['total']} | {gs['C']['top1']}/{gs['C']['total']} | {cr['ac_gap_pct']:.1f}%p{best_mark} |")
print()

# ── Best 모델 VLM 쿼리별 상세 ──
if best_cfg_idx >= 0:
    best_cr = config_results[best_cfg_idx]
    print(f"### Best 모델 VLM 쿼리별 상세 ({best_cr['config']})")
    print()
    print("| 그룹 | 이름 | 기대 | 결과 | 순위 | 점수 | Hit |")
    print("|------|------|------|------|------|------|-----|")
    for d in best_cr["details"]:
        hit = "O" if d["rank"] == 1 else ("△" if d["rank"] <= 3 else "X")
        print(f"| {d['name'][0]} | {d['name']} | {d['expected']} | {d['top1']} | {d['rank']} | {d['score']:.4f} | {hit} |")
    print()

print(f"\n{'=' * 120}")
if best_cfg_idx >= 0:
    print(f"  Best: [{best_cfg_idx+1}] {configs[best_cfg_idx]['label']}")
    print(f"  모델 저장 위치: {OUT_DIR}")
else:
    print("  개선된 모델 없음 (모든 config에서 키워드 Top1 < 7 또는 base 대비 성능 하락)")
print(f"{'=' * 120}")
