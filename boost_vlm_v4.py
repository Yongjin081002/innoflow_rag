"""
boost_vlm_v4: boost_best_vlm_v3 기반 추가 파인튜닝
- 학습 데이터: vlm_pairs (v1) + vlm_pairs_v2 + vlm_pairs_v3 + hard_negative_pairs + training_pairs (키워드)
- Base: boost_best_vlm_v3
- Config: lr=5e-6, epochs=4, warmup_steps=20
- 약한 카테고리/고질적 오답 보강 목적
"""
import json, torch, random, numpy as np, os, gc, sys, time

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
from vlm_training_data_v2 import vlm_pairs_v2
from vlm_training_data_v3 import vlm_pairs_v3, hard_negative_pairs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_dict = {c["id"]: c["content"] for c in chunks}
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

# VLM 스타일 테스트 30개
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
    vt1 = vlm_result["top1"]
    vn = vlm_result["total"]
    vavg = vlm_result["avg"]
    kw_factor = (keyword_top1 / 10) if keyword_top1 is not None else 1.0
    base_score = (vt1 / vn) * 0.55 + vavg * 0.35 + kw_factor * 0.10
    gs = vlm_result["group_stats"]
    a_rate = gs["A"]["top1"] / gs["A"]["total"] if gs["A"]["total"] > 0 else 0
    c_rate = gs["C"]["top1"] / gs["C"]["total"] if gs["C"]["total"] > 0 else 0
    gap = max(0, a_rate - c_rate)
    penalty = gap * 0.10
    return base_score - penalty


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  boost_vlm_v4: boost_best_vlm_v3 기반 추가 파인튜닝")
print("  데이터: vlm_v1 + vlm_v2 + vlm_v3 + hard_negatives + keywords")
print("  Config: lr=5e-6, epochs=4, warmup=20")
print("=" * 100)

base_model_path = os.path.join(BASE_DIR, "boost_best_vlm_v3")
out_dir = os.path.join(BASE_DIR, "boost_best_vlm_v4")
temp_dir = os.path.join(BASE_DIR, "boost_best_vlm_v4_tmp")

# ── 1) 베이스 성능 측정 ──
print("\n[1] boost_best_vlm_v3 베이스 성능 측정...")
model = SentenceTransformer(base_model_path)
model.max_seq_length = 384

chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
kw_top1_base, kw_avg_base = evaluate_keyword(model, chunk_emb)
base_vlm = evaluate_vlm(model, chunk_emb)

print(f"  [키워드10] Top1: {kw_top1_base}/10 | Avg: {kw_avg_base:.4f}")
print(f"  [VLM30]    Top1: {base_vlm['top1']}/{base_vlm['total']} | "
      f"Top3: {base_vlm['top3']}/{base_vlm['total']} | Avg: {base_vlm['avg']:.4f}")
for g, gs in base_vlm["group_stats"].items():
    print(f"    그룹{g}: Top1={gs['top1']}/{gs['total']} Avg={gs['avg']:.4f}")

# 고질적 오답 상세
print("\n  [고질적 오답 상세]")
for d in base_vlm["details"]:
    if d["rank"] != 1:
        print(f"    {d['name']}: 정답={d['expected']} Top1={d['top1']} rank={d['rank']} score={d['score']:.4f}")

# ── 2) 학습 데이터 준비 ──
print("\n[2] 학습 데이터 준비...")

# 전체 VLM pairs 통합
all_vlm = vlm_pairs + vlm_pairs_v2 + vlm_pairs_v3
print(f"  VLM 데이터: v1={len(vlm_pairs)} + v2={len(vlm_pairs_v2)} + v3={len(vlm_pairs_v3)} = {len(all_vlm)}쌍")
print(f"  키워드 데이터: {len(training_pairs)}쌍")
print(f"  Hard negative 명시쌍: {len(hard_negative_pairs)}쌍")

# Hard negative mining (auto)
all_q = [p["query"] for p in training_pairs] + [p["query"] for p in all_vlm]
all_p = [p["positive"] for p in training_pairs] + [p["positive"] for p in all_vlm]
print(f"  전체 학습 쌍: {len(all_q)}쌍 (hard negative mining 대상)")

mined = mine_hard_negatives(model, all_q, all_p)
del model, chunk_emb; gc.collect(); torch.cuda.empty_cache()

# ── 3) 학습 ──
print("\n[3] 파인튜닝 시작 (lr=5e-6, epochs=4, warmup=20)...")

random.seed(42); np.random.seed(42)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)

gc.collect(); torch.cuda.empty_cache()
model = SentenceTransformer(base_model_path)
model.max_seq_length = 384

vlm_repeat = 2  # VLM 데이터 반복 (과적합 방지를 위해 2회)

# ── Pair examples (CachedMNRL) ──
pair_examples = []

# 키워드 학습 데이터 (1회)
for p in training_pairs:
    c = chunk_dict.get(p["positive"], "")
    if c:
        pair_examples.append(InputExample(texts=[p["query"], c]))

# VLM 장문 학습 데이터 (v1+v2+v3, 반복 증강)
for p in all_vlm:
    c = chunk_dict.get(p["positive"], "")
    if c:
        for _ in range(vlm_repeat):
            pair_examples.append(InputExample(texts=[p["query"], c]))

# ── Triplet examples ──
triplet_examples = []

# Auto-mined hard negatives
for i, (q, pid) in enumerate(zip(all_q, all_p)):
    pc = chunk_dict.get(pid, "")
    if pc and i < len(mined):
        for nid in mined[i]:
            nc = chunk_dict.get(nid, "")
            if nc:
                triplet_examples.append(InputExample(texts=[q, pc, nc]))

# Explicit hard negative pairs (v3)
for hn in hard_negative_pairs:
    pc = chunk_dict.get(hn["positive"], "")
    nc = chunk_dict.get(hn["negative"], "")
    if pc and nc:
        for _ in range(3):
            triplet_examples.append(InputExample(texts=[hn["query"], pc, nc]))

# ── Scored pairs (AnglE) ──
scored_pairs = []
for p in training_pairs:
    c = chunk_dict.get(p["positive"], "")
    if c:
        scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
for p in all_vlm:
    c = chunk_dict.get(p["positive"], "")
    if c:
        scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
for i, (q, pid) in enumerate(zip(all_q, all_p)):
    if i < len(mined) and mined[i]:
        nc = chunk_dict.get(mined[i][0], "")
        if nc:
            scored_pairs.append(InputExample(texts=[q, nc], label=0.0))
for hn in hard_negative_pairs:
    nc = chunk_dict.get(hn["negative"], "")
    if nc:
        scored_pairs.append(InputExample(texts=[hn["query"], nc], label=0.0))

print(f"  Pairs: {len(pair_examples)} | Triplets: {len(triplet_examples)} | Scored: {len(scored_pairs)}")

# ── Loss 설정 ──
train_objectives = []

effective_bs = 2
effective_mini_bs = 16
dl1 = DataLoader(pair_examples, shuffle=True, batch_size=effective_bs)
train_objectives.append((dl1, losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=effective_mini_bs)))

dl2 = DataLoader(scored_pairs, shuffle=True, batch_size=effective_bs)
train_objectives.append((dl2, losses.AnglELoss(model)))

dl3 = DataLoader(triplet_examples, shuffle=True, batch_size=effective_bs)
train_objectives.append((dl3, losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

# ── Train ──
start_time = time.time()
model.fit(
    train_objectives=train_objectives,
    epochs=4,
    warmup_steps=20,
    output_path=temp_dir,
    show_progress_bar=True,
    optimizer_params={"lr": 5e-6},
    weight_decay=0.05,
)
train_time = time.time() - start_time
print(f"  학습 완료! ({train_time:.1f}초)")

# ── 4) 평가 ──
print("\n[4] 평가...")
chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
kw_t1, kw_a = evaluate_keyword(model, chunk_emb)
new_vlm = evaluate_vlm(model, chunk_emb)
combined = compute_combined_score(new_vlm, kw_t1)

print(f"  [키워드10] Top1: {kw_t1}/10 | Avg: {kw_a:.4f}")
print(f"  [VLM30]    Top1: {new_vlm['top1']}/{new_vlm['total']} | "
      f"Top3: {new_vlm['top3']}/{new_vlm['total']} | Avg: {new_vlm['avg']:.4f}")
for g, gs in new_vlm["group_stats"].items():
    base_gs = base_vlm["group_stats"][g]
    d_top1 = gs["top1"] - base_gs["top1"]
    d_avg = gs["avg"] - base_gs["avg"]
    print(f"    그룹{g}: Top1={gs['top1']}/{gs['total']}({d_top1:+d}) Avg={gs['avg']:.4f}({d_avg:+.4f})")

a_rate = new_vlm["group_stats"]["A"]["top1"] / new_vlm["group_stats"]["A"]["total"]
c_rate = new_vlm["group_stats"]["C"]["top1"] / new_vlm["group_stats"]["C"]["total"]
print(f"  [과적합] A-C gap: {(a_rate-c_rate)*100:.1f}%p | Combined(w/penalty): {combined:.4f}")

# 모델 저장
model.save(out_dir)
print(f"\n  모델 저장: {out_dir}")

# ══════════════════════════════════════════════════════════════
# [5] 비교표
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("  boost_best_vlm_v3 vs boost_best_vlm_v4 비교")
print(f"{'=' * 100}\n")

cw = 20
header = f"{'지표':<35}  {'v3 (Before)':>{cw}}  {'v4 (After)':>{cw}}  {'변화':>{cw}}"
print(header)
print("─" * len(header))

print(f"{'키워드 Top1 (/10)':<35}  {kw_top1_base:>{cw}}  {kw_t1:>{cw}}  {kw_t1-kw_top1_base:>+{cw}d}")
print(f"{'키워드 Avg Score':<35}  {kw_avg_base:>{cw}.4f}  {kw_a:>{cw}.4f}  {kw_a-kw_avg_base:>+{cw}.4f}")
print("─" * len(header))

d_t1 = new_vlm["top1"] - base_vlm["top1"]
d_t3 = new_vlm["top3"] - base_vlm["top3"]
d_avg = new_vlm["avg"] - base_vlm["avg"]
print(f"{'VLM Top1 (/30)':<35}  {base_vlm['top1']:>{cw}}  {new_vlm['top1']:>{cw}}  {d_t1:>+{cw}d}")
print(f"{'VLM Top3 (/30)':<35}  {base_vlm['top3']:>{cw}}  {new_vlm['top3']:>{cw}}  {d_t3:>+{cw}d}")
print(f"{'VLM Avg Score':<35}  {base_vlm['avg']:>{cw}.4f}  {new_vlm['avg']:>{cw}.4f}  {d_avg:>+{cw}.4f}")
print("─" * len(header))

for g in ["A", "B", "C"]:
    g_label = {"A": "A (학습 多)", "B": "B (학습 少)", "C": "C (극소/복합)"}[g]
    bs = base_vlm["group_stats"][g]
    ns = new_vlm["group_stats"][g]
    dt = ns["top1"] - bs["top1"]
    da = ns["avg"] - bs["avg"]
    print(f"{'그룹' + g_label + ' Top1':<35}  {bs['top1']:>{cw-3}}/{bs['total']}  {ns['top1']:>{cw-3}}/{ns['total']}  {dt:>+{cw}d}")
    print(f"{'그룹' + g_label + ' Avg':<35}  {bs['avg']:>{cw}.4f}  {ns['avg']:>{cw}.4f}  {da:>+{cw}.4f}")

print("─" * len(header))

base_combined = compute_combined_score(base_vlm, kw_top1_base)
print(f"{'Combined Score':<35}  {base_combined:>{cw}.4f}  {combined:>{cw}.4f}  {combined-base_combined:>+{cw}.4f}")

# ── 쿼리별 상세 ──
print(f"\n{'=' * 100}")
print("  쿼리별 상세 비교 (v3 -> v4)")
print(f"{'=' * 100}\n")

print(f"  {'그룹':<4} {'이름':<40} {'기대':>6} {'v3_Top1':>8} {'v4_Top1':>8} {'v3_rank':>7} {'v4_rank':>7} {'v3_score':>9} {'v4_score':>9}")
print(f"  {'─'*4} {'─'*40} {'─'*6} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*9} {'─'*9}")

base_detail_map = {d["name"]: d for d in base_vlm["details"]}

for d in new_vlm["details"]:
    bd = base_detail_map.get(d["name"], {})
    br = bd.get("rank", 999)
    bs_score = bd.get("score", 0)
    bt1 = bd.get("top1", "?")

    mark = ""
    if d["rank"] == 1 and br != 1:
        mark = " ++ FIX"
    elif d["rank"] != 1 and br == 1:
        mark = " -- REG"
    elif d["rank"] != 1 and br != 1:
        if d["rank"] < br:
            mark = " + IMPROVE"
        elif d["rank"] > br:
            mark = " - WORSE"
        else:
            mark = " = SAME"

    print(f"  {d['name'][0]:<4} {d['name']:<40} {d['expected']:>6} {bt1:>8} {d['top1']:>8} {br:>7} {d['rank']:>7} {bs_score:>9.4f} {d['score']:>9.4f}{mark}")

# ── 고질적 오답 변화 ──
print(f"\n{'=' * 100}")
print("  고질적 오답 5건 변화")
print(f"{'=' * 100}\n")

target_names = {
    "차20-2": "A5-차선변경끼어들기접촉",
    "차3-1": "B2-적색우회전녹색직진교차충돌",
    "차4-1": "B3-동일방향직진우회전교차충돌",
    "차31-2": "B5-일방통행로역주행정면충돌",
    "차11-2": "B9-비신호동일폭교차로우측차우선위반",
}

for chunk_id, test_name in target_names.items():
    bd = base_detail_map.get(test_name, {})
    nd = next((d for d in new_vlm["details"] if d["name"] == test_name), {})

    print(f"  {chunk_id} ({test_name}):")
    if bd:
        print(f"    v3: Top1={bd.get('top1','?')} rank={bd.get('rank','?')} score={bd.get('score',0):.4f}")
    if nd:
        print(f"    v4: Top1={nd.get('top1','?')} rank={nd.get('rank','?')} score={nd.get('score',0):.4f}")
        if bd and nd:
            improved = nd.get("rank", 999) < bd.get("rank", 999)
            fixed = nd.get("rank") == 1 and bd.get("rank") != 1
            if fixed:
                print(f"    >>> FIXED!")
            elif improved:
                print(f"    >>> IMPROVED (rank {bd['rank']} -> {nd['rank']})")
            elif nd.get("rank") == bd.get("rank"):
                print(f"    >>> NO CHANGE")
            else:
                print(f"    >>> REGRESSED (rank {bd['rank']} -> {nd['rank']})")
    print()

# ── 카테고리별 변화 ──
print(f"{'=' * 100}")
print("  약한 카테고리별 변화")
print(f"{'=' * 100}\n")

categories = {
    "차선변경/끼어들기": ["A5-차선변경끼어들기접촉"],
    "중앙선침범/역주행": ["B5-일방통행로역주행정면충돌"],
    "고속도로": ["A7-고속도로가속차로합류측면접촉", "B10-고속도로본선차선변경측면접촉", "C9-고속도로감속차로진입추돌"],
    "비신호교차로": ["A2-야간비신호교차로동시진입", "A10-비신호교차로골목좌회전vs직진", "B9-비신호동일폭교차로우측차우선위반"],
}

for cat_name, test_names in categories.items():
    print(f"  [{cat_name}]")
    v3_correct = 0
    v4_correct = 0
    for tn in test_names:
        bd = base_detail_map.get(tn, {})
        nd = next((d for d in new_vlm["details"] if d["name"] == tn), {})
        if bd.get("rank") == 1: v3_correct += 1
        if nd.get("rank") == 1: v4_correct += 1
        mark = "O" if nd.get("rank") == 1 else "X"
        mark_old = "O" if bd.get("rank") == 1 else "X"
        print(f"    {tn}: {mark_old} -> {mark} (score {bd.get('score',0):.4f} -> {nd.get('score',0):.4f})")
    print(f"    소계: {v3_correct}/{len(test_names)} -> {v4_correct}/{len(test_names)}")
    print()

# ── 결과 JSON 저장 ──
result_json = {
    "model": "boost_best_vlm_v4",
    "base": "boost_best_vlm_v3",
    "config": "lr=5e-6 ep=4 warmup=20 vr=2 wd=0.05",
    "train_time_sec": train_time,
    "data_counts": {
        "keyword_pairs": len(training_pairs),
        "vlm_v1": len(vlm_pairs),
        "vlm_v2": len(vlm_pairs_v2),
        "vlm_v3": len(vlm_pairs_v3),
        "hard_negative_explicit": len(hard_negative_pairs),
    },
    "v3_baseline": {
        "kw_top1": kw_top1_base, "kw_avg": kw_avg_base,
        "vlm_top1": base_vlm["top1"], "vlm_top3": base_vlm["top3"],
        "vlm_avg": base_vlm["avg"], "combined": base_combined,
        "group_stats": {g: {"top1": gs["top1"], "total": gs["total"], "avg": gs["avg"]}
                        for g, gs in base_vlm["group_stats"].items()},
    },
    "v4_result": {
        "kw_top1": kw_t1, "kw_avg": kw_a,
        "vlm_top1": new_vlm["top1"], "vlm_top3": new_vlm["top3"],
        "vlm_avg": new_vlm["avg"], "combined": combined,
        "group_stats": {g: {"top1": gs["top1"], "total": gs["total"], "avg": gs["avg"]}
                        for g, gs in new_vlm["group_stats"].items()},
        "details": new_vlm["details"],
    },
}

with open(os.path.join(BASE_DIR, "boost_vlm_v4_result.json"), "w") as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 100}")
print(f"  완료! 결과: boost_vlm_v4_result.json")
print(f"  모델: {out_dir}")
print(f"{'=' * 100}")

del model, chunk_emb; gc.collect(); torch.cuda.empty_cache()
