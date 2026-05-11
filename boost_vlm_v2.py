"""
boost_vlm_v2: boost_best 기반 VLM 장문 서술형 파인튜닝
- base: boost_best (과적합 적고 Avg Score 높아 일반화 출발점으로 적합)
- VLM 스타일 장문 서술 데이터 + 기존 키워드 데이터 혼합 학습
- 평가: 기존 키워드 10개 + VLM 스타일 30개
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

# ── 기존 키워드 테스트 10개 ──
TEST_QUERIES = [
    "신호위반 직진 충돌", "비신호교차로 직진 vs 좌회전", "추돌 사고 과실",
    "야간 교차로 충돌", "중앙선 침범 충돌", "끼어들기 충돌",
    "유턴 중 충돌", "고속도로 추돌 사고", "주차장 출차 중 충돌", "횡단보도 보행자 충돌"
]
TEST_POS = [
    "차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
    "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"
]

# ── VLM 스타일 테스트 30개 (compare_boost_v3.py와 동일) ──
from compare_boost_v3 import GROUP_A, GROUP_B, GROUP_C
VLM_TEST_ITEMS = GROUP_A + GROUP_B + GROUP_C


def evaluate(model):
    """기존 키워드 10개 테스트"""
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(TEST_QUERIES, convert_to_tensor=True, show_progress_bar=False)
    scores, top1_ok, top3_ok, details = [], 0, 0, []
    for i in range(10):
        pos_idx = chunk_ids.index(TEST_POS[i])
        s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(s)
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        sorted_idx = all_sim.argsort(descending=True)
        top_ids = [chunk_ids[j] for j in sorted_idx[:3]]
        ok = top_ids[0] == TEST_POS[i]
        ok3 = TEST_POS[i] in top_ids
        if ok: top1_ok += 1
        if ok3: top3_ok += 1
        details.append((TEST_QUERIES[i], TEST_POS[i], s, top_ids[0], ok))
    return top1_ok, top3_ok, sum(scores)/len(scores), scores, details


def evaluate_vlm(model, chunk_emb=None):
    """VLM 스타일 30개 테스트 (긴 문장 쿼리)"""
    if chunk_emb is None:
        chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    top1_ok, top3_ok = 0, 0
    scores = []
    for item in VLM_TEST_ITEMS:
        q_emb = model.encode([item["query"]], convert_to_tensor=True, show_progress_bar=False)
        sims = cos_sim(q_emb, chunk_emb)[0]
        sorted_idx = sims.argsort(descending=True)
        top_ids = [chunk_ids[j] for j in sorted_idx[:3]]
        expected = item["expected"]
        if expected in chunk_ids:
            exp_idx = chunk_ids.index(expected)
            s = sims[exp_idx].item()
        else:
            s = 0.0
        scores.append(s)
        if top_ids[0] == expected: top1_ok += 1
        if expected in top_ids: top3_ok += 1
    n = len(VLM_TEST_ITEMS)
    return top1_ok, top3_ok, n, sum(scores)/n, scores


def mine_hard_negatives(model, queries, pos_ids, top_k=2):
    """Hard negative mining"""
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


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

BASE_MODEL = os.path.join(BASE_DIR, "boost_best")
OUT_DIR = os.path.join(BASE_DIR, "boost_vlm_v2_best")
TEMP_DIR = os.path.join(BASE_DIR, "boost_vlm_v2_tmp")

print("=" * 100)
print(f"  boost_vlm_v2: VLM 장문 파인튜닝 (base: boost_best)")
print("=" * 100)

# ── 베이스 모델 성능 ──
model = SentenceTransformer(BASE_MODEL)
model.max_seq_length = 512  # VLM 긴 문장 대응

base_top1, base_top3, base_avg, base_scores, base_details = evaluate(model)
print(f"[Base 키워드10] Top1: {base_top1}/10 | Top3: {base_top3}/10 | Avg: {base_avg:.4f}")

vlm_top1, vlm_top3, vlm_n, vlm_avg, vlm_scores = evaluate_vlm(model)
print(f"[Base VLM30]    Top1: {vlm_top1}/{vlm_n} | Top3: {vlm_top3}/{vlm_n} | Avg: {vlm_avg:.4f}")

# ── Hard negative mining ──
all_q = [p["query"] for p in training_pairs] + [p["query"] for p in vlm_pairs]
all_p = [p["positive"] for p in training_pairs] + [p["positive"] for p in vlm_pairs]
print(f"\n학습 데이터: 키워드 {len(training_pairs)}쌍 + VLM {len(vlm_pairs)}쌍 = 총 {len(all_q)}쌍")

mined = mine_hard_negatives(model, all_q, all_p)
del model; gc.collect(); torch.cuda.empty_cache()

# ── 하이퍼파라미터 탐색 ──
configs = [
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 8,  "warmup": 25, "seed": 42,  "vlm_repeat": 4, "max_seq": 512, "label": "lr=3e-6 ep=8 vr=4 seq=512"},
    {"lr": 5e-6, "bs": 4, "mini_bs": 32, "epochs": 6,  "warmup": 20, "seed": 42,  "vlm_repeat": 5, "max_seq": 512, "label": "lr=5e-6 ep=6 vr=5 seq=512"},
    {"lr": 2e-6, "bs": 4, "mini_bs": 32, "epochs": 12, "warmup": 35, "seed": 42,  "vlm_repeat": 5, "max_seq": 512, "label": "lr=2e-6 ep=12 vr=5 seq=512"},
    {"lr": 4e-6, "bs": 4, "mini_bs": 32, "epochs": 8,  "warmup": 25, "seed": 77,  "vlm_repeat": 4, "max_seq": 512, "label": "lr=4e-6 ep=8 seed=77 vr=4"},
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 10, "warmup": 30, "seed": 42,  "vlm_repeat": 6, "max_seq": 512, "label": "lr=3e-6 ep=10 vr=6 seq=512"},
    {"lr": 1e-6, "bs": 4, "mini_bs": 32, "epochs": 15, "warmup": 40, "seed": 42,  "vlm_repeat": 5, "max_seq": 512, "label": "lr=1e-6 ep=15 vr=5 seq=512"},
    {"lr": 5e-6, "bs": 4, "mini_bs": 32, "epochs": 5,  "warmup": 15, "seed": 42,  "vlm_repeat": 3, "max_seq": 512, "label": "lr=5e-6 ep=5 vr=3 seq=512"},
    {"lr": 8e-6, "bs": 4, "mini_bs": 32, "epochs": 4,  "warmup": 10, "seed": 42,  "vlm_repeat": 3, "max_seq": 512, "label": "lr=8e-6 ep=4 vr=3 seq=512"},
]

print(f"\n총 {len(configs)}개 설정 탐색")
print("=" * 100)

best_vlm_top1 = 0
best_vlm_avg = 0.0
best_orig_top1 = 0
best_cfg_idx = -1
best_combined = 0.0

for ci, cfg in enumerate(configs):
    print(f"\n{'─'*80}")
    print(f"[Config {ci+1}/{len(configs)}] {cfg['label']}")
    print(f"{'─'*80}")

    gc.collect(); torch.cuda.empty_cache()

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = cfg["max_seq"]

    vlm_repeat = cfg["vlm_repeat"]

    # ── Pair examples (CachedMNRL) ──
    pair_examples = []

    # 기존 키워드 학습 데이터
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            pair_examples.append(InputExample(texts=[p["query"], c]))

    # VLM 장문 학습 데이터 (반복 증강)
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

    # ── Loss 설정 ──
    train_objectives = []

    dl1 = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl1, losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_bs"])))

    dl2 = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl2, losses.AnglELoss(model)))

    dl3 = DataLoader(triplet_examples, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl3, losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

    # ── Train ──
    model.fit(
        train_objectives=train_objectives,
        epochs=cfg["epochs"],
        warmup_steps=cfg["warmup"],
        output_path=TEMP_DIR,
        show_progress_bar=False,
        optimizer_params={"lr": cfg["lr"]},
        weight_decay=0.01,
    )

    # ── 평가 ──
    top1, top3, avg, scores, details = evaluate(model)
    vt1, vt3, vn, vavg, vscores = evaluate_vlm(model)

    # 종합 점수: 키워드 유지 + VLM Top1 최대화 + VLM Avg
    combined = (top1 / 10) * 0.25 + (vt1 / vn) * 0.45 + vavg * 0.30

    print(f"  [키워드10] Top1: {top1}/10 | Top3: {top3}/10 | Avg: {avg:.4f}")
    print(f"  [VLM30]    Top1: {vt1}/{vn} ({vt1/vn*100:.1f}%) | Top3: {vt3}/{vn} ({vt3/vn*100:.1f}%) | Avg: {vavg:.4f}")
    print(f"  [종합]     Combined: {combined:.4f}")

    # 기존 테스트 상세 (변화 추적)
    for q, pos, s, t1, ok in details:
        idx = TEST_QUERIES.index(q)
        d = s - base_scores[idx]
        print(f"    {q:<28} {s:.4f} ({d:+.4f}) {'O' if ok else 'X'}")

    # 최적 모델 선택: 키워드 Top1 >= 8 유지하면서 VLM 성능 최대
    if top1 >= 8 and combined > best_combined:
        best_combined = combined
        best_vlm_top1 = vt1
        best_vlm_avg = vavg
        best_orig_top1 = top1
        best_cfg_idx = ci
        model.save(OUT_DIR)
        print(f"  >>> New best! Combined={combined:.4f} (키워드Top1={top1}, VLM_Top1={vt1}/{vn}, VLM_Avg={vavg:.4f})")

    # 결과 저장
    with open(os.path.join(BASE_DIR, f"boost_vlm_v2_r{ci+1}_result.json"), "w") as f:
        json.dump({
            "top1": top1, "top3": top3, "avg": avg, "scores": scores,
            "vlm_top1": vt1, "vlm_top3": vt3, "vlm_avg": vavg, "vlm_scores": vscores,
            "combined": combined, "config": cfg["label"],
        }, f, indent=2, ensure_ascii=False)

    del model; gc.collect(); torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# 최종 결과
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("  최종 결과")
print(f"{'=' * 100}")

if best_cfg_idx >= 0:
    print(f"\nBest Config: [{best_cfg_idx+1}] {configs[best_cfg_idx]['label']}")
    print(f"  키워드10 Top1: {best_orig_top1}/10")
    print(f"  VLM30  Top1:   {best_vlm_top1}/30 ({best_vlm_top1/30*100:.1f}%)")
    print(f"  VLM30  Avg:    {best_vlm_avg:.4f}")
    print(f"  Combined:      {best_combined:.4f}")

    # 최종 검증
    model = SentenceTransformer(OUT_DIR)
    model.max_seq_length = 512
    top1, top3, avg, scores, details = evaluate(model)
    vt1, vt3, vn, vavg, vscores = evaluate_vlm(model)

    print(f"\n[최종 검증]")
    print(f"  키워드10 Top1: {top1}/10 | Avg: {avg:.4f}")
    print(f"  VLM30  Top1:   {vt1}/{vn} ({vt1/vn*100:.1f}%) | Top3: {vt3}/{vn} | Avg: {vavg:.4f}")

    print(f"\n  키워드 테스트 상세:")
    for q, pos, s, t1, ok in details:
        idx = TEST_QUERIES.index(q)
        d = s - base_scores[idx]
        print(f"    {q:<28} {pos:<8} {s:.4f} ({d:+.4f}) {t1:<8} {'O' if ok else 'X'}")

    # Base 대비 개선 확인
    print(f"\n  [Base 대비 변화]")
    print(f"    키워드 Top1: {base_top1}/10 → {top1}/10 ({top1-base_top1:+d})")
    print(f"    키워드 Avg:  {base_avg:.4f} → {avg:.4f} ({avg-base_avg:+.4f})")
    print(f"    VLM Top1:    {vlm_top1}/{vlm_n} → {vt1}/{vn} ({vt1-vlm_top1:+d})")
    print(f"    VLM Avg:     {vlm_avg:.4f} → {vavg:.4f} ({vavg-vlm_avg:+.4f})")

    del model
else:
    print("키워드 Top1 8 이상 유지하는 모델 없음 - 더 보수적인 설정 필요")

print(f"\n{'=' * 100}")
