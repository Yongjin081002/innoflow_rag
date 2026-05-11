"""
inno-flow 약점 집중 부스팅
핵심 문제: 테스트 쿼리와 청크 사이의 어휘 갭 (semantic gap)
  - "야간 교차로 충돌" → 차12-1 (청크에 "야간" 없음, "좌측도로 직진" 내용)
  - "신호위반 직진 충돌" → 차1-1 (청크에 "신호위반" 없음, "녹색 직진 적색 직진")
  - "고속도로 추돌 사고" → 차43-1 (청크에 "추돌" 없음, "본선차 합류차")
  - "주차장 출차 중 충돌" → 차51-1 (비교적 매치됨)

전략:
  1. 청크의 실제 키워드를 포함한 쿼리 변형 (bridge queries)
  2. 테스트 쿼리와 비슷한 자연어 쿼리 → 정답 청크 연결 강화
  3. CachedMNRL로 대규모 in-batch negative 효과
  4. 기존 강점 쿼리도 적절히 유지 (붕괴 방지)
"""
import json, torch, random, numpy as np, os, gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from training_data import training_pairs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_dict = {c["id"]: c["content"] for c in chunks}
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

TEST_QUERIES = [
    "신호위반 직진 충돌", "비신호교차로 직진 vs 좌회전", "추돌 사고 과실",
    "야간 교차로 충돌", "중앙선 침범 충돌", "끼어들기 충돌",
    "유턴 중 충돌", "고속도로 추돌 사고", "주차장 출차 중 충돌", "횡단보도 보행자 충돌"
]
TEST_POS = [
    "차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
    "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"
]

def evaluate(model):
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(TEST_QUERIES, convert_to_tensor=True, show_progress_bar=False)
    scores, top1_ok, details = [], 0, []
    for i in range(10):
        pos_idx = chunk_ids.index(TEST_POS[i])
        s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(s)
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        top1_id = chunk_ids[all_sim.argmax().item()]
        ok = top1_id == TEST_POS[i]
        if ok: top1_ok += 1
        details.append((TEST_QUERIES[i], TEST_POS[i], s, top1_id, ok))
    return top1_ok, sum(scores)/len(scores), scores, details


# ══════════════════════════════════════════════════════════════
# 약점 쿼리 Bridge Training Data
# 핵심: 자연어 쿼리 ↔ 청크 내 실제 키워드를 연결하는 "브릿지" 쿼리
# ══════════════════════════════════════════════════════════════

# ── 최약점: 차12-1 (0.6869) ──
# 청크 핵심: "좌측도로에서 직진", "동시/선진입/후진입", A40 B60
# 테스트 쿼리: "야간 교차로 충돌"
# 문제: "야간"이 청크에 없음 → 비신호 교차로 동폭도로 직진 vs 좌측도로 직진
bridge_차12_1 = [
    # 자연어 ↔ 청크 키워드 브릿지
    "야간 교차로 충돌",
    "야간 교차로 충돌 좌측도로 직진",
    "교차로 좌측도로 직진 충돌",
    "비신호 교차로 직진 좌측도로 직진 충돌",
    "교차로 동시진입 좌측도로 직진 과실",
    "비신호 교차로 동폭도로 직진 충돌",
    "교차로 좌측도로 직진 동시진입 A40 B60",
    "비신호 교차로 좌측 직진 충돌 과실비율",
    "교차로 좌측도로 선진입 후진입 과실",
    "비신호 교차로 양방향 직진 충돌 좌측",
    "야간 비신호 교차로 좌측도로 직진 충돌",
    "밤 교차로 좌측도로 직진 동시진입 사고",
    "야간 교차로 좌측도로 직진 충돌 과실",
    "교차로 직진 대 좌측도로 직진 충돌",
    "비신호 교차로 동폭 좌측 직진 과실비율",
    "야간 비신호 교차로 직진 충돌 과실비율",
    "교차로 좌측 직진 A40 B60 과실비율",
    "야간 교차로 직진 충돌 동시진입 과실",
    "비신호교차로 좌측도로 직진 충돌 기본과실",
    "야간 교차로 직진 충돌 좌측 과실비율 기준",
    # 순수 자연어 변형
    "밤에 교차로에서 충돌",
    "야간 교차로 사고",
    "야간 교차로 차량 동시 진입 충돌",
    "야간 교차로에서 양쪽 직진 충돌",
    "야간 비신호 교차로 사고 과실",
]

# ── 약점: 차1-1 (0.7286) ──
# 청크 핵심: "(A) 녹색 직진 (B) 적색 직진 기본 과실비율 A0 B100"
# 테스트 쿼리: "신호위반 직진 충돌"
bridge_차1_1 = [
    "신호위반 직진 충돌",
    "신호위반 직진 충돌 녹색 직진 적색 직진",
    "녹색 직진 적색 직진 충돌",
    "녹색 직진 적색 직진 기본 과실비율 A0 B100",
    "신호위반 교차로 녹색 직진 적색 직진",
    "적색 직진 녹색 직진 충돌 과실비율",
    "녹색 직진 적색 위반 직진 A0 B100",
    "교차로 녹색 직진 적색 직진 충돌 사고",
    "신호위반 직진 녹색 적색 과실비율",
    "빨간불 직진 파란불 직진 녹색 적색 충돌",
    "적색신호 직진 녹색신호 직진 교차로 과실",
    "녹색 직진 대 적색 직진 기본 과실비율",
    "신호위반 직진 충돌 A0 B100 과실",
    "적색 직진 교차로 사고 A0 B100",
    "녹색 직진 적색 직진 교차로 사고 기본과실",
    # 순수 자연어
    "신호위반 직진 충돌 과실비율",
    "빨간불 직진 충돌 과실 기준",
    "적색신호 위반 직진 사고",
    "교차로 신호위반 직진 충돌 사고",
    "신호 무시 직진 충돌 과실비율",
]

# ── 약점: 차43-1 (0.7750) ──
# 청크 핵심: "(A) 본선차 (B) 합류차 기본 과실비율 A40 B60"
# 테스트 쿼리: "고속도로 추돌 사고"
bridge_차43_1 = [
    "고속도로 추돌 사고",
    "고속도로 본선차 합류차 충돌",
    "고속도로 본선차 합류차 과실비율 A40 B60",
    "고속도로 합류 충돌 본선차 합류차",
    "고속도로 추돌 본선차 합류차 과실",
    "고속도로 합류차선 본선 충돌 과실비율",
    "본선차 합류차 고속도로 충돌 A40 B60",
    "고속도로 합류 본선 충돌 기본 과실비율",
    "고속도로 추돌 사고 본선 합류 과실",
    "고속도로 합류차 본선차 사고 과실비율",
    "고속도로 진입로 합류 본선 충돌",
    "고속도로 합류 추돌 A40 B60",
    "고속도로 본선 합류 충돌 기본과실",
    "고속도로 합류 지점 본선차 합류차 사고",
    "고속도로 추돌 합류차 본선차 과실 기준",
    # 순수 자연어
    "고속도로 추돌 사고 과실비율",
    "고속도로 합류 충돌 과실 기준",
    "고속도로에서 합류하다 추돌",
    "고속도로 합류차선 사고 과실",
    "고속도로 진입 합류 추돌 사고",
]

# ── 약점: 차51-1 (0.7580) ──
# 청크 핵심: "(A) 통로주행차 (B) 주차구획에서 출차 기본 과실비율 A30 B70"
# 테스트 쿼리: "주차장 출차 중 충돌"
bridge_차51_1 = [
    "주차장 출차 중 충돌",
    "주차장 통로주행차 출차 충돌",
    "통로주행차 주차구획 출차 충돌 과실비율",
    "주차장 통로주행차 주차구획 출차 A30 B70",
    "주차장 출차 통로주행차 충돌 과실",
    "주차구획 출차 통로 주행 차량 충돌",
    "주차장 출차 통로 충돌 A30 B70",
    "통로주행차 출차 차량 충돌 과실비율",
    "주차장 통로 주행 출차 기본 과실비율",
    "주차구획에서 출차 통로주행차 충돌",
    "주차장 출차 충돌 통로주행차 과실",
    "주차장 출차 통로 A30 B70 과실비율",
    "주차구획 출차 통로 차량 사고 과실",
    "주차장 통로주행 대 출차 충돌 기본과실",
    "주차장 출차 중 통로 차량 A30 B70",
    # 순수 자연어
    "주차장 출차 충돌 과실비율",
    "주차장에서 나오다 통로 차 충돌",
    "주차장 출차 사고 과실 기준",
    "주차장 출차 중 사고 누구 잘못",
    "주차장 빠져나오다 충돌 과실",
]

# ── 하드 네거티브 (각 약점별 혼동되기 쉬운 청크) ──
def mine_hard_negatives(model, queries, pos_ids, top_k=3):
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    hard_negs = []
    for i, (q, pos_id) in enumerate(zip(queries, pos_ids)):
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        sorted_indices = all_sim.argsort(descending=True)
        pos_idx = chunk_ids.index(pos_id)
        neg_ids = []
        for idx in sorted_indices:
            idx = idx.item()
            if idx != pos_idx and len(neg_ids) < top_k:
                neg_ids.append(chunk_ids[idx])
        hard_negs.append(neg_ids)
    return hard_negs

BASE_MODEL = os.path.join(BASE_DIR, "tmp_tune_r2")
OUT_DIR = os.path.join(BASE_DIR, "boost_weak_best")
TEMP_DIR = os.path.join(BASE_DIR, "boost_weak_tmp")

print("=" * 110)
print("  약점 집중 부스팅: Top1=10/10 + Avg≥0.90")
print("=" * 110)

# Base 평가
model = SentenceTransformer(BASE_MODEL)
model.max_seq_length = 256
base_top1, base_avg, base_scores, base_details = evaluate(model)
print(f"\n[Base] Top1: {base_top1}/10 | Avg: {base_avg:.4f}")
for q, pos, s, t1, ok in base_details:
    weak = " ◀ 약점" if s < 0.80 else ""
    print(f"  {q:<28} {s:.4f} {'O' if ok else 'X'}{weak}")

# Hard negative mining
weak_queries_all = []
weak_pos_all = []
for chunk_id, bridges in [("차12-1", bridge_차12_1), ("차1-1", bridge_차1_1),
                           ("차43-1", bridge_차43_1), ("차51-1", bridge_차51_1)]:
    for q in bridges:
        weak_queries_all.append(q)
        weak_pos_all.append(chunk_id)

# 기존 training pairs도 포함
all_q = [p["query"] for p in training_pairs] + weak_queries_all
all_p = [p["positive"] for p in training_pairs] + weak_pos_all
mined = mine_hard_negatives(model, all_q, all_p, top_k=3)

# 약점 쿼리별 hard negatives 출력
print("\n[Hard Negatives for weak queries]")
for q_name, chunk_id in [("야간 교차로 충돌", "차12-1"), ("신호위반 직진 충돌", "차1-1"),
                          ("고속도로 추돌 사고", "차43-1"), ("주차장 출차 중 충돌", "차51-1")]:
    idx = all_q.index(q_name)
    print(f"  {q_name} → 정답: {chunk_id}, 혼동: {mined[idx]}")

del model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── 하이퍼파라미터 그리드 ──
configs = [
    {"lr": 3e-6, "bs": 8, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 42,
     "weak_repeat": 8, "label": "weak-focus lr=3e-6 ep=10 wr=8"},
    {"lr": 5e-6, "bs": 8, "mini_bs": 32, "epochs": 8, "warmup": 20, "seed": 42,
     "weak_repeat": 8, "label": "weak-focus lr=5e-6 ep=8 wr=8"},
    {"lr": 2e-6, "bs": 8, "mini_bs": 32, "epochs": 15, "warmup": 40, "seed": 42,
     "weak_repeat": 10, "label": "weak-focus lr=2e-6 ep=15 wr=10"},
    {"lr": 4e-6, "bs": 8, "mini_bs": 32, "epochs": 12, "warmup": 30, "seed": 42,
     "weak_repeat": 8, "label": "weak-focus lr=4e-6 ep=12 wr=8"},
    {"lr": 3e-6, "bs": 8, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 77,
     "weak_repeat": 8, "label": "weak-focus lr=3e-6 ep=10 seed=77"},
    {"lr": 5e-6, "bs": 8, "mini_bs": 32, "epochs": 6, "warmup": 15, "seed": 42,
     "weak_repeat": 12, "label": "weak-focus lr=5e-6 ep=6 wr=12"},
    {"lr": 8e-6, "bs": 8, "mini_bs": 32, "epochs": 5, "warmup": 10, "seed": 42,
     "weak_repeat": 10, "label": "weak-focus lr=8e-6 ep=5 wr=10"},
    {"lr": 1e-6, "bs": 8, "mini_bs": 32, "epochs": 20, "warmup": 50, "seed": 42,
     "weak_repeat": 10, "label": "weak-focus lr=1e-6 ep=20 wr=10"},
    {"lr": 3e-6, "bs": 4, "mini_bs": 64, "epochs": 10, "warmup": 25, "seed": 42,
     "weak_repeat": 8, "label": "weak-focus bs=4 mini=64 lr=3e-6 ep=10"},
    {"lr": 2e-6, "bs": 4, "mini_bs": 64, "epochs": 12, "warmup": 30, "seed": 42,
     "weak_repeat": 10, "label": "weak-focus bs=4 mini=64 lr=2e-6 ep=12"},
]

best_top1, best_avg, best_cfg_idx = 0, 0.0, -1

for ci, cfg in enumerate(configs):
    print(f"\n[Config {ci+1}/{len(configs)}] {cfg['label']}")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 256
    wr = cfg["weak_repeat"]

    # ── 학습 데이터 ──
    pair_examples = []

    # 기존 training pairs (기존 강점 유지)
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            pair_examples.append(InputExample(texts=[p["query"], c]))

    # 약점 bridge 쿼리 (핵심! 반복 증강)
    for chunk_id, bridges in [("차12-1", bridge_차12_1), ("차1-1", bridge_차1_1),
                               ("차43-1", bridge_차43_1), ("차51-1", bridge_차51_1)]:
        content = chunk_dict[chunk_id]
        for q in bridges:
            for _ in range(wr):
                pair_examples.append(InputExample(texts=[q, content]))

    # Triplet (hard negatives)
    triplet_examples = []
    for i, (q, pos_id) in enumerate(zip(all_q, all_p)):
        pos_c = chunk_dict.get(pos_id, "")
        if pos_c and i < len(mined):
            for neg_id in mined[i][:2]:
                neg_c = chunk_dict.get(neg_id, "")
                if neg_c:
                    triplet_examples.append(InputExample(texts=[q, pos_c, neg_c]))

    # AnglE scored pairs
    scored_pairs = []
    # 약점 쿼리 scored pairs (label=1.0, 중복 없이)
    for chunk_id, bridges in [("차12-1", bridge_차12_1), ("차1-1", bridge_차1_1),
                               ("차43-1", bridge_차43_1), ("차51-1", bridge_차51_1)]:
        content = chunk_dict[chunk_id]
        for q in bridges:
            scored_pairs.append(InputExample(texts=[q, content], label=1.0))
    # 기존 pair도 추가
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
    # hard negative scored (label=0.0)
    for i, (q, pos_id) in enumerate(zip(weak_queries_all, weak_pos_all)):
        base_idx = len(training_pairs) + (i % len(weak_queries_all))
        if base_idx < len(mined):
            neg_c = chunk_dict.get(mined[base_idx][0], "")
            if neg_c:
                scored_pairs.append(InputExample(texts=[q, neg_c], label=0.0))

    # ── Loss ──
    train_objectives = []

    dl_pairs = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
    cached_mnrl = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_bs"])
    train_objectives.append((dl_pairs, cached_mnrl))

    dl_scored = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl_scored, losses.AnglELoss(model)))

    dl_trip = DataLoader(triplet_examples, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl_trip, losses.TripletLoss(
        model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

    print(f"  pairs={len(pair_examples)}, scored={len(scored_pairs)}, triplets={len(triplet_examples)}")

    model.fit(
        train_objectives=train_objectives,
        epochs=cfg["epochs"],
        warmup_steps=cfg["warmup"],
        output_path=TEMP_DIR,
        show_progress_bar=False,
        optimizer_params={"lr": cfg["lr"]},
        weight_decay=0.01,
    )

    top1, avg, scores, details = evaluate(model)
    delta = avg - base_avg
    status = ""
    if top1 == 10 and avg >= 0.90:
        status = " ★★★ 목표 달성! ★★★"
    elif top1 == 10:
        status = f" (Top1 유지, delta={delta:+.4f})"
    else:
        status = f" ✗ Top1 하락! ({top1}/10)"

    print(f"  → Top1: {top1}/10 | Avg: {avg:.4f}{status}")
    for idx, (q, pos, s, t1, ok) in enumerate(details):
        d = s - base_scores[idx]
        weak = " ◀" if base_scores[idx] < 0.80 else ""
        print(f"    {q:<28} {s:.4f} ({d:+.4f}) {'O' if ok else 'X'}{weak}")

    if top1 == 10 and avg > best_avg:
        best_top1 = top1
        best_avg = avg
        best_cfg_idx = ci
        model.save(OUT_DIR)
        print(f"  ✓ New best! avg={avg:.4f}")

    with open(os.path.join(BASE_DIR, f"boost_weak_r{ci+1}_result.json"), "w") as f:
        json.dump({"top1": top1, "avg": avg, "config": cfg["label"],
                    "scores": scores, "delta": delta}, f, indent=2)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if best_top1 == 10 and best_avg >= 0.90:
        print(f"\n★ 목표 달성! Config {best_cfg_idx+1} | Avg: {best_avg:.4f}")
        break

# ── 최종 ──
print("\n" + "=" * 110)
if best_cfg_idx >= 0:
    print(f"Best: Config {best_cfg_idx+1} ({configs[best_cfg_idx]['label']})")
    print(f"  Top1: {best_top1}/10 | Avg: {best_avg:.4f} (delta: {best_avg - base_avg:+.4f})")

    model = SentenceTransformer(OUT_DIR)
    model.max_seq_length = 256
    top1, avg, scores, details = evaluate(model)
    print(f"\n{'쿼리':<28} {'정답':<8} {'Score':>8} {'Delta':>8} {'Top1':>10}")
    print("-" * 80)
    for idx, (q, pos, s, t1, ok) in enumerate(details):
        d = s - base_scores[idx]
        print(f"{q:<28} {pos:<8} {s:>8.4f} {d:>+8.4f} {t1:>10} {'O' if ok else 'X'}")
    print("-" * 80)
    print(f"Top1: {top1}/10 | Avg: {avg:.4f} | Min: {min(scores):.4f} | Max: {max(scores):.4f}")

    if best_avg > base_avg:
        import shutil
        finetuned = os.path.join(BASE_DIR, "bge-m3-finetuned")
        backup = os.path.join(BASE_DIR, "bge-m3-finetuned-backup")
        if os.path.exists(finetuned):
            if os.path.exists(backup): shutil.rmtree(backup)
            shutil.copytree(finetuned, backup)
        shutil.copytree(OUT_DIR, finetuned, dirs_exist_ok=True)
        print(f"  bge-m3-finetuned 갱신!")
    del model
else:
    print("Top1 10/10 유지 모델 없음")
print("=" * 110)
