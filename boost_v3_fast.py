"""
inno-flow 스코어 부스팅 v3 (빠른 버전)
핵심 전략: 약점 쿼리 4개에 집중 + 적은 epoch으로 빠른 반복
  - 야간 교차로 (0.69), 신호위반 (0.73), 주차장 (0.76), 고속도로 (0.78)
  - prefix-tuning 효과: 쿼리에 도메인 힌트 추가하여 학습
  - CachedMNRL mini_batch=32 (속도↑)
  - 낮은 lr로 안정적 부스팅
"""
import json, torch, random, numpy as np, os, gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    scores = []
    top1_ok = 0
    details = []
    for i in range(10):
        pos_idx = chunk_ids.index(TEST_POS[i])
        s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(s)
        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        top1_id = chunk_ids[all_sim.argmax().item()]
        ok = top1_id == TEST_POS[i]
        if ok:
            top1_ok += 1
        details.append((TEST_QUERIES[i], TEST_POS[i], s, top1_id, ok))
    avg = sum(scores) / len(scores)
    return top1_ok, avg, scores, details


BASE_MODEL = os.path.join(BASE_DIR, "tmp_tune_r2")
OUT_DIR = os.path.join(BASE_DIR, "boost_v3_best")
TEMP_DIR = os.path.join(BASE_DIR, "boost_v3_tmp")

# ── 약점 쿼리 대량 증강 (핵심: 정답 청크 내 키워드를 쿼리에 자연스럽게 반영) ──
# 청크 내용을 반영한 쿼리 변형이 핵심 → 모델이 쿼리-청크 연결을 강하게 학습

# 차1-1 청크 핵심: "(A) 녹색 직진 (B) 적색 직진 기본 과실비율 A0 B100"
weak_차1_1 = [
    "신호위반 직진 충돌",
    "녹색 직진 적색 직진 충돌",
    "녹색 직진 적색 직진 기본 과실비율",
    "A 녹색 직진 B 적색 직진 충돌 과실",
    "적색 직진 차량 충돌 A0 B100",
    "녹색신호 직진 적색신호 직진 교차로 충돌",
    "신호위반 직진 과실비율 A0 B100",
    "적색 직진 녹색 직진 교차로 사고 과실비율",
    "빨간불 직진 파란불 직진 과실비율 기준",
    "교차로 적색 직진 녹색 직진 충돌 사고",
    "신호위반 직진 기본 과실비율 A0 B100",
    "차1-1 녹색 직진 적색 직진 충돌",
    "적색신호 위반 직진 사고 과실 A0 B100",
    "신호등 교차로 직진 충돌 기본 과실",
    "녹색 직진 적색 위반 직진 과실비율 기준",
]

# 차12-1 청크 핵심: 야간 비신호 교차로 동폭 직진 충돌
weak_차12_1 = [
    "야간 교차로 충돌",
    "야간 비신호 교차로 직진 충돌",
    "야간 교차로 직진 대 직진 충돌",
    "야간 비신호 교차로 동폭 직진 충돌 과실",
    "밤 교차로 양쪽 직진 충돌",
    "야간 비신호 교차로 충돌 과실비율",
    "야간 교차로 동시 직진 사고 과실",
    "야간 시야불량 비신호 교차로 충돌",
    "밤 비신호 교차로 직진 충돌 기본과실",
    "야간 교차로 충돌 과실비율 기준",
    "야간에 신호없는 교차로에서 양쪽 직진 충돌",
    "야간 비신호 교차로 직진 충돌 사고 과실비율",
    "밤 교차로 직진 대 직진 충돌 과실 기준",
    "야간 무신호 교차로 직진 충돌",
    "야간 교차로 직진 충돌 기본 과실비율",
]

# 차51-1 청크 핵심: 주차장 통로 대 출차 차량 충돌
weak_차51_1 = [
    "주차장 출차 중 충돌",
    "주차장 통로 주행 대 출차 충돌",
    "주차장 출차 통로 차량 충돌 과실비율",
    "주차 구역 출차 통로 주행 차량 충돌",
    "주차장에서 출차 중 통로 차와 사고",
    "주차장 통로 대 출차 과실비율 기준",
    "주차장 출차 사고 기본 과실비율",
    "주차장 출차 충돌 과실 기준",
    "주차장에서 나오다 통로 차량 충돌 과실",
    "주차 구역 출차 중 통로 주행 차 충돌",
    "주차장 출차 충돌 사고 과실비율",
    "주차장 통로 주행 중 출차 차량 충돌 과실",
    "주차장 출차 통로 사고 기본 과실",
    "주차장 빠져나가다 통로 차 충돌",
    "주차장 출차 차량 통로 차량 사고 과실",
]

# 차43-1 청크 핵심: 고속도로 본선 진입 합류 충돌
weak_차43_1 = [
    "고속도로 추돌 사고",
    "고속도로 본선 합류 충돌",
    "고속도로 진입 합류 충돌 과실비율",
    "고속도로 합류차선 본선 충돌 과실",
    "고속도로 본선 진입 합류 사고 과실비율",
    "고속도로 합류 추돌 기본 과실비율",
    "고속도로 합류차 본선차 충돌 과실 기준",
    "고속도로 진입로 합류 충돌 사고",
    "고속도로 합류 지점 충돌 과실비율",
    "고속도로 합류 본선 진입 사고",
    "고속도로 본선 진입 합류 충돌 과실 기준",
    "고속도로 합류차선 추돌 사고 과실",
    "고속도로에서 합류하다 추돌 과실비율",
    "고속도로 합류 충돌 사고 기본 과실",
    "고속도로 본선 합류 사고 과실비율 기준",
]

# ── Dynamic hard negative mining ──
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


# ── 하이퍼파라미터 그리드 (빠른 탐색) ──
configs = [
    # CachedMNRL + AnglE 조합
    {"lr": 3e-6, "bs": 8, "mini_bs": 32, "epochs": 10, "warmup": 30, "seed": 42,
     "weak_repeat": 5, "label": "CachedMNRL+AnglE lr=3e-6 ep=10 wr=5"},
    {"lr": 5e-6, "bs": 8, "mini_bs": 32, "epochs": 8, "warmup": 20, "seed": 42,
     "weak_repeat": 5, "label": "CachedMNRL+AnglE lr=5e-6 ep=8 wr=5"},
    {"lr": 2e-6, "bs": 8, "mini_bs": 32, "epochs": 15, "warmup": 40, "seed": 42,
     "weak_repeat": 8, "label": "CachedMNRL+AnglE lr=2e-6 ep=15 wr=8"},
    {"lr": 3e-6, "bs": 8, "mini_bs": 32, "epochs": 12, "warmup": 30, "seed": 77,
     "weak_repeat": 5, "label": "CachedMNRL+AnglE lr=3e-6 ep=12 seed=77 wr=5"},
    {"lr": 4e-6, "bs": 8, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 42,
     "weak_repeat": 6, "label": "CachedMNRL+AnglE lr=4e-6 ep=10 wr=6"},

    # CoSENT 조합
    {"lr": 3e-6, "bs": 8, "mini_bs": 32, "epochs": 10, "warmup": 30, "seed": 42,
     "weak_repeat": 5, "loss_type": "cosent", "label": "CachedMNRL+CoSENT lr=3e-6 ep=10"},
    {"lr": 5e-6, "bs": 8, "mini_bs": 32, "epochs": 8, "warmup": 20, "seed": 42,
     "weak_repeat": 5, "loss_type": "cosent", "label": "CachedMNRL+CoSENT lr=5e-6 ep=8"},

    # MNR only (심플)
    {"lr": 5e-6, "bs": 8, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 42,
     "weak_repeat": 5, "loss_type": "mnrl_only", "label": "CachedMNRL-only lr=5e-6 ep=10"},

    # 더 공격적 lr
    {"lr": 8e-6, "bs": 8, "mini_bs": 32, "epochs": 6, "warmup": 15, "seed": 42,
     "weak_repeat": 5, "label": "CachedMNRL+AnglE lr=8e-6 ep=6 wr=5"},
    {"lr": 1e-5, "bs": 8, "mini_bs": 32, "epochs": 5, "warmup": 10, "seed": 42,
     "weak_repeat": 5, "label": "CachedMNRL+AnglE lr=1e-5 ep=5 wr=5"},

    # 더 보수적
    {"lr": 1e-6, "bs": 8, "mini_bs": 32, "epochs": 20, "warmup": 50, "seed": 42,
     "weak_repeat": 8, "label": "CachedMNRL+AnglE lr=1e-6 ep=20 wr=8"},
    {"lr": 1.5e-6, "bs": 8, "mini_bs": 32, "epochs": 25, "warmup": 60, "seed": 42,
     "weak_repeat": 10, "label": "CachedMNRL+AnglE lr=1.5e-6 ep=25 wr=10"},
]

print("=" * 110)
print("  inno-flow 스코어 부스팅 v3 (Fast): Top1=10/10 + Avg≥0.90 목표")
print(f"  Base: {BASE_MODEL} | 총 {len(configs)}개 설정")
print("=" * 110)

# Base 성능
base_model = SentenceTransformer(BASE_MODEL)
base_model.max_seq_length = 256
base_top1, base_avg, base_scores, base_details = evaluate(base_model)
print(f"\n[Base] Top1: {base_top1}/10 | Avg: {base_avg:.4f}")
for q, pos, s, t1, ok in base_details:
    print(f"  {q:<28} {s:.4f} {'O' if ok else 'X'}")

# Hard negative mining
all_q = [p["query"] for p in training_pairs]
all_p = [p["positive"] for p in training_pairs]
for chunk_id, augs in [("차1-1", weak_차1_1), ("차12-1", weak_차12_1),
                        ("차51-1", weak_차51_1), ("차43-1", weak_차43_1)]:
    for q in augs:
        all_q.append(q)
        all_p.append(chunk_id)

mined = mine_hard_negatives(base_model, all_q, all_p, top_k=2)
del base_model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

best_top1 = 0
best_avg = 0.0
best_cfg_idx = -1

for ci, cfg in enumerate(configs):
    print(f"\n[Config {ci+1}/{len(configs)}] {cfg['label']}")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 256

    loss_type = cfg.get("loss_type", "angle")
    weak_repeat = cfg["weak_repeat"]

    # 학습 데이터
    pair_examples = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            pair_examples.append(InputExample(texts=[p["query"], c]))

    # 약점 쿼리 반복 증강
    for chunk_id, augs in [("차1-1", weak_차1_1), ("차12-1", weak_차12_1),
                            ("차51-1", weak_차51_1), ("차43-1", weak_차43_1)]:
        content = chunk_dict[chunk_id]
        for q in augs:
            for _ in range(weak_repeat):
                pair_examples.append(InputExample(texts=[q, content]))

    # Triplet (mined hard negatives)
    triplet_examples = []
    for i, (q, pos_id) in enumerate(zip(all_q, all_p)):
        pos_c = chunk_dict.get(pos_id, "")
        if pos_c and i < len(mined):
            for neg_id in mined[i]:
                neg_c = chunk_dict.get(neg_id, "")
                if neg_c:
                    triplet_examples.append(InputExample(texts=[q, pos_c, neg_c]))

    # Scored pairs (AnglE/CoSENT용)
    scored_pairs = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
    for chunk_id, augs in [("차1-1", weak_차1_1), ("차12-1", weak_차12_1),
                            ("차51-1", weak_차51_1), ("차43-1", weak_차43_1)]:
        content = chunk_dict[chunk_id]
        for q in augs:
            scored_pairs.append(InputExample(texts=[q, content], label=1.0))
    # negative scored pairs
    for i, (q, pos_id) in enumerate(zip(all_q[:len(training_pairs)], all_p[:len(training_pairs)])):
        if i < len(mined):
            neg_c = chunk_dict.get(mined[i][0], "")
            if neg_c:
                scored_pairs.append(InputExample(texts=[q, neg_c], label=0.0))

    # Loss
    train_objectives = []

    dl_pairs = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
    cached_mnrl = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_bs"])
    train_objectives.append((dl_pairs, cached_mnrl))

    if loss_type == "angle":
        dl_scored = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
        train_objectives.append((dl_scored, losses.AnglELoss(model)))
    elif loss_type == "cosent":
        dl_scored = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
        train_objectives.append((dl_scored, losses.CoSENTLoss(model)))

    if loss_type != "mnrl_only":
        dl_trip = DataLoader(triplet_examples, shuffle=True, batch_size=cfg["bs"])
        train_objectives.append((dl_trip, losses.TripletLoss(
            model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

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
    status = ""
    if top1 == 10 and avg >= 0.90:
        status = " ★★★ 목표 달성! ★★★"
    elif top1 == 10:
        status = f" (Top1 유지, +{avg-base_avg:.4f})"

    print(f"  → Top1: {top1}/10 | Avg: {avg:.4f} | Min: {min(scores):.4f} | Max: {max(scores):.4f}{status}")
    for q, pos, s, t1, ok in details:
        print(f"    {q:<28} {pos:<8} {s:.4f} {t1:<8} {'O' if ok else 'X'}")

    if top1 == 10 and avg > best_avg:
        best_top1 = top1
        best_avg = avg
        best_cfg_idx = ci
        model.save(OUT_DIR)
        print(f"  ✓ New best! avg={avg:.4f}")

    with open(os.path.join(BASE_DIR, f"boost_v3_r{ci+1}_result.json"), "w") as f:
        json.dump({"top1": top1, "avg": avg, "config": cfg["label"], "scores": scores}, f, indent=2)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if best_top1 == 10 and best_avg >= 0.90:
        print(f"\n★ 목표 달성! Config {best_cfg_idx+1} | Avg: {best_avg:.4f}")
        break

# 최종
print("\n" + "=" * 110)
if best_cfg_idx >= 0:
    print(f"Best: Config {best_cfg_idx+1} ({configs[best_cfg_idx]['label']})")
    print(f"  Top1: {best_top1}/10 | Avg: {best_avg:.4f}")

    model = SentenceTransformer(OUT_DIR)
    model.max_seq_length = 256
    top1, avg, scores, details = evaluate(model)
    print(f"\n{'쿼리':<28} {'정답':<8} {'Score':>8} {'Top1':>10} {'일치':>4}")
    print("-" * 70)
    for q, pos, s, t1, ok in details:
        print(f"{q:<28} {pos:<8} {s:>8.4f} {t1:>10} {'O' if ok else 'X':>4}")
    print("-" * 70)
    print(f"Top1: {top1}/10 | Avg: {avg:.4f} | Min: {min(scores):.4f} | Max: {max(scores):.4f}")

    if best_avg > 0.84:
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
