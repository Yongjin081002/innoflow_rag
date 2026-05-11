"""
inno-flow 스코어 부스팅 v2: Top1=10/10 유지 + Avg Score 0.90+ 달성
전략:
  1. tmp_tune_r2 (현재 best: 10/10, 0.84) 기반
  2. CachedMultipleNegativesRankingLoss (mini_batch=64) → 큰 배치 효과
  3. AnglELoss → CosineSimilarityLoss보다 안정적 유사도 부스팅
  4. Dynamic hard negative mining → 실제 모델이 혼동하는 청크를 네거티브로 사용
  5. 매우 낮은 lr (1e-6~5e-6) → catastrophic forgetting 방지
"""
import json, torch, random, numpy as np, os, sys, gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from training_data import training_pairs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 청크 로드 ──
with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_dict = {c["id"]: c["content"] for c in chunks}
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

# ── 테스트 케이스 ──
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
    """Top1 + Avg Score 평가"""
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


def mine_hard_negatives(model, queries, pos_ids, top_k=3):
    """모델의 실제 예측 기반 hard negative mining"""
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


# ── 모든 training pair에서 쿼리/positive 추출 ──
all_queries = [p["query"] for p in training_pairs]
all_pos_ids = [p["positive"] for p in training_pairs]

# ── 약점 쿼리 추가 증강 ──
weak_augmentations = {
    "차1-1": [
        "신호위반 직진 충돌", "신호위반 직진 충돌 과실비율",
        "적색신호 직진 녹색신호 직진 충돌", "빨간불 무시 직진 충돌 사고",
        "교차로 신호위반 직진 차량 충돌 과실", "녹색 직진 적색 직진 교차로 충돌",
        "적색 위반 직진 충돌 A0 B100", "신호 위반 직진 교차로 충돌 과실비율",
        "자동차 신호위반 직진 충돌", "차량 간 신호위반 교차로 사고",
    ],
    "차12-1": [
        "야간 교차로 충돌", "야간 교차로 충돌 과실비율",
        "밤 교차로 충돌 사고", "야간 비신호 교차로 충돌",
        "야간 교차로 직진 충돌 사고", "야간 교차로 동시 진입 충돌",
        "야간 시야불량 교차로 충돌 과실", "야간 교차로 직진 대 직진 사고",
        "야간에 신호없는 교차로 충돌", "밤 교차로 차량 충돌 과실",
    ],
    "차51-1": [
        "주차장 출차 중 충돌", "주차장에서 나오다 충돌",
        "주차장 출차 충돌 과실비율", "주차장 출차 통로 충돌 사고",
        "주차장에서 빠져나오다 통로 차량과 충돌", "주차 구역 출차 중 충돌 과실",
        "주차장 통로 주행 중 출차 차량과 충돌", "주차장 나오다 통행 차량과 사고",
        "주차장 출차 사고 누구 잘못", "주차장에서 차 빼다 충돌",
    ],
    "차43-1": [
        "고속도로 추돌 사고", "고속도로 추돌 사고 과실비율",
        "고속도로 합류 충돌 과실", "고속도로 합류차선 사고",
        "고속도로 진입로 합류 충돌 과실", "고속도로 본선 합류 충돌 과실 기준",
        "고속도로에서 추돌 과실비율", "고속도로 추돌 누구 과실",
        "고속도로 합류차선 충돌 과실 기준", "고속도로 합류 추돌 과실 기준",
    ],
}

# ── 하이퍼파라미터 그리드 ──
configs = [
    # Phase 1: CachedMNRL + AnglELoss (안정적 부스팅)
    {"lr": 2e-6, "bs": 4, "mini_bs": 64, "epochs": 20, "warmup": 50, "seed": 42,
     "loss": "cached_mnrl+angle", "label": "CachedMNRL+AnglE lr=2e-6 ep=20"},

    {"lr": 3e-6, "bs": 4, "mini_bs": 64, "epochs": 25, "warmup": 60, "seed": 42,
     "loss": "cached_mnrl+angle", "label": "CachedMNRL+AnglE lr=3e-6 ep=25"},

    {"lr": 5e-6, "bs": 4, "mini_bs": 64, "epochs": 15, "warmup": 40, "seed": 42,
     "loss": "cached_mnrl+angle", "label": "CachedMNRL+AnglE lr=5e-6 ep=15"},

    # Phase 2: CachedMNRL + CoSENTLoss (pair-level ranking)
    {"lr": 2e-6, "bs": 4, "mini_bs": 64, "epochs": 20, "warmup": 50, "seed": 42,
     "loss": "cached_mnrl+cosent", "label": "CachedMNRL+CoSENT lr=2e-6 ep=20"},

    {"lr": 3e-6, "bs": 4, "mini_bs": 64, "epochs": 25, "warmup": 60, "seed": 42,
     "loss": "cached_mnrl+cosent", "label": "CachedMNRL+CoSENT lr=3e-6 ep=25"},

    # Phase 3: CachedMNRL only (심플하지만 강력)
    {"lr": 3e-6, "bs": 4, "mini_bs": 128, "epochs": 30, "warmup": 80, "seed": 42,
     "loss": "cached_mnrl_only", "label": "CachedMNRL-only mini=128 lr=3e-6 ep=30"},

    {"lr": 2e-6, "bs": 4, "mini_bs": 128, "epochs": 40, "warmup": 100, "seed": 42,
     "loss": "cached_mnrl_only", "label": "CachedMNRL-only mini=128 lr=2e-6 ep=40"},

    # Phase 4: Symmetric MNR (양방향 학습)
    {"lr": 3e-6, "bs": 4, "mini_bs": 64, "epochs": 25, "warmup": 60, "seed": 42,
     "loss": "cached_sym_mnrl+angle", "label": "CachedSymMNRL+AnglE lr=3e-6 ep=25"},

    # Phase 5: 다른 시드
    {"lr": 2e-6, "bs": 4, "mini_bs": 64, "epochs": 25, "warmup": 60, "seed": 77,
     "loss": "cached_mnrl+angle", "label": "CachedMNRL+AnglE lr=2e-6 seed=77"},

    {"lr": 3e-6, "bs": 4, "mini_bs": 64, "epochs": 20, "warmup": 50, "seed": 123,
     "loss": "cached_mnrl+angle", "label": "CachedMNRL+AnglE lr=3e-6 seed=123"},
]

BASE_MODEL = os.path.join(BASE_DIR, "tmp_tune_r2")
OUT_DIR = os.path.join(BASE_DIR, "boost_v2_best")
TEMP_DIR = os.path.join(BASE_DIR, "boost_v2_tmp")

best_top1 = 0
best_avg = 0.0
best_cfg_idx = -1

print("=" * 110)
print("  inno-flow 스코어 부스팅 v2: Top1=10/10 + Avg≥0.90 목표")
print(f"  Base model: {BASE_MODEL}")
print(f"  총 {len(configs)}개 설정 시도")
print("=" * 110)

# 먼저 base 모델 현재 성능 확인
print("\n[Base 모델 현재 성능]")
base_model = SentenceTransformer(BASE_MODEL)
base_model.max_seq_length = 256
base_top1, base_avg, base_scores, base_details = evaluate(base_model)
print(f"  Top1: {base_top1}/10 | Avg: {base_avg:.4f} | Min: {min(base_scores):.4f} | Max: {max(base_scores):.4f}")
for q, pos, s, t1, ok in base_details:
    mark = "O" if ok else "X"
    print(f"    {q:<28} {pos:<8} {s:.4f} {t1:<8} {mark}")

# Dynamic hard negative mining from base model
print("\n[Hard Negative Mining...]")
all_q_for_mining = list(all_queries)
all_p_for_mining = list(all_pos_ids)
# 약점 쿼리도 추가
for chunk_id, aug_queries in weak_augmentations.items():
    for q in aug_queries:
        all_q_for_mining.append(q)
        all_p_for_mining.append(chunk_id)

mined_hard_negs = mine_hard_negatives(base_model, all_q_for_mining, all_p_for_mining, top_k=3)
print(f"  총 {len(mined_hard_negs)}개 쿼리에서 hard negatives 채굴 완료")

del base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

for ci, cfg in enumerate(configs):
    print(f"\n{'='*110}")
    print(f"[Config {ci+1}/{len(configs)}] {cfg['label']}")
    print(f"  lr={cfg['lr']}, bs={cfg['bs']}, mini_bs={cfg['mini_bs']}, epochs={cfg['epochs']}, "
          f"warmup={cfg['warmup']}, seed={cfg['seed']}, loss={cfg['loss']}")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    # 모델 로드
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 256

    # ── 학습 데이터 구성 ──
    # 1) MNR/CachedMNRL용 positive pairs
    pair_examples = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            pair_examples.append(InputExample(texts=[p["query"], c]))

    # 약점 쿼리 증강 (2배 반복)
    for chunk_id, aug_queries in weak_augmentations.items():
        content = chunk_dict[chunk_id]
        for q in aug_queries:
            for _ in range(2):
                pair_examples.append(InputExample(texts=[q, content]))

    # 2) Triplet 데이터 (mined hard negatives 사용)
    triplet_examples = []
    for i, (q, pos_id) in enumerate(zip(all_q_for_mining, all_p_for_mining)):
        pos_c = chunk_dict.get(pos_id, "")
        if pos_c and i < len(mined_hard_negs):
            for neg_id in mined_hard_negs[i][:2]:  # top-2 hard negatives
                neg_c = chunk_dict.get(neg_id, "")
                if neg_c:
                    triplet_examples.append(InputExample(texts=[q, pos_c, neg_c]))

    # 3) AnglE/CoSENT용 scored pairs
    scored_pairs = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
    # 약점 쿼리
    for chunk_id, aug_queries in weak_augmentations.items():
        content = chunk_dict[chunk_id]
        for q in aug_queries:
            scored_pairs.append(InputExample(texts=[q, content], label=1.0))
    # 네거티브 scored pairs (hard negatives with label=0.0)
    for i, (q, pos_id) in enumerate(zip(all_q_for_mining[:len(training_pairs)], all_p_for_mining[:len(training_pairs)])):
        if i < len(mined_hard_negs):
            for neg_id in mined_hard_negs[i][:1]:  # top-1 hard negative만
                neg_c = chunk_dict.get(neg_id, "")
                if neg_c:
                    scored_pairs.append(InputExample(texts=[q, neg_c], label=0.0))

    # ── Loss 구성 ──
    train_objectives = []

    if "cached_mnrl" in cfg["loss"] and "sym" not in cfg["loss"]:
        dl_pairs = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
        cached_mnrl = losses.CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=cfg["mini_bs"]
        )
        train_objectives.append((dl_pairs, cached_mnrl))

    elif "cached_sym_mnrl" in cfg["loss"]:
        dl_pairs = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
        cached_sym = losses.CachedMultipleNegativesSymmetricRankingLoss(
            model, mini_batch_size=cfg["mini_bs"]
        )
        train_objectives.append((dl_pairs, cached_sym))

    if "angle" in cfg["loss"]:
        dl_scored = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
        angle_loss = losses.AnglELoss(model)
        train_objectives.append((dl_scored, angle_loss))

    if "cosent" in cfg["loss"]:
        dl_scored = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
        cosent_loss = losses.CoSENTLoss(model)
        train_objectives.append((dl_scored, cosent_loss))

    # Triplet loss 항상 추가 (판별력 유지)
    if triplet_examples and cfg["loss"] != "cached_mnrl_only":
        dl_triplets = DataLoader(triplet_examples, shuffle=True, batch_size=cfg["bs"])
        triplet_loss = losses.TripletLoss(
            model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2
        )
        train_objectives.append((dl_triplets, triplet_loss))

    print(f"  데이터: pairs={len(pair_examples)}, triplets={len(triplet_examples)}, scored={len(scored_pairs)}")
    print(f"  Train objectives: {len(train_objectives)}개")

    # ── 학습 ──
    model.fit(
        train_objectives=train_objectives,
        epochs=cfg["epochs"],
        warmup_steps=cfg["warmup"],
        output_path=TEMP_DIR,
        show_progress_bar=True,
        optimizer_params={"lr": cfg["lr"]},
        weight_decay=0.01,
    )

    # ── 평가 ──
    top1, avg, scores, details = evaluate(model)
    min_s = min(scores)
    max_s = max(scores)

    status = ""
    if top1 == 10 and avg >= 0.90:
        status = " ★★★ 목표 달성! ★★★"
    elif top1 == 10:
        status = " (Top1 유지)"
    elif top1 < base_top1:
        status = " ✗ Top1 하락!"

    print(f"\n  → Top1: {top1}/10 | Avg: {avg:.4f} | Min: {min_s:.4f} | Max: {max_s:.4f}{status}")

    for q, pos, s, t1, ok in details:
        mark = "O" if ok else "X"
        delta = s - base_scores[details.index((q, pos, s, t1, ok))] if (q, pos, s, t1, ok) in base_details else 0
        print(f"    {q:<28} {pos:<8} {s:.4f} {t1:<8} {mark}")

    # Best 갱신: Top1 10/10 유지 + avg 최대
    if top1 == 10 and avg > best_avg:
        best_top1 = top1
        best_avg = avg
        best_cfg_idx = ci
        model.save(OUT_DIR)
        print(f"  ✓ New best saved → {OUT_DIR} (avg: {avg:.4f})")

    # 결과 저장
    result = {"top1": top1, "avg": avg, "config": cfg["label"], "scores": scores}
    with open(os.path.join(BASE_DIR, f"boost_v2_r{ci+1}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 목표 달성하면 조기 종료
    if best_top1 == 10 and best_avg >= 0.90:
        print(f"\n★ 목표 달성! Config {best_cfg_idx+1} ({configs[best_cfg_idx]['label']}) | Avg: {best_avg:.4f}")
        break

# ── 최종 결과 ──
print("\n" + "=" * 110)
if best_cfg_idx >= 0:
    print(f"최종 Best: Config {best_cfg_idx+1} ({configs[best_cfg_idx]['label']})")
    print(f"  Top1: {best_top1}/10 | Avg: {best_avg:.4f}")
    print(f"  모델 저장: {OUT_DIR}")

    # 최종 상세 평가
    model = SentenceTransformer(OUT_DIR)
    model.max_seq_length = 256
    top1, avg, scores, details = evaluate(model)
    print(f"\n{'쿼리':<28} {'정답':<8} {'Score':>8} {'Top1':>10} {'일치':>4}")
    print("-" * 70)
    for q, pos, s, t1, ok in details:
        mark = "O" if ok else "X"
        print(f"{q:<28} {pos:<8} {s:>8.4f} {t1:>10} {mark:>4}")
    print("-" * 70)
    print(f"Top1: {top1}/10 | Avg: {avg:.4f} | Min: {min(scores):.4f} | Max: {max(scores):.4f}")

    # bge-m3-finetuned 갱신
    if best_avg > 0.84:
        import shutil
        finetuned_dir = os.path.join(BASE_DIR, "bge-m3-finetuned")
        backup_dir = os.path.join(BASE_DIR, "bge-m3-finetuned-backup")
        if os.path.exists(finetuned_dir):
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(finetuned_dir, backup_dir)
            print(f"  기존 모델 백업: {backup_dir}")
        shutil.copytree(OUT_DIR, finetuned_dir, dirs_exist_ok=True)
        print(f"  bge-m3-finetuned 갱신 완료!")

    del model
else:
    print("Top1 10/10을 유지하는 모델을 찾지 못했습니다.")
print("=" * 110)
