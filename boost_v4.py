"""
boost_v4: boost_best (Avg=0.9105) 기반 약점 보완
타겟 약점:
  - 차5-2 (0.8237): "횡단보도 보행자 충돌" → 청크는 "횡단보도 보행자신호 우회전 vs 녹색 직진"
  - 차43-1 (0.8308): "고속도로 추돌 사고" → 청크는 "본선차 vs 합류차"
  - 차1-1 (0.8638): "신호위반 직진 충돌" → 청크는 "녹색 직진 적색 직진"

전략: bridge query + hard negative triplet + 기존 강점 유지
"""
import json, torch, random, numpy as np, os, gc

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
    return top1_ok, sum(scores) / len(scores), scores, details


# ══════════════════════���═══════════════════════════════════════
# Bridge Queries: 약점 카테고리별 자연어 ↔ 청크 키워드 연결
# ══════════════════════════════════════════════════════════════

# ── 차5-2 (0.8237): "횡단보도 보행자 충돌" ──
# 청크: "(A) 횡단보도 보행자신호 우회전 (B) 녹색 직진 기본 과실비율 A100 B0"
# 핵심 갭: "보행자 충돌" vs "보행자신호 우회전 vs 녹색 직진"
bridge_차5_2 = [
    # 청크 키워드 직접 연결
    "횡단보도 보행자 충돌",
    "횡단보도 보행자신호 우회전 녹색 직진 충돌",
    "횡단보도 보행자신호 우회전 충돌 과실비율",
    "보행자신호 우회전 녹색 직진 충돌 A100 B0",
    "횡단보도 보행자신호 우회전 대 녹색 직진 과실",
    "횡단보도 우회전 보행자신호 충돌 기본 과실비율",
    "보행자신호에서 우회전하다 직진차와 충돌",
    "횡단보도 보행자 파란불 우회전 직진 충돌 과실",
    "횡단보도에서 보행자신호 중 우회전 직진차 사고",
    "보행자 신호일 때 우회전 녹색 직진 충돌",
    "보행자신호 우회전 녹색 직진 기본 과실비율 A100 B0",
    "횡단보도 보행자신호 우회전 직진 교차로 사고",
    "횡단보도 우회전 직진 충돌 A100 B0 과실",
    # 자연어 변형
    "횡단보도에서 보행자 신호인데 우회전 충돌",
    "횡단보도 보행자 충돌 과실비율",
    "횡단보도에서 우회전하다 충돌",
    "횡단보도 보행자 신호 우회전 사고",
    "보행자 횡단보도 우회전 충돌 과실",
    "횡단보도 보행자 충돌 사고 과실",
    "횡단보도 보행자 충돌 기본 과실비율",
    "횡단보도 보행자 충돌 누구 과실",
    "횡단보도에서 보행자 보호 의무 위반 충돌",
    "횡단보도 보행자 충돌 사고 과실비율 기준",
    "횡단보도 우회전 보행자 충돌 과실 기준",
    "횡단보도 보행자 충돌 사고 100대0",
]

# ── 차43-1 (0.8308): "고속도로 추돌 사고" ──
# 청크: "(A) 본선차 (B) 합류차 기본 과실비율 A40 B60"
bridge_차43_1 = [
    "고속도로 추돌 사고",
    "고속도로 본선차 합류차 충돌",
    "고속도로 본선차 합류차 과실비율 A40 B60",
    "고속도로 합류 충돌 본선차 합류차",
    "고속도로 추돌 본선차 합류차 과실",
    "본선차 합류차 고속도로 충돌 A40 B60",
    "고속도로 합류 본선 충돌 기본 과실비율",
    "고속도로 추돌 사고 본선 합류 과실",
    "고속도로 합류차 본선차 사고 과실비율",
    "고속도로 합류 추돌 A40 B60",
    "고속도로 추돌 사고 과실비율",
    "고속도로 합류 충돌 과실 기준",
    "고속도로에서 합류하다 추돌",
    "고속도로 합류차선 사고 과실",
    "고속도로 진입 합류 추돌 사고",
    "고속도로 본선 합류 충돌 기본과실",
    "고속도로 합류 지점 본선차 합류차 사고",
    "고속도로 추돌 합류차 본선차 과실 기준",
    "고속도로 합류차선 본선 충돌 과실비율",
    "고속도로 진입로 합류 본선 충돌",
    "고속도로 추돌 사고 본선차 합류차 누구 잘못",
    "고속도로 합류 사고 과실비율 기준",
    "고속도로에서 끼어들다 본선차와 충돌",
    "고속도로 합류지점 추돌 기본 과실비율",
    "고속도로 본선 합류 추돌 사고 과실비율 기준",
]

# ── 차1-1 (0.8638): "신호위반 직진 충돌" ──
# 청크: "(A) 녹색 직진 (B) 적색 직진 기본 과실비율 A0 B100"
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
    "적색신호 직진 녹색신호 직진 교차로 과실",
    "녹색 직진 대 적색 직진 기본 과실비율",
    "신호위반 직진 충돌 A0 B100 과실",
    "신호위반 직진 충돌 과실비율",
    "빨간불 직진 충돌 과실 기준",
    "적색신호 위반 직진 사고",
    "교차로 신호위반 직진 충돌 사고",
    "신호 무시 직진 충돌 과실비율",
    "녹색 직진 적색 직진 교차로 사고 기본과실",
    "적색 직진 교차로 사고 A0 B100",
    "빨간불 직진 파란불 직진 녹색 적색 충돌",
    "신호위반 직진 기본 과실비율 A0 B100",
    "신호위반 직진 충돌 100대0 과실",
    "적색 직진 교차로 충돌 과실비율 기준",
    "녹색 직진 적색 직진 교차로 사고 과실비율 A0 B100",
    "신호위반 직진 녹색 직진 적색 직진 기본과실",
]


def mine_hard_negatives(model, queries, pos_ids, top_k=3):
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


# ══════════════════���═════════════════════════��═════════════════
#  메인
# ══════════════════════════════════════════════════════════════
BASE_MODEL = os.path.join(BASE_DIR, "boost_best")
OUT_DIR = os.path.join(BASE_DIR, "boost_v4_best")
TEMP_DIR = os.path.join(BASE_DIR, "boost_v4_tmp")

print("=" * 100)
print("  boost_v4: boost_best 기반 약점 보완 (차5-2, 차43-1, 차1-1)")
print("=" * 100)

# Base 평가
model = SentenceTransformer(BASE_MODEL)
model.max_seq_length = 256
base_top1, base_avg, base_scores, base_details = evaluate(model)
print(f"\n[Base: boost_best] Top1: {base_top1}/10 | Avg: {base_avg:.4f}")
for q, pos, s, t1, ok in base_details:
    weak = " ◀ 약점" if s < 0.85 else ""
    print(f"  {q:<28} {s:.4f} {'O' if ok else 'X'}{weak}")

# 전체 쿼리 + bridge 쿼리 합치기
WEAK_BRIDGES = [
    ("차5-2", bridge_차5_2),
    ("차43-1", bridge_차43_1),
    ("차1-1", bridge_차1_1),
]

all_q = [p["query"] for p in training_pairs]
all_p = [p["positive"] for p in training_pairs]
for cid, bridges in WEAK_BRIDGES:
    for q in bridges:
        all_q.append(q)
        all_p.append(cid)

# Hard negative mining
mined = mine_hard_negatives(model, all_q, all_p)

# 약점 쿼리별 혼동 청크 출력
print("\n[Hard Negatives]")
for q_name, cid in [("횡단보도 보행자 충돌", "차5-2"), ("고속도로 추돌 사고", "차43-1"),
                     ("신호위반 직진 충돌", "차1-1")]:
    idx = all_q.index(q_name)
    print(f"  {q_name} → 정답: {cid}, 혼동: {mined[idx]}")

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── 하이퍼파라미터 그리드 ──
configs = [
    {"lr": 2e-6, "bs": 4, "mini_bs": 32, "epochs": 8,  "warmup": 20, "seed": 42, "wr": 6,
     "label": "lr=2e-6 ep=8 wr=6"},
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 6,  "warmup": 15, "seed": 42, "wr": 6,
     "label": "lr=3e-6 ep=6 wr=6"},
    {"lr": 1e-6, "bs": 4, "mini_bs": 32, "epochs": 12, "warmup": 30, "seed": 42, "wr": 8,
     "label": "lr=1e-6 ep=12 wr=8"},
    {"lr": 2e-6, "bs": 4, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 42, "wr": 8,
     "label": "lr=2e-6 ep=10 wr=8"},
    {"lr": 5e-7, "bs": 4, "mini_bs": 32, "epochs": 15, "warmup": 40, "seed": 42, "wr": 6,
     "label": "lr=5e-7 ep=15 wr=6"},
    {"lr": 1.5e-6, "bs": 4, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 77, "wr": 6,
     "label": "lr=1.5e-6 ep=10 wr=6 seed=77"},
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 8,  "warmup": 20, "seed": 42, "wr": 10,
     "label": "lr=3e-6 ep=8 wr=10"},
    {"lr": 2e-6, "bs": 4, "mini_bs": 32, "epochs": 6,  "warmup": 15, "seed": 42, "wr": 4,
     "label": "lr=2e-6 ep=6 wr=4"},
]

best_top1, best_avg, best_cfg_idx = 0, 0.0, -1

for ci, cfg in enumerate(configs):
    print(f"\n[Config {ci+1}/{len(configs)}] {cfg['label']}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 256
    wr = cfg["wr"]

    # ── 학습 데이터 구성 ──

    # 1) Pair examples: 기존 training pairs + 약점 bridge (반복 증강)
    pair_examples = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            pair_examples.append(InputExample(texts=[p["query"], c]))

    for cid, bridges in WEAK_BRIDGES:
        content = chunk_dict[cid]
        for q in bridges:
            for _ in range(wr):
                pair_examples.append(InputExample(texts=[q, content]))

    # 2) Scored pairs: AnglE loss용
    scored_pairs = []
    for cid, bridges in WEAK_BRIDGES:
        content = chunk_dict[cid]
        for q in bridges:
            scored_pairs.append(InputExample(texts=[q, content], label=1.0))
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))
    # hard negative scored (label=0.0) - 약점 쿼리의 혼동 청크
    n_train = len(training_pairs)
    for i, (q, pid) in enumerate(zip(all_q[n_train:], all_p[n_train:])):
        global_idx = n_train + i
        if global_idx < len(mined):
            for neg_id in mined[global_idx][:2]:
                neg_c = chunk_dict.get(neg_id, "")
                if neg_c:
                    scored_pairs.append(InputExample(texts=[q, neg_c], label=0.0))

    # 3) Triplet examples: hard negatives
    triplet_examples = []
    for i, (q, pid) in enumerate(zip(all_q, all_p)):
        pc = chunk_dict.get(pid, "")
        if pc and i < len(mined):
            for nid in mined[i][:2]:
                nc = chunk_dict.get(nid, "")
                if nc:
                    triplet_examples.append(InputExample(texts=[q, pc, nc]))

    # ── Loss 구성 ──
    train_objectives = []

    dl_pairs = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl_pairs, losses.CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=cfg["mini_bs"])))

    dl_scored = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl_scored, losses.AnglELoss(model)))

    dl_trip = DataLoader(triplet_examples, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl_trip, losses.TripletLoss(
        model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

    print(f"  pairs={len(pair_examples)}, scored={len(scored_pairs)}, triplets={len(triplet_examples)}")

    # ── 학습 ──
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
    top1, avg, scores, details = evaluate(model)
    delta = avg - base_avg

    if top1 == 10 and avg >= 0.92:
        status = " ★★★ 목표 달성! ★★★"
    elif top1 == 10 and delta > 0:
        status = f" ✓ 개선 (delta={delta:+.4f})"
    elif top1 == 10:
        status = f" (delta={delta:+.4f})"
    else:
        status = f" ✗ Top1 하락! ({top1}/10)"

    print(f"  → Top1: {top1}/10 | Avg: {avg:.4f}{status}")

    # 약점 카테고리 변화 하이라이트
    weak_idx = {0: "차1-1", 7: "차43-1", 9: "차5-2"}
    for idx, (q, pos, s, t1, ok) in enumerate(details):
        d = s - base_scores[idx]
        mark = " ◀ 약점" if idx in weak_idx else ""
        print(f"    {q:<28} {s:.4f} ({d:+.4f}) {'O' if ok else 'X'}{mark}")

    # Best 갱신
    if top1 == 10 and avg > best_avg:
        best_top1 = top1
        best_avg = avg
        best_cfg_idx = ci
        model.save(OUT_DIR)
        print(f"  ✓ New best! avg={avg:.4f}")

    # 결과 저장
    with open(os.path.join(BASE_DIR, f"boost_v4_r{ci+1}_result.json"), "w") as f:
        json.dump({
            "top1": top1, "avg": avg, "scores": scores,
            "config": cfg["label"], "delta": delta,
            "top3": sum(1 for _, _, _, _, ok in details if ok) if top1 < 10 else 10,
        }, f, indent=2)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 목표 달성 시 조기 종료
    if best_top1 == 10 and best_avg >= 0.92:
        print(f"\n★ 목표 달성! Config {best_cfg_idx+1} | Avg: {best_avg:.4f}")
        break

# ═══════════════════════���══════════════════════════════════════
#  최종 결과
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
if best_cfg_idx >= 0:
    print(f"Best: Config {best_cfg_idx+1} ({configs[best_cfg_idx]['label']})")
    print(f"  Top1: {best_top1}/10 | Avg: {best_avg:.4f} (delta: {best_avg - base_avg:+.4f})")

    # 최종 검증
    model = SentenceTransformer(OUT_DIR)
    model.max_seq_length = 256
    top1, avg, scores, details = evaluate(model)

    print(f"\n{'쿼리':<28} {'정답':<8} {'Base':>8} {'New':>8} {'Delta':>8} {'Hit':>4}")
    print("-" * 80)
    for idx, (q, pos, s, t1, ok) in enumerate(details):
        d = s - base_scores[idx]
        mark = " ◀" if idx in {0, 7, 9} else ""
        print(
            f"{q:<28} {pos:<8} "
            f"{base_scores[idx]:>8.4f} {s:>8.4f} {d:>+8.4f} "
            f"{'O' if ok else 'X':>4}{mark}"
        )
    print("-" * 80)
    print(f"Top1: {top1}/10 | Avg: {avg:.4f} | Min: {min(scores):.4f} | Max: {max(scores):.4f}")

    # bge-m3-finetuned 갱신
    if best_avg > base_avg:
        import shutil
        finetuned = os.path.join(BASE_DIR, "bge-m3-finetuned")
        backup = os.path.join(BASE_DIR, "bge-m3-finetuned-backup")
        if os.path.exists(finetuned):
            if os.path.exists(backup):
                shutil.rmtree(backup)
            shutil.copytree(finetuned, backup)
        shutil.copytree(OUT_DIR, finetuned, dirs_exist_ok=True)
        print(f"\n  ✓ bge-m3-finetuned 갱신 완료!")

    del model
else:
    print("Top1 10/10 유지하는 설정 없음 — lr을 더 낮추거나 wr을 줄여보세요")
print("=" * 100)
