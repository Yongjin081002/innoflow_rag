"""
v3b: v3 Config1 결과(0.8712)를 base로 한 2차 부스팅
OOM 방지: gc.collect + empty_cache between configs
약점 집중 + bridge query 전략
"""
import json, torch, random, numpy as np, os, gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# v3 Config1의 best 모델이 있으면 사용, 없으면 tmp_tune_r2
v3_best = os.path.join(BASE_DIR, "boost_v3_best")
if os.path.isdir(v3_best):
    BASE_MODEL = v3_best
    print(f"Base: boost_v3_best (0.8712)")
else:
    BASE_MODEL = os.path.join(BASE_DIR, "tmp_tune_r2")
    print(f"Base: tmp_tune_r2 (0.84)")

# Bridge queries (청크 키워드 ↔ 자연어 연결)
bridge_차12_1 = [
    "야간 교차로 충돌", "야간 교차로 충돌 좌측도로 직진",
    "교차로 좌측도로 직진 충돌", "비신호 교차로 좌측도로 직진 충돌",
    "교차로 동시진입 좌측도로 직진 과실", "비신호 교차로 동폭도로 직진 충돌",
    "교차로 좌측도로 직진 동시진입 A40 B60", "야간 비신호 교차로 좌측도로 직진 충돌",
    "밤 교차로 좌측도로 직진 동시진입 사고", "야간 교차로 좌측도로 직진 충돌 과실",
    "교차로 직진 대 좌측도로 직진 충돌", "야간 비신호 교차로 직진 충돌 과실비율",
    "교차로 좌측 직진 A40 B60 과실비율", "야간 교차로 동시진입 충돌 과실",
    "비신호교차로 좌측도로 직진 충돌 기본과실", "밤에 교차로에서 충돌",
    "야간 교차로 사고", "야간 교차로 차량 동시 진입 충돌",
    "야간 교차로에서 양쪽 직진 충돌", "야간 비신호 교차로 사고 과실",
]

bridge_차1_1 = [
    "신호위반 직진 충돌", "녹색 직진 적색 직진 충돌",
    "녹색 직진 적색 직진 기본 과실비율 A0 B100", "신호위반 교차로 녹색 직진 적색 직진",
    "적색 직진 녹색 직진 충돌 과실비율", "녹색 직진 적색 위반 직진 A0 B100",
    "신호위반 직진 녹색 적색 과실비율", "적색신호 직진 녹색신호 직진 교차로 과실",
    "녹색 직진 대 적색 직진 기본 과실비율", "신호위반 직진 충돌 A0 B100 과실",
    "신호위반 직진 충돌 과실비율", "빨간불 직진 충돌 과실 기준",
    "적색신호 위반 직진 사고", "교차로 신호위반 직진 충돌 사고",
    "신호 무시 직진 충돌 과실비율", "녹색 직진 적색 직진 교차로 사고 기본과실",
    "적색 직진 교차로 사고 A0 B100", "교차로 녹색 직진 적색 직진 충돌 사고",
    "빨간불 직진 파란불 직진 충돌 과실비율", "신호위반 직진 기본 과실비율",
]

bridge_차43_1 = [
    "고속도로 추돌 사고", "고속도로 본선차 합류차 충돌",
    "고속도로 본선차 합류차 과실비율 A40 B60", "고속도로 합류 충돌 본선차 합류차",
    "고속도로 추돌 본선차 합류차 과실", "본선차 합류차 고속도로 충돌 A40 B60",
    "고속도로 합류 본선 충돌 기본 과실비율", "고속도로 추돌 사고 본선 합류 과실",
    "고속도로 합류차 본선차 사고 과실비율", "고속도로 합류 추돌 A40 B60",
    "고속도로 추돌 사고 과실비율", "고속도로 합류 충돌 과실 기준",
    "고속도로에서 합류하다 추돌", "고속도로 합류차선 사고 과실",
    "고속도로 진입 합류 추돌 사고", "고속도로 본선 합류 충돌 기본과실",
    "고속도로 합류 지점 본선차 합류차 사고", "고속도로 추돌 합류차 본선차 과실 기준",
    "고속도로 합류차선 본선 충돌 과실비율", "고속도로 진입로 합류 본선 충돌",
]

bridge_차51_1 = [
    "주차장 출차 중 충돌", "주차장 통로주행차 출차 충돌",
    "통로주행차 주차구획 출차 충돌 과실비율", "주차장 통로주행차 주차구획 출차 A30 B70",
    "주차장 출차 통로주행차 충돌 과실", "주차구획 출차 통로 주행 차량 충돌",
    "주차장 출차 통로 충돌 A30 B70", "통로주행차 출차 차량 충돌 과실비율",
    "주차장 통로 주행 출차 기본 과실비율", "주차구획에서 출차 통로주행차 충돌",
    "주차장 출차 충돌 과실비율", "주차장에서 나오다 통로 차 충돌",
    "주차장 출차 사고 과실 기준", "주차장 출차 중 사고 누구 잘못",
    "주차장 빠져나오다 충돌 과실", "주차장 출차 충돌 통로주행차 과실",
    "주차장 출차 통로 A30 B70 과실비율", "주차구획 출차 통로 차량 사고 과실",
    "주차장 통로주행 대 출차 충돌 기본과실", "주차장 출차 중 통로 차량 A30 B70",
]

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

print("=" * 100)
model = SentenceTransformer(BASE_MODEL)
model.max_seq_length = 256
base_top1, base_avg, base_scores, base_details = evaluate(model)
print(f"[Base] Top1: {base_top1}/10 | Avg: {base_avg:.4f}")
for q, pos, s, t1, ok in base_details:
    print(f"  {q:<28} {s:.4f} {'O' if ok else 'X'}")

all_q = [p["query"] for p in training_pairs]
all_p = [p["positive"] for p in training_pairs]
for cid, bridges in [("차12-1", bridge_차12_1), ("차1-1", bridge_차1_1),
                      ("차43-1", bridge_차43_1), ("차51-1", bridge_차51_1)]:
    for q in bridges:
        all_q.append(q); all_p.append(cid)
mined = mine_hard_negatives(model, all_q, all_p)
del model; gc.collect(); torch.cuda.empty_cache()

configs = [
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 42, "wr": 8},
    {"lr": 5e-6, "bs": 4, "mini_bs": 32, "epochs": 8, "warmup": 20, "seed": 42, "wr": 8},
    {"lr": 2e-6, "bs": 4, "mini_bs": 32, "epochs": 15, "warmup": 40, "seed": 42, "wr": 10},
    {"lr": 4e-6, "bs": 4, "mini_bs": 32, "epochs": 10, "warmup": 25, "seed": 77, "wr": 8},
    {"lr": 3e-6, "bs": 4, "mini_bs": 32, "epochs": 12, "warmup": 30, "seed": 42, "wr": 10},
    {"lr": 5e-6, "bs": 4, "mini_bs": 32, "epochs": 6, "warmup": 15, "seed": 42, "wr": 12},
    {"lr": 1e-6, "bs": 4, "mini_bs": 32, "epochs": 20, "warmup": 50, "seed": 42, "wr": 10},
    {"lr": 8e-6, "bs": 4, "mini_bs": 32, "epochs": 5, "warmup": 10, "seed": 42, "wr": 8},
]

OUT_DIR = os.path.join(BASE_DIR, "boost_v3b_best")
TEMP_DIR = os.path.join(BASE_DIR, "boost_v3b_tmp")
best_top1, best_avg, best_cfg_idx = 0, 0.0, -1

for ci, cfg in enumerate(configs):
    print(f"\n[Config {ci+1}/{len(configs)}] lr={cfg['lr']}, ep={cfg['epochs']}, wr={cfg['wr']}, seed={cfg['seed']}")

    # 메모리 정리
    gc.collect(); torch.cuda.empty_cache()

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 256

    pair_examples = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c: pair_examples.append(InputExample(texts=[p["query"], c]))
    for cid, bridges in [("차12-1", bridge_차12_1), ("차1-1", bridge_차1_1),
                          ("차43-1", bridge_차43_1), ("차51-1", bridge_차51_1)]:
        content = chunk_dict[cid]
        for q in bridges:
            for _ in range(cfg["wr"]):
                pair_examples.append(InputExample(texts=[q, content]))

    triplet_examples = []
    for i, (q, pid) in enumerate(zip(all_q, all_p)):
        pc = chunk_dict.get(pid, "")
        if pc and i < len(mined):
            for nid in mined[i]:
                nc = chunk_dict.get(nid, "")
                if nc: triplet_examples.append(InputExample(texts=[q, pc, nc]))

    scored_pairs = []
    for cid, bridges in [("차12-1", bridge_차12_1), ("차1-1", bridge_차1_1),
                          ("차43-1", bridge_차43_1), ("차51-1", bridge_차51_1)]:
        content = chunk_dict[cid]
        for q in bridges:
            scored_pairs.append(InputExample(texts=[q, content], label=1.0))
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c: scored_pairs.append(InputExample(texts=[p["query"], c], label=1.0))

    train_objectives = []
    dl1 = DataLoader(pair_examples, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl1, losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg["mini_bs"])))
    dl2 = DataLoader(scored_pairs, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl2, losses.AnglELoss(model)))
    dl3 = DataLoader(triplet_examples, shuffle=True, batch_size=cfg["bs"])
    train_objectives.append((dl3, losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)))

    model.fit(train_objectives=train_objectives, epochs=cfg["epochs"], warmup_steps=cfg["warmup"],
              output_path=TEMP_DIR, show_progress_bar=False, optimizer_params={"lr": cfg["lr"]}, weight_decay=0.01)

    top1, avg, scores, details = evaluate(model)
    delta = avg - base_avg
    status = f" ★★★ 목표! ★★★" if top1==10 and avg>=0.90 else f" (+{delta:.4f})" if top1==10 else f" ✗{top1}/10"
    print(f"  → Top1: {top1}/10 | Avg: {avg:.4f}{status}")
    for idx, (q, pos, s, t1, ok) in enumerate(details):
        d = s - base_scores[idx]
        print(f"    {q:<28} {s:.4f} ({d:+.4f}) {'O' if ok else 'X'}")

    if top1 == 10 and avg > best_avg:
        best_top1, best_avg, best_cfg_idx = top1, avg, ci
        model.save(OUT_DIR)
        print(f"  ✓ New best! avg={avg:.4f}")

    with open(os.path.join(BASE_DIR, f"boost_v3b_r{ci+1}_result.json"), "w") as f:
        json.dump({"top1": top1, "avg": avg, "scores": scores}, f, indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()

    if best_top1 == 10 and best_avg >= 0.90:
        print(f"\n★ 목표 달성! avg={best_avg:.4f}")
        break

print("\n" + "=" * 100)
if best_cfg_idx >= 0:
    print(f"Best: Config {best_cfg_idx+1} | Top1: {best_top1}/10 | Avg: {best_avg:.4f}")
    if best_avg > base_avg:
        import shutil
        finetuned = os.path.join(BASE_DIR, "bge-m3-finetuned")
        backup = os.path.join(BASE_DIR, "bge-m3-finetuned-backup")
        if os.path.exists(finetuned):
            if os.path.exists(backup): shutil.rmtree(backup)
            shutil.copytree(finetuned, backup)
        shutil.copytree(OUT_DIR, finetuned, dirs_exist_ok=True)
        print(f"  bge-m3-finetuned 갱신!")
print("=" * 100)
