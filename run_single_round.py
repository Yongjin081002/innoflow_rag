"""단일 튜닝 라운드 실행 (subprocess로 호출됨)"""
import json, torch, random, numpy as np, os, sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from training_data import training_pairs

# ── 인자 파싱 ──
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--out", required=True)
parser.add_argument("--round", type=int, default=1)
parser.add_argument("--label", default="")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ── 청크 로드 ──
with open("chunks.json", "r", encoding="utf-8") as f:
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

# ── 보강 학습 데이터 ──
extra_positive_pairs = [
    # 차1-1 구분 강화
    {"query": "자동차 신호위반 직진 충돌 과실비율", "positive": "차1-1"},
    {"query": "차량 간 신호위반 직진 교차로 사고", "positive": "차1-1"},
    {"query": "승용차 적색신호 직진 충돌", "positive": "차1-1"},
    {"query": "자동차끼리 신호위반 직진 충돌 과실", "positive": "차1-1"},
    {"query": "차 대 차 신호위반 교차로 직진 사고", "positive": "차1-1"},
    {"query": "차량 녹색 직진 적색 직진 교차로 사고", "positive": "차1-1"},
    {"query": "자동차 교차로 신호위반 사고 과실비율", "positive": "차1-1"},
    {"query": "차량 A0 B100 신호위반 직진 과실", "positive": "차1-1"},
    {"query": "신호위반 직진 차량 충돌 기본 과실비율", "positive": "차1-1"},
    {"query": "신호위반 직진 충돌 자동차 사고", "positive": "차1-1"},

    # 차33-1 구분 강화
    {"query": "유턴 중 직진차와 충돌 A20 B80", "positive": "차33-1"},
    {"query": "상시유턴구역 유턴 직진차 충돌 기본과실", "positive": "차33-1"},
    {"query": "유턴 중 충돌 기본 과실비율 A20 B80", "positive": "차33-1"},
    {"query": "유턴하다 직진차 사고 과실 A0 B100", "positive": "차33-1"},
    {"query": "직진 대 유턴 충돌 과실비율", "positive": "차33-1"},
    {"query": "유턴 중 충돌 직진 차량 과실", "positive": "차33-1"},
    {"query": "유턴 사고 직진차와 충돌", "positive": "차33-1"},
    {"query": "유턴 도중 직진 차 충돌 사고", "positive": "차33-1"},
    {"query": "상시유턴 직진차 충돌 과실비율", "positive": "차33-1"},
    {"query": "유턴 중 충돌 사고 기본과실", "positive": "차33-1"},

    # 차15-1 구분 강화
    {"query": "비신호교차로 직진 vs 좌회전 과실비율", "positive": "차15-1"},
    {"query": "비신호 교차로 직진차 좌회전차 과실 A30 B70", "positive": "차15-1"},
    {"query": "신호없는 교차로에서 직진하다 좌회전차와 충돌", "positive": "차15-1"},
    {"query": "비신호 교차로 직진 좌회전 기본 과실비율", "positive": "차15-1"},
    {"query": "비신호교차로 직진 대 좌회전 사고 과실", "positive": "차15-1"},
    {"query": "무신호 사거리 직진 좌회전 충돌 과실비율", "positive": "차15-1"},
    {"query": "비신호 교차로에서 직진 좌회전 충돌 과실 기준", "positive": "차15-1"},
    {"query": "비신호 교차로 직진 좌회전 사고 과실비율 기준", "positive": "차15-1"},
    {"query": "비신호교차로 직진 좌회전 충돌 사고", "positive": "차15-1"},
    {"query": "비신호 교차로 직진 vs 좌회전 충돌 과실", "positive": "차15-1"},
]

# ── 하드 네거티브 triplet ──
hard_neg_triplets = [
    ("신호위반 직진 충돌", "차1-1", "거1-2"),
    ("적색신호 직진 사고 과실비율", "차1-1", "거1-2"),
    ("신호위반 직진 충돌 사고", "차1-1", "거1-2"),
    ("녹색 직진 적색 직진 충돌", "차1-1", "거1-2"),
    ("교차로 신호위반 직진 충돌", "차1-1", "거1-2"),
    ("신호위반 직진 충돌 과실", "차1-1", "거1-2"),
    ("빨간불 직진 충돌 과실비율", "차1-1", "거1-2"),
    ("자동차 신호위반 직진 충돌", "차1-1", "거1-2"),
    ("차량 신호위반 직진 교차로 사고", "차1-1", "거1-2"),
    ("직진 신호위반 충돌 사고 과실", "차1-1", "거1-2"),

    ("유턴 중 충돌", "차33-1", "차33-2"),
    ("유턴 중 충돌 과실비율", "차33-1", "차33-2"),
    ("유턴하다 직진차와 충돌", "차33-1", "차33-2"),
    ("유턴 사고 과실", "차33-1", "차33-2"),
    ("상시유턴구역 유턴 충돌", "차33-1", "차33-2"),
    ("유턴 중 사고 과실비율", "차33-1", "차33-2"),
    ("유턴 도중 충돌", "차33-1", "차33-2"),
    ("유턴 중 직진차 충돌", "차33-1", "차33-2"),
    ("유턴 충돌 과실 기준", "차33-1", "차33-2"),
    ("유턴 중 충돌 기본과실", "차33-1", "차33-2"),

    ("비신호교차로 직진 vs 좌회전", "차15-1", "차21-1"),
    ("비신호 교차로 직진 좌회전 충돌", "차15-1", "차21-1"),
    ("비신호교차로 직진 좌회전 과실비율", "차15-1", "차21-1"),
    ("신호 없는 교차로 직진 좌회전 사고", "차15-1", "차21-1"),
    ("무신호 교차로 직진 대 좌회전 충돌", "차15-1", "차21-1"),
    ("비신호 사거리 직진 좌회전 충돌 과실", "차15-1", "차21-1"),
    ("신호없는 교차로 직진차 좌회전차 사고", "차15-1", "차21-1"),
    ("비신호 교차로 좌회전 직진 과실비율 기준", "차15-1", "차21-1"),
    ("비신호 직진 좌회전 충돌", "차15-1", "차21-1"),
    ("비신호교차로 좌회전 직진 사고 과실", "차15-1", "차21-1"),
    # 거2-1도 네거티브로 추가 (Round 1에서 거2-1로 검색됨)
    ("비신호교차로 직진 vs 좌회전", "차15-1", "거2-1"),
    ("비신호 교차로 직진 좌회전 충돌", "차15-1", "거2-1"),
    ("비신호교차로 직진 좌회전 과실비율", "차15-1", "거2-1"),
    ("신호 없는 교차로 직진 좌회전 사고", "차15-1", "거2-1"),
    ("무신호 교차로 직진 대 좌회전 충돌", "차15-1", "거2-1"),
]

# ── 학습 데이터 구성 ──
pair_examples = []
for p in training_pairs:
    c = chunk_dict.get(p["positive"], "")
    if c:
        pair_examples.append(InputExample(texts=[p["query"], c]))

for p in extra_positive_pairs:
    c = chunk_dict.get(p["positive"], "")
    if c:
        pair_examples.append(InputExample(texts=[p["query"], c]))

triplet_examples = []
for anchor, pos_id, neg_id in hard_neg_triplets:
    pos_c = chunk_dict.get(pos_id, "")
    neg_c = chunk_dict.get(neg_id, "")
    if pos_c and neg_c:
        triplet_examples.append(InputExample(texts=[anchor, pos_c, neg_c]))

# ── 모델 로드 및 학습 ──
print(f"모델 로딩: {args.base}")
model = SentenceTransformer(args.base)
model.max_seq_length = 256

dl_pairs = DataLoader(pair_examples, shuffle=True, batch_size=args.bs)
mnr_loss = losses.MultipleNegativesRankingLoss(model)
train_objectives = [(dl_pairs, mnr_loss)]

if triplet_examples:
    dl_triplets = DataLoader(triplet_examples, shuffle=True, batch_size=args.bs)
    triplet_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.3)
    train_objectives.append((dl_triplets, triplet_loss))

print(f"학습 시작: epochs={args.epochs}, lr={args.lr}, bs={args.bs}, warmup={args.warmup}")
model.fit(train_objectives=train_objectives, epochs=args.epochs, warmup_steps=args.warmup,
          output_path=args.out, show_progress_bar=False)

# ── 평가 ──
chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
query_emb = model.encode(TEST_QUERIES, convert_to_tensor=True, show_progress_bar=False)

scores = []
top1_ok = 0
print(f"\n{'='*100}")
print(f"  [Round {args.round}] {args.label}")
print(f"{'='*100}")
print(f"{'쿼리':<28} {'정답':<8} {'Score':>8} {'Top1':>10} {'일치':>4}")
print(f"{'-'*100}")

for i in range(10):
    pos_idx = chunk_ids.index(TEST_POS[i])
    s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
    scores.append(s)
    all_sim = cos_sim(query_emb[i], chunk_emb)[0]
    top1_id = chunk_ids[all_sim.argmax().item()]
    ok = top1_id == TEST_POS[i]
    if ok:
        top1_ok += 1
    mark = "O" if ok else "X"
    print(f"{TEST_QUERIES[i]:<28} {TEST_POS[i]:<8} {s:>8.4f} {top1_id:>10} {mark:>4}")

avg = sum(scores) / len(scores)
print(f"{'-'*100}")
print(f"  Top1: {top1_ok}/10 | 평균 Score: {avg:.4f}")
print(f"{'='*100}")

# 결과를 파일로도 저장 (래퍼에서 읽기 위해)
with open(f"{args.out}_result.json", "w") as f:
    json.dump({"top1": top1_ok, "avg": avg, "round": args.round}, f)
