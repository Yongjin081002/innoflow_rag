import json, torch, random, numpy as np, os, sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from training_data import training_pairs

SEED = int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_dict = {c["id"]: c["content"] for c in chunks}
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

examples = []
for p in training_pairs:
    c = chunk_dict.get(p["positive"], "")
    if c:
        examples.append(InputExample(texts=[p["query"], c]))

model = SentenceTransformer("BAAI/bge-m3")
model.max_seq_length = 256
dl = DataLoader(examples, shuffle=True, batch_size=2)
loss_fn = losses.MultipleNegativesRankingLoss(model)
out_path = f"./tmp_model_{SEED}"
model.fit(train_objectives=[(dl, loss_fn)], epochs=5, warmup_steps=20,
          output_path=out_path, show_progress_bar=False)

chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
test_queries = ["신호위반 직진 충돌", "비신호교차로 직진 vs 좌회전", "추돌 사고 과실",
                "야간 교차로 충돌", "중앙선 침범 충돌", "끼어들기 충돌",
                "유턴 중 충돌", "고속도로 추돌 사고", "주차장 출차 중 충돌", "횡단보도 보행자 충돌"]
test_pos = ["차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
            "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"]
query_emb = model.encode(test_queries, convert_to_tensor=True, show_progress_bar=False)

scores = []
top1_ok = 0
details = []
for i in range(10):
    pos_idx = chunk_ids.index(test_pos[i])
    s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
    scores.append(s)
    top1_id = chunk_ids[cos_sim(query_emb[i], chunk_emb)[0].argmax().item()]
    ok = top1_id == test_pos[i]
    if ok:
        top1_ok += 1
    details.append(f"{test_pos[i]}:{s:.4f}:{'O' if ok else top1_id}")

avg = sum(scores) / len(scores)
print(f"RESULT:{top1_ok},{avg:.6f}")
for d in details:
    print(f"DETAIL:{d}")
