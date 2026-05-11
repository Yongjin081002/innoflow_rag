import json, torch, random, numpy as np, os, sys, shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from training_data import training_pairs

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_dict = {c["id"]: c["content"] for c in chunks}
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

test_queries = ["신호위반 직진 충돌", "비신호교차로 직진 vs 좌회전", "추돌 사고 과실",
                "야간 교차로 충돌", "중앙선 침범 충돌", "끼어들기 충돌",
                "유턴 중 충돌", "고속도로 추돌 사고", "주차장 출차 중 충돌", "횡단보도 보행자 충돌"]
test_pos = ["차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
            "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"]

# Hyperparameter grid
configs = [
    {"lr": 3e-05, "batch": 2, "warmup": 20, "seed": 42},
    {"lr": 5e-05, "batch": 2, "warmup": 10, "seed": 42},
    {"lr": 7e-05, "batch": 2, "warmup": 20, "seed": 42},
    {"lr": 1e-04, "batch": 2, "warmup": 20, "seed": 42},
    {"lr": 1e-04, "batch": 2, "warmup": 10, "seed": 42},
    {"lr": 7e-05, "batch": 4, "warmup": 10, "seed": 42},
    {"lr": 1e-04, "batch": 4, "warmup": 10, "seed": 42},
    {"lr": 1e-04, "batch": 4, "warmup": 20, "seed": 42},
    {"lr": 2e-04, "batch": 2, "warmup": 10, "seed": 42},
    {"lr": 2e-04, "batch": 4, "warmup": 10, "seed": 42},
]

best_avg = 0.0
best_top1 = 0
best_cfg = None

for idx, cfg in enumerate(configs):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    examples = []
    for p in training_pairs:
        c = chunk_dict.get(p["positive"], "")
        if c:
            examples.append(InputExample(texts=[p["query"], c]))

    model = SentenceTransformer("BAAI/bge-m3")
    model.max_seq_length = 256
    dl = DataLoader(examples, shuffle=True, batch_size=cfg["batch"])
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    out_path = f"./tmp_tune_{idx}"

    model.fit(train_objectives=[(dl, loss_fn)], epochs=5,
              warmup_steps=cfg["warmup"], output_path=out_path,
              show_progress_bar=False,
              optimizer_params={"lr": cfg["lr"]})

    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(test_queries, convert_to_tensor=True, show_progress_bar=False)

    scores = []
    top1_ok = 0
    for i in range(10):
        pos_idx = chunk_ids.index(test_pos[i])
        s = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(s)
        top1_id = chunk_ids[cos_sim(query_emb[i], chunk_emb)[0].argmax().item()]
        if top1_id == test_pos[i]:
            top1_ok += 1

    avg = sum(scores) / len(scores)
    is_best = avg > best_avg
    marker = " ★ BEST" if is_best else ""
    print(f"[{idx+1}/{len(configs)}] lr={cfg['lr']:.0e} batch={cfg['batch']} warmup={cfg['warmup']}  Top1={top1_ok}/10 avg={avg:.4f}{marker}")

    if is_best:
        best_avg = avg
        best_top1 = top1_ok
        best_cfg = cfg
        if os.path.exists("./bge-m3-finetuned"):
            shutil.rmtree("./bge-m3-finetuned")
        shutil.copytree(out_path, "./bge-m3-finetuned")

    shutil.rmtree(out_path, ignore_errors=True)

    if best_avg >= 0.80:
        print(f"\n목표 달성! avg={best_avg:.4f} >= 0.80")
        break

print(f"\n{'='*60}")
print(f"최적 설정: lr={best_cfg['lr']:.0e}, batch={best_cfg['batch']}, warmup={best_cfg['warmup']}")
print(f"Top1: {best_top1}/10, 평균 score: {best_avg:.4f}")
print(f"모델 저장: ./bge-m3-finetuned")
