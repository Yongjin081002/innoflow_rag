import json
import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# 시드 고정
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 1. 학습 데이터 불러오기
from training_data import training_pairs

# 2. chunks.json 불러오기 (정답 청크 내용 가져오기)
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_dict = {c["id"]: c["content"] for c in chunks}

# 3. InputExample 형식으로 변환
train_examples = []
for pair in training_pairs:
    query = pair["query"]
    positive_id = pair["positive"]
    positive_content = chunk_dict.get(positive_id, "")
    if positive_content:
        train_examples.append(InputExample(
            texts=[query, positive_content]
        ))

print(f"학습 데이터: {len(train_examples)}개")

# 4. 모델 로드
model = SentenceTransformer("BAAI/bge-m3")
model.max_seq_length = 256

# 5. DataLoader 및 Loss 설정
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=2
)
train_loss = losses.MultipleNegativesRankingLoss(model)

# 6. 튜닝 실행
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=20,
    output_path="./bge-m3-finetuned",
    show_progress_bar=True
)

print("튜닝 완료! 모델 저장: ./bge-m3-finetuned")