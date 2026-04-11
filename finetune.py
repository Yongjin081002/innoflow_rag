import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# 1. 학습 데이터 불러오기
training_pairs = [
    # training_data.py에서 만든 148쌍 전부 여기 붙여넣기
]

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

# 5. DataLoader 및 Loss 설정
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=8
)
train_loss = losses.MultipleNegativesRankingLoss(model)

# 6. 튜닝 실행
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    output_path="./bge-m3-finetuned",
    show_progress_bar=True
)

print("튜닝 완료! 모델 저장: ./bge-m3-finetuned")