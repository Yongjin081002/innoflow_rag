import json
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 쿼리와 정답 청크 매핑
test_cases = [
    {"query": "신호위반 직진 충돌", "positive": "차1-1"},
    {"query": "비신호교차로 직진 vs 좌회전", "positive": "차15-1"},
    {"query": "추돌 사고 과실", "positive": "차41-1"},
    {"query": "야간 교차로 충돌", "positive": "차12-1"},
    {"query": "중앙선 침범 충돌", "positive": "차31-1"},
    {"query": "끼어들기 충돌", "positive": "차20-2"},
    {"query": "유턴 중 충돌", "positive": "차33-1"},
    {"query": "고속도로 추돌 사고", "positive": "차43-1"},
    {"query": "주차장 출차 중 충돌", "positive": "차51-1"},
    {"query": "횡단보도 보행자 충돌", "positive": "차5-2"},
]

# chunks.json 로드
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

# 모델 로드
print("원본 모델 로딩...")
model_orig = SentenceTransformer("BAAI/bge-m3")
model_orig.max_seq_length = 256

print("튜닝 모델 로딩...")
model_ft = SentenceTransformer("./bge-m3-finetuned")
model_ft.max_seq_length = 256

# 청크 임베딩 생성
print("청크 임베딩 생성 중...")
chunk_emb_orig = model_orig.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
chunk_emb_ft = model_ft.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)

queries = [tc["query"] for tc in test_cases]

# 쿼리 임베딩 생성
query_emb_orig = model_orig.encode(queries, convert_to_tensor=True, show_progress_bar=False)
query_emb_ft = model_ft.encode(queries, convert_to_tensor=True, show_progress_bar=False)

# 결과 출력
print("\n" + "=" * 110)
print(f"{'쿼리':<25} {'정답청크':<10} {'원본 score':>10} {'튜닝 score':>10} {'변화':>8} {'원본 Top1':<10} {'튜닝 Top1':<10} {'Top1 일치':>8}")
print("=" * 110)

orig_scores = []
ft_scores = []

for i, tc in enumerate(test_cases):
    pos_id = tc["positive"]
    pos_idx = chunk_ids.index(pos_id)

    # 정답 청크와의 유사도
    sim_orig = cos_sim(query_emb_orig[i], chunk_emb_orig[pos_idx]).item()
    sim_ft = cos_sim(query_emb_ft[i], chunk_emb_ft[pos_idx]).item()

    # Top1 검색
    all_sim_orig = cos_sim(query_emb_orig[i], chunk_emb_orig)[0]
    all_sim_ft = cos_sim(query_emb_ft[i], chunk_emb_ft)[0]

    top1_orig_idx = all_sim_orig.argmax().item()
    top1_ft_idx = all_sim_ft.argmax().item()

    top1_orig_id = chunk_ids[top1_orig_idx]
    top1_ft_id = chunk_ids[top1_ft_idx]

    diff = sim_ft - sim_orig
    sign = "+" if diff > 0 else ""

    top1_match_orig = "O" if top1_orig_id == pos_id else "X"
    top1_match_ft = "O" if top1_ft_id == pos_id else "X"

    orig_scores.append(sim_orig)
    ft_scores.append(sim_ft)

    print(f"{tc['query']:<25} {pos_id:<10} {sim_orig:>10.4f} {sim_ft:>10.4f} {sign}{diff:>7.4f} {top1_orig_id:<10} {top1_ft_id:<10} {top1_match_orig}→{top1_match_ft}")

print("=" * 110)
avg_orig = sum(orig_scores) / len(orig_scores)
avg_ft = sum(ft_scores) / len(ft_scores)
avg_diff = avg_ft - avg_orig
sign = "+" if avg_diff > 0 else ""
print(f"{'평균':<25} {'':<10} {avg_orig:>10.4f} {avg_ft:>10.4f} {sign}{avg_diff:>7.4f}")
