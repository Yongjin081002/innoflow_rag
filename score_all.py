"""
inno-flow 전체 모델 스코어링 스크립트
- 원본 BAAI/bge-m3 + 파인튜닝 모델들 비교 평가
- Top-1 정확도, 평균 유사도, Top-3/Top-5 정확도 측정
"""
import json
import os
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ── 테스트 케이스 ──
test_cases = [
    {"query": "신호위반 직진 충돌",           "positive": "차1-1"},
    {"query": "비신호교차로 직진 vs 좌회전",  "positive": "차15-1"},
    {"query": "추돌 사고 과실",               "positive": "차41-1"},
    {"query": "야간 교차로 충돌",             "positive": "차12-1"},
    {"query": "중앙선 침범 충돌",             "positive": "차31-1"},
    {"query": "끼어들기 충돌",                "positive": "차20-2"},
    {"query": "유턴 중 충돌",                 "positive": "차33-1"},
    {"query": "고속도로 추돌 사고",           "positive": "차43-1"},
    {"query": "주차장 출차 중 충돌",          "positive": "차51-1"},
    {"query": "횡단보도 보행자 충돌",         "positive": "차5-2"},
]

# ── 데이터 로드 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]
queries = [tc["query"] for tc in test_cases]

# ── 평가할 모델 목록 ──
models_to_eval = [
    ("BAAI/bge-m3 (원본)", "BAAI/bge-m3"),
    ("bge-m3-finetuned", os.path.join(BASE_DIR, "bge-m3-finetuned")),
    ("tmp_model_42", os.path.join(BASE_DIR, "tmp_model_42")),
    ("tmp_tune_r1", os.path.join(BASE_DIR, "tmp_tune_r1")),
    ("tmp_tune_r2", os.path.join(BASE_DIR, "tmp_tune_r2")),
    ("tmp_tune_r3", os.path.join(BASE_DIR, "tmp_tune_r3")),
    ("tmp_tune_r4", os.path.join(BASE_DIR, "tmp_tune_r4")),
    ("tmp_tune_r5", os.path.join(BASE_DIR, "tmp_tune_r5")),
    ("tmp_tune_r6", os.path.join(BASE_DIR, "tmp_tune_r6")),
    ("tmp_tune_best", os.path.join(BASE_DIR, "tmp_tune_best")),
    ("boost_tmp", os.path.join(BASE_DIR, "boost_tmp")),
    ("boost_best", os.path.join(BASE_DIR, "boost_best")),
]

# 존재하지 않는 체크포인트 제외 (원본 제외)
models_to_eval = [
    (name, path) for name, path in models_to_eval
    if path == "BAAI/bge-m3" or os.path.isdir(path)
]


def evaluate_model(model_name, model_path):
    """모델 로드 → 임베딩 → Top-K 정확도 + 평균 유사도 계산"""
    print(f"  평가 중: {model_name} ...", flush=True)

    # GPU 메모리 정리 후 로드, OOM 시 CPU fallback
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        print(f"    GPU OOM → CPU로 전환", flush=True)
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 256

    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)

    # 모델 즉시 해제
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    scores = []
    per_query = []

    for i, tc in enumerate(test_cases):
        pos_id = tc["positive"]
        pos_idx = chunk_ids.index(pos_id)

        sim = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(sim)

        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        sorted_indices = all_sim.argsort(descending=True)
        top_ids = [chunk_ids[idx] for idx in sorted_indices[:5]]

        rank = top_ids.index(pos_id) + 1 if pos_id in top_ids else 999

        if rank == 1:
            top1_correct += 1
        if rank <= 3:
            top3_correct += 1
        if rank <= 5:
            top5_correct += 1

        per_query.append({
            "query": tc["query"],
            "positive": pos_id,
            "score": sim,
            "rank": rank,
            "top1": top_ids[0],
        })

    n = len(test_cases)
    return {
        "model": model_name,
        "top1_acc": top1_correct / n,
        "top1_count": f"{top1_correct}/{n}",
        "top3_acc": top3_correct / n,
        "top3_count": f"{top3_correct}/{n}",
        "top5_acc": top5_correct / n,
        "top5_count": f"{top5_correct}/{n}",
        "avg_score": sum(scores) / n,
        "min_score": min(scores),
        "max_score": max(scores),
        "per_query": per_query,
    }


# ── 전체 평가 실행 ──
print("=" * 60)
print("  inno-flow 모델 스코어링")
print("=" * 60)

results = []
for name, path in models_to_eval:
    res = evaluate_model(name, path)
    results.append(res)

# ── 요약 표 출력 ──
print("\n")
print("=" * 110)
print(f"{'모델':<28} {'Top1':>8} {'Top3':>8} {'Top5':>8} {'Avg Score':>10} {'Min':>8} {'Max':>8}")
print("=" * 110)
for r in results:
    print(
        f"{r['model']:<28} "
        f"{r['top1_count']:>8} "
        f"{r['top3_count']:>8} "
        f"{r['top5_count']:>8} "
        f"{r['avg_score']:>10.4f} "
        f"{r['min_score']:>8.4f} "
        f"{r['max_score']:>8.4f}"
    )
print("=" * 110)

# ── 쿼리별 상세 (Best 모델 기준) ──
best = max(results, key=lambda r: (r["top1_acc"], r["avg_score"]))
print(f"\n** Best 모델: {best['model']} **")
print(f"{'쿼리':<28} {'정답청크':<10} {'Top1 결과':<10} {'Rank':>5} {'Score':>8}")
print("-" * 70)
for pq in best["per_query"]:
    match = "O" if pq["rank"] == 1 else "X"
    print(
        f"{pq['query']:<28} "
        f"{pq['positive']:<10} "
        f"{pq['top1']:<10} "
        f"{pq['rank']:>5} "
        f"{pq['score']:>8.4f} "
        f"  {match}"
    )
print("-" * 70)
print(f"평균 Score: {best['avg_score']:.4f}  |  Top1: {best['top1_count']}  |  Top3: {best['top3_count']}  |  Top5: {best['top5_count']}")
