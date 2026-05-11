"""boost_best_vlm_v4 VLM 텍스트 쿼리 테스트 + 표 출력"""
import json, os
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 로드
with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

from compare_boost_v3 import GROUP_A, GROUP_B, GROUP_C

# 모델 로드
print("boost_best_vlm_v4 로딩 중...")
model = SentenceTransformer(os.path.join(BASE_DIR, "boost_best_vlm_v4"))
model.max_seq_length = 512
chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)

all_groups = [("A", GROUP_A), ("B", GROUP_B), ("C", GROUP_C)]
results = []

for gname, gdata in all_groups:
    for item in gdata:
        q_emb = model.encode([item["query"]], convert_to_tensor=True, show_progress_bar=False)
        sims = cos_sim(q_emb, chunk_emb)[0]
        sorted_idx = torch.argsort(sims, descending=True)
        top_ids = [chunk_ids[i] for i in sorted_idx[:5]]
        top_scores = [sims[i].item() for i in sorted_idx[:5]]

        exp = item["expected"]
        exp_idx = chunk_ids.index(exp)
        exp_score = sims[exp_idx].item()
        rank = (top_ids.index(exp) + 1) if exp in top_ids else 999

        results.append({
            "group": gname,
            "name": item["name"],
            "expected": exp,
            "top1": top_ids[0],
            "rank": rank,
            "score": exp_score,
            "top1_score": top_scores[0],
            "hit": rank == 1,
        })

# 결과 출력
print(f"\n{'='*90}")
print(f"  boost_best_vlm_v4 - VLM 텍스트 쿼리 테스트 결과")
print(f"{'='*90}\n")

print(f"  {'#':<4} {'그룹':<4} {'테스트명':<40} {'정답':<8} {'Top1':<8} {'Rank':>5} {'Score':>8} {'판정'}")
print(f"  {'-'*4} {'-'*4} {'-'*40} {'-'*8} {'-'*8} {'-'*5} {'-'*8} {'-'*4}")

for i, r in enumerate(results, 1):
    mark = "O" if r["hit"] else ("T3" if r["rank"] <= 3 else "X")
    print(f"  {i:<4} {r['group']:<4} {r['name']:<40} {r['expected']:<8} {r['top1']:<8} {r['rank']:>5} {r['score']:>8.4f} {mark}")

# 그룹별 요약
print(f"\n{'='*90}")
print(f"  그룹별 요약")
print(f"{'='*90}\n")

for gname in ["A", "B", "C"]:
    g_items = [r for r in results if r["group"] == gname]
    top1 = sum(1 for r in g_items if r["hit"])
    top3 = sum(1 for r in g_items if r["rank"] <= 3)
    avg = sum(r["score"] for r in g_items) / len(g_items)
    label = {"A": "학습 多 (6~30쌍)", "B": "학습 少 (1~3쌍)", "C": "극소/복합"}[gname]
    print(f"  그룹 {gname} ({label}): Top1={top1}/10, Top3={top3}/10, Avg Score={avg:.4f}")

# 전체 요약
total_top1 = sum(1 for r in results if r["hit"])
total_top3 = sum(1 for r in results if r["rank"] <= 3)
total_avg = sum(r["score"] for r in results) / len(results)
print(f"\n  전체: Top1={total_top1}/30, Top3={total_top3}/30, Avg Score={total_avg:.4f}")

# 오답 상세
wrong = [r for r in results if not r["hit"]]
print(f"\n{'='*90}")
print(f"  오답 상세 ({len(wrong)}건)")
print(f"{'='*90}\n")

for r in wrong:
    print(f"  [{r['group']}] {r['name']}: 정답={r['expected']}, Top1={r['top1']} (rank={r['rank']}, score={r['score']:.4f})")

del model
print("\n완료.")
