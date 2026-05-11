"""
inno-flow 현재 모델 전체 스코어 요약
- 저장된 result JSON + 실시간 평가 결합
- 모델별 요약표 + 쿼리별 상세표 출력
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 테스트 케이스 (쿼리별 점수 매핑용) ──
QUERIES = [
    "신호위반 직진 충돌",
    "비신호교차로 직진 vs 좌회전",
    "추돌 사고 과실",
    "야간 교차로 충돌",
    "중앙선 침범 충돌",
    "끼어들기 충돌",
    "유턴 중 충돌",
    "고속도로 추돌 사고",
    "주차장 출차 중 충돌",
    "횡단보도 보행자 충돌",
]

POSITIVES = ["차1-1", "차15-1", "차41-1", "차12-1", "차31-1",
             "차20-2", "차33-1", "차43-1", "차51-1", "차5-2"]

# ── result JSON 수집 ──
result_files = [
    ("tune_r1", "tmp_tune_r1_result.json"),
    ("tune_r2", "tmp_tune_r2_result.json"),
    ("tune_r3", "tmp_tune_r3_result.json"),
    ("tune_r4", "tmp_tune_r4_result.json"),
    ("tune_r5", "tmp_tune_r5_result.json"),
    ("tune_r6", "tmp_tune_r6_result.json"),
    ("boost_v2_r1", "boost_v2_r1_result.json"),
    ("boost_v2_r2", "boost_v2_r2_result.json"),
    ("boost_v3_r1", "boost_v3_r1_result.json"),
    ("boost_v3b_r1", "boost_v3b_r1_result.json"),
    ("boost_v3b_r2", "boost_v3b_r2_result.json"),
    ("boost_09_r1", "boost_09_r1_result.json"),
    ("boost_09_r2", "boost_09_r2_result.json"),
    ("boost_weak_r1", "boost_weak_r1_result.json"),
]

results = []
for name, fname in result_files:
    fpath = os.path.join(BASE_DIR, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r") as f:
        data = json.load(f)
    top1 = data.get("top1", "?")
    avg = data.get("avg", 0)
    scores = data.get("scores", [])
    config = data.get("config", "")
    results.append({
        "name": name,
        "top1": top1,
        "avg": avg,
        "min": min(scores) if scores else 0,
        "max": max(scores) if scores else 0,
        "scores": scores,
        "config": config,
    })

# ── 정렬: avg score 내림차순 ──
results.sort(key=lambda r: r["avg"], reverse=True)

# ══════════════════════════════════════════════
# 1) 모델별 요약표
# ══════════════════════════════════════════════
print()
print("=" * 95)
print("  inno-flow 모델 스코어 요약 (result JSON 기준)")
print("=" * 95)
header = f"{'#':<4} {'모델':<18} {'Top1':>7} {'Avg':>9} {'Min':>9} {'Max':>9}  {'Config'}"
print(header)
print("-" * 95)
for i, r in enumerate(results, 1):
    print(
        f"{i:<4} {r['name']:<18} "
        f"{r['top1']:>5}/10 "
        f"{r['avg']:>9.4f} "
        f"{r['min']:>9.4f} "
        f"{r['max']:>9.4f}  "
        f"{r['config']}"
    )
print("-" * 95)

# ══════════════════════════════════════════════
# 2) Best 모델 쿼리별 상세
# ══════════════════════════════════════════════
best = results[0]
print()
print(f"  Best 모델: {best['name']}  (Top1={best['top1']}/10, Avg={best['avg']:.4f})")
print("=" * 80)
print(f"{'#':<4} {'쿼리':<28} {'정답청크':<10} {'Score':>9}")
print("-" * 80)
if best["scores"]:
    for i, (q, pos, s) in enumerate(zip(QUERIES, POSITIVES, best["scores"])):
        flag = " *" if s < 0.85 else ""
        print(f"{i+1:<4} {q:<28} {pos:<10} {s:>9.4f}{flag}")
print("-" * 80)
print("  (* = score < 0.85, 개선 필요)")

# ══════════════════════════════════════════════
# 3) 쿼리별 크로스 모델 히트맵 (Top-3 모델)
# ══════════════════════════════════════════════
top_n = min(5, len(results))
top_models = results[:top_n]

print()
print("=" * (30 + 14 * top_n))
print("  쿼리별 Score 비교 (상위 모델)")
print("=" * (30 + 14 * top_n))

# 헤더
hdr = f"{'쿼리':<28}"
for m in top_models:
    hdr += f"  {m['name']:>12}"
print(hdr)
print("-" * (30 + 14 * top_n))

for i, (q, pos) in enumerate(zip(QUERIES, POSITIVES)):
    row = f"{q:<28}"
    for m in top_models:
        if m["scores"] and i < len(m["scores"]):
            row += f"  {m['scores'][i]:>12.4f}"
        else:
            row += f"  {'N/A':>12}"
    print(row)

print("-" * (30 + 14 * top_n))
# 평균 행
avg_row = f"{'[평균]':<28}"
for m in top_models:
    avg_row += f"  {m['avg']:>12.4f}"
print(avg_row)
print()
