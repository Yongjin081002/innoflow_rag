"""
boost_best vs boost_v3b_best 비교 테스트
- VLM 스타일 쿼리 (블랙박스 영상 분석 출력 시뮬레이션)
- 키워드 형식 쿼리 (짧은 검색어)
- 두 모델의 성능을 표로 정리

Usage:
  python compare_vlm_keyword.py
"""
import json
import os
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 데이터 로드 ──
with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

# ═══════════════════════════════════════════════════
# 테스트 케이스: 동일 시나리오에 대한 두 가지 쿼리 형식
# ═══════════════════════════════════════════════════

TEST_PAIRS = [
    {
        "id": 1,
        "scenario": "녹색 직진 vs 적색 직진",
        "expected": "차1-1",
        "keyword": "신호위반 직진 충돌",
        "vlm": (
            "A 차량이 녹색 신호에 따라 정상적으로 교차로를 직진하던 중, "
            "B 차량이 적색 신호를 무시하고 우측에서 교차로에 진입하여 "
            "A 차량의 조수석 측면을 충격하였습니다."
        ),
    },
    {
        "id": 2,
        "scenario": "비보호 좌회전 vs 직진",
        "expected": "차2-6",
        "keyword": "비보호 좌회전 맞은편 직진 충돌",
        "vlm": (
            "A 차량이 비보호 좌회전을 시도하면서 맞은편에서 직진하는 "
            "B 트럭을 미처 확인하지 못하고 좌회전을 개시하여, "
            "B 트럭이 A 차량의 운전석 측면을 충격하였습니다."
        ),
    },
    {
        "id": 3,
        "scenario": "비신호 교차로 직진 vs 좌회전",
        "expected": "차15-1",
        "keyword": "비신호교차로 직진 vs 좌회전",
        "vlm": (
            "신호가 없는 교차로에서 B 차량이 골목에서 좌회전하여 "
            "주 도로에 진입하면서, 주 도로를 직진하던 A 차량과 충돌하였습니다."
        ),
    },
    {
        "id": 4,
        "scenario": "추돌 사고",
        "expected": "차41-1",
        "keyword": "추돌 사고 과실",
        "vlm": (
            "선행하던 B 트럭이 전방 교통 정체로 정차하였으나, "
            "후방의 A 차량이 전방 주시를 태만히 하여 B 트럭의 후미를 "
            "추돌하였습니다. A 차량의 제동 흔적은 충돌 직전 약 5m 정도만 확인됩니다."
        ),
    },
    {
        "id": 5,
        "scenario": "야간 비신호 교차로 동시진입",
        "expected": "차12-1",
        "keyword": "야간 교차로 충돌",
        "vlm": (
            "야간에 신호기가 설치되지 않은 교차로에서 양쪽 도로에서 동시에 "
            "진입한 두 차량이 교차로 중앙에서 측면 충돌하였습니다. "
            "양쪽 모두 서행하지 않은 것으로 보입니다."
        ),
    },
    {
        "id": 6,
        "scenario": "중앙선 침범 정면충돌",
        "expected": "차31-1",
        "keyword": "중앙선 침범 충돌",
        "vlm": (
            "비가 오는 커브 구간에서 B 화물차가 중앙선을 침범하여 "
            "반대 차로의 A 차량과 정면으로 충돌하였습니다. "
            "A 차량 운전자가 우측으로 회피를 시도하였으나 미처 피하지 못하였습니다."
        ),
    },
    {
        "id": 7,
        "scenario": "끼어들기 충돌",
        "expected": "차20-2",
        "keyword": "끼어들기 충돌",
        "vlm": (
            "A 차량이 충분한 안전거리를 확보하지 않은 채 1차로로 끼어들면서 "
            "B 차량의 우측 전면부와 A 차량의 좌측 후면부가 접촉하였습니다."
        ),
    },
    {
        "id": 8,
        "scenario": "유턴 vs 직진 충돌",
        "expected": "차33-1",
        "keyword": "유턴 중 충돌",
        "vlm": (
            "A 차량이 유턴을 하면서 반대편 차로로 진입하던 중, "
            "반대편에서 직진하던 B 차량과 충돌하였습니다. "
            "A 차량이 유턴 완료 전에 B 차량의 진행 경로를 차단한 형태입니다."
        ),
    },
    {
        "id": 9,
        "scenario": "고속도로 합류 충돌",
        "expected": "차43-1",
        "keyword": "고속도로 추돌 사고",
        "vlm": (
            "B 차량이 가속차로에서 본선으로 합류하면서 3차로의 A 차량과 "
            "나란히 진행하게 되었고, 가속차로가 끝나는 지점에서 "
            "B 차량이 A 차량의 옆으로 들어오면서 측면 접촉 사고가 발생하였습니다."
        ),
    },
    {
        "id": 10,
        "scenario": "주차장 출차 충돌",
        "expected": "차51-1",
        "keyword": "주차장 출차 중 충돌",
        "vlm": (
            "A 차량이 주차 공간에서 후진으로 출차하는 과정에서 "
            "통로를 지나가던 B 차량의 측면과 충돌하였습니다. "
            "A 차량 운전자가 후방 및 좌우 확인을 충분히 하지 않은 것으로 보입니다."
        ),
    },
    {
        "id": 11,
        "scenario": "횡단보도 보행자 충돌",
        "expected": "차5-2",
        "keyword": "횡단보도 보행자 충돌",
        "vlm": (
            "A 승합차가 녹색 신호에 우회전을 하면서 횡단보도를 건너고 있던 "
            "노인 보행자 B를 미처 확인하지 못하고 충돌하였습니다. "
            "보행자 신호는 녹색이었습니다."
        ),
    },
    {
        "id": 12,
        "scenario": "황색 vs 적색 직진 충돌",
        "expected": "차1-3",
        "keyword": "황색신호 적색신호 직진 충돌",
        "vlm": (
            "황색신호에 교차로에 진입한 A 차량과 적색신호에 교차로에 진입한 "
            "B 차량이 교차로 중앙에서 충돌하였습니다."
        ),
    },
    {
        "id": 13,
        "scenario": "적색 vs 적색 직진 충돌",
        "expected": "차1-4",
        "keyword": "양쪽 적색신호 위반 직진 충돌",
        "vlm": (
            "양쪽 차량 모두 적색신호를 위반하여 직진하다 "
            "교차로 중앙에서 충돌하였습니다."
        ),
    },
    {
        "id": 14,
        "scenario": "녹색 직진 vs 신호위반 좌회전",
        "expected": "차2-2",
        "keyword": "녹색 직진 신호위반 좌회전 충돌",
        "vlm": (
            "A 차량이 녹색 신호에 정상 직진하던 중, "
            "B 차량이 적색 신호임에도 좌회전을 시도하여 "
            "교차로에서 A 차량과 충돌하였습니다."
        ),
    },
    {
        "id": 15,
        "scenario": "차도 보행자 충돌",
        "expected": "보27-1",
        "keyword": "보행자 차도 보행 차량 충돌",
        "vlm": (
            "차도 위를 걸어가던 보행자 A를 차도를 주행하던 "
            "B 차량이 충돌하였습니다. 보행자가 보도가 아닌 "
            "차도 위를 보행하고 있었습니다."
        ),
    },
    {
        "id": 16,
        "scenario": "우회전 vs 직진 충돌",
        "expected": "차3-1",
        "keyword": "우회전 직진 교차로 충돌",
        "vlm": (
            "B 차량이 적색 신호에서 우회전을 하면서 교차로에 진입하였고, "
            "좌측에서 녹색 신호에 직진하던 A 차량과 교차로 내에서 충돌하였습니다."
        ),
    },
    {
        "id": 17,
        "scenario": "차선변경 직진 충돌",
        "expected": "차20-1",
        "keyword": "차선변경 사고 과실",
        "vlm": (
            "A 차량이 2차로에서 1차로로 차선을 변경하는 과정에서 "
            "1차로 후방에서 직진하던 B 차량과 접촉하였습니다. "
            "A 차량이 사이드미러 확인 없이 급하게 차선을 변경한 것으로 보입니다."
        ),
    },
    {
        "id": 18,
        "scenario": "역주행 정면충돌",
        "expected": "차31-2",
        "keyword": "역주행 일방통행로 정면충돌",
        "vlm": (
            "A 차량이 일방통행 도로를 역방향으로 주행하면서 "
            "정상 방향으로 진행하던 B 차량과 정면으로 충돌하였습니다."
        ),
    },
    {
        "id": 19,
        "scenario": "연쇄추돌 (3중)",
        "expected": "차42-1",
        "keyword": "연쇄 추돌 3중 사고",
        "vlm": (
            "교통 정체로 A, B 차량이 순서대로 정차해 있던 상황에서, "
            "후방의 C 트럭이 감속하지 못하고 B 차량 후미를 추돌하였고, "
            "그 충격으로 B 차량이 A 차량 후미를 재추돌하는 연쇄 추돌 사고가 "
            "발생하였습니다."
        ),
    },
    {
        "id": 20,
        "scenario": "점멸신호 교차로 충돌",
        "expected": "차1-5",
        "keyword": "점멸신호 교차로 충돌",
        "vlm": (
            "심야 시간 점멸 신호가 작동하는 교차로에서 적색 점멸 도로의 "
            "A 차량이 일시정지를 하지 않고 교차로에 진입하여, "
            "황색 점멸 도로에서 직진하던 B 차량과 충돌하였습니다."
        ),
    },
]


def load_model(model_path):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 256
    return model


def evaluate(model, queries, expected_ids):
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    q_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)

    results = []
    for i, exp_id in enumerate(expected_ids):
        sims = cos_sim(q_emb[i], chunk_emb)[0]
        sorted_idx = torch.argsort(sims, descending=True)
        top_ids = [chunk_ids[j] for j in sorted_idx[:5]]

        exp_idx = chunk_ids.index(exp_id)
        score = sims[exp_idx].item()
        rank = (top_ids.index(exp_id) + 1) if exp_id in top_ids else 999

        results.append({
            "expected": exp_id,
            "top1": top_ids[0],
            "rank": rank,
            "score": score,
            "hit": rank == 1,
        })
    return results


def main():
    models_info = [
        ("boost_best", os.path.join(BASE_DIR, "boost_best")),
        ("boost_v3b_best", os.path.join(BASE_DIR, "boost_v3b_best")),
    ]

    keyword_queries = [t["keyword"] for t in TEST_PAIRS]
    vlm_queries = [t["vlm"] for t in TEST_PAIRS]
    expected_ids = [t["expected"] for t in TEST_PAIRS]
    scenarios = [t["scenario"] for t in TEST_PAIRS]
    N = len(TEST_PAIRS)

    all_results = {}

    for model_name, model_path in models_info:
        print(f"  [{model_name}] 로딩 중...", flush=True)
        model = load_model(model_path)

        kw_res = evaluate(model, keyword_queries, expected_ids)
        vlm_res = evaluate(model, vlm_queries, expected_ids)

        all_results[model_name] = {"keyword": kw_res, "vlm": vlm_res}

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{model_name}] 완료", flush=True)

    # ═══════════════════════════════════════════════
    #  결과 출력
    # ═══════════════════════════════════════════════

    W = 120
    print()
    print("=" * W)
    print("  boost_best vs boost_v3b_best — 키워드 / VLM 쿼리 비교 테스트")
    print("=" * W)

    # ── [1] 요약표 ──
    print()
    print("-" * W)
    print("  [1] 모델별 요약")
    print("-" * W)
    print(f"{'모델':<20} {'쿼리 형식':<12} {'Top1':>8} {'Avg Score':>12} {'Min':>10} {'Max':>10}")
    print("-" * W)

    for mname in ["boost_best", "boost_v3b_best"]:
        for qtype, label in [("keyword", "키워드"), ("vlm", "VLM")]:
            res = all_results[mname][qtype]
            top1 = sum(1 for r in res if r["hit"])
            avg = sum(r["score"] for r in res) / N
            mn = min(r["score"] for r in res)
            mx = max(r["score"] for r in res)
            print(f"{mname:<20} {label:<12} {top1:>6}/{N} {avg:>12.4f} {mn:>10.4f} {mx:>10.4f}")
        print("-" * W)

    # ── [2] 쿼리별 상세 비교 (키워드) ──
    print()
    print("-" * W)
    print("  [2] 키워드 쿼리 — 쿼리별 상세")
    print("-" * W)
    print(
        f"{'#':<4} {'시나리오':<28} {'정답':<8} "
        f"{'boost_best':>12} {'Hit':>4} "
        f"{'v3b_best':>12} {'Hit':>4} "
        f"{'차이':>8}"
    )
    print("-" * W)
    for i, tc in enumerate(TEST_PAIRS):
        r1 = all_results["boost_best"]["keyword"][i]
        r2 = all_results["boost_v3b_best"]["keyword"][i]
        h1 = "O" if r1["hit"] else "X"
        h2 = "O" if r2["hit"] else "X"
        diff = r1["score"] - r2["score"]
        print(
            f"{tc['id']:<4} {tc['scenario']:<28} {tc['expected']:<8} "
            f"{r1['score']:>12.4f} {h1:>4} "
            f"{r2['score']:>12.4f} {h2:>4} "
            f"{diff:>+8.4f}"
        )
    print("-" * W)

    # ── [3] 쿼리별 상세 비교 (VLM) ──
    print()
    print("-" * W)
    print("  [3] VLM 스타일 쿼리 — 쿼리별 상세")
    print("-" * W)
    print(
        f"{'#':<4} {'시나리오':<28} {'정답':<8} "
        f"{'boost_best':>12} {'Hit':>4} "
        f"{'v3b_best':>12} {'Hit':>4} "
        f"{'차이':>8}"
    )
    print("-" * W)
    for i, tc in enumerate(TEST_PAIRS):
        r1 = all_results["boost_best"]["vlm"][i]
        r2 = all_results["boost_v3b_best"]["vlm"][i]
        h1 = "O" if r1["hit"] else "X"
        h2 = "O" if r2["hit"] else "X"
        diff = r1["score"] - r2["score"]
        print(
            f"{tc['id']:<4} {tc['scenario']:<28} {tc['expected']:<8} "
            f"{r1['score']:>12.4f} {h1:>4} "
            f"{r2['score']:>12.4f} {h2:>4} "
            f"{diff:>+8.4f}"
        )
    print("-" * W)

    # ── [4] 쿼리 형식별 성능 차이 분석 ──
    print()
    print("-" * W)
    print("  [4] 쿼리 형식별 성능 차이 분석")
    print("-" * W)
    print(
        f"{'#':<4} {'시나리오':<28} {'정답':<8} "
        f"{'BB키워드':>10} {'BB VLM':>10} {'Gap':>8} "
        f"{'V3키워드':>10} {'V3 VLM':>10} {'Gap':>8}"
    )
    print("-" * W)
    for i, tc in enumerate(TEST_PAIRS):
        bk = all_results["boost_best"]["keyword"][i]["score"]
        bv = all_results["boost_best"]["vlm"][i]["score"]
        vk = all_results["boost_v3b_best"]["keyword"][i]["score"]
        vv = all_results["boost_v3b_best"]["vlm"][i]["score"]
        print(
            f"{tc['id']:<4} {tc['scenario']:<28} {tc['expected']:<8} "
            f"{bk:>10.4f} {bv:>10.4f} {bk-bv:>+8.4f} "
            f"{vk:>10.4f} {vv:>10.4f} {vk-vv:>+8.4f}"
        )
    print("-" * W)

    # ── [5] 틀린 쿼리 상세 ──
    print()
    print("-" * W)
    print("  [5] Top1 틀린 쿼리 상세")
    print("-" * W)
    has_wrong = False
    for mname in ["boost_best", "boost_v3b_best"]:
        for qtype, label in [("keyword", "키워드"), ("vlm", "VLM")]:
            res = all_results[mname][qtype]
            for i, r in enumerate(res):
                if not r["hit"]:
                    has_wrong = True
                    tc = TEST_PAIRS[i]
                    print(f"  [{mname} / {label}] #{tc['id']} {tc['scenario']}")
                    print(f"    기대: {r['expected']}  →  실제 Top1: {r['top1']} (score={r['score']:.4f})")
    if not has_wrong:
        print("  모든 쿼리 Top1 정답!")
    print("-" * W)

    # ═══════════════════════════════════════════════
    #  Markdown 표 (복사용)
    # ═══════════════════════════════════════════════

    print()
    print("=" * W)
    print("  Markdown 표 (복사용)")
    print("=" * W)

    # 요약표
    print()
    print("### 모델별 요약")
    print()
    print("| 모델 | 쿼리 형식 | Top1 | Avg Score | Min | Max |")
    print("|------|----------|------|-----------|-----|-----|")
    for mname in ["boost_best", "boost_v3b_best"]:
        for qtype, label in [("keyword", "키워드"), ("vlm", "VLM")]:
            res = all_results[mname][qtype]
            top1 = sum(1 for r in res if r["hit"])
            avg = sum(r["score"] for r in res) / N
            mn = min(r["score"] for r in res)
            mx = max(r["score"] for r in res)
            print(f"| {mname} | {label} | {top1}/{N} | {avg:.4f} | {mn:.4f} | {mx:.4f} |")
    print()

    # 키워드 상세
    print("### 키워드 쿼리 상세")
    print()
    print("| # | 시나리오 | 정답 | boost_best | Hit | v3b_best | Hit | 차이 |")
    print("|---|---------|------|-----------|-----|---------|-----|------|")
    for i, tc in enumerate(TEST_PAIRS):
        r1 = all_results["boost_best"]["keyword"][i]
        r2 = all_results["boost_v3b_best"]["keyword"][i]
        h1 = "O" if r1["hit"] else "X"
        h2 = "O" if r2["hit"] else "X"
        diff = r1["score"] - r2["score"]
        print(f"| {tc['id']} | {tc['scenario']} | {tc['expected']} | {r1['score']:.4f} | {h1} | {r2['score']:.4f} | {h2} | {diff:+.4f} |")
    print()

    # VLM 상세
    print("### VLM 스타일 쿼리 상세")
    print()
    print("| # | 시나리오 | 정답 | boost_best | Hit | v3b_best | Hit | 차이 |")
    print("|---|---------|------|-----------|-----|---------|-----|------|")
    for i, tc in enumerate(TEST_PAIRS):
        r1 = all_results["boost_best"]["vlm"][i]
        r2 = all_results["boost_v3b_best"]["vlm"][i]
        h1 = "O" if r1["hit"] else "X"
        h2 = "O" if r2["hit"] else "X"
        diff = r1["score"] - r2["score"]
        print(f"| {tc['id']} | {tc['scenario']} | {tc['expected']} | {r1['score']:.4f} | {h1} | {r2['score']:.4f} | {h2} | {diff:+.4f} |")
    print()

    # 형식별 Gap 분석
    print("### 키워드 vs VLM 성능 Gap 분석")
    print()
    print("| # | 시나리오 | boost_best 키워드 | boost_best VLM | Gap | v3b_best 키워드 | v3b_best VLM | Gap |")
    print("|---|---------|------------------|---------------|-----|----------------|-------------|-----|")
    for i, tc in enumerate(TEST_PAIRS):
        bk = all_results["boost_best"]["keyword"][i]["score"]
        bv = all_results["boost_best"]["vlm"][i]["score"]
        vk = all_results["boost_v3b_best"]["keyword"][i]["score"]
        vv = all_results["boost_v3b_best"]["vlm"][i]["score"]
        print(f"| {tc['id']} | {tc['scenario']} | {bk:.4f} | {bv:.4f} | {bk-bv:+.4f} | {vk:.4f} | {vv:.4f} | {vk-vv:+.4f} |")
    print()

    # 종합 비교
    bb_kw_top1 = sum(1 for r in all_results["boost_best"]["keyword"] if r["hit"])
    bb_vlm_top1 = sum(1 for r in all_results["boost_best"]["vlm"] if r["hit"])
    v3_kw_top1 = sum(1 for r in all_results["boost_v3b_best"]["keyword"] if r["hit"])
    v3_vlm_top1 = sum(1 for r in all_results["boost_v3b_best"]["vlm"] if r["hit"])

    bb_kw_avg = sum(r["score"] for r in all_results["boost_best"]["keyword"]) / N
    bb_vlm_avg = sum(r["score"] for r in all_results["boost_best"]["vlm"]) / N
    v3_kw_avg = sum(r["score"] for r in all_results["boost_v3b_best"]["keyword"]) / N
    v3_vlm_avg = sum(r["score"] for r in all_results["boost_v3b_best"]["vlm"]) / N

    print("### 종합 비교")
    print()
    print("| 지표 | boost_best | boost_v3b_best |")
    print("|------|-----------|---------------|")
    print(f"| 키워드 Top1 | {bb_kw_top1}/{N} | {v3_kw_top1}/{N} |")
    print(f"| 키워드 Avg | {bb_kw_avg:.4f} | {v3_kw_avg:.4f} |")
    print(f"| VLM Top1 | {bb_vlm_top1}/{N} | {v3_vlm_top1}/{N} |")
    print(f"| VLM Avg | {bb_vlm_avg:.4f} | {v3_vlm_avg:.4f} |")
    print(f"| 키워드-VLM Gap | {bb_kw_avg - bb_vlm_avg:+.4f} | {v3_kw_avg - v3_vlm_avg:+.4f} |")
    print()


if __name__ == "__main__":
    main()
