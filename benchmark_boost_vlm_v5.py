import json
import os

from benchmark_final_v4 import (
    aggregate_category_stats,
    build_new_vlm_items,
    build_vlm_30_items,
    build_wrong_answer_rows,
    compute_overfit_gap,
    evaluate_query_set,
    load_chunks,
    load_model,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "boost_best_vlm_v5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
BASELINE_JSON = os.path.join(BASE_DIR, "boost_best_vlm_v4_final_benchmark.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "boost_best_vlm_v5_benchmark.json")

CHRONIC_TARGET_ITEMS = [
    ("차11-2", "B9-비신호동일폭교차로우측차우선위반"),
    ("차44-1", "N13-고속도로추돌(새표현)"),
    ("차61-1", "N18-개문사고(새표현)"),
    ("차3-1", "B2-적색우회전녹색직진교차충돌"),
    ("차4-1", "B3-동일방향직진우회전교차충돌"),
    ("차31-2", "B5-일방통행로역주행정면충돌"),
    ("차33-2", "B6-좌회전중후방유턴차충돌"),
    ("차42-1", "B7-정체구간연쇄3중추돌"),
    ("차51-2", "C4-아파트주차장통로교차지점충돌"),
]

CATEGORY_ORDER = [
    "신호교차로",
    "비신호교차로",
    "추돌",
    "중앙선역주행",
    "차선변경끼어들기",
    "유턴",
    "주차장",
    "고속도로",
    "보행자",
    "자전거",
]


KEYWORD_TESTS = [
    {"name": "K1-신호위반직진충돌", "expected": "차1-1", "query": "신호위반 직진 충돌"},
    {"name": "K2-비신호교차로직진좌회전", "expected": "차15-1", "query": "비신호교차로 직진 vs 좌회전"},
    {"name": "K3-추돌사고과실", "expected": "차41-1", "query": "추돌 사고 과실"},
    {"name": "K4-야간교차로충돌", "expected": "차12-1", "query": "야간 교차로 충돌"},
    {"name": "K5-중앙선침범충돌", "expected": "차31-1", "query": "중앙선 침범 충돌"},
    {"name": "K6-끼어들기충돌", "expected": "차20-2", "query": "끼어들기 충돌"},
    {"name": "K7-유턴중충돌", "expected": "차33-1", "query": "유턴 중 충돌"},
    {"name": "K8-고속도로추돌사고", "expected": "차43-1", "query": "고속도로 추돌 사고"},
    {"name": "K9-주차장출차중충돌", "expected": "차51-1", "query": "주차장 출차 중 충돌"},
    {"name": "K10-횡단보도보행자충돌", "expected": "차5-2", "query": "횡단보도 보행자 충돌"},
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_detail_map(details):
    return {item["name"]: item for item in details}


def build_chronic_item_report(target_items, baseline_details, current_details):
    rows = []
    for chunk_id, query_name in target_items:
        baseline = baseline_details.get(query_name, {})
        current = current_details.get(query_name, {})

        baseline_rank = baseline.get("rank", 999)
        current_rank = current.get("rank", 999)

        if current_rank == 1 and baseline_rank != 1:
            status = "FIXED"
        elif current_rank < baseline_rank:
            status = "IMPROVED"
        elif current_rank == baseline_rank:
            status = "UNCHANGED"
        else:
            status = "REGRESSED"

        rows.append(
            {
                "chunk_id": chunk_id,
                "query_name": query_name,
                "baseline_top1": baseline.get("top1"),
                "baseline_rank": baseline_rank,
                "baseline_score": baseline.get("score"),
                "current_top1": current.get("top1"),
                "current_rank": current_rank,
                "current_score": current.get("score"),
                "status": status,
            }
        )

    return rows


def build_category_comparison(baseline_stats, current_stats):
    rows = []
    for category in CATEGORY_ORDER:
        baseline = baseline_stats.get(category)
        current = current_stats.get(category)
        if not baseline or not current:
            continue
        rows.append(
            {
                "category": category,
                "baseline_top1": baseline["top1"],
                "baseline_total": baseline["total"],
                "baseline_avg": baseline["avg_score"],
                "current_top1": current["top1"],
                "current_total": current["total"],
                "current_avg": current["avg_score"],
                "top1_delta": current["top1"] - baseline["top1"],
                "avg_delta": round(current["avg_score"] - baseline["avg_score"], 4),
            }
        )
    return rows


def print_summary_comparison(baseline_tests, current_tests):
    print("| 테스트 유형 | v4 Top1 | v5 Top1 | v4 Top3 | v5 Top3 | v4 Top5 | v5 Top5 | v4 Avg | v5 Avg |")
    print("|-----------|---------|---------|---------|---------|---------|---------|--------|--------|")
    rows = [
        ("키워드 10개", baseline_tests["keyword_10"]["summary"], current_tests["keyword_10"]["summary"]),
        ("VLM 30개", baseline_tests["vlm_30"]["summary"], current_tests["vlm_30"]["summary"]),
        ("새 VLM 20개", baseline_tests["new_vlm_20"]["summary"], current_tests["new_vlm_20"]["summary"]),
    ]
    for label, baseline, current in rows:
        print(
            f"| {label} | {baseline['top1']}/{baseline['total']} | {current['top1']}/{current['total']} | "
            f"{baseline['top3']}/{baseline['total']} | {current['top3']}/{current['total']} | "
            f"{baseline['top5']}/{baseline['total']} | {current['top5']}/{current['total']} | "
            f"{baseline['avg_score']:.4f} | {current['avg_score']:.4f} |"
        )


def print_category_comparison(rows):
    print("\n카테고리별 비교 (VLM 30개 기준):")
    print("| 카테고리 | v4 Top1 | v5 Top1 | Delta | v4 Avg | v5 Avg | Delta |")
    print("|----------|---------|---------|-------|--------|--------|-------|")
    for row in rows:
        print(
            f"| {row['category']} | {row['baseline_top1']}/{row['baseline_total']} | "
            f"{row['current_top1']}/{row['current_total']} | {row['top1_delta']:+d} | "
            f"{row['baseline_avg']:.4f} | {row['current_avg']:.4f} | {row['avg_delta']:+.4f} |"
        )


def print_chronic_report(rows):
    print("\n고질적 오답 9건 개선 여부:")
    print("| chunk | 쿼리명 | v4 rank | v5 rank | v4 예측 | v5 예측 | v4 score | v5 score | 상태 |")
    print("|-------|--------|---------|---------|---------|---------|----------|----------|------|")
    for row in rows:
        print(
            f"| {row['chunk_id']} | {row['query_name']} | {row['baseline_rank']} | {row['current_rank']} | "
            f"{row['baseline_top1']} | {row['current_top1']} | {row['baseline_score']:.4f} | "
            f"{row['current_score']:.4f} | {row['status']} |"
        )


def main():
    import torch

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {MODEL_PATH}")

    baseline = load_json(BASELINE_JSON)
    chunk_ids, chunk_contents = load_chunks()
    model = load_model(MODEL_PATH)
    chunk_embeddings = model.encode(
        chunk_contents,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=32,
    )

    keyword_result = evaluate_query_set(
        model=model,
        chunk_ids=chunk_ids,
        chunk_embeddings=chunk_embeddings,
        items=KEYWORD_TESTS,
        test_type="keyword_10",
    )
    vlm_30_result = evaluate_query_set(
        model=model,
        chunk_ids=chunk_ids,
        chunk_embeddings=chunk_embeddings,
        items=build_vlm_30_items(),
        test_type="vlm_30",
    )
    new_vlm_result = evaluate_query_set(
        model=model,
        chunk_ids=chunk_ids,
        chunk_embeddings=chunk_embeddings,
        items=build_new_vlm_items(),
        test_type="new_vlm_20",
    )

    current_tests = {
        "keyword_10": keyword_result,
        "vlm_30": vlm_30_result,
        "new_vlm_20": new_vlm_result,
    }
    current_category_stats = aggregate_category_stats(vlm_30_result["details"])
    current_overfit = compute_overfit_gap(vlm_30_result["details"])
    wrong_answers = build_wrong_answer_rows(
        keyword_result["details"] + vlm_30_result["details"] + new_vlm_result["details"]
    )

    baseline_details = build_detail_map(
        baseline["tests"]["keyword_10"]["details"]
        + baseline["tests"]["vlm_30"]["details"]
        + baseline["tests"]["new_vlm_20"]["details"]
    )
    current_details = build_detail_map(
        keyword_result["details"] + vlm_30_result["details"] + new_vlm_result["details"]
    )
    category_comparison = build_category_comparison(
        baseline["vlm_30_category_stats"], current_category_stats
    )
    chronic_report = build_chronic_item_report(
        target_items=CHRONIC_TARGET_ITEMS,
        baseline_details=baseline_details,
        current_details=current_details,
    )

    print_summary_comparison(baseline["tests"], current_tests)
    print_category_comparison(category_comparison)
    print_chronic_report(chronic_report)
    print("\n과적합 비교:")
    print(
        f"v4 A-C Gap {baseline['overfit']['avg_score_gap']:+.4f} | "
        f"v5 A-C Gap {current_overfit['avg_score_gap']:+.4f}"
    )

    result = {
        "model": MODEL_NAME,
        "base_benchmark": os.path.basename(BASELINE_JSON),
        "embedding_mode": "content_full_embedding",
        "tests": current_tests,
        "vlm_30_category_stats": current_category_stats,
        "overfit": current_overfit,
        "wrong_answers": wrong_answers,
        "comparison_vs_v4": {
            "summary": {
                "keyword_10": {
                    "v4": baseline["tests"]["keyword_10"]["summary"],
                    "v5": keyword_result["summary"],
                },
                "vlm_30": {
                    "v4": baseline["tests"]["vlm_30"]["summary"],
                    "v5": vlm_30_result["summary"],
                },
                "new_vlm_20": {
                    "v4": baseline["tests"]["new_vlm_20"]["summary"],
                    "v5": new_vlm_result["summary"],
                },
            },
            "category_comparison": category_comparison,
            "chronic_items": chronic_report,
            "overfit": {
                "v4_avg_score_gap": baseline["overfit"]["avg_score_gap"],
                "v5_avg_score_gap": current_overfit["avg_score_gap"],
            },
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"\n결과 JSON 저장: {OUTPUT_JSON}")

    del model, chunk_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
