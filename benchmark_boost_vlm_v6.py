import json
import os

from benchmark_final_v4 import (
    KEYWORD_TESTS,
    aggregate_category_stats,
    build_new_vlm_items,
    build_vlm_30_items,
    build_wrong_answer_rows,
    compute_overfit_gap,
    evaluate_query_set,
    load_chunks,
    load_model,
)
from benchmark_boost_vlm_v5 import build_chronic_item_report, build_detail_map


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_V5 = os.path.join(BASE_DIR, "boost_best_vlm_v5")
MODEL_V6 = os.path.join(BASE_DIR, "boost_best_vlm_v6")
OUTPUT_JSON = os.path.join(BASE_DIR, "boost_best_vlm_v6_benchmark.json")

CHRONIC_TARGET_ITEMS = [
    ("차3-1", "B2-적색우회전녹색직진교차충돌"),
    ("차4-1", "B3-동일방향직진우회전교차충돌"),
    ("차31-2", "B5-일방통행로역주행정면충돌"),
    ("차11-2", "B9-비신호동일폭교차로우측차우선위반"),
    ("차15-1", "A10-비신호교차로골목좌회전vs직진"),
    ("차41-1", "N13-고속도로추돌(새표현)"),
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


def evaluate_model(model_path):
    model = load_model(model_path)
    chunk_ids, chunk_contents = load_chunks()
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
    result = {
        "tests": {
            "keyword_10": keyword_result,
            "vlm_30": vlm_30_result,
            "new_vlm_20": new_vlm_result,
        },
        "vlm_30_category_stats": aggregate_category_stats(vlm_30_result["details"]),
        "overfit": compute_overfit_gap(vlm_30_result["details"]),
        "wrong_answers": build_wrong_answer_rows(
            keyword_result["details"] + vlm_30_result["details"] + new_vlm_result["details"]
        ),
    }
    return result


def build_category_comparison(v5_stats, v6_stats):
    rows = []
    for category in CATEGORY_ORDER:
        baseline = v5_stats.get(category)
        current = v6_stats.get(category)
        if not baseline or not current:
            continue
        rows.append(
            {
                "category": category,
                "v5_top1": baseline["top1"],
                "v5_total": baseline["total"],
                "v5_avg": baseline["avg_score"],
                "v6_top1": current["top1"],
                "v6_total": current["total"],
                "v6_avg": current["avg_score"],
                "top1_delta": current["top1"] - baseline["top1"],
                "avg_delta": round(current["avg_score"] - baseline["avg_score"], 4),
            }
        )
    return rows


def print_summary_comparison(v5_tests, v6_tests):
    print("| 테스트 유형 | v5 Top1 | v6 Top1 | v5 Top3 | v6 Top3 | v5 Top5 | v6 Top5 | v5 Avg | v6 Avg |")
    print("|-----------|---------|---------|---------|---------|---------|---------|--------|--------|")
    for label, baseline, current in [
        ("키워드 10개", v5_tests["keyword_10"]["summary"], v6_tests["keyword_10"]["summary"]),
        ("VLM 30개", v5_tests["vlm_30"]["summary"], v6_tests["vlm_30"]["summary"]),
        ("새 VLM 20개", v5_tests["new_vlm_20"]["summary"], v6_tests["new_vlm_20"]["summary"]),
    ]:
        print(
            f"| {label} | {baseline['top1']}/{baseline['total']} | {current['top1']}/{current['total']} | "
            f"{baseline['top3']}/{baseline['total']} | {current['top3']}/{current['total']} | "
            f"{baseline['top5']}/{baseline['total']} | {current['top5']}/{current['total']} | "
            f"{baseline['avg_score']:.4f} | {current['avg_score']:.4f} |"
        )


def print_category_comparison(rows):
    print("\n카테고리별 비교 (VLM 30개 기준):")
    print("| 카테고리 | v5 Top1 | v6 Top1 | Delta | v5 Avg | v6 Avg | Delta |")
    print("|----------|---------|---------|-------|--------|--------|-------|")
    for row in rows:
        print(
            f"| {row['category']} | {row['v5_top1']}/{row['v5_total']} | "
            f"{row['v6_top1']}/{row['v6_total']} | {row['top1_delta']:+d} | "
            f"{row['v5_avg']:.4f} | {row['v6_avg']:.4f} | {row['avg_delta']:+.4f} |"
        )


def print_chronic_report(rows):
    print("\n고질 오답 개선 여부:")
    print("| chunk | 쿼리명 | v5 rank | v6 rank | v5 예측 | v6 예측 | v5 score | v6 score | 상태 |")
    print("|-------|--------|---------|---------|---------|---------|----------|----------|------|")
    for row in rows:
        print(
            f"| {row['chunk_id']} | {row['query_name']} | {row['baseline_rank']} | {row['current_rank']} | "
            f"{row['baseline_top1']} | {row['current_top1']} | {row['baseline_score']:.4f} | "
            f"{row['current_score']:.4f} | {row['status']} |"
        )


def main():
    import torch

    if not os.path.isdir(MODEL_V5):
        raise FileNotFoundError(f"v5 모델 디렉토리를 찾을 수 없습니다: {MODEL_V5}")
    if not os.path.isdir(MODEL_V6):
        raise FileNotFoundError(f"v6 모델 디렉토리를 찾을 수 없습니다: {MODEL_V6}")

    v5_result = evaluate_model(MODEL_V5)
    v6_result = evaluate_model(MODEL_V6)

    v5_details = build_detail_map(
        v5_result["tests"]["keyword_10"]["details"]
        + v5_result["tests"]["vlm_30"]["details"]
        + v5_result["tests"]["new_vlm_20"]["details"]
    )
    v6_details = build_detail_map(
        v6_result["tests"]["keyword_10"]["details"]
        + v6_result["tests"]["vlm_30"]["details"]
        + v6_result["tests"]["new_vlm_20"]["details"]
    )
    category_comparison = build_category_comparison(
        v5_result["vlm_30_category_stats"],
        v6_result["vlm_30_category_stats"],
    )
    chronic_report = build_chronic_item_report(
        target_items=CHRONIC_TARGET_ITEMS,
        baseline_details=v5_details,
        current_details=v6_details,
    )

    print_summary_comparison(v5_result["tests"], v6_result["tests"])
    print_category_comparison(category_comparison)
    print_chronic_report(chronic_report)
    print("\n과적합 비교:")
    print(
        f"v5 A-C Gap {v5_result['overfit']['avg_score_gap']:+.4f} | "
        f"v6 A-C Gap {v6_result['overfit']['avg_score_gap']:+.4f}"
    )

    payload = {
        "model": "boost_best_vlm_v6",
        "baseline_model": "boost_best_vlm_v5",
        "embedding_mode": "content_full_embedding",
        "v5_result": v5_result,
        "v6_result": v6_result,
        "comparison_vs_v5": {
            "summary": {
                "keyword_10": {
                    "v5": v5_result["tests"]["keyword_10"]["summary"],
                    "v6": v6_result["tests"]["keyword_10"]["summary"],
                },
                "vlm_30": {
                    "v5": v5_result["tests"]["vlm_30"]["summary"],
                    "v6": v6_result["tests"]["vlm_30"]["summary"],
                },
                "new_vlm_20": {
                    "v5": v5_result["tests"]["new_vlm_20"]["summary"],
                    "v6": v6_result["tests"]["new_vlm_20"]["summary"],
                },
            },
            "category_comparison": category_comparison,
            "chronic_items": chronic_report,
            "overfit": {
                "v5_avg_score_gap": v5_result["overfit"]["avg_score_gap"],
                "v6_avg_score_gap": v6_result["overfit"]["avg_score_gap"],
            },
            "label_alignment_notes": [
                "N13 expected label corrected from 차44-1 to 차41-1",
                "N18 expected label corrected from 차61-1 to 차52-1",
                "C4 query rewritten to match 차51-2 semantics",
            ],
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\n결과 JSON 저장: {OUTPUT_JSON}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
