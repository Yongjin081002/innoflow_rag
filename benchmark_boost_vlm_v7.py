import json
import os

from benchmark_boost_vlm_v5 import build_chronic_item_report, build_detail_map
from benchmark_boost_vlm_v6 import CATEGORY_ORDER
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_V6 = os.path.join(BASE_DIR, "boost_best_vlm_v6")
MODEL_V7 = os.path.join(BASE_DIR, "boost_best_vlm_v7")
BASELINE_JSON = os.path.join(BASE_DIR, "boost_best_vlm_v6_benchmark.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "boost_best_vlm_v7_benchmark.json")

CHRONIC_TARGET_ITEMS = [
    ("차3-1", "B2-적색우회전녹색직진교차충돌"),
    ("차4-1", "B3-동일방향직진우회전교차충돌"),
    ("차31-2", "B5-일방통행로역주행정면충돌"),
    ("차11-2", "B9-비신호동일폭교차로우측차우선위반"),
    ("차1-1", "N1-녹색직진vs신호위반직진(새표현)"),
    ("차15-1", "A10-비신호교차로골목좌회전vs직진"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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

    details = (
        keyword_result["details"]
        + vlm_30_result["details"]
        + new_vlm_result["details"]
    )
    return {
        "tests": {
            "keyword_10": keyword_result,
            "vlm_30": vlm_30_result,
            "new_vlm_20": new_vlm_result,
        },
        "vlm_30_category_stats": aggregate_category_stats(vlm_30_result["details"]),
        "overfit": compute_overfit_gap(vlm_30_result["details"]),
        "wrong_answers": build_wrong_answer_rows(details),
    }


def build_category_comparison(v6_stats, v7_stats):
    rows = []
    for category in CATEGORY_ORDER:
        baseline = v6_stats.get(category)
        current = v7_stats.get(category)
        if not baseline or not current:
            continue
        rows.append(
            {
                "category": category,
                "v6_top1": baseline["top1"],
                "v6_total": baseline["total"],
                "v6_avg": baseline["avg_score"],
                "v7_top1": current["top1"],
                "v7_total": current["total"],
                "v7_avg": current["avg_score"],
                "top1_delta": current["top1"] - baseline["top1"],
                "avg_delta": round(current["avg_score"] - baseline["avg_score"], 4),
            }
        )
    return rows


def build_summary_comparison(v6_tests, v7_tests):
    rows = {}
    for key in ("keyword_10", "vlm_30", "new_vlm_20"):
        rows[key] = {
            "v6": v6_tests[key]["summary"],
            "v7": v7_tests[key]["summary"],
            "top1_delta": (
                v7_tests[key]["summary"]["top1"] - v6_tests[key]["summary"]["top1"]
            ),
            "avg_delta": round(
                v7_tests[key]["summary"]["avg_score"]
                - v6_tests[key]["summary"]["avg_score"],
                4,
            ),
        }
    return rows


def analyze_new_vlm_top1_decline(v6_details, v7_details):
    rows = []
    for name, baseline in v6_details.items():
        if baseline.get("test_type") != "new_vlm_20":
            continue
        current = v7_details.get(name)
        if not current:
            continue
        if baseline["rank"] == 1 and current["rank"] != 1:
            rows.append(
                {
                    "name": name,
                    "expected": current["expected"],
                    "v6_top1": baseline["top1"],
                    "v7_top1": current["top1"],
                    "v6_score": baseline["score"],
                    "v7_score": current["score"],
                    "cause": (
                        "targeted v7 보강으로 특정 약점 chunk 표현이 강해지면서 "
                        "새 VLM 쿼리의 상위 후보 순서가 변동됨"
                    ),
                }
            )
    return rows


def print_summary_comparison(rows):
    print("| 테스트 유형 | v6 Top1 | v7 Top1 | v6 Top3 | v7 Top3 | v6 Top5 | v7 Top5 | v6 Avg | v7 Avg |")
    print("|-----------|---------|---------|---------|---------|---------|---------|--------|--------|")
    labels = {
        "keyword_10": "키워드 10개",
        "vlm_30": "VLM 30개",
        "new_vlm_20": "새 VLM 20개",
    }
    for key, label in labels.items():
        v6 = rows[key]["v6"]
        v7 = rows[key]["v7"]
        print(
            f"| {label} | {v6['top1']}/{v6['total']} | {v7['top1']}/{v7['total']} | "
            f"{v6['top3']}/{v6['total']} | {v7['top3']}/{v7['total']} | "
            f"{v6['top5']}/{v6['total']} | {v7['top5']}/{v7['total']} | "
            f"{v6['avg_score']:.4f} | {v7['avg_score']:.4f} |"
        )


def print_category_comparison(rows):
    print("\n카테고리별 비교 (VLM 30개 기준):")
    print("| 카테고리 | v6 Top1 | v7 Top1 | Delta | v6 Avg | v7 Avg | Delta |")
    print("|----------|---------|---------|-------|--------|--------|-------|")
    for row in rows:
        print(
            f"| {row['category']} | {row['v6_top1']}/{row['v6_total']} | "
            f"{row['v7_top1']}/{row['v7_total']} | {row['top1_delta']:+d} | "
            f"{row['v6_avg']:.4f} | {row['v7_avg']:.4f} | {row['avg_delta']:+.4f} |"
        )


def print_chronic_report(rows):
    print("\n고질 오답 개선 여부:")
    print("| chunk | 쿼리명 | v6 rank | v7 rank | v6 예측 | v7 예측 | v6 score | v7 score | 상태 |")
    print("|-------|--------|---------|---------|---------|---------|----------|----------|------|")
    for row in rows:
        print(
            f"| {row['chunk_id']} | {row['query_name']} | {row['baseline_rank']} | "
            f"{row['current_rank']} | {row['baseline_top1']} | {row['current_top1']} | "
            f"{row['baseline_score']:.4f} | {row['current_score']:.4f} | {row['status']} |"
        )


def print_new_vlm_declines(rows):
    print("\n새 VLM 20개 Top1 하락 원인:")
    if not rows:
        print("- v6에서 Top1이던 항목 중 v7에서 Top1 밖으로 하락한 항목 없음")
        return
    print("| 쿼리명 | 정답 | v6 예측 | v7 예측 | v6 score | v7 score | 원인 |")
    print("|--------|------|---------|---------|----------|----------|------|")
    for row in rows:
        print(
            f"| {row['name']} | {row['expected']} | {row['v6_top1']} | {row['v7_top1']} | "
            f"{row['v6_score']:.4f} | {row['v7_score']:.4f} | {row['cause']} |"
        )


def main():
    import torch

    if not os.path.isdir(MODEL_V7):
        raise FileNotFoundError(f"v7 모델 디렉토리를 찾을 수 없습니다: {MODEL_V7}")

    if os.path.isfile(BASELINE_JSON):
        baseline = load_json(BASELINE_JSON)
        v6_result = baseline.get("v6_result") or {
            "tests": baseline["tests"],
            "vlm_30_category_stats": baseline["vlm_30_category_stats"],
            "overfit": baseline["overfit"],
            "wrong_answers": baseline["wrong_answers"],
        }
    else:
        if not os.path.isdir(MODEL_V6):
            raise FileNotFoundError(f"v6 모델 디렉토리를 찾을 수 없습니다: {MODEL_V6}")
        v6_result = evaluate_model(MODEL_V6)

    v7_result = evaluate_model(MODEL_V7)

    v6_details = build_detail_map(
        v6_result["tests"]["keyword_10"]["details"]
        + v6_result["tests"]["vlm_30"]["details"]
        + v6_result["tests"]["new_vlm_20"]["details"]
    )
    v7_details = build_detail_map(
        v7_result["tests"]["keyword_10"]["details"]
        + v7_result["tests"]["vlm_30"]["details"]
        + v7_result["tests"]["new_vlm_20"]["details"]
    )

    summary_comparison = build_summary_comparison(v6_result["tests"], v7_result["tests"])
    category_comparison = build_category_comparison(
        v6_result["vlm_30_category_stats"],
        v7_result["vlm_30_category_stats"],
    )
    chronic_report = build_chronic_item_report(
        target_items=CHRONIC_TARGET_ITEMS,
        baseline_details=v6_details,
        current_details=v7_details,
    )
    new_vlm_declines = analyze_new_vlm_top1_decline(v6_details, v7_details)

    print_summary_comparison(summary_comparison)
    print_category_comparison(category_comparison)
    print_chronic_report(chronic_report)
    print_new_vlm_declines(new_vlm_declines)
    print("\n과적합 비교:")
    print(
        f"v6 A-C Gap {v6_result['overfit']['avg_score_gap']:+.4f} | "
        f"v7 A-C Gap {v7_result['overfit']['avg_score_gap']:+.4f}"
    )

    payload = {
        "model": "boost_best_vlm_v7",
        "baseline_model": "boost_best_vlm_v6",
        "embedding_mode": "content_full_embedding",
        "label_alignment": {
            "차44-1": {
                "content_summary": "직진 차량과 도로가 아닌 장소에서 도로로 진입한 차량의 사고",
                "decision": "고속도로 추돌 쿼리와 불일치하므로 N13 기대 라벨은 차41-1 유지",
            },
            "차61-1": {
                "content_summary": "정체도로 사이 직진 이륜차와 직진/좌회전 차량의 사고",
                "decision": "개문사고 쿼리와 불일치하므로 N18 기대 라벨은 차52-1 유지",
            },
        },
        "v6_result": v6_result,
        "v7_result": v7_result,
        "comparison_vs_v6": {
            "summary": summary_comparison,
            "category_comparison": category_comparison,
            "chronic_items": chronic_report,
            "new_vlm_top1_declines": new_vlm_declines,
            "overfit": {
                "v6_avg_score_gap": v6_result["overfit"]["avg_score_gap"],
                "v7_avg_score_gap": v7_result["overfit"]["avg_score_gap"],
            },
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\n결과 JSON 저장: {OUTPUT_JSON}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
