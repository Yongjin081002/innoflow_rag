import json
import os
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "boost_best_vlm_v4"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
OUTPUT_JSON = os.path.join(BASE_DIR, "boost_best_vlm_v4_final_benchmark.json")

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

VLM_30_CATEGORY_BY_NAME = {
    "A1-녹색직진vs적색직진": "신호교차로",
    "A2-야간비신호교차로동시진입": "비신호교차로",
    "A3-선행차정지후추돌": "추돌",
    "A4-유턴차량vs반대편직진": "유턴",
    "A5-차선변경끼어들기접촉": "차선변경끼어들기",
    "A6-커브구간중앙선침범정면충돌": "중앙선역주행",
    "A7-고속도로가속차로합류측면접촉": "고속도로",
    "A8-주차장후진출차통행차량충돌": "주차장",
    "A9-비보호좌회전맞은편직진충돌": "신호교차로",
    "A10-비신호교차로골목좌회전vs직진": "비신호교차로",
    "B1-녹색좌회전진입후황색직진충돌": "신호교차로",
    "B2-적색우회전녹색직진교차충돌": "신호교차로",
    "B3-동일방향직진우회전교차충돌": "신호교차로",
    "B4-급차선변경후방직진접촉": "차선변경끼어들기",
    "B5-일방통행로역주행정면충돌": "중앙선역주행",
    "B6-좌회전중후방유턴차충돌": "유턴",
    "B7-정체구간연쇄3중추돌": "추돌",
    "B8-보행자전용도로차량침범충돌": "보행자",
    "B9-비신호동일폭교차로우측차우선위반": "비신호교차로",
    "B10-고속도로본선차선변경측면접촉": "고속도로",
    "C1-자전거도로역통행정면충돌": "자전거",
    "C2-자전거전용도로후방추돌": "자전거",
    "C3-자전거무리한도로횡단충돌": "자전거",
    "C4-아파트주차장통로교차지점충돌": "주차장",
    "C5-야간보도없는도로가장자리보행자충돌": "보행자",
    "C6-자전거급차선변경후방접촉": "자전거",
    "C7-우회전시횡단보도보행자충돌": "보행자",
    "C8-심야점멸신호적색점멸미정지충돌": "신호교차로",
    "C9-고속도로감속차로진입추돌": "고속도로",
    "C10-주간차도보행자차량충돌": "보행자",
}


def round4(value):
    return round(float(value), 4)


def summarize_details(details):
    total = len(details)
    avg_score = sum(item["score"] for item in details) / total if total else 0.0
    return {
        "total": total,
        "top1": sum(1 for item in details if item["rank"] == 1),
        "top3": sum(1 for item in details if item["rank"] <= 3),
        "top5": sum(1 for item in details if item["rank"] <= 5),
        "avg_score": round4(avg_score),
    }


def aggregate_category_stats(details):
    grouped = defaultdict(list)
    for item in details:
        grouped[item["category"]].append(item)

    stats = {}
    for category, items in grouped.items():
        stats[category] = summarize_details(items)
    return dict(stats)


def compute_overfit_gap(details):
    group_a_scores = [item["score"] for item in details if item.get("group") == "A"]
    group_c_scores = [item["score"] for item in details if item.get("group") == "C"]
    a_avg = sum(group_a_scores) / len(group_a_scores) if group_a_scores else 0.0
    c_avg = sum(group_c_scores) / len(group_c_scores) if group_c_scores else 0.0
    return {
        "group_a_avg_score": round4(a_avg),
        "group_c_avg_score": round4(c_avg),
        "avg_score_gap": round4(a_avg - c_avg),
    }


def rule_id_to_category(rule_id):
    if rule_id.startswith("보"):
        return "보행자"
    if rule_id.startswith("거"):
        return "자전거"
    if not rule_id.startswith("차") or "-" not in rule_id:
        return None

    try:
        number = int(rule_id[1 : rule_id.index("-")])
    except ValueError:
        return None

    if number in {1, 2, 3, 4, 5, 7}:
        return "신호교차로"
    if number in {11, 12, 13, 15, 16}:
        return "비신호교차로"
    if number == 20:
        return "차선변경끼어들기"
    if number == 31:
        return "중앙선역주행"
    if number == 33:
        return "유턴"
    if number in {41, 42}:
        return "추돌"
    if number in {43, 44}:
        return "고속도로"
    if number == 51:
        return "주차장"
    if number == 61:
        return "개문기타"
    return None


def infer_wrong_cause(item):
    expected = item["expected"]
    predicted = item["top1"]
    if item["rank"] > 5:
        return "Top5 밖 오답"
    if expected.split("-")[0] == predicted.split("-")[0]:
        return "유사 유형과 혼동"
    expected_category = item.get("category") or rule_id_to_category(expected)
    predicted_category = rule_id_to_category(predicted)
    if expected_category and predicted_category and expected_category == predicted_category:
        return "동일 분류 내 세부 기준 혼동"
    return "다른 카테고리로 오검색"


def build_wrong_answer_rows(details):
    rows = []
    for item in details:
        if item["rank"] == 1:
            continue
        rows.append(
            {
                "test_type": item.get("test_type", "unknown"),
                "name": item["name"],
                "expected": item["expected"],
                "predicted": item["top1"],
                "score": round4(item["score"]),
                "cause": infer_wrong_cause(item),
            }
        )
    return rows


def load_chunks():
    with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    chunk_ids = [chunk["id"] for chunk in chunks]
    chunk_contents = [chunk["content"] for chunk in chunks]
    return chunk_ids, chunk_contents


def build_vlm_30_items():
    from compare_boost_v3 import GROUP_A, GROUP_B, GROUP_C

    items = []
    for group_name, group_items in (("A", GROUP_A), ("B", GROUP_B), ("C", GROUP_C)):
        for item in group_items:
            items.append(
                {
                    "group": group_name,
                    "category": VLM_30_CATEGORY_BY_NAME[item["name"]],
                    "name": item["name"],
                    "expected": item["expected"],
                    "query": item["query"],
                }
            )
    return items


def build_new_vlm_items():
    from test_vlm_new_queries import NEW_VLM_QUERIES

    return [
        {
            "name": item["name"],
            "expected": item["expected"],
            "query": item["query"],
        }
        for item in NEW_VLM_QUERIES
    ]


def load_model(model_path):
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 512
    return model


def evaluate_query_set(model, chunk_ids, chunk_embeddings, items, test_type):
    import torch
    from sentence_transformers.util import cos_sim

    query_embeddings = model.encode(
        [item["query"] for item in items],
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=32,
    )

    details = []
    for index, item in enumerate(items):
        similarities = cos_sim(query_embeddings[index], chunk_embeddings)[0]
        sorted_indices = torch.argsort(similarities, descending=True)
        top5_ids = [chunk_ids[i] for i in sorted_indices[:5]]
        top5_scores = [round4(similarities[i].item()) for i in sorted_indices[:5]]

        expected = item["expected"]
        expected_index = chunk_ids.index(expected)
        expected_score = similarities[expected_index].item()
        rank = top5_ids.index(expected) + 1 if expected in top5_ids else 999

        detail = {
            "test_type": test_type,
            "name": item["name"],
            "expected": expected,
            "top1": top5_ids[0],
            "rank": rank,
            "score": round4(expected_score),
            "top5": [
                {"id": top_id, "score": score}
                for top_id, score in zip(top5_ids, top5_scores)
            ],
        }
        if "group" in item:
            detail["group"] = item["group"]
        if "category" in item:
            detail["category"] = item["category"]

        details.append(detail)

    return {
        "summary": summarize_details(details),
        "details": details,
    }


def save_json(payload, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_summary_table(keyword_result, vlm_30_result, new_vlm_result):
    print("| 테스트 유형 | Top1 | Top3 | Top5 | Avg Score |")
    print("|-----------|------|------|------|-----------|")
    for label, result in [
        ("키워드 10개", keyword_result),
        ("VLM 30개", vlm_30_result),
        ("새 VLM 20개", new_vlm_result),
    ]:
        summary = result["summary"]
        total = summary["total"]
        print(
            f"| {label} | {summary['top1']}/{total} | {summary['top3']}/{total} | "
            f"{summary['top5']}/{total} | {summary['avg_score']:.4f} |"
        )


def print_category_stats(category_stats):
    print("\n카테고리별 (VLM 30개 기준):")
    ordered_categories = [
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
    for category in ordered_categories:
        stats = category_stats.get(category)
        if not stats:
            continue
        print(
            f"- {category}: Top1 {stats['top1']}/{stats['total']}, "
            f"Top3 {stats['top3']}/{stats['total']}, Top5 {stats['top5']}/{stats['total']}, "
            f"Avg {stats['avg_score']:.4f}"
        )


def print_wrong_answers(rows):
    print("\n오답 전체:")
    print("| 쿼리명 | 정답 | 예측 | Score | 원인 |")
    print("|-------|------|------|-------|------|")
    for row in rows:
        print(
            f"| {row['name']} | {row['expected']} | {row['predicted']} | "
            f"{row['score']:.4f} | {row['cause']} |"
        )


def main():
    import torch

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {MODEL_PATH}")

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

    category_stats = aggregate_category_stats(vlm_30_result["details"])
    overfit = compute_overfit_gap(vlm_30_result["details"])
    wrong_answers = build_wrong_answer_rows(
        keyword_result["details"] + vlm_30_result["details"] + new_vlm_result["details"]
    )

    result = {
        "model": MODEL_NAME,
        "embedding_mode": "content_full_embedding",
        "tests": {
            "keyword_10": keyword_result,
            "vlm_30": vlm_30_result,
            "new_vlm_20": new_vlm_result,
        },
        "vlm_30_category_stats": category_stats,
        "overfit": overfit,
        "wrong_answers": wrong_answers,
    }

    print_summary_table(keyword_result, vlm_30_result, new_vlm_result)
    print_category_stats(category_stats)
    print("\n과적합 판정:")
    print(
        f"A그룹 Avg Score {overfit['group_a_avg_score']:.4f} vs "
        f"C그룹 Avg Score {overfit['group_c_avg_score']:.4f} "
        f"(차이 {overfit['avg_score_gap']:+.4f})"
    )
    print_wrong_answers(wrong_answers)

    save_json(result, OUTPUT_JSON)
    print(f"\n결과 JSON 저장: {OUTPUT_JSON}")

    del model, chunk_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
