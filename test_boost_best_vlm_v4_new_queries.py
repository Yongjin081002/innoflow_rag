"""boost_best_vlm_v4: 새 VLM 질의 20개 기준 전용 평가 스크립트"""
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "boost_best_vlm_v4"
OUTPUT_JSON = os.path.join(BASE_DIR, "boost_best_vlm_v4_new_queries_result.json")


def load_new_vlm_queries():
    from test_vlm_new_queries import NEW_VLM_QUERIES

    return NEW_VLM_QUERIES


def load_chunks():
    with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_ids = [c["id"] for c in chunks]
    chunk_contents = [c["content"] for c in chunks]
    return chunk_ids, chunk_contents


def compute_summary(details):
    total = len(details)
    top1 = sum(1 for item in details if item["rank"] == 1)
    top3 = sum(1 for item in details if item["rank"] <= 3)
    top5 = sum(1 for item in details if item["rank"] <= 5)
    avg_score = (sum(item["score"] for item in details) / total) if total else 0.0
    return {
        "total": total,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "avg_score": avg_score,
    }


def build_table_rows(details):
    rows = []
    for index, item in enumerate(details, 1):
        if item["rank"] == 1:
            judgment = "O"
        elif item["rank"] <= 3:
            judgment = "T3"
        elif item["rank"] <= 5:
            judgment = "T5"
        else:
            judgment = "X"

        rows.append(
            {
                "index": index,
                "name": item["name"],
                "expected": item["expected"],
                "top1": item["top1"],
                "rank": item["rank"],
                "score": item["score"],
                "judgment": judgment,
            }
        )
    return rows


def evaluate_model(model_path):
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    queries = load_new_vlm_queries()
    chunk_ids, chunk_contents = load_chunks()

    print(f"{MODEL_NAME} 로딩 중...")
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 512

    chunk_embeddings = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    details = []

    for item in queries:
        query_embedding = model.encode([item["query"]], convert_to_tensor=True, show_progress_bar=False)
        similarities = cos_sim(query_embedding, chunk_embeddings)[0]
        sorted_indices = torch.argsort(similarities, descending=True)

        top5_ids = [chunk_ids[i] for i in sorted_indices[:5]]
        top5_scores = [similarities[i].item() for i in sorted_indices[:5]]

        expected = item["expected"]
        expected_index = chunk_ids.index(expected)
        expected_score = similarities[expected_index].item()
        rank = (top5_ids.index(expected) + 1) if expected in top5_ids else 999

        details.append(
            {
                "name": item["name"],
                "query": item["query"],
                "expected": expected,
                "top1": top5_ids[0],
                "rank": rank,
                "score": expected_score,
                "top5": [
                    {"id": chunk_id, "score": round(score, 4)}
                    for chunk_id, score in zip(top5_ids, top5_scores)
                ],
            }
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = compute_summary(details)
    return {
        "model": MODEL_NAME,
        "query_count": len(queries),
        "summary": summary,
        "details": details,
    }


def save_result(result, output_path=OUTPUT_JSON):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def print_report(result):
    rows = build_table_rows(result["details"])
    summary = result["summary"]

    print(f"\n{'=' * 100}")
    print(f"  {MODEL_NAME} - 새 VLM 질의 20개 테스트 결과")
    print(f"{'=' * 100}\n")
    print(f"  {'#':<4} {'테스트명':<40} {'정답':<8} {'Top1':<8} {'Rank':>5} {'Score':>8} {'판정'}")
    print(f"  {'-' * 4} {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 5} {'-' * 8} {'-' * 4}")

    for row in rows:
        print(
            f"  {row['index']:<4} {row['name']:<40} {row['expected']:<8} "
            f"{row['top1']:<8} {row['rank']:>5} {row['score']:>8.4f} {row['judgment']}"
        )

    print(f"\n{'=' * 100}")
    print("  요약")
    print(f"{'=' * 100}\n")
    print(f"  Top1: {summary['top1']}/{summary['total']}")
    print(f"  Top3: {summary['top3']}/{summary['total']}")
    print(f"  Top5: {summary['top5']}/{summary['total']}")
    print(f"  Avg Score: {summary['avg_score']:.4f}")

    wrong = [item for item in result["details"] if item["rank"] != 1]
    print(f"\n{'=' * 100}")
    print(f"  오답 상세 ({len(wrong)}건)")
    print(f"{'=' * 100}\n")

    if not wrong:
        print("  전부 Top1 정답입니다.")
        return

    for item in wrong:
        top5_text = " | ".join(f"{entry['id']}({entry['score']:.4f})" for entry in item["top5"])
        print(
            f"  {item['name']}: 정답={item['expected']} Top1={item['top1']} "
            f"rank={item['rank']} score={item['score']:.4f}"
        )
        print(f"    Top5: {top5_text}")


def main():
    model_path = os.path.join(BASE_DIR, MODEL_NAME)
    result = evaluate_model(model_path)
    print_report(result)
    save_result(result)
    print(f"\n결과 JSON 저장: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
