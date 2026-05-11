"""
inno-flow 전체 학습 모델 스코어링 + 요약 표
- 저장된 result JSON 수집
- 미평가 모델 디렉토리 실시간 평가
- 모델별 요약표, 쿼리별 크로스 모델 비교, Markdown 표 출력

Usage:
  python score_summary.py                 # 캐시 + 실시간 평가
  python score_summary.py --cached-only   # 캐시된 결과만 (GPU 불필요)
  python score_summary.py --no-cache      # 전부 실시간 평가
"""
import json
import os
import sys
import glob as glob_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 테스트 케이스 ──
TEST_CASES = [
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

QUERIES = [tc["query"] for tc in TEST_CASES]
POSITIVES = [tc["positive"] for tc in TEST_CASES]
N = len(TEST_CASES)

# ── 데이터 로드 ──
with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

SKIP_DIRS = {"__pycache__", "checkpoints", ".git"}


def collect_cached_results():
    """*_result.json 파일 전부 수집"""
    results = []
    for fpath in sorted(glob_mod.glob(os.path.join(BASE_DIR, "*_result.json"))):
        fname = os.path.basename(fpath)
        name = fname.replace("_result.json", "")
        with open(fpath) as f:
            data = json.load(f)
        scores = data.get("scores", [])
        results.append({
            "name": name,
            "top1": data.get("top1", 0),
            "top3": data.get("top3", None),
            "top5": data.get("top5", None),
            "avg": data.get("avg", 0),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "scores": scores,
            "config": data.get("config", ""),
            "per_query": [],
            "source": "cached",
        })
    return results


def discover_unevaluated_models(cached_names):
    """result JSON이 없는 모델 디렉토리 탐색"""
    models = []
    for entry in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(full) or entry in SKIP_DIRS:
            continue
        if not os.path.isfile(os.path.join(full, "config.json")):
            continue
        if entry not in cached_names:
            models.append((entry, full))
    return models


def evaluate_model(model_name, model_path):
    """모델 로드 → 임베딩 → Top-K 정확도 + 평균 유사도"""
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim as _cos_sim

    print(f"  [EVAL] {model_name} ...", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        print(f"    GPU OOM → CPU fallback", flush=True)
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 256

    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(QUERIES, convert_to_tensor=True, show_progress_bar=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    top1 = top3 = top5 = 0
    scores = []
    per_query = []

    for i, tc in enumerate(TEST_CASES):
        pos_id = tc["positive"]
        pos_idx = chunk_ids.index(pos_id)

        sim = _cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(sim)

        all_sim = _cos_sim(query_emb[i], chunk_emb)[0]
        sorted_idx = all_sim.argsort(descending=True)
        top_ids = [chunk_ids[idx] for idx in sorted_idx[:5]]
        rank = (top_ids.index(pos_id) + 1) if pos_id in top_ids else 999

        if rank == 1: top1 += 1
        if rank <= 3: top3 += 1
        if rank <= 5: top5 += 1

        per_query.append({
            "query": tc["query"],
            "positive": pos_id,
            "top1_result": top_ids[0],
            "rank": rank,
            "score": sim,
        })

    return {
        "name": model_name,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "avg": sum(scores) / N,
        "min": min(scores),
        "max": max(scores),
        "scores": scores,
        "config": "",
        "per_query": per_query,
        "source": "live",
    }


def print_summary_table(results):
    print("\n")
    print("=" * 115)
    print("  inno-flow 학습 모델 스코어 요약")
    print("=" * 115)
    header = f"{'#':<4} {'모델':<24} {'Top1':>7} {'Top3':>7} {'Top5':>7} {'Avg':>9} {'Min':>9} {'Max':>9} {'Config'}"
    print(header)
    print("-" * 115)

    for i, r in enumerate(results, 1):
        top1_s = f"{r['top1']}/{N}"
        top3_s = f"{r['top3']}/{N}" if isinstance(r['top3'], int) else "  -  "
        top5_s = f"{r['top5']}/{N}" if isinstance(r['top5'], int) else "  -  "
        min_s = f"{r['min']:.4f}" if r.get('min') is not None else "  -  "
        max_s = f"{r['max']:.4f}" if r.get('max') is not None else "  -  "
        cfg = r.get("config", "")

        print(
            f"{i:<4} {r['name']:<24} "
            f"{top1_s:>7} "
            f"{top3_s:>7} "
            f"{top5_s:>7} "
            f"{r['avg']:>9.4f} "
            f"{min_s:>9} "
            f"{max_s:>9}  "
            f"{cfg}"
        )
    print("-" * 115)


def print_cross_model(results):
    scored = [r for r in results if r["scores"]]
    top_models = scored[:min(5, len(scored))]
    if not top_models:
        return

    col_w = 14
    total_w = 30 + col_w * len(top_models)
    print("\n")
    print("=" * total_w)
    print("  쿼리별 Score 비교 (상위 모델)")
    print("=" * total_w)

    hdr = f"{'쿼리':<28}"
    for m in top_models:
        hdr += f"  {m['name'][:12]:>12}"
    print(hdr)
    print("-" * total_w)

    for i, (q, pos) in enumerate(zip(QUERIES, POSITIVES)):
        row = f"{q:<28}"
        for m in top_models:
            if i < len(m["scores"]):
                s = m["scores"][i]
                flag = "*" if s < 0.80 else " "
                row += f"  {s:>11.4f}{flag}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    print("-" * total_w)
    avg_row = f"{'[평균]':<28}"
    for m in top_models:
        avg_row += f"  {m['avg']:>12.4f}"
    print(avg_row)
    print("  (* = score < 0.80)")


def print_best_detail(results):
    live = [r for r in results if r.get("per_query")]
    if not live:
        return
    best = live[0]
    print("\n")
    print("=" * 85)
    print(f"  Best 모델 상세: {best['name']}  (Top1={best['top1']}/{N}, Avg={best['avg']:.4f})")
    print("=" * 85)
    print(f"{'#':<4} {'쿼리':<28} {'정답':<8} {'Top1결과':<8} {'Rank':>5} {'Score':>9} {'Hit':>4}")
    print("-" * 85)
    for j, pq in enumerate(best["per_query"], 1):
        hit = "O" if pq["rank"] == 1 else "X"
        print(
            f"{j:<4} {pq['query']:<28} "
            f"{pq['positive']:<8} "
            f"{pq['top1_result']:<8} "
            f"{pq['rank']:>5} "
            f"{pq['score']:>9.4f} "
            f"{hit:>4}"
        )
    print("-" * 85)


def print_markdown(results):
    print("\n")
    print("=" * 80)
    print("  Markdown 표 (복사용)")
    print("=" * 80)
    print()
    print("| # | 모델 | Top1 | Avg Score | Min | Max | Config |")
    print("|---|------|------|-----------|-----|-----|--------|")
    for i, r in enumerate(results, 1):
        min_s = f"{r['min']:.4f}" if r.get('min') is not None else "-"
        max_s = f"{r['max']:.4f}" if r.get('max') is not None else "-"
        cfg = r.get("config", "")
        print(f"| {i} | {r['name']} | {r['top1']}/{N} | {r['avg']:.4f} | {min_s} | {max_s} | {cfg} |")
    print()


def main():
    only_cached = "--cached-only" in sys.argv
    no_cache = "--no-cache" in sys.argv

    # Step 1: 캐시된 결과 수집
    if no_cache:
        cached_results = []
    else:
        cached_results = collect_cached_results()
        print(f"\n  캐시된 결과: {len(cached_results)}개")

    cached_names = {r["name"] for r in cached_results}

    # Step 2: 미평가 모델 실시간 평가
    live_results = []
    if not only_cached:
        unevaluated = discover_unevaluated_models(cached_names)
        # 원본 bge-m3 추가
        unevaluated.insert(0, ("BAAI/bge-m3 (원본)", "BAAI/bge-m3"))
        print(f"  실시간 평가 대상: {len(unevaluated)}개")

        for name, path in unevaluated:
            res = evaluate_model(name, path)
            live_results.append(res)

            # 결과 자동 저장
            if "원본" not in name:
                out_path = os.path.join(BASE_DIR, f"{name}_result.json")
                if not os.path.isfile(out_path):
                    save_data = {
                        "top1": res["top1"], "avg": res["avg"],
                        "scores": res["scores"],
                        "top3": res["top3"], "top5": res["top5"],
                    }
                    with open(out_path, "w") as f:
                        json.dump(save_data, f, indent=2)
                    print(f"    → 저장: {os.path.basename(out_path)}")

    # 전체 결과 합치기
    all_results = cached_results + live_results
    all_results.sort(key=lambda r: (r["top1"], r["avg"]), reverse=True)

    print(f"\n  총 {len(all_results)}개 모델 결과")

    # 출력
    print_summary_table(all_results)
    print_cross_model(all_results)
    print_best_detail(all_results)
    print_markdown(all_results)


if __name__ == "__main__":
    main()
