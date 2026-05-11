"""
inno-flow 학습 모델 종합 스코어링
- 캐시된 result JSON 수집 + 현재 배포 모델(bge-m3-finetuned) 실시간 평가
- 모델별 요약표, 쿼리별 상세, 실제 사용 시나리오 테스트
- Markdown 표 출력

Usage:
  python score_models.py                 # 캐시 + 실시간 평가
  python score_models.py --cached-only   # 캐시된 결과만 (GPU 불필요)
"""
import json
import os
import sys
import glob as glob_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 기본 테스트 케이스 (10개) ──
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

# ── 실제 사용 시나리오 쿼리 (VLM 출력 → RAG 검색 시뮬레이션) ──
SCENARIO_QUERIES = [
    {
        "scenario": "블랙박스: 교차로에서 파란불에 직진하던 중 빨간불 무시하고 들어온 차와 충돌",
        "query": "교차로에서 녹색신호 직진 중 적색신호 위반 차량과 충돌한 사고",
        "expected": "차1-1",
        "description": "녹색 직진 vs 적색 직진 (A0:B100)",
    },
    {
        "scenario": "블랙박스: 비보호 좌회전 중 맞은편 직진 차량과 충돌",
        "query": "비보호 좌회전하다 맞은편에서 직진하는 차량과 충돌한 사고",
        "expected": "차2-6",
        "description": "비보호 좌회전 vs 녹색 직진 (A90:B10)",
    },
    {
        "scenario": "블랙박스: 신호 없는 교차로에서 좌회전하다 직진 차량과 충돌",
        "query": "신호가 없는 교차로에서 좌회전하다 직진 차량과 충돌한 사고",
        "expected": "차15-1",
        "description": "비신호교차로 직진 vs 좌회전 (A30:B70)",
    },
    {
        "scenario": "블랙박스: 앞차가 급정거해서 뒤에서 추돌한 사고",
        "query": "앞차가 갑자기 정지해서 뒤에서 추돌한 사고 과실비율",
        "expected": "차41-1",
        "description": "추돌 사고 (A0:B100)",
    },
    {
        "scenario": "블랙박스: 중앙선을 넘어온 차량과 정면충돌",
        "query": "중앙선을 침범하여 반대편 차선의 차량과 정면충돌한 사고",
        "expected": "차31-1",
        "description": "중앙선 침범 충돌 (A0:B100)",
    },
    {
        "scenario": "블랙박스: 고속도로 합류 지점에서 본선 차량과 충돌",
        "query": "고속도로 합류차선에서 본선으로 진입하다 본선 차량과 충돌한 사고",
        "expected": "차43-1",
        "description": "고속도로 본선 vs 합류 (A40:B60)",
    },
    {
        "scenario": "블랙박스: 유턴하던 차량이 직진 차량과 충돌",
        "query": "유턴하던 차량이 직진하는 차량과 충돌한 사고",
        "expected": "차33-1",
        "description": "유턴 vs 직진 충돌",
    },
    {
        "scenario": "블랙박스: 주차장에서 출차하다 통행 차량과 충돌",
        "query": "주차장에서 후진하여 출차하다가 통행 중인 차량과 충돌한 사고",
        "expected": "차51-1",
        "description": "주차장 출차 충돌",
    },
    {
        "scenario": "블랙박스: 횡단보도에서 보행자 신호일 때 우회전하다 직진차와 충돌",
        "query": "횡단보도에서 보행자 신호일 때 우회전하다 직진 차량과 충돌한 사고",
        "expected": "차5-2",
        "description": "횡단보도 보행자신호 우회전 vs 녹색 직진 (A100:B0)",
    },
    {
        "scenario": "블랙박스: 끼어들기하다 옆 차선 차량과 충돌",
        "query": "차선 변경하며 끼어들다가 옆 차선 차량과 충돌한 사고",
        "expected": "차20-2",
        "description": "끼어들기 충돌",
    },
    {
        "scenario": "블랙박스: 야간에 교차로에서 양쪽 차량이 충돌",
        "query": "야간에 신호가 없는 교차로에서 양쪽에서 진입한 차량이 충돌한 사고",
        "expected": "차12-1",
        "description": "야간 교차로 동시진입 충돌",
    },
    {
        "scenario": "블랙박스: 노란불에 진입한 차와 빨간불에 진입한 차가 충돌",
        "query": "황색신호에 교차로 진입한 차량과 적색신호에 진입한 차량이 충돌한 사고",
        "expected": "차1-3",
        "description": "황색 직진 vs 적색 직진 (A30:B70)",
    },
    {
        "scenario": "블랙박스: 둘 다 빨간불인데 양쪽에서 직진하다 충돌",
        "query": "양쪽 차량 모두 적색신호를 위반하여 직진하다 교차로에서 충돌한 사고",
        "expected": "차1-4",
        "description": "적색 직진 vs 적색 직진 (A50:B50)",
    },
    {
        "scenario": "블랙박스: 파란불 직진 중 불법 좌회전 차량과 충돌",
        "query": "녹색 신호에 직진하다 신호위반 좌회전 차량과 충돌한 사고",
        "expected": "차2-2",
        "description": "녹색 직진 vs 신호위반 좌회전 (A0:B100)",
    },
    {
        "scenario": "블랙박스: 보행자가 차도를 걷다가 차에 치인 사고",
        "query": "보행자가 차도를 걷다가 주행 차량에 충돌당한 사고",
        "expected": "보27-1",
        "description": "차도 보행 중 차량 충돌 (보행자 과실 0)",
    },
]

QUERIES = [tc["query"] for tc in TEST_CASES]
POSITIVES = [tc["positive"] for tc in TEST_CASES]
N = len(TEST_CASES)

# ── 데이터 로드 ──
with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]
chunk_dict = {c["id"]: c["content"] for c in chunks}


def collect_cached_results():
    """*_result.json 파일 수집 (tmp_*, boost_* 등 학습 라운드별 결과)"""
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
            "avg": data.get("avg", 0),
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "scores": scores,
            "config": data.get("config", ""),
        })
    return results


def evaluate_live(model_path, model_name="live"):
    """모델 실시간 평가 (기본 10개 테스트 + 시나리오 쿼리)"""
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    print(f"  [EVAL] {model_name} 로딩 중...", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        print(f"    GPU OOM → CPU fallback", flush=True)
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 256

    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)

    # ── 1) 기본 10개 테스트 ──
    query_emb = model.encode(QUERIES, convert_to_tensor=True, show_progress_bar=False)
    top1_ok = 0
    scores = []
    per_query = []

    for i, tc in enumerate(TEST_CASES):
        pos_id = tc["positive"]
        pos_idx = chunk_ids.index(pos_id)
        sim = cos_sim(query_emb[i], chunk_emb[pos_idx]).item()
        scores.append(sim)

        all_sim = cos_sim(query_emb[i], chunk_emb)[0]
        sorted_idx = all_sim.argsort(descending=True)
        top_ids = [chunk_ids[idx] for idx in sorted_idx[:5]]
        rank = (top_ids.index(pos_id) + 1) if pos_id in top_ids else 999
        if rank == 1:
            top1_ok += 1

        per_query.append({
            "query": tc["query"],
            "positive": pos_id,
            "top1_result": top_ids[0],
            "rank": rank,
            "score": sim,
            "top5": top_ids,
        })

    # ── 2) 시나리오 쿼리 테스트 ──
    scenario_queries = [sq["query"] for sq in SCENARIO_QUERIES]
    scenario_emb = model.encode(scenario_queries, convert_to_tensor=True, show_progress_bar=False)
    scenario_results = []

    for i, sq in enumerate(SCENARIO_QUERIES):
        expected = sq["expected"]
        exp_idx = chunk_ids.index(expected)
        sim = cos_sim(scenario_emb[i], chunk_emb[exp_idx]).item()

        all_sim = cos_sim(scenario_emb[i], chunk_emb)[0]
        sorted_idx = all_sim.argsort(descending=True)
        top_ids = [chunk_ids[idx] for idx in sorted_idx[:5]]
        top_scores = [all_sim[idx].item() for idx in sorted_idx[:5]]
        rank = (top_ids.index(expected) + 1) if expected in top_ids else 999

        scenario_results.append({
            "scenario": sq["scenario"],
            "query": sq["query"],
            "expected": expected,
            "description": sq["description"],
            "score": sim,
            "rank": rank,
            "top1_result": top_ids[0],
            "top1_score": top_scores[0],
            "top5": list(zip(top_ids, top_scores)),
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": model_name,
        "top1": top1_ok,
        "avg": sum(scores) / N,
        "min": min(scores),
        "max": max(scores),
        "scores": scores,
        "per_query": per_query,
        "scenario_results": scenario_results,
    }


def print_divider(char="=", width=120):
    print(char * width)


def print_model_summary(results):
    """모델별 요약표"""
    print()
    print_divider()
    print("  [1] 학습 모델 스코어 요약 (avg score 내림차순)")
    print_divider()
    header = f"{'#':<4} {'모델':<28} {'Top1':>7} {'Avg':>9} {'Min':>9} {'Max':>9}  {'Config'}"
    print(header)
    print_divider("-")
    for i, r in enumerate(results, 1):
        top1_s = f"{r['top1']}/{N}"
        print(
            f"{i:<4} {r['name']:<28} "
            f"{top1_s:>7} "
            f"{r['avg']:>9.4f} "
            f"{r['min']:>9.4f} "
            f"{r['max']:>9.4f}  "
            f"{r.get('config', '')}"
        )
    print_divider("-")


def print_query_detail(live_result):
    """기본 10개 쿼리별 상세"""
    print()
    print_divider()
    name = live_result["name"]
    print(f"  [2] 쿼리별 상세 — {name} (Top1={live_result['top1']}/{N}, Avg={live_result['avg']:.4f})")
    print_divider()
    print(f"{'#':<4} {'쿼리':<28} {'정답':<8} {'Top1결과':<8} {'Rank':>5} {'Score':>9} {'Hit':>4}")
    print_divider("-")
    for j, pq in enumerate(live_result["per_query"], 1):
        hit = "O" if pq["rank"] == 1 else "X"
        flag = " *" if pq["score"] < 0.85 else ""
        print(
            f"{j:<4} {pq['query']:<28} "
            f"{pq['positive']:<8} "
            f"{pq['top1_result']:<8} "
            f"{pq['rank']:>5} "
            f"{pq['score']:>9.4f} "
            f"{hit:>4}{flag}"
        )
    print_divider("-")
    print("  (* = score < 0.85)")


def print_scenario_test(scenario_results):
    """실제 시나리오 쿼리 테스트 결과"""
    print()
    print_divider()
    print("  [3] 실제 사용 시나리오 테스트 (VLM 출력 → RAG 검색 시뮬레이션)")
    print_divider()

    hit_count = sum(1 for sr in scenario_results if sr["rank"] == 1)
    total = len(scenario_results)
    avg_score = sum(sr["score"] for sr in scenario_results) / total

    print(f"  Top1 정확도: {hit_count}/{total} ({hit_count/total*100:.1f}%)")
    print(f"  평균 Score: {avg_score:.4f}")
    print()

    print(f"{'#':<4} {'시나리오 (축약)':<44} {'기대':<8} {'결과':<8} {'Score':>9} {'Hit':>4}")
    print_divider("-")
    for j, sr in enumerate(scenario_results, 1):
        # 시나리오 축약 (40자)
        desc = sr["description"]
        if len(desc) > 40:
            desc = desc[:38] + ".."
        hit = "O" if sr["rank"] == 1 else "X"
        flag = " *" if sr["score"] < 0.80 else ""
        print(
            f"{j:<4} {desc:<44} "
            f"{sr['expected']:<8} "
            f"{sr['top1_result']:<8} "
            f"{sr['score']:>9.4f} "
            f"{hit:>4}{flag}"
        )
    print_divider("-")
    print("  (* = score < 0.80)")

    # 틀린 쿼리 상세
    wrong = [sr for sr in scenario_results if sr["rank"] != 1]
    if wrong:
        print()
        print("  [틀린 쿼리 상세]")
        for sr in wrong:
            print(f"  쿼리: {sr['query']}")
            print(f"  기대: {sr['expected']}  |  실제 Top1: {sr['top1_result']} (score={sr['top1_score']:.4f})")
            print(f"  Top5: {', '.join(f'{cid}({s:.4f})' for cid, s in sr['top5'])}")
            print()


def print_scenario_detail(scenario_results):
    """시나리오별 Top-5 검색 결과 상세"""
    print()
    print_divider()
    print("  [4] 시나리오별 Top-5 검색 결과 상세")
    print_divider()

    for j, sr in enumerate(scenario_results, 1):
        hit = "O" if sr["rank"] == 1 else "X"
        print(f"\n  [{j}] {sr['scenario']}")
        print(f"      쿼리: {sr['query']}")
        print(f"      기대 청크: {sr['expected']} ({sr['description']})")
        print(f"      정답 Score: {sr['score']:.4f}  |  Hit: {hit}  |  Rank: {sr['rank']}")
        print(f"      Top-5:")
        for rank, (cid, score) in enumerate(sr["top5"], 1):
            marker = " ◀ 정답" if cid == sr["expected"] else ""
            # 청크 내용 첫 줄
            content = chunk_dict.get(cid, "")
            first_line = content.split("\n")[1] if "\n" in content else content[:50]
            print(f"        {rank}. {cid} ({score:.4f}) — {first_line}{marker}")


def print_scenario_comparison(live_results):
    """여러 모델의 시나리오 테스트 비교"""
    print()
    print_divider()
    print("  [5] 시나리오 테스트 모델 간 비교 (일반화 성능)")
    print_divider()

    # 요약 행
    header = f"{'지표':<24}"
    for lr in live_results:
        header += f"  {lr['name'][:16]:>16}"
    print(header)
    print_divider("-")

    # 기본 테스트 Top1
    row = f"{'기본 10개 Top1':<24}"
    for lr in live_results:
        row += f"  {lr['top1']:>14}/{N}"
    print(row)

    # 기본 테스트 Avg
    row = f"{'기본 10개 Avg':<24}"
    for lr in live_results:
        row += f"  {lr['avg']:>16.4f}"
    print(row)

    # 시나리오 Top1
    row = f"{'시나리오 Top1':<24}"
    for lr in live_results:
        sr = lr["scenario_results"]
        hit = sum(1 for s in sr if s["rank"] == 1)
        row += f"  {hit:>13}/{len(sr)}"
    print(row)

    # 시나리오 Avg
    row = f"{'시나리오 Avg Score':<24}"
    for lr in live_results:
        sr = lr["scenario_results"]
        avg = sum(s["score"] for s in sr) / len(sr)
        row += f"  {avg:>16.4f}"
    print(row)

    # 과적합 지표: 기본Avg - 시나리오Avg (클수록 과적합 의심)
    row = f"{'기본-시나리오 Gap':<24}"
    for lr in live_results:
        sr = lr["scenario_results"]
        sc_avg = sum(s["score"] for s in sr) / len(sr)
        gap = lr["avg"] - sc_avg
        row += f"  {gap:>+16.4f}"
    print(row)

    print_divider("-")

    # 쿼리별 비교
    print()
    n_sc = len(SCENARIO_QUERIES)
    col_w = 18
    header2 = f"{'#':<4} {'시나리오':<32} {'기대':<8}"
    for lr in live_results:
        header2 += f"  {lr['name'][:14]:>14}"
    print(header2)
    print_divider("-", 48 + col_w * len(live_results))

    for j in range(n_sc):
        sq = SCENARIO_QUERIES[j]
        desc = sq["description"][:28]
        row = f"{j+1:<4} {desc:<32} {sq['expected']:<8}"
        for lr in live_results:
            sr = lr["scenario_results"][j]
            hit = "O" if sr["rank"] == 1 else "X"
            row += f"  {sr['score']:>10.4f} {hit:>2}"
        print(row)

    print_divider("-", 48 + col_w * len(live_results))


def print_markdown_scenario_comparison(live_results):
    """시나리오 비교 Markdown"""
    print()
    print("### 시나리오 테스트 모델 비교 (일반화 성능 — 과적합 검증)")
    print()

    # 요약
    print("| 지표 |", end="")
    for lr in live_results:
        print(f" {lr['name']} |", end="")
    print()
    print("|------|" + "------|" * len(live_results))

    row = "| 기본 10개 Top1 |"
    for lr in live_results:
        row += f" {lr['top1']}/{N} |"
    print(row)

    row = "| 기본 10개 Avg |"
    for lr in live_results:
        row += f" {lr['avg']:.4f} |"
    print(row)

    row = "| 시나리오 Top1 |"
    for lr in live_results:
        sr = lr["scenario_results"]
        hit = sum(1 for s in sr if s["rank"] == 1)
        row += f" {hit}/{len(sr)} |"
    print(row)

    row = "| 시나리오 Avg |"
    for lr in live_results:
        sr = lr["scenario_results"]
        avg = sum(s["score"] for s in sr) / len(sr)
        row += f" {avg:.4f} |"
    print(row)

    row = "| 기본-시나리오 Gap |"
    for lr in live_results:
        sr = lr["scenario_results"]
        sc_avg = sum(s["score"] for s in sr) / len(sr)
        gap = lr["avg"] - sc_avg
        row += f" {gap:+.4f} |"
    print(row)

    print()


def print_markdown_summary(results):
    """Markdown 표 (복사용)"""
    print()
    print_divider()
    print("  [5] Markdown 표 (복사용)")
    print_divider()
    print()
    print("### 모델별 스코어 요약")
    print()
    print("| # | 모델 | Top1 | Avg Score | Min | Max | Config |")
    print("|---|------|------|-----------|-----|-----|--------|")
    for i, r in enumerate(results, 1):
        cfg = r.get("config", "")
        print(f"| {i} | {r['name']} | {r['top1']}/{N} | {r['avg']:.4f} | {r['min']:.4f} | {r['max']:.4f} | {cfg} |")
    print()


def print_markdown_scenario(scenario_results):
    """시나리오 테스트 Markdown"""
    hit_count = sum(1 for sr in scenario_results if sr["rank"] == 1)
    total = len(scenario_results)
    avg_score = sum(sr["score"] for sr in scenario_results) / total

    print("### 실제 사용 시나리오 테스트")
    print()
    print(f"**Top1 정확도: {hit_count}/{total} ({hit_count/total*100:.1f}%)  |  평균 Score: {avg_score:.4f}**")
    print()
    print("| # | 시나리오 | 기대 청크 | 검색 결과 | Score | Hit |")
    print("|---|---------|----------|----------|-------|-----|")
    for j, sr in enumerate(scenario_results, 1):
        hit = "O" if sr["rank"] == 1 else "X"
        print(f"| {j} | {sr['description']} | {sr['expected']} | {sr['top1_result']} | {sr['score']:.4f} | {hit} |")
    print()


def main():
    only_cached = "--cached-only" in sys.argv

    print()
    print_divider("=")
    print("  inno-flow 학습 모델 종합 스코어링")
    print_divider("=")

    # Step 1: 캐시된 결과 수집
    cached = collect_cached_results()
    print(f"  캐시된 result JSON: {len(cached)}개")

    # Step 2: 실시간 평가 (bge-m3-finetuned + 주요 모델들)
    live_results = []
    if not only_cached:
        eval_targets = [
            ("bge-m3-finetuned (현재)", os.path.join(BASE_DIR, "bge-m3-finetuned")),
            ("boost_best", os.path.join(BASE_DIR, "boost_best")),
            ("boost_09_best", os.path.join(BASE_DIR, "boost_09_best")),
            ("boost_v3b_best", os.path.join(BASE_DIR, "boost_v3b_best")),
        ]
        for name, path in eval_targets:
            if os.path.isdir(path):
                res = evaluate_live(path, name)
                live_results.append(res)
                print(f"  {name}: Top1={res['top1']}/{N}, Avg={res['avg']:.4f}")
    live_result = live_results[0] if live_results else None

    # 전체 결과 (중복 제거 + 정렬)
    seen = set()
    all_results = []
    if live_result:
        all_results.append(live_result)
        seen.add("bge-m3-finetuned")
    for r in cached:
        # bge-m3-finetuned 중복 방지, tmp 제외
        base_name = r["name"].replace("_result", "")
        if base_name in seen:
            continue
        # tmp 모델 결과 제외 (너무 많음)
        if r["name"].startswith("boost_") and "_tmp" in r["name"]:
            continue
        seen.add(base_name)
        all_results.append(r)

    all_results.sort(key=lambda r: (r["top1"], r["avg"]), reverse=True)

    # 출력
    print_model_summary(all_results)

    if live_result:
        print_query_detail(live_result)
        print_scenario_test(live_result["scenario_results"])
        print_scenario_detail(live_result["scenario_results"])

    # 시나리오 비교표 (여러 모델)
    if len(live_results) > 1:
        print_scenario_comparison(live_results)

    print_markdown_summary(all_results)
    if live_result:
        print_markdown_scenario(live_result["scenario_results"])
    if len(live_results) > 1:
        print_markdown_scenario_comparison(live_results)


if __name__ == "__main__":
    main()
