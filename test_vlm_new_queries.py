"""
boost_best vs boost_v3b_best: 새로운 VLM 스타일 쿼리 테스트
- 기존 학습/테스트 데이터에 없는 완전히 새로운 서술형 쿼리
- Qwen3.5-27B VLM 출력 스타일 (블랙박스 영상 분석 결과 형식)
"""
import json, os, torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)
chunk_ids = [c["id"] for c in chunks]
chunk_contents = [c["content"] for c in chunks]

# ══════════════════════════════════════════════════════════════
# 새로운 VLM 스타일 쿼리 (학습/테스트 데이터에 없는 표현)
# 실제 블랙박스 영상 분석 VLM 출력처럼 구체적 서술
# ══════════════════════════════════════════════════════════════

NEW_VLM_QUERIES = [
    # ── 신호 교차로 ──
    {
        "name": "N1-녹색직진vs신호위반직진(새표현)",
        "expected": "차1-1",
        "query": "블랙박스 전방 카메라 영상에서 A 차량이 사거리 교차로에 진입하는 시점에 전방 신호등이 녹색으로 확인됩니다. A 차량이 교차로 중앙을 통과하는 순간 좌측에서 B 승용차가 빠른 속도로 진입하여 A 차량 운전석 측 후방 도어를 충격하였습니다. B 차량 진행 방향 신호등은 적색 상태였습니다.",
    },
    {
        "name": "N2-쌍방적색직진(새표현)",
        "expected": "차1-4",
        "query": "CCTV 영상 분석 결과, 양쪽 도로의 신호등 모두 적색인 상태에서 A 차량과 B 차량이 거의 동시에 교차로에 진입하였습니다. A 차량은 남측에서 북측으로, B 차량은 동측에서 서측으로 각각 직진하다 교차로 중앙 부근에서 T자 형태로 측면 충돌이 발생하였습니다.",
    },
    {
        "name": "N3-비보호좌회전(새표현)",
        "expected": "차2-6",
        "query": "오전 출근 시간대 왕복 6차로 교차로에서 발생한 사고입니다. A 택시가 비보호 좌회전 표시가 있는 교차로에서 녹색 신호에 좌회전을 시도하였으나, 맞은편에서 동일한 녹색 신호에 약 55km/h로 직진하던 B 버스와 교차로 내에서 충돌하였습니다. A 택시 운전자가 맞은편 직진 차량의 속도와 거리를 잘못 판단한 것으로 보입니다.",
    },
    {
        "name": "N4-적색직진vs좌회전화살표(새표현)",
        "expected": "차2-1",
        "query": "영상 분석 결과 B 차량 방향의 좌회전 화살표 신호가 점등된 상태에서 B 차량이 교차로를 좌회전하고 있었습니다. 이때 A 차량이 적색 신호를 무시하고 교차로에 직진 진입하면서 좌회전 중인 B 차량의 후면부를 충격하였습니다. A 차량의 명백한 적색 신호 위반 사고입니다.",
    },

    # ── 추돌 ──
    {
        "name": "N5-정차차량후미추돌(새표현)",
        "expected": "차41-1",
        "query": "블랙박스 후방 카메라에 기록된 영상입니다. A 차량이 빨간불 신호 대기로 교차로 앞에 정차해 있는 상태에서, 후방에서 접근하던 B 대형 SUV가 제동 없이 A 차량 후면 범퍼를 강하게 추돌하였습니다. 추돌 충격으로 A 차량이 전방으로 약 3미터 밀려났으며, B 운전자는 휴대전화 사용으로 전방 주시를 하지 못한 것으로 추정됩니다.",
    },
    {
        "name": "N6-3중연쇄추돌(새표현)",
        "expected": "차42-1",
        "query": "비 오는 오후 편도 3차로 간선도로 2차로에서 A 소형차, B 중형차가 순차적으로 정차해 있던 중, 후방의 C 1톤 화물차가 젖은 노면에서 미끄러지며 B 차량 후미를 추돌하였고, 그 충격에 의해 B 차량이 전방의 A 차량을 재추돌하였습니다. 3대 연쇄 추돌 사고입니다.",
    },

    # ── 중앙선 침범 ──
    {
        "name": "N7-추월중중앙선침범(새표현)",
        "expected": "차31-3",
        "query": "왕복 2차로 지방도로에서 A 차량이 전방의 저속 트랙터를 추월하기 위해 중앙선(황색 실선)을 넘어 반대편 차로로 진입하였다가, 마주오던 B 승용차와 정면 충돌하였습니다. A 차량이 시야가 확보되지 않은 상태에서 무리하게 추월을 시도한 것이 사고의 직접적 원인입니다.",
    },
    {
        "name": "N8-중앙선침범정면충돌(새표현)",
        "expected": "차31-1",
        "query": "새벽 시간대 안개가 짙은 편도 1차로 국도에서 B 11톤 화물차가 곡선 구간을 주행하면서 차체가 중앙선을 넘어 반대편 차로로 이탈하였고, 반대편에서 정상 주행 중이던 A 승합차와 정면으로 충돌하였습니다. A 승합차는 충돌 직전 브레이크 흔적이 약 8미터 남아 있으나 회피하지 못하였습니다.",
    },

    # ── 차선변경/끼어들기 ──
    {
        "name": "N9-합류구간끼어들기(새표현)",
        "expected": "차20-2",
        "query": "램프에서 본선으로 합류하는 구간에서 A 차량이 가속 없이 저속으로 본선에 끼어들면서 본선 1차로에서 약 70km/h로 직진하던 B 차량의 우측 앞범퍼와 A 차량의 좌측 뒤 펜더가 접촉하였습니다. A 차량의 합류 시 안전거리 미확보 및 가속 부족이 사고 원인입니다.",
    },

    # ── 비신호 교차로 ──
    {
        "name": "N10-비신호교차로직진vs좌회전(새표현)",
        "expected": "차15-1",
        "query": "주택가 골목의 신호 없는 삼거리에서 B 승합차가 좁은 골목에서 좌회전하여 주 도로로 나오면서, 주 도로 좌측에서 직진하던 A 오토바이와 충돌하였습니다. B 차량이 좌회전 시 주 도로의 교통 상황을 확인하지 않고 급하게 진입한 것이 확인됩니다.",
    },
    {
        "name": "N11-비신호교차로대로vs소로(새표현)",
        "expected": "차12-2",
        "query": "신호등이 없는 교차로에서 넓은 4차로 도로를 직진하던 A 차량과 좁은 2차로 골목에서 직진 진입한 B 차량이 교차로에서 충돌하였습니다. B 차량이 소로에서 대로로 진입하면서 대로의 직진 차량에 진로를 양보하지 않았습니다.",
    },

    # ── 고속도로 ──
    {
        "name": "N12-고속도로합류측면접촉(새표현)",
        "expected": "차43-1",
        "query": "고속도로 IC 진입 램프에서 B 차량이 가속차로를 통해 본선에 합류하는 과정에서, 본선 바깥 차로를 약 100km/h로 주행하던 A 화물차 옆으로 나란히 진행하다가 가속차로 끝 지점에서 A 화물차 우측면과 B 차량 좌측면이 접촉하였습니다. B 차량의 합류 시 본선 차량 확인이 미흡하였습니다.",
    },
    {
        "name": "N13-고속도로추돌(새표현)",
        "expected": "차41-1",
        "query": "고속도로 본선 2차로에서 전방 사고로 인한 정체가 발생하여 A 차량이 감속 후 서행하고 있던 중, 후방에서 약 110km/h로 접근하던 B 차량이 급제동하였으나 미처 정지하지 못하고 A 차량 후미를 추돌하였습니다. 고속 주행 중 안전거리 미확보가 원인입니다.",
    },

    # ── 유턴 ──
    {
        "name": "N14-유턴vs직진(새표현)",
        "expected": "차33-1",
        "query": "왕복 4차로 도로의 유턴 허용 구간에서 A 승용차가 유턴을 개시하여 반대편 차로 2차로까지 진입한 상태에서, 반대편 1차로에서 약 45km/h로 직진하던 B 택시가 A 차량을 피하지 못하고 A 차량 우측 앞문 부분을 충격하였습니다.",
    },

    # ── 주차장 ──
    {
        "name": "N15-주차장후진출차충돌(새표현)",
        "expected": "차51-1",
        "query": "백화점 지상 주차장에서 A 차량이 주차 구획에서 후진으로 빠져나오면서 주차장 통로를 서행하던 B 차량과 충돌하였습니다. A 차량이 후진 시 좌우 확인 없이 빠르게 후진한 것으로 보이며, A 차량 후면부와 B 차량 우측 전면 휀더가 파손되었습니다.",
    },

    # ── 보행자/자전거 ──
    {
        "name": "N16-차도보행자충돌(새표현)",
        "expected": "보27-1",
        "query": "밤 10시경 편도 2차로 시내 도로에서 어두운 옷을 입은 보행자가 보도가 아닌 2차로 차도 위를 보행하고 있었고, 1차로에서 2차로로 차선을 변경하던 B 승용차가 보행자를 미처 발견하지 못하고 충돌한 사고입니다.",
    },
    {
        "name": "N17-자전거vs차량신호교차로(새표현)",
        "expected": "거1-1",
        "query": "자전거 이용자가 자전거도로에서 녹색 신호에 따라 교차로를 직진 통과하던 중, 교차 도로에서 적색 신호를 위반하고 우회전한 승용차와 교차로 모서리 부근에서 충돌하였습니다. 차량 운전자의 신호 위반이 명백합니다.",
    },

    # ── 개문/기타 ──
    {
        "name": "N18-개문사고(새표현)",
        "expected": "차52-1",
        "query": "편도 2차로 도로 갓길에 정차한 A 택시에서 뒷좌석 승객이 하차하면서 후방을 확인하지 않고 문을 열었고, 1차로에서 주행하다 갓길 옆으로 접근하던 B 배달 오토바이가 열린 문에 충돌하였습니다.",
    },

    # ── 횡단보도/우회전 ──
    {
        "name": "N19-우회전시직진차충돌(새표현)",
        "expected": "차5-2",
        "query": "신호 교차로에서 A 대형버스가 우회전을 시도하면서 횡단보도 보행자 신호가 녹색인 것을 무시하고 진행하다가, 교차 도로에서 녹색 신호에 직진하던 B 승용차와 교차로 모서리에서 충돌한 사고입니다. A 버스의 횡단보도 보행자 신호 확인 미흡과 우회전 시 안전 확인 부주의가 원인입니다.",
    },

    # ── 일시정지 위반 ──
    {
        "name": "N20-일시정지위반직진충돌(새표현)",
        "expected": "차7-1",
        "query": "일시정지 표지판이 설치된 이면도로 교차로에서 B 차량이 일시정지를 하지 않고 교차로에 직진 진입하면서, 교차 도로에서 직진하던 A 승용차와 교차로 중앙에서 측면 충돌이 발생하였습니다. B 차량의 일시정지 의무 위반이 확인됩니다.",
    },
]


def test_model(model_path, model_name):
    print(f"\n  [{model_name}] 로딩 중...", flush=True)
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 512

    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)

    top1_ok, top3_ok, top5_ok = 0, 0, 0
    scores = []
    details = []

    for item in NEW_VLM_QUERIES:
        q_emb = model.encode([item["query"]], convert_to_tensor=True, show_progress_bar=False)
        sims = cos_sim(q_emb, chunk_emb)[0]
        sorted_idx = torch.argsort(sims, descending=True)
        top5_ids = [chunk_ids[i] for i in sorted_idx[:5]]
        top5_scores = [sims[i].item() for i in sorted_idx[:5]]

        exp = item["expected"]
        exp_idx = chunk_ids.index(exp)
        exp_score = sims[exp_idx].item()
        scores.append(exp_score)

        rank = (top5_ids.index(exp) + 1) if exp in top5_ids else 999
        if rank == 1: top1_ok += 1
        if rank <= 3: top3_ok += 1
        if rank <= 5: top5_ok += 1

        details.append({
            "name": item["name"],
            "expected": exp,
            "rank": rank,
            "score": exp_score,
            "top3": list(zip(top5_ids[:3], [f"{s:.4f}" for s in top5_scores[:3]])),
        })

    n = len(NEW_VLM_QUERIES)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "top1": top1_ok, "top3": top3_ok, "top5": top5_ok,
        "total": n, "avg": sum(scores) / n,
        "details": details, "scores": scores,
    }


def main():
    models = [
        ("boost_best", os.path.join(BASE_DIR, "boost_best")),
        ("boost_v3b_best", os.path.join(BASE_DIR, "boost_v3b_best")),
    ]

    # VLM v3 학습 후 모델도 있으면 추가
    vlm_v3_path = os.path.join(BASE_DIR, "boost_best_vlm_v3")
    if os.path.isdir(vlm_v3_path) and os.path.exists(os.path.join(vlm_v3_path, "config.json")):
        models.append(("boost_best_vlm_v3", vlm_v3_path))

    print("=" * 100)
    print("  새로운 VLM 스타일 쿼리 테스트 (학습/테스트 데이터에 없는 표현)")
    print(f"  쿼리 수: {len(NEW_VLM_QUERIES)}개")
    print("=" * 100)

    all_results = {}
    for name, path in models:
        if not os.path.isdir(path):
            print(f"  {name} 디렉토리 없음 — 스킵")
            continue
        all_results[name] = test_model(path, name)
        print(f"  [{name}] 완료")

    # ══════════════════════════════════════════════════════
    # 종합 비교표
    # ══════════════════════════════════════════════════════
    model_names = list(all_results.keys())
    cw = 20

    print(f"\n{'=' * 100}")
    print("  종합 비교 결과")
    print(f"{'=' * 100}\n")

    print(f"{'지표':<25}", end="")
    for mn in model_names:
        print(f"  {mn:>{cw}}", end="")
    print()
    print("-" * (25 + (cw + 2) * len(model_names)))

    n = len(NEW_VLM_QUERIES)
    for metric, key in [("Top1", "top1"), ("Top3", "top3"), ("Top5", "top5")]:
        print(f"{f'VLM {metric} (/{n})':25}", end="")
        for mn in model_names:
            r = all_results[mn]
            pct = r[key] / n * 100
            print(f"  {r[key]:>{cw-7}}/{n} ({pct:.0f}%)", end="")
        print()

    print(f"{'VLM Avg Score':<25}", end="")
    for mn in model_names:
        print(f"  {all_results[mn]['avg']:>{cw}.4f}", end="")
    print()

    print("-" * (25 + (cw + 2) * len(model_names)))

    # ══════════════════════════════════════════════════════
    # 쿼리별 상세
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("  쿼리별 상세 결과")
    print(f"{'=' * 100}")

    for mn in model_names:
        r = all_results[mn]
        print(f"\n  [{mn}] Top1: {r['top1']}/{r['total']} | Top3: {r['top3']}/{r['total']} | Avg: {r['avg']:.4f}")
        print(f"  {'이름':<42} {'정답':>6} {'순위':>4} {'점수':>7}  Top3 결과")
        print("  " + "-" * 95)

        for d in r["details"]:
            mark = "O" if d["rank"] == 1 else ("△" if d["rank"] <= 3 else "X")
            top3_str = " | ".join([f"{tid}({ts})" for tid, ts in d["top3"]])
            print(f"  {d['name']:<42} {d['expected']:>6} {d['rank']:>4} {d['score']:>7.4f}  {mark} {top3_str}")

    # ══════════════════════════════════════════════════════
    # 오답 비교
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("  모델별 오답 비교")
    print(f"{'=' * 100}")

    for mn in model_names:
        wrong = [d for d in all_results[mn]["details"] if d["rank"] != 1]
        if not wrong:
            print(f"\n  [{mn}] 전부 정답!")
            continue
        print(f"\n  [{mn}] 오답 {len(wrong)}건:")
        for d in wrong:
            top3_str = " | ".join([f"{tid}({ts})" for tid, ts in d["top3"]])
            print(f"    {d['name']}: 정답={d['expected']} 순위={d['rank']} 점수={d['score']:.4f}")
            print(f"      Top3: {top3_str}")

    print(f"\n{'=' * 100}")
    print("  테스트 완료!")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
