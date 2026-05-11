"""
boost_best vs boost_v3b_best 비교 평가
- VLM 출력 스타일 긴 문장 쿼리로 일반화 성능 테스트
- 과적합 검증 (그룹 A/B/C 성능 차이)
- 키워드 X, 실제 VLM 출력처럼 긴 서술형 쿼리 사용

Usage:
  python compare_boost_v3.py
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

# ══════════════════════════════════════════════════════════════
# VLM 스타일 긴 문장 쿼리 (키워드 형식 절대 금지)
# 실제 Qwen3.5-27B VLM이 산출하는 사고 분석 텍스트와 동일한 형태
# ══════════════════════════════════════════════════════════════

# ── 그룹 A: 학습 많이 된 chunk (6~30 pairs) ──
GROUP_A = [
    {
        "name": "A1-녹색직진vs적색직진",
        "expected": "차1-1",
        "train_count": 22,
        "query": "A 차량이 녹색 신호에 따라 정상적으로 교차로를 직진하던 중, B 차량이 적색 신호를 무시하고 우측에서 교차로에 진입하여 A 차량의 조수석 측면을 충격하였습니다. 양 차량 모두 교차로 중앙에 정지하였으며, A 차량 우측 문짝 파손, B 차량 전면부 파손 확인됩니다. B 차량의 적색 신호 위반이 명확합니다.",
    },
    {
        "name": "A2-야간비신호교차로동시진입",
        "expected": "차12-1",
        "train_count": 30,
        "query": "야간에 신호기가 설치되지 않은 이면도로 교차로에서 양쪽 도로에서 동시에 진입한 두 차량이 교차로 중앙에서 측면 충돌하였습니다. 가로등 일부만 점등되어 시야가 제한되었고 건물 모서리로 좌측 확인이 어려운 환경이었습니다. 양쪽 차량 모두 서행하지 않은 것으로 보이며, A 차량 좌측 전면부와 B 차량 우측 측면이 파손되었습니다.",
    },
    {
        "name": "A3-선행차정지후추돌",
        "expected": "차41-1",
        "train_count": 18,
        "query": "편도 3차로 직선 도로에서 선행하던 B 트럭이 전방 교통 정체로 정차하였으나, 후방의 A 차량이 전방 주시를 태만히 하여 약 60km/h의 속도로 B 트럭의 후미를 추돌하였습니다. A 차량의 제동 흔적은 충돌 직전 약 5m 정도만 확인되며, A 차량 전면부가 심하게 파손되고 B 트럭 후미 범퍼가 파손되었습니다.",
    },
    {
        "name": "A4-유턴차량vs반대편직진",
        "expected": "차33-1",
        "train_count": 17,
        "query": "왕복 6차로 도로의 유턴 구역에서 A 차량이 유턴을 하면서 반대편 차로로 진입하던 중, 반대편에서 직진하던 B 차량과 충돌하였습니다. A 차량이 유턴 완료 전에 B 차량의 진행 경로를 차단한 형태이며, A 차량 운전석 측면과 B 차량 전면 우측이 파손되었습니다. A 차량의 유턴 시 안전 미확인이 사고 원인으로 판단됩니다.",
    },
    {
        "name": "A5-차선변경끼어들기접촉",
        "expected": "차20-2",
        "train_count": 9,
        "query": "편도 3차로 도시 도로에서 A 차량이 2차로에서 갑자기 좌측 방향지시등을 켜고 1차로로 진로 변경을 시도하면서 충분한 안전거리를 확보하지 않은 채 끼어들어 1차로에서 정상 속도로 직진 중이던 B 차량의 우측 전면부와 A 차량의 좌측 후면부가 접촉하였습니다. 양 차량 갓길로 이동하여 정차하였습니다.",
    },
    {
        "name": "A6-커브구간중앙선침범정면충돌",
        "expected": "차31-1",
        "train_count": 14,
        "query": "비가 오는 왕복 2차로 국도의 커브 구간에서 B 화물차가 속도를 줄이지 않고 중앙선(황색 실선)을 침범하여 반대 차로의 A 차량과 정면으로 충돌하였습니다. A 차량 운전자가 우측으로 회피를 시도하였으나 미처 피하지 못하였으며, 양 차량 모두 심각한 전면 파손이 발생하여 도로 위에 정지하였습니다.",
    },
    {
        "name": "A7-고속도로가속차로합류측면접촉",
        "expected": "차43-1",
        "train_count": 15,
        "query": "고속도로 합류 구간에서 B 차량이 가속차로에서 본선으로 합류하면서 본선 3차로(가장 바깥 차로)에서 약 90km/h로 주행하던 A 차량과 나란히 진행하게 되었고, 가속차로가 끝나는 지점에서 B 차량이 A 차량의 옆으로 들어오면서 측면 접촉 사고가 발생하였습니다. 양 차량 갓길에 정차하였으며 경미한 측면 스크래치 파손이 확인됩니다.",
    },
    {
        "name": "A8-주차장후진출차통행차량충돌",
        "expected": "차51-1",
        "train_count": 8,
        "query": "대형마트 지하주차장에서 A 미니밴이 주차 공간에서 후진하여 출차하는 과정에서 주차장 통로를 저속으로 주행하던 B 세단의 측면과 충돌하였습니다. A 차량 운전자가 후방 및 좌우 확인을 충분히 하지 않은 것으로 보이며, A 차량 후면 범퍼와 B 차량 좌측 뒷문이 파손되었습니다.",
    },
    {
        "name": "A9-비보호좌회전맞은편직진충돌",
        "expected": "차2-6",
        "train_count": 6,
        "query": "왕복 4차로 교차로에서 비보호 좌회전 구간(별도 좌회전 화살표 없음)에서 오후 퇴근 시간대에 A 세단이 녹색 신호에 비보호 좌회전 대기 후 좌회전을 개시하면서 맞은편에서 녹색 신호에 약 50km/h로 직진하는 B 트럭을 미처 확인하지 못하고 좌회전을 시도하여, B 트럭이 A 차량의 운전석 측면을 충격하였습니다.",
    },
    {
        "name": "A10-비신호교차로골목좌회전vs직진",
        "expected": "차15-1",
        "train_count": 14,
        "query": "신호가 없는 주택가 이면도로 T자 교차로에서 B 차량이 우측 골목에서 좌회전하여 주 도로에 진입하면서, 주 도로를 직진하던 A 경차와 충돌하였습니다. B 차량이 교차로 진입 시 일시정지 및 좌우 확인을 소홀히 한 것으로 보이며, A 차량 전면 좌측과 B 차량 우측 측면이 파손되었습니다.",
    },
]

# ── 그룹 B: 학습 적게 된 chunk (1~3 pairs) ──
GROUP_B = [
    {
        "name": "B1-녹색좌회전진입후황색직진충돌",
        "expected": "차2-3",
        "train_count": 2,
        "query": "왕복 4차로 신호 교차로에서 B 차량이 녹색 신호에 좌회전을 위해 교차로에 진입하여 대기하던 중 신호가 황색으로 변경되었고, 이때 A 차량이 황색 신호에 교차로에 진입하여 직진하면서 좌회전을 완료하려던 B 차량과 충돌하였습니다. A 차량의 황색 신호 진입이 문제로 보입니다.",
    },
    {
        "name": "B2-적색우회전녹색직진교차충돌",
        "expected": "차3-1",
        "train_count": 3,
        "query": "신호 교차로에서 B 차량이 좌회전 화살표 신호에 따라 정상적으로 좌회전을 시작하던 중, A 세단이 적색 신호를 위반한 채 직진으로 교차로에 진입하여 B 차량과 충돌하였습니다. 횡단보도 우회전이 아니라 적색 직진 대 좌회전 화살표 충돌입니다.",
    },
    {
        "name": "B3-동일방향직진우회전교차충돌",
        "expected": "차4-1",
        "train_count": 2,
        "query": "왕복 4차로 교차로에서 A 승용차가 녹색 화살표 좌회전 신호를 받고 좌회전을 하던 중, 맞은편에서 우회전하던 B 트럭이 크게 선회하며 A 차량의 진행 경로를 침범하여 충돌하였습니다. A 차량의 좌회전 신호와 맞은편에서 우회전한 B 차량이 핵심입니다.",
    },
    {
        "name": "B4-급차선변경후방직진접촉",
        "expected": "차20-1",
        "train_count": 3,
        "query": "편도 3차로 일반 도로에서 A 승용차가 2차로에서 1차로로 차선을 변경하는 과정에서 1차로 후방에서 직진하던 B 승용차와 접촉하였습니다. A 차량이 사이드미러 확인 없이 급하게 차선을 변경한 것으로 보이며, 진로 변경 시 안전 확인이 불충분한 것으로 판단됩니다.",
    },
    {
        "name": "B5-일방통행로역주행정면충돌",
        "expected": "차31-2",
        "train_count": 2,
        "query": "주차장 출구처럼 도로가 아닌 장소에서 나오던 B 승용차가 도로로 합류하기 위해 중앙선 침범 좌회전을 시도하였고, 본선을 따라 직진하던 A 승용차와 충돌하였습니다. 일방통행 역주행이 아니라 도로가 아닌 장소에서의 중앙선 침범 진입입니다.",
    },
    {
        "name": "B6-좌회전중후방유턴차충돌",
        "expected": "차33-2",
        "train_count": 2,
        "query": "왕복 4차로 교차로에서 A 승용차가 좌회전을 하던 중, 같은 방향에서 유턴을 하던 B 승용차와 충돌하였습니다. B 차량이 유턴 시 후방의 좌회전 차량을 미처 확인하지 못한 것이 사고 원인으로 보입니다.",
    },
    {
        "name": "B7-정체구간연쇄3중추돌",
        "expected": "차42-1",
        "train_count": 3,
        "query": "편도 2차로 도시 도로의 교통 정체 구간에서 A 승용차와 B SUV가 순서대로 정차해 있던 상황에서, 후방의 C 트럭이 감속하지 못하고 B 차량 후미를 추돌하였고, 그 충격으로 B 차량이 A 차량 후미를 재추돌하는 연쇄 추돌 사고가 발생하였습니다. C 차량의 안전거리 미확보와 전방주시 태만이 원인입니다.",
    },
    {
        "name": "B8-보행자전용도로차량침범충돌",
        "expected": "보29-1",
        "train_count": 1,
        "query": "보행자 전용도로(차량 진입 금지 구역)를 정상적으로 보행하던 A 보행자를 B 승용차가 보행자 전용도로에 불법으로 진입하여 주행하다가 충돌하였습니다. B 차량의 보행자 전용도로 침범 주행이 명확한 위반 행위입니다.",
    },
    {
        "name": "B9-비신호동일폭교차로우측차우선위반",
        "expected": "차11-2",
        "train_count": 2,
        "query": "교차로 바닥 노면표시를 보면 A 차량은 직진·좌회전 병용 차로에서 직진 중이었고, B 차량은 직진 전용 차로 노면표시를 무시한 채 좌회전을 하다가 A 차량과 충돌하였습니다. 우측차 우선이 아니라 노면표시와 직진 전용 차로 위반이 핵심입니다.",
    },
    {
        "name": "B10-고속도로본선차선변경측면접촉",
        "expected": "차43-2",
        "train_count": 3,
        "query": "고속도로 본선 3차로에서 A 세단이 3차로에서 2차로로 차선을 변경하면서 2차로를 주행하던 B SUV의 측면과 접촉하였습니다. 고속 주행 중 충분한 안전거리를 확보하지 않고 차선 변경을 시도한 것이 사고 원인이며, A 차량의 진로 변경 시 안전거리 미확보가 확인됩니다.",
    },
]

# ── 그룹 C: 학습 적음/복합/비전형 시나리오 ──
GROUP_C = [
    {
        "name": "C1-자전거도로역통행정면충돌",
        "expected": "거21-1",
        "train_count": 1,
        "query": "자전거 도로 겸용 보도에서 A 자전거가 역방향으로 주행하면서 정상 방향으로 오던 B 자전거와 정면 충돌하였습니다. A 자전거의 자전거 도로 역통행이 사고의 직접적 원인입니다.",
    },
    {
        "name": "C2-자전거전용도로후방추돌",
        "expected": "거31-1",
        "train_count": 1,
        "query": "자전거 전용도로에서 저속으로 주행하던 A 자전거를 후방에서 빠르게 접근하던 B 자전거가 추돌하였습니다. B 자전거의 안전거리 미확보가 사고 원인이며, 전방 자전거와의 속도 차이를 인지하지 못한 것으로 보입니다.",
    },
    {
        "name": "C3-자전거무리한도로횡단충돌",
        "expected": "거41-1",
        "train_count": 1,
        "query": "자전거 횡단도로가 아닌 일반 도로 구간에서 A 자전거가 도로를 횡단하려다 정상 주행하던 B 자전거와 충돌하였습니다. A 자전거의 무리한 도로 횡단이 사고의 원인으로 판단됩니다.",
    },
    {
        "name": "C4-아파트주차장통로교차지점충돌",
        "expected": "차51-2",
        "train_count": 2,
        "query": "아파트 지하주차장에서 A 승용차가 빈 주차 구획으로 진입하기 위해 통로 가장자리에서 속도를 줄이며 선행 주차를 진행하던 중, 뒤따르던 B 승용차가 이를 기다리지 않고 좌측으로 비켜 추월하려다가 A 차량의 측면과 접촉하였습니다. 선행 주차 진행 차량과 후행 추월 차량 사이의 주차장 사고입니다.",
    },
    {
        "name": "C5-야간보도없는도로가장자리보행자충돌",
        "expected": "보27-2",
        "train_count": 1,
        "query": "야간에 보도가 없는 간선도로 가장자리를 보행하던 A 보행자를 B 승용차가 미처 발견하지 못하고 충돌하였습니다. 야간 시야 제한 상황에서 B 차량의 전방주시 태만이 사고 원인이며, 보행자는 차도 가장자리를 보행하고 있었습니다.",
    },
    {
        "name": "C6-자전거급차선변경후방접촉",
        "expected": "거32-1",
        "train_count": 1,
        "query": "2차선 자전거 도로에서 A 자전거가 갑자기 좌측으로 진로를 변경하면서 후방에서 직진하던 B 자전거와 접촉하였습니다. A 자전거의 진로 변경 시 후방 미확인이 사고 원인입니다.",
    },
    {
        "name": "C7-우회전시횡단보도보행자충돌",
        "expected": "차5-2",
        "train_count": 11,
        "query": "신호 교차로에서 A 승합차가 녹색 신호에 우회전을 하면서 횡단보도를 건너고 있던 노인 보행자 B를 미처 확인하지 못하고 충돌하였습니다. 보행자 신호는 녹색이었으며, A 차량 운전자가 우회전에 집중하느라 횡단보도의 보행자를 주시하지 못한 것으로 보이며, 우회전 시 횡단보도 보행자 보호 의무를 위반한 것입니다.",
    },
    {
        "name": "C8-심야점멸신호적색점멸미정지충돌",
        "expected": "차1-5",
        "train_count": 3,
        "query": "심야 시간 점멸 신호가 작동하는 교차로에서 적색 점멸 도로의 A 승용차가 일시정지를 하지 않고 교차로에 진입하여, 황색 점멸 도로에서 직진하던 B 승용차와 충돌하였습니다. A 차량의 적색 점멸 신호에서 일시정지 의무 위반이 사고 원인입니다.",
    },
    {
        "name": "C9-고속도로감속차로진입추돌",
        "expected": "차43-4",
        "train_count": 2,
        "query": "고속도로 진출 구간에서 A 승용차가 본선에서 감속차로로 진입하면서 이미 감속차로를 주행 중이던 B 트럭의 후미를 추돌하였습니다. A 차량의 감속차로 진입 시 안전 확인이 불충분했던 것이 사고 원인입니다.",
    },
    {
        "name": "C10-주간차도보행자차량충돌",
        "expected": "보27-1",
        "train_count": 1,
        "query": "주간 편도 2차로 일반 도로에서 차도 위를 걸어가던 보행자 A를 차도를 주행하던 B 승용차가 충돌하였습니다. 보행자가 보도가 아닌 차도 위를 보행하고 있었으며, B 차량의 전방주시 의무 소홀도 사고에 기여한 것으로 보입니다.",
    },
]


def evaluate_model(model_path: str, model_name: str):
    """모델 평가: 3그룹 VLM 스타일 쿼리 + 기존 키워드 쿼리 비교"""
    print(f"\n  [{model_name}] 로딩 중...", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        print(f"    GPU OOM → CPU fallback", flush=True)
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 512  # VLM 긴 문장 대응

    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)

    all_groups = {
        "A (학습 多, 6~30쌍)": GROUP_A,
        "B (학습 少, 1~3쌍)": GROUP_B,
        "C (학습 극소/복합)": GROUP_C,
    }

    results = {}
    detail_rows = []

    for gname, gdata in all_groups.items():
        top1_ok = 0
        top3_ok = 0
        top5_ok = 0
        scores = []

        for item in gdata:
            q_emb = model.encode([item["query"]], convert_to_tensor=True, show_progress_bar=False)
            sims = cos_sim(q_emb, chunk_emb)[0]
            sorted_idx = torch.argsort(sims, descending=True)
            top_ids = [chunk_ids[i] for i in sorted_idx[:5]]
            top_scores = [sims[i].item() for i in sorted_idx[:5]]

            exp = item["expected"]
            exp_idx = chunk_ids.index(exp)
            exp_score = sims[exp_idx].item()
            scores.append(exp_score)

            rank = (top_ids.index(exp) + 1) if exp in top_ids else 999
            if rank == 1: top1_ok += 1
            if rank <= 3: top3_ok += 1
            if rank <= 5: top5_ok += 1

            detail_rows.append({
                "group": gname[0],  # A, B, C
                "name": item["name"],
                "expected": exp,
                "top1": top_ids[0],
                "rank": rank,
                "score": exp_score,
                "hit": rank == 1,
                "top3": top_ids[:3],
                "top3_scores": top_scores[:3],
                "train_count": item["train_count"],
            })

        n = len(gdata)
        results[gname] = {
            "top1": top1_ok, "top3": top3_ok, "top5": top5_ok,
            "total": n, "avg": sum(scores) / n,
            "min": min(scores), "max": max(scores),
            "scores": scores,
        }

    # 기존 키워드 쿼리 (비교용)
    keyword_queries = [
        ("신호위반 직진 충돌", "차1-1"),
        ("비신호교차로 직진 vs 좌회전", "차15-1"),
        ("추돌 사고 과실", "차41-1"),
        ("야간 교차로 충돌", "차12-1"),
        ("중앙선 침범 충돌", "차31-1"),
        ("끼어들기 충돌", "차20-2"),
        ("유턴 중 충돌", "차33-1"),
        ("고속도로 추돌 사고", "차43-1"),
        ("주차장 출차 중 충돌", "차51-1"),
        ("횡단보도 보행자 충돌", "차5-2"),
    ]
    kw_queries = [q for q, _ in keyword_queries]
    kw_expected = [e for _, e in keyword_queries]
    kw_emb = model.encode(kw_queries, convert_to_tensor=True, show_progress_bar=False)
    kw_sims = cos_sim(kw_emb, chunk_emb)

    kw_top1 = 0
    kw_scores = []
    for i, (q, exp) in enumerate(keyword_queries):
        sorted_idx = torch.argsort(kw_sims[i], descending=True)
        top_id = chunk_ids[sorted_idx[0]]
        exp_idx = chunk_ids.index(exp)
        score = kw_sims[i][exp_idx].item()
        kw_scores.append(score)
        if top_id == exp:
            kw_top1 += 1

    kw_result = {
        "top1": kw_top1, "total": 10,
        "avg": sum(kw_scores) / len(kw_scores),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, detail_rows, kw_result


def main():
    models = [
        ("boost_best", os.path.join(BASE_DIR, "boost_best")),
        ("boost_v3b_best", os.path.join(BASE_DIR, "boost_v3b_best")),
    ]

    all_model_results = {}
    all_detail = {}
    all_kw = {}

    print("=" * 80)
    print("  boost_best vs boost_v3b_best 비교 평가")
    print("  (VLM 스타일 긴 문장 쿼리 / 과적합 검증)")
    print("=" * 80)

    for name, path in models:
        if not os.path.isdir(path):
            print(f"  ⚠ {name} 디렉토리 없음: {path}")
            continue
        results, detail, kw = evaluate_model(path, name)
        all_model_results[name] = results
        all_detail[name] = detail
        all_kw[name] = kw
        print(f"  [{name}] 평가 완료")

    if len(all_model_results) < 2:
        print("  모델이 부족하여 비교 불가")
        return

    # ══════════════════════════════════════════════════════
    # [1] 종합 비교표
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  [1] 종합 비교표 (VLM 스타일 긴 문장 쿼리)")
    print(f"{'=' * 80}\n")

    model_names = list(all_model_results.keys())
    group_names = list(all_model_results[model_names[0]].keys())

    # 표 헤더
    col_w = 20
    print(f"{'지표':<30}", end="")
    for mn in model_names:
        print(f"  {mn:>{col_w}}", end="")
    print()
    print("-" * (30 + (col_w + 2) * len(model_names)))

    # 기존 키워드 테스트
    print(f"{'[키워드] Top1 (10개)':<30}", end="")
    for mn in model_names:
        kw = all_kw[mn]
        print(f"  {kw['top1']:>16}/10", end="")
    print()
    print(f"{'[키워드] Avg Score':<30}", end="")
    for mn in model_names:
        kw = all_kw[mn]
        print(f"  {kw['avg']:>{col_w}.4f}", end="")
    print()
    print("-" * (30 + (col_w + 2) * len(model_names)))

    # VLM 그룹별
    for gname in group_names:
        n = all_model_results[model_names[0]][gname]["total"]
        print(f"{'[VLM] ' + gname[:22] + ' Top1':<30}", end="")
        for mn in model_names:
            r = all_model_results[mn][gname]
            print(f"  {r['top1']:>{col_w-3}}/{n}", end="")
        print()
        print(f"{'[VLM] ' + gname[:22] + ' Top3':<30}", end="")
        for mn in model_names:
            r = all_model_results[mn][gname]
            print(f"  {r['top3']:>{col_w-3}}/{n}", end="")
        print()
        print(f"{'[VLM] ' + gname[:22] + ' Avg':<30}", end="")
        for mn in model_names:
            r = all_model_results[mn][gname]
            print(f"  {r['avg']:>{col_w}.4f}", end="")
        print()

    print("-" * (30 + (col_w + 2) * len(model_names)))

    # VLM 전체
    print(f"{'[VLM] 전체 Top1 (30개)':<30}", end="")
    for mn in model_names:
        total_top1 = sum(r["top1"] for r in all_model_results[mn].values())
        print(f"  {total_top1:>16}/30", end="")
    print()
    print(f"{'[VLM] 전체 Avg Score':<30}", end="")
    for mn in model_names:
        all_scores = []
        for r in all_model_results[mn].values():
            all_scores.extend(r["scores"])
        print(f"  {sum(all_scores)/len(all_scores):>{col_w}.4f}", end="")
    print()

    # ══════════════════════════════════════════════════════
    # [2] 과적합 분석
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  [2] 과적합 분석")
    print(f"{'=' * 80}\n")

    print(f"{'지표':<40}", end="")
    for mn in model_names:
        print(f"  {mn:>{col_w}}", end="")
    print()
    print("-" * (40 + (col_w + 2) * len(model_names)))

    for mn in model_names:
        pass  # header already printed

    # 키워드 vs VLM Gap
    print(f"{'키워드 Avg - VLM 전체 Avg (Gap)':<40}", end="")
    for mn in model_names:
        kw_avg = all_kw[mn]["avg"]
        all_scores = []
        for r in all_model_results[mn].values():
            all_scores.extend(r["scores"])
        vlm_avg = sum(all_scores) / len(all_scores)
        gap = kw_avg - vlm_avg
        print(f"  {gap:>+{col_w}.4f}", end="")
    print()

    # A-B Gap
    print(f"{'A그룹 Avg - B그룹 Avg':<40}", end="")
    for mn in model_names:
        a_avg = all_model_results[mn][group_names[0]]["avg"]
        b_avg = all_model_results[mn][group_names[1]]["avg"]
        print(f"  {a_avg - b_avg:>+{col_w}.4f}", end="")
    print()

    # A-C Gap
    print(f"{'A그룹 Avg - C그룹 Avg':<40}", end="")
    for mn in model_names:
        a_avg = all_model_results[mn][group_names[0]]["avg"]
        c_avg = all_model_results[mn][group_names[2]]["avg"]
        print(f"  {a_avg - c_avg:>+{col_w}.4f}", end="")
    print()

    # A-C Top1 Gap
    print(f"{'A그룹 Top1% - C그룹 Top1%':<40}", end="")
    for mn in model_names:
        a = all_model_results[mn][group_names[0]]
        c = all_model_results[mn][group_names[2]]
        gap = (a["top1"]/a["total"] - c["top1"]/c["total"]) * 100
        print(f"  {gap:>+{col_w-1}.1f}%p", end="")
    print()

    print("-" * (40 + (col_w + 2) * len(model_names)))

    # 판정
    for mn in model_names:
        a_avg = all_model_results[mn][group_names[0]]["avg"]
        c_avg = all_model_results[mn][group_names[2]]["avg"]
        a = all_model_results[mn][group_names[0]]
        c = all_model_results[mn][group_names[2]]
        score_gap = a_avg - c_avg
        rate_gap = a["top1"]/a["total"] - c["top1"]/c["total"]

        if rate_gap > 0.3 or score_gap > 0.10:
            verdict = "⚠ 과적합 의심"
        elif rate_gap > 0.15 or score_gap > 0.05:
            verdict = "△ 약한 과적합"
        else:
            verdict = "✓ 양호"
        print(f"  {mn}: {verdict} (A-C score gap={score_gap:+.4f}, top1 gap={rate_gap*100:+.1f}%p)")

    # ══════════════════════════════════════════════════════
    # [3] 쿼리별 상세 (틀린 것 위주)
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  [3] 모델별 오답 상세")
    print(f"{'=' * 80}")

    for mn in model_names:
        wrong = [d for d in all_detail[mn] if not d["hit"]]
        if not wrong:
            print(f"\n  [{mn}] 전부 정답!")
            continue

        print(f"\n  [{mn}] 오답 {len(wrong)}건:")
        print(f"  {'그룹':<4} {'이름':<36} {'정답':<10} {'Top1':<10} {'Score':>8} {'학습쌍':>6}")
        print("  " + "-" * 78)
        for d in wrong:
            print(f"  {d['group']:<4} {d['name']:<36} {d['expected']:<10} {d['top1']:<10} {d['score']:>8.4f} {d['train_count']:>6}")
            for rank, (tid, ts) in enumerate(zip(d["top3"], d["top3_scores"]), 1):
                marker = " ◀" if tid == d["expected"] else ""
                print(f"       #{rank}: {tid} ({ts:.4f}){marker}")

    # ══════════════════════════════════════════════════════
    # [4] Markdown 표 (복사용)
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  [4] Markdown 표 (복사용)")
    print(f"{'=' * 80}\n")

    print("### boost_best vs boost_v3b_best 종합 비교")
    print()
    print("| 지표 |", " | ".join(model_names), "|")
    print("|------|" + "------|" * len(model_names))

    print(f"| 키워드 Top1 (10개) |", end="")
    for mn in model_names:
        print(f" {all_kw[mn]['top1']}/10 |", end="")
    print()

    print(f"| 키워드 Avg Score |", end="")
    for mn in model_names:
        print(f" {all_kw[mn]['avg']:.4f} |", end="")
    print()

    for gname in group_names:
        short = gname.split("(")[0].strip()
        n = all_model_results[model_names[0]][gname]["total"]
        print(f"| VLM {short} Top1 |", end="")
        for mn in model_names:
            r = all_model_results[mn][gname]
            print(f" {r['top1']}/{n} |", end="")
        print()
        print(f"| VLM {short} Avg |", end="")
        for mn in model_names:
            r = all_model_results[mn][gname]
            print(f" {r['avg']:.4f} |", end="")
        print()

    print(f"| **VLM 전체 Top1** |", end="")
    for mn in model_names:
        total_top1 = sum(r["top1"] for r in all_model_results[mn].values())
        print(f" **{total_top1}/30** |", end="")
    print()

    print(f"| **VLM 전체 Avg** |", end="")
    for mn in model_names:
        all_scores = []
        for r in all_model_results[mn].values():
            all_scores.extend(r["scores"])
        print(f" **{sum(all_scores)/len(all_scores):.4f}** |", end="")
    print()

    print()
    print("### 과적합 지표")
    print()
    print("| 지표 |", " | ".join(model_names), "|")
    print("|------|" + "------|" * len(model_names))

    print(f"| 키워드-VLM Gap |", end="")
    for mn in model_names:
        kw_avg = all_kw[mn]["avg"]
        all_scores = []
        for r in all_model_results[mn].values():
            all_scores.extend(r["scores"])
        vlm_avg = sum(all_scores) / len(all_scores)
        print(f" {kw_avg - vlm_avg:+.4f} |", end="")
    print()

    print(f"| A-C Score Gap |", end="")
    for mn in model_names:
        a_avg = all_model_results[mn][group_names[0]]["avg"]
        c_avg = all_model_results[mn][group_names[2]]["avg"]
        print(f" {a_avg - c_avg:+.4f} |", end="")
    print()

    print(f"| A-C Top1 Gap |", end="")
    for mn in model_names:
        a = all_model_results[mn][group_names[0]]
        c = all_model_results[mn][group_names[2]]
        gap = (a["top1"]/a["total"] - c["top1"]/c["total"]) * 100
        print(f" {gap:+.1f}%p |", end="")
    print()

    for mn in model_names:
        a_avg = all_model_results[mn][group_names[0]]["avg"]
        c_avg = all_model_results[mn][group_names[2]]["avg"]
        a = all_model_results[mn][group_names[0]]
        c = all_model_results[mn][group_names[2]]
        score_gap = a_avg - c_avg
        rate_gap = a["top1"]/a["total"] - c["top1"]/c["total"]
        if rate_gap > 0.3 or score_gap > 0.10:
            verdict = "⚠ 과적합 의심"
        elif rate_gap > 0.15 or score_gap > 0.05:
            verdict = "△ 약한 과적합"
        else:
            verdict = "✓ 양호"
        print(f"\n**{mn}**: {verdict}")

    print()


if __name__ == "__main__":
    main()
