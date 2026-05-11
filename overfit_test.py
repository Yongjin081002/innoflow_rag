"""
과적합 검증 테스트
- VLM(Qwen3.5-27B) 출력 스타일의 테스트 데이터로 모델 일반화 능력 평가
- 3개 그룹으로 나눠서 비교:
  A) 학습 많이 된 chunk (10+ pairs) - 하지만 VLM 스타일 새 쿼리
  B) 학습 적게 된 chunk (1-3 pairs) - VLM 스타일 새 쿼리
  C) 학습 안 된 chunk (0 pairs) - VLM 스타일 새 쿼리
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

# ═══════════════════════════════════════════════════════════
# VLM 스타일 테스트 데이터 (블랙박스 영상 분석 출력 시뮬레이션)
# ═══════════════════════════════════════════════════════════

# ── 그룹 A: 학습 많이 된 chunk (10+ pairs) ──
# 기존 학습 데이터와 완전히 다른 VLM 출력 스타일
GROUP_A_VLM = [
    {
        "name": "A1-신호위반직진",
        "expected": "차1-1",
        "train_count": 22,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 왕복 4차로 신호 교차로
- **신호등**: 있음 (3색 신호등)
- **날씨**: 맑음, 주간

### 2. 관련 대상
- **사고 유형**: 차대차 사고
- **A 차량**: 흰색 SUV (블랙박스 장착 차량)
- **B 차량**: 검정색 세단

### 3. 사고 전 상황
- A 차량: 녹색 신호에 교차로를 직진으로 통과 중, 속도 약 40km/h
- B 차량: 우측 도로에서 적색 신호임에도 불구하고 감속 없이 교차로 진입

### 4. 사고 발생 경위
A 차량이 녹색 신호에 따라 정상적으로 교차로를 직진하던 중, B 차량이 적색 신호를 무시하고 우측에서 교차로에 진입하여 A 차량의 조수석 측면을 충격하였습니다.

### 5. 사고 후 상황
양 차량 모두 교차로 중앙에 정지하였으며, A 차량 우측 문짝 파손, B 차량 전면부 파손 확인됩니다.

### 6. 관찰된 위반 행위
- B 차량: 적색 신호 위반 (도로교통법 제5조)""",
    },
    {
        "name": "A2-야간비신호교차로",
        "expected": "차12-1",
        "train_count": 30,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 이면도로 교차로
- **신호등**: 없음
- **시간대**: 야간 (가로등 일부 점등)
- **특이사항**: 시야 제한, 건물 모서리로 좌측 확인 어려움

### 2. 관련 대상
- **A 차량**: 승용차 (블랙박스 차량)
- **B 차량**: 승용차

### 3. 사고 전 상황
- A 차량: 동쪽에서 서쪽으로 직진 중, 약 30km/h
- B 차량: 남쪽에서 북쪽으로 직진 중

### 4. 사고 발생 경위
야간에 신호기가 설치되지 않은 교차로에서 양쪽 도로에서 동시에 진입한 두 차량이 교차로 중앙에서 측면 충돌하였습니다. 양쪽 모두 서행하지 않은 것으로 보입니다.

### 5. 사고 후 상황
충돌 후 양 차량 정지, A 차량 좌측 전면부, B 차량 우측 측면 파손.

### 6. 관찰된 위반 행위
- 양쪽 차량 모두 서행 의무 불이행 의심""",
    },
    {
        "name": "A3-추돌",
        "expected": "차41-1",
        "train_count": 18,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 편도 3차로 직선 도로
- **날씨**: 흐림
- **시간대**: 주간

### 2. 관련 대상
- **A 차량**: 승용차 (블랙박스 차량, 2차로 주행)
- **B 차량**: 트럭 (2차로 선행)

### 3. 사고 전 상황
- B 차량(트럭)이 전방 정체로 인해 감속 후 정지
- A 차량이 후방에서 약 60km/h로 접근 중

### 4. 사고 발생 경위
선행하던 B 트럭이 전방 교통 정체로 정차하였으나, 후방의 A 차량이 전방 주시를 태만히 하여 B 트럭의 후미를 추돌하였습니다. A 차량의 제동 흔적은 충돌 직전 약 5m 정도만 확인됩니다.

### 5. 사고 후 상황
A 차량 전면부 심하게 파손, B 트럭 후미 범퍼 파손.

### 6. 관찰된 위반 행위
- A 차량: 안전거리 미확보, 전방주시 태만""",
    },
    {
        "name": "A4-유턴충돌",
        "expected": "차33-1",
        "train_count": 17,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 왕복 6차로 도로, 유턴 구역 표시 있음
- **신호등**: 유턴 신호 있음

### 2. 관련 대상
- **A 차량**: 은색 세단 (유턴 시도)
- **B 차량**: 흰색 승용차 (반대편에서 직진)

### 3. 사고 전 상황
- A 차량이 유턴 구역에서 유턴을 시작
- B 차량이 반대편 차로에서 직진으로 접근 중

### 4. 사고 발생 경위
A 차량이 유턴을 하면서 반대편 차로로 진입하던 중, 반대편에서 직진하던 B 차량과 충돌하였습니다. A 차량이 유턴 완료 전에 B 차량의 진행 경로를 차단한 형태입니다.

### 5. 사고 후 상황
A 차량 운전석 측면 파손, B 차량 전면 우측 파손.

### 6. 관찰된 위반 행위
- A 차량: 유턴 시 안전 미확인""",
    },
    {
        "name": "A5-끼어들기",
        "expected": "차20-2",
        "train_count": 9,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 편도 3차로 도시 도로
- **교통량**: 보통

### 2. 관련 대상
- **A 차량**: SUV (2차로에서 1차로로 차선 변경 시도)
- **B 차량**: 세단 (1차로 직진)

### 3. 사고 전 상황
- A 차량: 2차로에서 주행 중 갑자기 좌측 방향지시등을 켜고 1차로로 진로 변경 시도
- B 차량: 1차로에서 정상 속도로 직진 중

### 4. 사고 발생 경위
A 차량이 충분한 안전거리를 확보하지 않은 채 1차로로 끼어들면서 B 차량의 우측 전면부와 A 차량의 좌측 후면부가 접촉하였습니다.

### 5. 사고 후 상황
양 차량 갓길로 이동하여 정차.

### 6. 관찰된 위반 행위
- A 차량: 안전거리 미확보 상태에서 차선 변경""",
    },
    {
        "name": "A6-중앙선침범",
        "expected": "차31-1",
        "train_count": 14,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 왕복 2차로 국도
- **도로 상태**: 커브 구간, 황색 실선(중앙선)
- **날씨**: 비

### 2. 관련 대상
- **A 차량**: 승용차 (블랙박스 차량, 정상 주행)
- **B 차량**: 화물차 (반대편에서 접근)

### 3. 사고 전 상황
- A 차량: 자기 차로에서 정상 주행 중
- B 차량: 커브 구간에서 속도를 줄이지 않고 중앙선을 넘어 반대 차로로 진입

### 4. 사고 발생 경위
비가 오는 커브 구간에서 B 화물차가 중앙선을 침범하여 반대 차로의 A 차량과 정면으로 충돌하였습니다. A 차량 운전자가 우측으로 회피를 시도하였으나 미처 피하지 못하였습니다.

### 5. 사고 후 상황
양 차량 심각한 전면 파손, 도로 위에 정지.

### 6. 관찰된 위반 행위
- B 차량: 중앙선 침범 (도로교통법 제13조 제3항)""",
    },
    {
        "name": "A7-고속도로합류",
        "expected": "차43-1",
        "train_count": 15,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 고속도로 합류 구간
- **차선**: 본선 3차로 + 가속 차로
- **교통량**: 본선 다소 혼잡

### 2. 관련 대상
- **A 차량**: 승용차 (본선 3차로 주행)
- **B 차량**: SUV (가속차로에서 본선 합류 시도)

### 3. 사고 전 상황
- A 차량: 본선 3차로(가장 바깥 차로)에서 약 90km/h로 주행
- B 차량: 가속차로에서 속도를 높이며 본선 합류 시도

### 4. 사고 발생 경위
B 차량이 가속차로에서 본선으로 합류하면서 3차로의 A 차량과 나란히 진행하게 되었고, 가속차로가 끝나는 지점에서 B 차량이 A 차량의 옆으로 들어오면서 측면 접촉 사고가 발생하였습니다.

### 5. 사고 후 상황
양 차량 갓길 정차, 경미한 측면 스크래치 파손.

### 6. 관찰된 위반 행위
- B 차량: 합류 시 안전 미확인""",
    },
    {
        "name": "A8-주차장출차",
        "expected": "차51-1",
        "train_count": 8,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 대형마트 지하주차장
- **특이사항**: 주차구역 밀집, 시야 제한

### 2. 관련 대상
- **A 차량**: 미니밴 (주차 공간에서 후진 출차 중)
- **B 차량**: 세단 (주차장 내 통로 주행 중)

### 3. 사고 전 상황
- A 차량: 주차된 상태에서 후진하여 출차 시도
- B 차량: 주차장 통로를 저속으로 주행 중

### 4. 사고 발생 경위
A 차량이 주차 공간에서 후진으로 출차하는 과정에서 통로를 지나가던 B 차량의 측면과 충돌하였습니다. A 차량 운전자가 후방 및 좌우 확인을 충분히 하지 않은 것으로 보입니다.

### 5. 사고 후 상황
A 차량 후면 범퍼, B 차량 좌측 뒷문 파손.

### 6. 관찰된 위반 행위
- A 차량: 후진 시 안전 확인 의무 불이행""",
    },
    {
        "name": "A9-비보호좌회전",
        "expected": "차2-6",
        "train_count": 6,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 왕복 4차로 교차로
- **신호등**: 비보호 좌회전 구간 (별도 좌회전 화살표 없음)
- **시간대**: 오후, 퇴근 시간대

### 2. 관련 대상
- **A 차량**: 세단 (좌회전 시도)
- **B 차량**: 트럭 (맞은편에서 직진)

### 3. 사고 전 상황
- A 차량: 교차로에서 녹색 신호에 비보호 좌회전 대기 후 좌회전 개시
- B 차량: 맞은편 차로에서 녹색 신호에 직진 중, 약 50km/h

### 4. 사고 발생 경위
A 차량이 비보호 좌회전을 시도하면서 맞은편에서 직진하는 B 트럭을 미처 확인하지 못하고 좌회전을 개시하여, B 트럭이 A 차량의 운전석 측면을 충격하였습니다.

### 5. 사고 후 상황
A 차량 운전석 측면 심하게 파손, B 트럭 전면 범퍼 파손.

### 6. 관찰된 위반 행위
- A 차량: 비보호 좌회전 시 맞은편 직진 차량 확인 소홀""",
    },
    {
        "name": "A10-비신호교차로좌회전",
        "expected": "차15-1",
        "train_count": 14,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 주택가 이면도로 T자 교차로
- **신호등**: 없음

### 2. 관련 대상
- **A 차량**: 경차 (직진)
- **B 차량**: SUV (좌회전)

### 3. 사고 전 상황
- A 차량: 주 도로에서 직진 중
- B 차량: 우측 골목에서 좌회전하여 주 도로에 진입 시도

### 4. 사고 발생 경위
신호가 없는 교차로에서 B 차량이 골목에서 좌회전하여 주 도로에 진입하면서, 주 도로를 직진하던 A 차량과 충돌하였습니다.

### 5. 사고 후 상황
A 차량 전면 좌측, B 차량 우측 측면 파손.

### 6. 관찰된 위반 행위
- B 차량: 교차로 진입 시 일시정지 및 좌우 확인 소홀""",
    },
]

# ── 그룹 B: 학습 적게 된 chunk (1~3 pairs) ──
GROUP_B_VLM = [
    {
        "name": "B1-녹색좌회전후황색충돌",
        "expected": "차2-3",
        "train_count": 2,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 왕복 4차로 신호 교차로

### 2. 관련 대상
- **A 차량**: 승용차 (황색 신호에 직진)
- **B 차량**: 승용차 (녹색 신호에 좌회전 진입 후 황색으로 바뀐 뒤 충돌)

### 3. 사고 전 상황
- B 차량이 녹색 신호에서 좌회전을 위해 교차로에 진입하여 대기
- 신호가 황색으로 변경
- A 차량이 황색 신호에 교차로에 진입하여 직진

### 4. 사고 발생 경위
B 차량이 녹색 신호에 좌회전을 위해 교차로에 진입한 뒤, 신호가 황색으로 바뀐 후 좌회전을 완료하려는 순간 황색에 직진 진입한 A 차량과 충돌하였습니다.

### 6. 관찰된 위반 행위
- A 차량: 황색 신호 진입""",
    },
    {
        "name": "B2-우회전중직진충돌",
        "expected": "차3-1",
        "train_count": 3,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 신호 교차로, 우회전 차로 있음

### 2. 관련 대상
- **A 차량**: 세단 (녹색 신호 직진)
- **B 차량**: SUV (적색 신호에 우회전)

### 4. 사고 발생 경위
B 차량이 적색 신호에서 우회전을 하면서 교차로에 진입하였고, 좌측에서 녹색 신호에 직진하던 A 차량과 교차로 내에서 충돌하였습니다. B 차량이 보행자 신호 확인 후 우회전을 시도하였으나 직진 차량 확인이 미흡하였습니다.

### 6. 관찰된 위반 행위
- B 차량: 우회전 시 교차 도로 직진 차량 확인 불충분""",
    },
    {
        "name": "B3-동일방향우회전직진",
        "expected": "차4-1",
        "train_count": 2,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 왕복 4차로 교차로

### 2. 관련 대상
- **A 차량**: 승용차 (직진, 1차로)
- **B 차량**: 트럭 (우회전, 2차로)

### 4. 사고 발생 경위
교차로에서 A 차량이 1차로에서 직진하고, B 트럭이 같은 방향 2차로에서 우회전하면서 A 차량의 진행 경로를 차단하여 충돌하였습니다. B 차량이 우회전 시 내측 차로의 직진 차량을 미처 확인하지 못한 것으로 보입니다.

### 6. 관찰된 위반 행위
- B 차량: 우회전 시 내측 차로 직진 차량 미확인""",
    },
    {
        "name": "B4-차로변경직진충돌",
        "expected": "차20-1",
        "train_count": 3,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 편도 3차로 일반 도로

### 2. 관련 대상
- **A 차량**: 승용차 (2차로→1차로 차선 변경)
- **B 차량**: 승용차 (1차로 직진)

### 4. 사고 발생 경위
A 차량이 2차로에서 1차로로 차선을 변경하는 과정에서 1차로 후방에서 직진하던 B 차량과 접촉하였습니다. A 차량이 사이드미러 확인 없이 급하게 차선을 변경한 것으로 보입니다.

### 6. 관찰된 위반 행위
- A 차량: 진로 변경 시 안전 확인 불충분""",
    },
    {
        "name": "B5-역주행충돌",
        "expected": "차31-2",
        "train_count": 2,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 일방통행로

### 2. 관련 대상
- **A 차량**: 승용차 (일방통행로 역주행)
- **B 차량**: 승용차 (정상 방향 주행)

### 4. 사고 발생 경위
A 차량이 일방통행 도로를 역방향으로 주행하면서 정상 방향으로 진행하던 B 차량과 정면으로 충돌하였습니다.

### 6. 관찰된 위반 행위
- A 차량: 일방통행로 역주행""",
    },
    {
        "name": "B6-좌회전중유턴충돌",
        "expected": "차33-2",
        "train_count": 2,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 왕복 4차로 교차로

### 2. 관련 대상
- **A 차량**: 승용차 (좌회전)
- **B 차량**: 승용차 (유턴)

### 4. 사고 발생 경위
교차로에서 A 차량이 좌회전을 하던 중, 같은 방향에서 유턴을 하던 B 차량과 충돌하였습니다. B 차량이 유턴 시 후방의 좌회전 차량을 미처 확인하지 못한 것으로 보입니다.

### 6. 관찰된 위반 행위
- B 차량: 유턴 시 후방 좌회전 차량 미확인""",
    },
    {
        "name": "B7-연속추돌(3중)",
        "expected": "차42-1",
        "train_count": 3,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 편도 2차로 도시 도로, 교통 정체 구간

### 2. 관련 대상
- **A 차량**: 승용차 (선두, 정차 중)
- **B 차량**: SUV (중간)
- **C 차량**: 트럭 (후미, 추돌 가해)

### 4. 사고 발생 경위
교통 정체로 A, B 차량이 순서대로 정차해 있던 상황에서, 후방의 C 트럭이 감속하지 못하고 B 차량 후미를 추돌하였고, 그 충격으로 B 차량이 A 차량 후미를 재추돌하는 연쇄 추돌 사고가 발생하였습니다.

### 6. 관찰된 위반 행위
- C 차량: 안전거리 미확보, 전방주시 태만""",
    },
    {
        "name": "B8-보행자전용도로침범",
        "expected": "보29-1",
        "train_count": 1,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 보행자 전용도로 (차량 진입 금지 구역)

### 2. 관련 대상
- **사고 유형**: 차대보행자 사고
- **A 대상**: 보행자 (보행자 전용도로 보행 중)
- **B 차량**: 승용차 (보행자 전용도로 침범 주행)

### 4. 사고 발생 경위
보행자 전용도로를 정상적으로 보행하던 A 보행자를 B 차량이 보행자 전용도로에 불법으로 진입하여 주행하다가 충돌하였습니다.

### 6. 관찰된 위반 행위
- B 차량: 보행자 전용도로 침범 주행""",
    },
    {
        "name": "B9-동시진입우측차우선",
        "expected": "차11-2",
        "train_count": 2,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 비신호 교차로 (동일 폭 도로)

### 2. 관련 대상
- **A 차량**: 승용차 (좌측에서 진입)
- **B 차량**: 승용차 (우측에서 진입)

### 4. 사고 발생 경위
신호가 없는 교차로에서 폭이 동일한 양쪽 도로에서 동시에 진입한 두 차량이 충돌하였습니다. A 차량이 우측 도로에서 진입하는 B 차량에 진로를 양보하지 않았습니다.

### 6. 관찰된 위반 행위
- A 차량: 우측 차량 진로 양보 의무 위반""",
    },
    {
        "name": "B10-고속도로끼어들기",
        "expected": "차43-2",
        "train_count": 3,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 고속도로 본선 3차로

### 2. 관련 대상
- **A 차량**: 세단 (3차로→2차로 차선 변경)
- **B 차량**: SUV (2차로 직진)

### 4. 사고 발생 경위
고속도로 본선에서 A 차량이 3차로에서 2차로로 차선을 변경하면서 2차로를 주행하던 B 차량의 측면과 접촉하였습니다. 고속 주행 중 충분한 안전거리를 확보하지 않고 차선 변경을 시도한 것으로 보입니다.

### 6. 관찰된 위반 행위
- A 차량: 진로 변경 시 안전거리 미확보""",
    },
]

# ── 그룹 C: 학습 안 된 또는 매우 적은 chunk + 복합/비전형 시나리오 ──
GROUP_C_VLM = [
    {
        "name": "C1-자전거역주행",
        "expected": "거21-1",
        "train_count": 1,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 자전거 도로 겸용 보도

### 2. 관련 대상
- **사고 유형**: 자전거 대 자전거
- **A**: 자전거 (역방향 주행)
- **B**: 자전거 (정상 방향 주행)

### 4. 사고 발생 경위
자전거 도로에서 A 자전거가 역방향으로 주행하면서 정상 방향으로 오던 B 자전거와 정면 충돌하였습니다.

### 6. 관찰된 위반 행위
- A 자전거: 자전거 도로 역통행""",
    },
    {
        "name": "C2-자전거추돌",
        "expected": "거31-1",
        "train_count": 1,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 자전거 전용도로

### 2. 관련 대상
- **A**: 자전거 (선행, 저속)
- **B**: 자전거 (후행, 고속)

### 4. 사고 발생 경위
자전거 도로에서 저속으로 주행하던 A 자전거를 후방에서 빠르게 접근하던 B 자전거가 추돌하였습니다.

### 6. 관찰된 위반 행위
- B 자전거: 안전거리 미확보""",
    },
    {
        "name": "C3-자전거횡단",
        "expected": "거41-1",
        "train_count": 1,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 일반 도로 (자전거 횡단도로 아님)

### 2. 관련 대상
- **A**: 자전거 (도로 횡단 시도)
- **B**: 자전거 (정상 주행)

### 4. 사고 발생 경위
A 자전거가 도로를 횡단하려다 정상 주행하던 B 자전거와 충돌하였습니다.

### 6. 관찰된 위반 행위
- A 자전거: 무리한 횡단""",
    },
    {
        "name": "C4-주차장통로교차",
        "expected": "차51-2",
        "train_count": 2,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 아파트 지하주차장, 통로 교차 지점

### 2. 관련 대상
- **A 차량**: 승용차 (통로 직진)
- **B 차량**: 승용차 (교차 통로에서 진입)

### 4. 사고 발생 경위
주차장 내 통로가 교차하는 지점에서 A 차량이 직진하고 B 차량이 교차 통로에서 진입하면서 충돌하였습니다. 양 차량 모두 교차 지점에서 서행하지 않은 것으로 보입니다.

### 6. 관찰된 위반 행위
- 양쪽 모두 주차장 내 교차 지점에서 서행 의무 불이행""",
    },
    {
        "name": "C5-차도가장자리보행",
        "expected": "보27-2",
        "train_count": 1,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 간선도로, 보도 없는 구간
- **시간대**: 야간

### 2. 관련 대상
- **사고 유형**: 차대보행자 사고
- **A 대상**: 보행자 (차도 가장자리 보행 중)
- **B 차량**: 승용차 (차도 주행)

### 4. 사고 발생 경위
야간에 보도가 없는 간선도로 가장자리를 보행하던 A 보행자를 B 차량이 미처 발견하지 못하고 충돌하였습니다.

### 6. 관찰된 위반 행위
- B 차량: 전방주시 태만""",
    },
    {
        "name": "C6-자전거차선변경",
        "expected": "거32-1",
        "train_count": 1,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 자전거 도로 (2차선)

### 2. 관련 대상
- **A**: 자전거 (좌측으로 진로 변경)
- **B**: 자전거 (후방에서 직진)

### 4. 사고 발생 경위
A 자전거가 갑자기 좌측으로 진로를 변경하면서 후방에서 직진하던 B 자전거와 접촉하였습니다.

### 6. 관찰된 위반 행위
- A 자전거: 진로 변경 시 후방 미확인""",
    },
    {
        "name": "C7-교차로우회전보행자",
        "expected": "차5-2",
        "train_count": 11,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 신호 교차로, 횡단보도 있음
- **신호**: 보행자 녹색 신호

### 2. 관련 대상
- **사고 유형**: 차대보행자 사고
- **A 차량**: 승합차 (녹색 신호에 우회전 시도)
- **B 대상**: 노인 보행자 (횡단보도 보행 중)

### 4. 사고 발생 경위
A 승합차가 녹색 신호에 우회전을 하면서 횡단보도를 건너고 있던 노인 보행자 B를 미처 확인하지 못하고 충돌하였습니다. 보행자 신호는 녹색이었으며, A 차량 운전자가 우회전에 집중하느라 횡단보도의 보행자를 주시하지 못한 것으로 보입니다.

### 6. 관찰된 위반 행위
- A 차량: 우회전 시 횡단보도 보행자 보호 의무 위반""",
    },
    {
        "name": "C8-점멸신호교차로",
        "expected": "차1-5",
        "train_count": 3,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 교차로
- **신호등**: 점멸 신호 작동 중 (A 도로: 적색 점멸, B 도로: 황색 점멸)
- **시간대**: 심야

### 2. 관련 대상
- **A 차량**: 승용차 (적색 점멸 신호 도로에서 직진)
- **B 차량**: 승용차 (황색 점멸 신호 도로에서 직진)

### 4. 사고 발생 경위
심야 시간 점멸 신호가 작동하는 교차로에서 적색 점멸 도로의 A 차량이 일시정지를 하지 않고 교차로에 진입하여, 황색 점멸 도로에서 직진하던 B 차량과 충돌하였습니다.

### 6. 관찰된 위반 행위
- A 차량: 적색 점멸 신호에서 일시정지 의무 위반""",
    },
    {
        "name": "C9-고속도로진출입충돌",
        "expected": "차43-4",
        "train_count": 2,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 고속도로 진출 구간 (감속차로)

### 2. 관련 대상
- **A 차량**: 승용차 (본선에서 감속차로로 진출 시도)
- **B 차량**: 트럭 (감속차로 주행 중)

### 4. 사고 발생 경위
A 차량이 고속도로 본선에서 감속차로로 진입하면서 이미 감속차로를 주행 중이던 B 트럭의 후미를 추돌하였습니다.

### 6. 관찰된 위반 행위
- A 차량: 감속차로 진입 시 안전 확인 불충분""",
    },
    {
        "name": "C10-차도보행자충돌",
        "expected": "보27-1",
        "train_count": 1,
        "vlm_output": """## 사고 상황 분석

### 1. 도로 환경
- **도로 유형**: 편도 2차로 일반 도로
- **시간대**: 주간

### 2. 관련 대상
- **사고 유형**: 차대보행자 사고
- **A 대상**: 보행자 (차도 위를 걸어서 이동 중)
- **B 차량**: 승용차 (차도 주행)

### 4. 사고 발생 경위
차도 위를 걸어가던 보행자 A를 차도를 주행하던 B 차량이 충돌하였습니다. 보행자가 보도가 아닌 차도 위를 보행하고 있었습니다.

### 6. 관찰된 위반 행위
- 보행자: 차도 보행
- B 차량: 전방주시 의무""",
    },
]


def extract_query_from_vlm(vlm_text: str) -> str:
    """VLM 출력에서 사고 경위 섹션을 추출하여 RAG 검색 쿼리로 사용"""
    lines = vlm_text.strip().split("\n")
    # '사고 발생 경위' 섹션 추출
    capture = False
    query_parts = []
    for line in lines:
        if "사고 발생 경위" in line:
            capture = True
            continue
        if capture:
            if line.startswith("###") or line.startswith("## "):
                break
            stripped = line.strip()
            if stripped and not stripped.startswith("-"):
                query_parts.append(stripped)
            elif stripped.startswith("- "):
                query_parts.append(stripped[2:])

    query = " ".join(query_parts)
    # 너무 길면 앞부분만
    if len(query) > 300:
        query = query[:300]
    return query


def run_test(model_path: str, model_name: str):
    """과적합 검증 테스트 실행"""
    print(f"\n{'='*70}")
    print(f"  과적합 검증 테스트: {model_name}")
    print(f"  모델: {model_path}")
    print(f"{'='*70}\n")

    # 모델 로드
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(model_path)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        model = SentenceTransformer(model_path, device="cpu")
    model.max_seq_length = 256

    # chunk 임베딩
    chunk_emb = model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)

    all_groups = [
        ("A: 학습 많이 됨 (6~30 pairs)", GROUP_A_VLM),
        ("B: 학습 적게 됨 (1~3 pairs)", GROUP_B_VLM),
        ("C: 학습 적음/복합 시나리오", GROUP_C_VLM),
    ]

    group_results = {}

    for group_name, group_data in all_groups:
        print(f"\n── {group_name} ──")
        print(f"{'이름':<30} {'정답':>8} {'Top1':>8} {'Top1맞음':>8} {'Score':>8} {'Top3':>6} {'Top5':>6}")
        print("-" * 90)

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        scores = []

        for item in group_data:
            query = extract_query_from_vlm(item["vlm_output"])
            q_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
            sims = cos_sim(q_emb, chunk_emb)[0]

            # 정렬
            sorted_indices = torch.argsort(sims, descending=True)
            top_ids = [chunk_ids[i] for i in sorted_indices[:5]]
            top_scores = [sims[i].item() for i in sorted_indices[:5]]

            expected = item["expected"]
            is_top1 = top_ids[0] == expected
            is_top3 = expected in top_ids[:3]
            is_top5 = expected in top_ids[:5]

            # 정답 chunk의 score
            if expected in chunk_ids:
                exp_idx = chunk_ids.index(expected)
                exp_score = sims[exp_idx].item()
            else:
                exp_score = 0.0

            scores.append(exp_score)
            if is_top1: top1_correct += 1
            if is_top3: top3_correct += 1
            if is_top5: top5_correct += 1

            status = "O" if is_top1 else "X"
            t3 = "O" if is_top3 else "X"
            t5 = "O" if is_top5 else "X"
            print(f"{item['name']:<30} {expected:>8} {top_ids[0]:>8} {status:>8} {exp_score:>8.4f} {t3:>6} {t5:>6}")

            # 틀렸으면 상위 3개 보여주기
            if not is_top1:
                for rank, (tid, ts) in enumerate(zip(top_ids[:3], top_scores[:3]), 1):
                    print(f"  └ #{rank}: {tid} ({ts:.4f})")

        n = len(group_data)
        avg_score = sum(scores) / n if n else 0
        print(f"\n  >> Top1: {top1_correct}/{n} ({top1_correct/n*100:.1f}%) | "
              f"Top3: {top3_correct}/{n} ({top3_correct/n*100:.1f}%) | "
              f"Top5: {top5_correct}/{n} ({top5_correct/n*100:.1f}%) | "
              f"Avg Score: {avg_score:.4f}")

        group_results[group_name] = {
            "top1": top1_correct, "top3": top3_correct, "top5": top5_correct,
            "total": n, "avg": avg_score, "scores": scores,
        }

    # ── 종합 판정 ──
    print(f"\n{'='*70}")
    print("  종합 과적합 판정")
    print(f"{'='*70}")

    print(f"\n{'그룹':<40} {'Top1':>10} {'Top3':>10} {'Avg Score':>12}")
    print("-" * 75)
    for gname, gres in group_results.items():
        t1 = f"{gres['top1']}/{gres['total']}"
        t3 = f"{gres['top3']}/{gres['total']}"
        print(f"{gname:<40} {t1:>10} {t3:>10} {gres['avg']:>12.4f}")

    # 과적합 판단 기준
    a_res = list(group_results.values())[0]
    b_res = list(group_results.values())[1]
    c_res = list(group_results.values())[2]

    a_rate = a_res["top1"] / a_res["total"]
    b_rate = b_res["top1"] / b_res["total"]
    c_rate = c_res["top1"] / c_res["total"]

    score_gap_ab = a_res["avg"] - b_res["avg"]
    score_gap_ac = a_res["avg"] - c_res["avg"]
    rate_gap_ab = a_rate - b_rate
    rate_gap_ac = a_rate - c_rate

    print(f"\n  A-B Top1 차이: {rate_gap_ab*100:+.1f}%p | Score 차이: {score_gap_ab:+.4f}")
    print(f"  A-C Top1 차이: {rate_gap_ac*100:+.1f}%p | Score 차이: {score_gap_ac:+.4f}")

    print("\n  [판정 결과]")
    if rate_gap_ac > 0.3 or score_gap_ac > 0.10:
        print("  ⚠ 과적합 의심: 학습 많은 그룹과 적은 그룹 간 성능 차이가 큼")
        print("  → 추가 파인튜닝 필요: 다양한 VLM 스타일 데이터로 재학습 권장")
    elif rate_gap_ac > 0.15 or score_gap_ac > 0.05:
        print("  △ 약한 과적합: 그룹 간 성능 차이 존재하나 심각하지 않음")
        print("  → VLM 스타일 데이터로 보충 학습 검토")
    else:
        print("  ✓ 양호: 그룹 간 성능 차이가 작아 일반화 능력 양호")
        print("  → 현재 모델로 배포 가능")

    overall_top1 = sum(r["top1"] for r in group_results.values())
    overall_total = sum(r["total"] for r in group_results.values())
    overall_avg = sum(sum(r["scores"]) for r in group_results.values()) / overall_total

    print(f"\n  전체 Top1: {overall_top1}/{overall_total} ({overall_top1/overall_total*100:.1f}%)")
    print(f"  전체 Avg Score: {overall_avg:.4f}")

    # 기존 테스트 10개와 비교
    print(f"\n  [참고] 기존 테스트 10개 결과:")
    test_queries = [
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
    orig_queries = [q for q, _ in test_queries]
    orig_expected = [e for _, e in test_queries]

    q_emb = model.encode(orig_queries, convert_to_tensor=True, show_progress_bar=False)
    sims_all = cos_sim(q_emb, chunk_emb)
    orig_top1 = 0
    orig_scores_list = []
    for i, (q, exp) in enumerate(test_queries):
        sorted_idx = torch.argsort(sims_all[i], descending=True)
        top_id = chunk_ids[sorted_idx[0]]
        exp_idx = chunk_ids.index(exp)
        score = sims_all[i][exp_idx].item()
        orig_scores_list.append(score)
        if top_id == exp:
            orig_top1 += 1

    orig_avg = sum(orig_scores_list) / len(orig_scores_list)
    print(f"  기존 Top1: {orig_top1}/10 ({orig_top1/10*100:.1f}%)")
    print(f"  기존 Avg Score: {orig_avg:.4f}")
    print(f"  VLM 테스트 Avg Score: {overall_avg:.4f}")
    print(f"  차이: {orig_avg - overall_avg:+.4f}")

    if orig_avg - overall_avg > 0.10:
        print("\n  ⚠ 기존 테스트 대비 VLM 스타일 쿼리 성능이 현저히 낮음")
        print("  → VLM 출력 형태에 맞는 추가 파인튜닝 강력 권장")

    return group_results


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "boost_v3_best")
    model_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(model_path)

    # boost_v3_best 테스트
    results_v3 = run_test(model_path, model_name)

    # boost_v3b_best도 있으면 비교
    v3b_path = os.path.join(BASE_DIR, "boost_v3b_best")
    if os.path.exists(v3b_path) and model_path != v3b_path:
        results_v3b = run_test(v3b_path, "boost_v3b_best")

    print(f"\n{'='*70}")
    print("  VLM 스타일 테스트 데이터 샘플 (실제 사용 시 이런 텍스트가 입력됨)")
    print(f"{'='*70}")
    # 3개만 예시 출력
    for item in [GROUP_A_VLM[0], GROUP_B_VLM[0], GROUP_C_VLM[0]]:
        print(f"\n[{item['name']}] 정답: {item['expected']} (학습 {item['train_count']}쌍)")
        print(f"VLM 출력 → RAG 쿼리로 변환:")
        print(f"  \"{extract_query_from_vlm(item['vlm_output'])[:120]}...\"")
