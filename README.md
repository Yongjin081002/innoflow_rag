# InnoFlow RAG - 과실비율 검색 시스템

## 개요

블랙박스 영상 분석 VLM(Qwen3.5-27B) 출력을 쿼리로,
교통사고 과실비율 인정기준 162개 chunk에서 가장 유사한 사고유형을 검색하는 시스템.

### 아키텍처

```
블랙박스 영상 → Qwen3.5-27B VLM → 사고 상황 서술 텍스트
                                          ↓
                               boost_best_vlm_v4 (임베딩)
                                          ↓
                               Qdrant (fault_rules 컬렉션)
                                          ↓
                           Top-K 과실비율 기준 + base_fault + modifiers
```

### 주요 파일

| 파일 | 역할 |
|------|------|
| `parse_chunks.py` | chunks.json 파싱 (embed_text, base_fault, modifiers 추출) |
| `insert_qdrant.py` | Qdrant에 구조화 payload insert |
| `search_final.py` | 공통 스키마 형식 검색 API |
| `chunks.json` | 과실비율 인정기준 원본 (162건) |
| `boost_best_vlm_v4/` | 파인튜닝된 sentence-transformer 모델 |
| `qdrant_fault_data/` | Qdrant 로컬 DB |

### 모델 학습 파일 (참고용)

| 파일 | 역할 |
|------|------|
| `training_data.py` | 키워드 학습 데이터 (475쌍) |
| `vlm_training_data.py` | VLM 장문 학습 데이터 v1 (120쌍) |
| `vlm_training_data_v2.py` | VLM 장문 학습 데이터 v2 (35쌍) |
| `vlm_training_data_v3.py` | VLM 장문 학습 데이터 v3 + hard negatives (35쌍 + 16 hard neg) |
| `boost_vlm_v3.py` | v3 학습 스크립트 (8 config 탐색) |
| `boost_vlm_v4.py` | v4 학습 스크립트 (v1+v2+v3 통합) |

---

## 1. 실행 방법

### 의존 라이브러리 설치

```bash
pip install sentence-transformers==3.2.1 qdrant-client==1.12.1 torch>=2.0 numpy transformers
```

### Qdrant 실행 (로컬 파일 모드 / Docker 불필요)

현재 `qdrant-client`의 로컬 파일 모드를 사용합니다.
별도 Qdrant 서버 없이 `qdrant_fault_data/` 디렉토리에서 직접 읽습니다.

Docker로 Qdrant 서버를 별도 운영하려면:

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v /home/minsung0830/innoflow_rag/qdrant_fault_data:/qdrant/storage \
  qdrant/qdrant:latest
```

Docker 사용 시 `search_final.py`의 클라이언트를 아래로 변경:

```python
# 로컬 파일 모드 (현재)
client = QdrantClient(path="./qdrant_fault_data")

# Docker 서버 모드
client = QdrantClient(host="localhost", port=6333)
```

### Qdrant 데이터 초기 insert

```bash
cd /home/minsung0830/innoflow_rag
python3 insert_qdrant.py
```

출력:
```
153건 insert 완료
```

### 검색 테스트

```bash
python3 search_final.py
```

---

## 2. 입력/출력 형식

### 입력

VLM이 생성한 사고 상황 서술 텍스트 (또는 키워드 쿼리):

```python
from search_final import FaultRuleSearcher

searcher = FaultRuleSearcher()
results = searcher.search("교차로에서 A 차량이 녹색 신호에 직진하던 중 B 차량이 적색 신호를 무시하고 직진 진입하여 충돌하였습니다.", top_k=3)
```

### 출력

공통 스키마 형식:

```json
{
  "type": "fault_rule",
  "id": "차1-1",
  "content": "차1-1\n(A) 녹색 직진\n(B) 적색 직진\n기본 과실비율 A0 B100\n...",
  "base_fault": {"A": 0, "B": 100},
  "modifiers": [
    {"target": "A", "condition": "현저한_과실", "value": 10},
    {"target": "A", "condition": "중대한_과실", "value": 20},
    {"target": "B", "condition": "중대한_과실", "value": 20},
    {"target": "B", "condition": "현저한_과실", "value": "비적용"}
  ],
  "category": "자동차",
  "source": "과실비율 인정기준",
  "score": 0.85,
  "metadata": {
    "embed_text": "차1-1 녹색 직진 대 적색 직진 (A0:B100)"
  }
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `type` | string | 항상 `"fault_rule"` |
| `id` | string | 과실비율 기준 ID (예: 차1-1, 보27-1, 거4-4) |
| `content` | string | 원본 청크 텍스트 전체 |
| `base_fault` | object | 기본 과실비율. 자동차: `{"A": n, "B": m}`, 보행자: `{"보행자": n, "차량": m}` |
| `modifiers` | array | 수정요소 목록. 각 항목: `{target, condition, value}` |
| `category` | string | `"자동차"` / `"보행자"` / `"자전거"` |
| `source` | string | 항상 `"과실비율 인정기준"` |
| `score` | float | 코사인 유사도 (0~1) |
| `metadata` | object | 추가 정보 (embed_text 등) |

---

## 3. base_fault + modifiers 계산 예시

### 예시: 차1-1 (녹색 직진 vs 적색 직진)

```
기본 과실비율: A 0% : B 100%
```

**시나리오: A에 현저한 과실이 있고, B에도 중대한 과실이 있는 경우**

```
A 과실 = 0 + 10(현저한 과실) = 10%
B 과실 = 100 + 20(중대한 과실) = 120% → 100%로 cap

최종: A 10% : B 90%
```

**시나리오: B에 현저한 과실 주장 → 비적용**

```
B의 "현저한_과실" modifier의 value가 "비적용"
→ B가 적색 신호 위반이므로 현저한 과실 수정요소 적용 불가
→ 기본 비율 유지: A 0% : B 100%
```

### 계산 로직 (의사코드)

```python
def calculate_fault(base_fault, modifiers, applicable_conditions):
    """
    base_fault: {"A": 0, "B": 100}
    modifiers: [{"target": "A", "condition": "현저한_과실", "value": 10}, ...]
    applicable_conditions: {"A": ["현저한_과실"], "B": ["중대한_과실"]}
    """
    fault_a = base_fault["A"]
    fault_b = base_fault["B"]

    for mod in modifiers:
        target = mod["target"]
        condition = mod["condition"]
        value = mod["value"]

        # "비적용"이면 해당 수정요소 무시
        if value == "비적용":
            continue

        # 해당 당사자에게 적용 가능한 조건인지 확인
        if condition in applicable_conditions.get(target, []):
            if target == "A":
                fault_a += value
            elif target == "B":
                fault_b += value

    # 합계 100% 맞추기
    total = fault_a + fault_b
    if total != 100:
        # 비율 조정
        fault_a = round(fault_a / total * 100)
        fault_b = 100 - fault_a

    return {"A": fault_a, "B": fault_b}
```

### 보행자 사고 예시: 보27-1 (차도 보행 vs 차도 주행)

```
기본: 보행자 0% : 차량 100%

적용 가능한 수정요소:
  야간_기타_시야장애  → 보행자 +5
  차량 중대한_과실    → 차량 -20 (보행자 과실에서 차감 = 보행자에 유리)

보행자 과실 = 0 + 5(야간) = 5%
차량 과실 감경 = -20(중대한 과실) → 보행자 과실에서 차감

최종: 보행자 과실 5% → 차량에 중대한 과실이 있으므로 보행자 -20 적용
→ 보행자 max(0, 5-20) = 0% : 차량 100%
```

---

## 4. 의존 라이브러리

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `sentence-transformers` | 3.2.1 | 임베딩 모델 로딩 및 인코딩 |
| `qdrant-client` | 1.12.1 | 벡터 DB 클라이언트 |
| `torch` | 2.4.1 | PyTorch (sentence-transformers 의존) |
| `numpy` | 1.24.4 | 수치 연산 |
| `transformers` | 4.44.2 | HuggingFace 모델 (sentence-transformers 의존) |

### Python 버전

- Python 3.8 이상 (시스템 python3 사용 가능)

---

## 5. 모델 성능 요약

### VLM 30개 쿼리 테스트 (2026-04-21 기준)

| 지표 | v3 (content) | v4 (content) | v4 (embed_text) |
|------|-------------|-------------|-----------------|
| 키워드 Top1 (/10) | 10 | 10 | 10 |
| VLM Top1 (/30) | 17 | **20** | 12 |
| VLM Top3 (/30) | 22 | **24** | 23 |
| Combined Score | 0.6345 | **0.6825** | 0.5127 |
| A-C Gap | +20%p | +10%p | +10%p |

- **v4(content)**: v3 대비 Top1 +3, 고속도로/유턴 카테고리 대폭 개선
- **v4(embed_text)**: Top1은 낮지만 Top3은 23/30으로 견조, payload 구조화 이점

### 고질적 오답 5건 현황

| Chunk | v3 | v4(content) | v4(embed_text) |
|-------|-----|------------|----------------|
| 차20-2 (끼어들기) | X (rank 999) | rank 2 | rank 2 |
| 차3-1 (적색직진) | X | X | X |
| 차4-1 (좌회전-우회전) | X | X | X |
| 차31-2 (중앙선침범) | X | X | X |
| 차11-2 (노면표시위반) | X | X | X |

---

## 6. 디렉토리 구조

```
innoflow_rag/
├── README.md                    # 이 파일
├── chunks.json                  # 과실비율 인정기준 원본 (162건)
├── chunks_structured.json       # 파싱된 구조화 데이터
├── parse_chunks.py              # 청크 파서
├── insert_qdrant.py             # Qdrant insert
├── search_final.py              # 검색 API
├── test_qdrant_vlm.py           # VLM 테스트 스크립트
├── qdrant_fault_data/           # Qdrant 로컬 DB
├── boost_best_vlm_v4/           # 최신 파인튜닝 모델
├── boost_best_vlm_v3/           # 이전 모델
├── boost_best/                  # 베이스 모델
├── training_data.py             # 키워드 학습 데이터
├── vlm_training_data.py         # VLM 학습 데이터 v1
├── vlm_training_data_v2.py      # VLM 학습 데이터 v2
├── vlm_training_data_v3.py      # VLM 학습 데이터 v3 + hard negatives
├── boost_vlm_v3.py              # v3 학습 스크립트
├── boost_vlm_v4.py              # v4 학습 스크립트
└── compare_boost_v3.py          # 테스트 쿼리 정의 (GROUP_A/B/C)
```
