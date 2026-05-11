# InnoFlow RAG — 과실기준 검색 시스템

## 개요

블랙박스 영상 분석 VLM(Qwen3.5-27B)이 생성한 사고 장면 텍스트를 쿼리로 받아, 자동차사고 과실비율 인정기준 chunk 중 가장 유사한 사고 유형을 검색하는 과실기준 RAG 시스템입니다.

`chunks.json`의 원본 과실기준 문서를 `parse_chunks.py`에서 구조화하여 `base_fault`, `modifiers`, `category`, `embed_text`를 추출하고, `insert_qdrant.py`가 임베딩과 함께 Qdrant `fault_rules` 컬렉션에 저장합니다. 검색 시 `search_final.py`의 `FaultRuleSearcher`가 쿼리를 임베딩한 뒤 Top-K 과실기준과 기본과실, 수정요소를 공통 스키마로 반환합니다.

담당자: 김용진 (과실기준 RAG 파트)

## 전체 파이프라인

```text
블랙박스 영상
    ↓
VLM (Qwen3.5-27B) → 사고 장면 텍스트 변환
    ↓
RAG 검색 (과실기준 / 법률 / 판례) 병렬
    ↓
병합 모듈 (이동휘) → 3개 RAG 결과 통합
    ↓
LLM → 사고 유형 분류
    ↓
코드 → 기본과실 + 수정요소 계산
    ↓
최종 리포트
```

## 기술 스택

| 구분 | 라이브러리/도구 | 사용 위치 | 역할 |
|------|----------------|----------|------|
| 임베딩 모델 | `sentence-transformers` | `search_final.py`, `insert_qdrant.py`, `boost_vlm_v*.py` | BGE-M3 계열 모델 로드, 쿼리/청크 임베딩 |
| 벡터 DB | `qdrant-client` | `insert_qdrant.py`, `search_final.py`, `test_qdrant_vlm.py` | `fault_rules` 컬렉션 생성, 벡터 검색 |
| 딥러닝 | `torch` | 학습/벤치마크 스크립트 | CUDA 학습, tensor 유사도 계산 |
| 학습 데이터 처리 | `numpy` | `boost_vlm_v5.py`~`boost_vlm_v7.py` 등 | 시드 고정, 학습 재현성 보조 |
| 데이터 로딩 | Python `json`, `re`, `os` | 전반 | chunk 파싱, benchmark JSON 저장 |
| 모델 학습 | `torch.utils.data.DataLoader` | `boost_vlm_v*.py` | `InputExample` 배치 구성 |
| 유사도 평가 | `sentence_transformers.util.cos_sim` | `compare_*`, `benchmark_*`, `test_*` | Top1/Top3/Top5 평가 |
| 컨테이너 실행 | Docker Qdrant | 실행 환경 | Qdrant 서버 모드 실행 시 사용 |

## 실행 방법

### 1. 의존 라이브러리 설치

```bash
pip install sentence-transformers==3.2.1 qdrant-client==1.12.1 torch==2.4.1 numpy==1.24.4 transformers==4.44.2 tqdm==4.67.1
```

### 2. Qdrant 실행

현재 코드는 `QdrantClient(path=QDRANT_PATH)` 로컬 파일 모드를 기본으로 사용합니다. Docker 서버 모드로 실행하려면 아래 명령으로 Qdrant를 띄운 뒤 코드의 클라이언트를 `QdrantClient(host="localhost", port=6333)` 형태로 바꾸면 됩니다.

```bash
docker run -p 6333:6333 \
  -v $(pwd)/qdrant_fault_data:/qdrant/storage \
  qdrant/qdrant
```

### 3. 과실기준 chunk 파싱

```bash
python3 parse_chunks.py
```

`chunks.json`을 읽어 `chunks_structured.json`을 생성합니다. 각 rule에는 `id`, `content`, `embed_text`, `base_fault`, `modifiers`, `category`, `is_rule`이 포함됩니다.

### 4. Qdrant insert

```bash
python3 insert_qdrant.py
```

`parse_all_chunks()` 결과 중 `is_rule=True`인 과실기준을 임베딩하고, Qdrant `fault_rules` 컬렉션에 저장합니다. payload는 `id`, `content`, `embed_text`, `base_fault`, `modifiers`, `category`로 구성됩니다.

### 5. 검색 실행

```python
from search_final import FaultRuleSearcher

searcher = FaultRuleSearcher()
results = searcher.search("신호위반 직진 충돌", top_k=5)
searcher.close()
```

CLI 테스트:

```bash
python3 search_final.py
```

### 6. 최종 모델 벤치마크

```bash
python3 benchmark_boost_vlm_v7.py
```

결과는 `boost_best_vlm_v7_benchmark.json`에 저장됩니다.

## 검색 결과 스키마

`search_final.py`의 `FaultRuleSearcher.search()`는 아래 공통 스키마의 리스트를 반환합니다.

```json
{
  "type": "fault_rule",
  "id": "차1-1",
  "content": "원본 과실기준 청크 텍스트",
  "base_fault": {
    "A": 0,
    "B": 100
  },
  "modifiers": [
    {
      "target": "A",
      "condition": "현저한_과실",
      "value": 10
    }
  ],
  "category": "자동차",
  "source": "과실비율 인정기준",
  "score": 0.85,
  "metadata": {
    "embed_text": "차1-1 녹색 직진 대 적색 직진 (A0:B100)"
  }
}
```

| 필드 | 설명 |
|------|------|
| `type` | 과실기준 RAG 결과를 뜻하는 `"fault_rule"` |
| `id` | 과실기준 chunk ID. 예: `차1-1`, `보27-1`, `거4-4` |
| `content` | 원본 과실기준 텍스트 |
| `base_fault` | 기본 과실비율. 자동차/자전거는 A/B, 보행자는 보행자/차량 기준 |
| `modifiers` | 수정요소 목록. 각 항목은 `target`, `condition`, `value` 포함 |
| `category` | `자동차`, `보행자`, `자전거` |
| `source` | `"과실비율 인정기준"` |
| `score` | Qdrant cosine similarity 점수 |
| `metadata.embed_text` | 임베딩에 사용한 사고 상황 요약 텍스트 |

## 수정요소 계산 방법

검색 결과의 `base_fault`를 시작값으로 두고, 실제 사고 상황에 해당하는 `modifiers`만 선택해 더합니다. `value`가 `"비적용"`인 항목은 해당 수정요소를 적용하지 않습니다.

합계가 100을 초과하거나 미달하면 최종 비율이 합계 100이 되도록 정규화합니다.

```python
def calculate_fault(base_fault, modifiers, applicable_conditions):
    fault = dict(base_fault)

    for modifier in modifiers:
        target = modifier["target"]
        condition = modifier["condition"]
        value = modifier["value"]

        if value == "비적용":
            continue

        if condition in applicable_conditions.get(target, []):
            fault[target] = fault.get(target, 0) + value

    total = sum(fault.values())
    if total != 100 and total > 0:
        normalized = {}
        keys = list(fault.keys())
        running = 0

        for key in keys[:-1]:
            normalized[key] = round(fault[key] / total * 100)
            running += normalized[key]

        normalized[keys[-1]] = 100 - running
        fault = normalized

    return fault


result = calculate_fault(
    base_fault={"A": 0, "B": 100},
    modifiers=[
        {"target": "A", "condition": "현저한_과실", "value": 10},
        {"target": "B", "condition": "중대한_과실", "value": 20}
    ],
    applicable_conditions={
        "A": ["현저한_과실"],
        "B": ["중대한_과실"]
    }
)
```

## 최종 모델 성능 (boost_best_vlm_v7)

| 테스트 | Top1 | Top3 | Top5 | Avg |
|-------|------|------|------|-----|
| 키워드 10개 | 10/10 (100%) | 10/10 | 10/10 | 0.887 |
| VLM 30개 | 22/30 (73%) | 29/30 | 29/30 | 0.774 |
| 새 VLM 20개 | 14/20 (70%) | 19/20 | 20/20 | 0.681 |

## 파인튜닝 히스토리

| 모델 | VLM Top1 | 주요 변경 |
|------|---------|---------|
| 원본 BGE-M3 | 2/10 (20%) | - |
| boost_best | 10/10 키워드 | 키워드 475쌍 |
| v3 | 17/30 | VLM 120쌍 추가 |
| v4 | 20/30 | hard negative 추가 |
| v5 | 21/30 | 약한 chunk 집중 보강 |
| v6 | 21/30 | 라벨 불일치 정리 |
| v7 | 22/30 | 고질 오답 집중 보강 |

## 의존 라이브러리

| 라이브러리 | 버전 | 확인 기준 |
|-----------|------|----------|
| `sentence-transformers` | `3.2.1` | `SentenceTransformer`, `InputExample`, `losses`, `cos_sim` |
| `qdrant-client` | `1.12.1` | `QdrantClient`, `Distance`, `VectorParams`, `PointStruct` |
| `torch` | `2.4.1` | CUDA 학습, `DataLoader`, tensor similarity |
| `numpy` | `1.24.4` | 학습 시드 고정 및 수치 처리 |
| `transformers` | `4.44.2` | `sentence-transformers` 백엔드 |
| `tqdm` | `4.67.1` | 임베딩/학습 progress bar |
| `scikit-learn` | `1.1.2` | `sentence-transformers` 의존성 |

