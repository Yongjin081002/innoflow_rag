"""
chunks.json 파싱: 각 청크에서 embed_text, base_fault, modifiers, category 추출
"""
import json
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_chunk(chunk_id: str, content: str) -> dict:
    """청크 content를 파싱하여 구조화된 데이터 반환"""
    lines = content.strip().split("\n")

    # category
    if chunk_id.startswith("보"):
        category = "보행자"
    elif chunk_id.startswith("거"):
        category = "자전거"
    else:
        category = "자동차"

    # 인덱스/참조 테이블인지 확인 (대표유형에 통합 등)
    is_index_table = bool(re.search(r"대표유형에?\s*통합|대표유형$", content, re.MULTILINE))
    # 발행처/개정 메타 청크
    if re.search(r"발\s*행\s*처|개\s*정\s*:", content):
        is_index_table = True

    # A/B 또는 보/차 역할 설명 추출
    role_a = ""
    role_b = ""
    for line in lines:
        line_s = line.strip()
        if category == "보행자":
            m = re.match(r"^\(보\)\s*(.+)", line_s)
            if m:
                role_a = m.group(1).strip()
            m = re.match(r"^\(차\)\s*(.+)", line_s)
            if m:
                role_b = m.group(1).strip()
        else:
            # 표준: (A) ... / (B) ...
            m = re.match(r"^\(A\)\s*(.+)", line_s)
            if m:
                role_a = m.group(1).strip()
            m = re.match(r"^\(B\)\s*(.+)", line_s)
            if m:
                role_b = m.group(1).strip()
            # "자동차 A : ..." 형식 (차1-5 등)
            if not role_a:
                m = re.match(r"^자동차\s*A\s*[:：]\s*(.+)", line_s)
                if m:
                    role_a = m.group(1).strip()
            if not role_b:
                m = re.match(r"^자동차\s*B\s*[:：]\s*(.+)", line_s)
                if m:
                    role_b = m.group(1).strip()

    # (가)/(나) 시나리오 형식에서 role_a 보완 (차12-1, 차33-1 등)
    if not role_a and not is_index_table:
        for line in lines:
            line_s = line.strip()
            # "(가)동시 (나)선진입 (다)후진입" → A 역할은 첫 줄에 있음
            m = re.match(r"^\(가\)\s*(.+)", line_s)
            if m and "(나)" in line_s:
                role_a = m.group(0).strip()  # 전체 시나리오 설명
                break
            # "(B) 유턴" 있는데 A 없는 경우 → A는 직진 (차33-1)
            if re.match(r"^\(B\)\s*유턴", line_s) and not role_a:
                role_a = "직진"
                break

    # 기본 과실비율 추출
    base_fault = {}
    for line in lines:
        line_s = line.strip()
        # 차/거: "기본 과실비율 A{num} B{num}" 패턴 (여러 줄에 걸칠 수 있음)
        m = re.search(r"기본\s*과실비율.*?A\s*(\d+)\s*B\s*(\d+)", line_s)
        if m:
            base_fault = {"A": int(m.group(1)), "B": int(m.group(2))}
            break
        # 보행자: "보행자 기본 과실비율 {num}" 패턴
        m = re.search(r"보행자\s*기본\s*과실비율\s*(\d+)", line_s)
        if m:
            base_fault = {"보행자": int(m.group(1)), "차량": 100 - int(m.group(1))}
            break
        # (가)(나)(다) 형태 - 여러 상황
        m = re.search(r"\(가\)\s*A\s*(\d+)\s*B\s*(\d+)", line_s)
        if m:
            base_fault = {"A": int(m.group(1)), "B": int(m.group(2))}
            # 추가 (나)(다)도 있을 수 있지만 기본은 (가) 사용
            break

    # 수정요소 추출
    modifiers = []
    # content에서 수정요소 패턴 매칭
    # 패턴: "A/B/보행자/차 + 조건 + (+/-숫자 or 비적용)"
    modifier_patterns = [
        # "A 현저한 과실 +10" 등
        (r"(A|B)\s+(현저한\s*과실)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(중대한\s*과실)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(현저한\s*과실)\s*비적용", "target", "condition", "na"),
        (r"(A|B)\s+(중���한\s*과실)\s*비적용", "target", "condition", "na"),
        # "A 서행불이행 +10" 등 일반 수정요소
        (r"(A|B)\s+(서행불이행)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(노면표시위반)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(대좌회전|소좌회전|대우회전|소우회전)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(명확한\s*선진입)\s*[-](\d+)", "target", "condition", "value_minus"),
        (r"(A|B)\s+(일시정지위반)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(진로변경\s*(?:신호불이행|신호불이행·지연))\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(좌회전\s*방법\s*위반)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(우회전\s*방법\s*위반)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(좌회전\s*금지\s*위반)\s*\+(\d+)", "target", "condition", "value_plus"),
        (r"(A|B)\s+(서행\(일시정지\s*포함\))\s*[-](\d+)", "target", "condition", "value_minus"),
        (r"(A|B)\s+(서행)\s*[-](\d+)", "target", "condition", "value_minus"),
        (r"(A|B)\s+(무리한\s*끼어들기[/·]진로방해)\s*\+(\d+)", "target", "condition", "value_plus"),
    ]

    # 더 포괄적인 패턴으로 수정요소 추출
    full_text = content.replace("\n", " ")

    # A/B 타겟의 +숫자 패턴
    for m in re.finditer(r"(A|B)\s+([\w·/\s\(\)]+?)\s*\+(\d+)", full_text):
        target = m.group(1)
        condition = re.sub(r"\s+", "_", m.group(2).strip())
        # 불필요한 접두사 제거
        condition = re.sub(r"^[①②③④⑤⑥⑦⑧]\s*", "", condition)
        if condition and condition not in ("과", "실", "비", "율", "조", "정", "예", "시"):
            value = int(m.group(3))
            mod = {"target": target, "condition": condition, "value": value}
            if mod not in modifiers:
                modifiers.append(mod)

    # A/B 타겟의 -숫자 패턴
    for m in re.finditer(r"(A|B)\s+([\w·/\s\(\)]+?)\s*[-](\d+)", full_text):
        target = m.group(1)
        condition = re.sub(r"\s+", "_", m.group(2).strip())
        condition = re.sub(r"^[①②③④⑤⑥⑦⑧]\s*", "", condition)
        if condition and condition not in ("과", "실", "비", "율", "조", "정", "예", "시"):
            value = -int(m.group(3))
            mod = {"target": target, "condition": condition, "value": value}
            if mod not in modifiers:
                modifiers.append(mod)

    # A/B 비적용 패턴
    for m in re.finditer(r"(A|B)\s+([\w·/\s\(\)]+?)\s*비적용", full_text):
        target = m.group(1)
        condition = re.sub(r"\s+", "_", m.group(2).strip())
        condition = re.sub(r"^[①②③④⑤⑥⑦⑧]\s*", "", condition)
        if condition and condition not in ("과", "실", "비", "율", "조", "정", "예", "시"):
            mod = {"target": target, "condition": condition, "value": "비적용"}
            if mod not in modifiers:
                modifiers.append(mod)

    # 보행자 수정요소
    if category == "보행자":
        for m in re.finditer(r"(야간[·\s]*기타\s*시야장애)\s*\+(\d+)", full_text):
            modifiers.append({"target": "보행자", "condition": "야간_기타_시야장애", "value": int(m.group(2))})
        for m in re.finditer(r"(ㄹ자\s*보행)\s*\+(\d+)", full_text):
            modifiers.append({"target": "보행자", "condition": "ㄹ자_보행", "value": int(m.group(2))})
        for m in re.finditer(r"(고의로\s*차량\s*진행\s*방해)\s*\+(\d+)", full_text):
            modifiers.append({"target": "보행자", "condition": "고의_차량_방해", "value": int(m.group(2))})
        for m in re.finditer(r"(간선도로)\s*\+(\d+)", full_text):
            modifiers.append({"target": "보행자", "condition": "간선도로", "value": int(m.group(2))})
        # 차량 쪽 수정요소
        for m in re.finditer(r"차의\s*(현저한\s*과실)\s*[-](\d+)", full_text):
            modifiers.append({"target": "차량", "condition": "현저한_과실", "value": -int(m.group(2))})
        for m in re.finditer(r"차의\s*(중대한\s*과실)\s*[-](\d+)", full_text):
            modifiers.append({"target": "차량", "condition": "중대한_과실", "value": -int(m.group(2))})
        # 감경 요소
        for m in re.finditer(r"(주택[·\s]*상점가[·\s]*학교)\s*[-](\d+)", full_text):
            modifiers.append({"target": "보행자", "condition": "주택_상점가_학교", "value": -int(m.group(2))})
        for m in re.finditer(r"(집단보행)\s*[-](\d+)", full_text):
            modifiers.append({"target": "보행자", "condition": "집단보행", "value": -int(m.group(2))})
        for m in re.finditer(r"(어린이[·\s]*노인[·\s]*장애인보호구역)\s*[-](\d+)", full_text):
            modifiers.append({"target": "보행자", "condition": "보호구역", "value": -int(m.group(2))})
        for m in re.finditer(r"(어린이[·\s]*노인[·\s]*장애인)\s*[-](\d+)", full_text):
            if "보호구역" not in m.group(0):
                modifiers.append({"target": "보행자", "condition": "어린이_노인_장애인", "value": -int(m.group(2))})
        if "보행자 급진입 비적용" in full_text or "보행자 급 진입" in full_text:
            modifiers.append({"target": "보행자", "condition": "급진입", "value": "비적용"})

    # 중복 제거
    seen = set()
    unique_modifiers = []
    for mod in modifiers:
        key = (mod["target"], mod["condition"], str(mod["value"]))
        if key not in seen:
            seen.add(key)
            unique_modifiers.append(mod)
    modifiers = unique_modifiers

    # embed_text 생성: ID + 사고 상황 핵심만 (수정요소 제외)
    if category == "보행자":
        embed_text = f"{chunk_id} {role_a} 대 {role_b}"
    else:
        embed_text = f"{chunk_id} {role_a} 대 {role_b}"

    # 사고상황 해설이 있으면 첫 문장 추가
    situation_match = re.search(r"사고\s*상황\s*\n?\s*⊙\s*(.+?)(?:\n|$)", content)
    if situation_match:
        situation = situation_match.group(1).strip()
        embed_text += f" - {situation}"

    # base_fault 정보도 embed_text에 포함
    if base_fault:
        if category == "보행자":
            embed_text += f" (보행자과실 {base_fault.get('보행자', '?')}%)"
        else:
            embed_text += f" (A{base_fault.get('A', '?')}:B{base_fault.get('B', '?')})"

    return {
        "id": chunk_id,
        "content": content,
        "embed_text": embed_text,
        "base_fault": base_fault,
        "modifiers": modifiers,
        "category": category,
        "is_rule": not is_index_table and bool(base_fault),
    }


def parse_all_chunks(chunks_path=None):
    if chunks_path is None:
        chunks_path = os.path.join(BASE_DIR, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [parse_chunk(c["id"], c["content"]) for c in chunks]


if __name__ == "__main__":
    parsed = parse_all_chunks()
    print(f"Total parsed: {len(parsed)}")

    # 카테고리별 통계
    cats = {}
    for p in parsed:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}건")

    # 샘플 출력
    samples = ["차1-1", "보27-1", "거4-4", "차12-1", "차43-4"]
    for sid in samples:
        p = next((x for x in parsed if x["id"] == sid), None)
        if p:
            print(f"\n{'='*60}")
            print(f"ID: {p['id']}")
            print(f"Category: {p['category']}")
            print(f"Embed text: {p['embed_text']}")
            print(f"Base fault: {p['base_fault']}")
            print(f"Modifiers ({len(p['modifiers'])}개):")
            for m in p["modifiers"]:
                print(f"  {m}")

    # 구조화 JSON 저장
    out_path = os.path.join(BASE_DIR, "chunks_structured.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")
