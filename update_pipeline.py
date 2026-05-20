"""
과실비율 인정기준 자동 업데이트 파이프라인

동작 순서:
  1. 손보협 사이트에서 PDF URL 탐색 후 다운로드
  2. SHA-256 해시 비교 → 변경 없으면 종료
  3. pdfplumber로 텍스트 추출 → output.txt 저장
  4. 청킹 → chunks.json 업데이트
  5. parse_chunks + insert_qdrant → Qdrant 재삽입
  6. update_log.json 기록
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STANDARD_PAGE_URL = "https://accident.knia.or.kr/standard"
PDF_FALLBACK_URL = "https://www.knia.or.kr/file-manager/105815"

PDF_PATH = os.path.join(BASE_DIR, "fault_standard.pdf")
PDF_HASH_PATH = os.path.join(BASE_DIR, "fault_standard.sha256")
OUTPUT_TXT_PATH = os.path.join(BASE_DIR, "output.txt")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.json")
UPDATE_LOG_PATH = os.path.join(BASE_DIR, "update_log.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "update_pipeline.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. PDF 다운로드
# ─────────────────────────────────────────────

def _find_pdf_url() -> str:
    """손보협 기준정보 페이지에서 최신 PDF URL 탐색"""
    try:
        resp = requests.get(STANDARD_PAGE_URL, timeout=15)
        resp.raise_for_status()
        # 과실비율 인정기준 PDF 링크 패턴
        match = re.search(
            r'href=["\']([^"\']*file-manager/\d+)["\'][^>]*>[^<]*과실비율\s*인정기준',
            resp.text,
        )
        if match:
            url = match.group(1)
            if url.startswith("//"):
                url = "https:" + url
            log.info(f"페이지에서 PDF URL 탐색 성공: {url}")
            return url
    except Exception as e:
        log.warning(f"PDF URL 탐색 실패, fallback 사용: {e}")
    return PDF_FALLBACK_URL


def download_pdf() -> bytes:
    """PDF 다운로드 후 바이트 반환"""
    url = _find_pdf_url()
    log.info(f"PDF 다운로드 시작: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; InnoflowBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type and len(resp.content) < 1000:
        raise ValueError(f"PDF 응답이 아닙니다. Content-Type: {content_type}")
    log.info(f"PDF 다운로드 완료: {len(resp.content):,} bytes")
    return resp.content


# ─────────────────────────────────────────────
# 2. 해시 비교
# ─────────────────────────────────────────────

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_stored_hash() -> str:
    if os.path.exists(PDF_HASH_PATH):
        with open(PDF_HASH_PATH, "r") as f:
            return f.read().strip()
    return ""


def save_hash(h: str) -> None:
    with open(PDF_HASH_PATH, "w") as f:
        f.write(h)


# ─────────────────────────────────────────────
# 3. 텍스트 추출
# ─────────────────────────────────────────────

def extract_text(pdf_bytes: bytes) -> str:
    """pdfplumber로 PDF 전체 텍스트 추출"""
    import pdfplumber
    import io

    log.info("PDF 텍스트 추출 중...")
    all_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            all_text.append(text)
            if i % 10 == 0:
                log.info(f"  {i}/{len(pdf.pages)} 페이지 처리 중...")

    result = "\n".join(all_text)
    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    log.info(f"텍스트 추출 완료: {len(result):,}자 → {OUTPUT_TXT_PATH}")
    return result


# ─────────────────────────────────────────────
# 4. 청킹 (case ID 기준 분할)
# ─────────────────────────────────────────────

# 차N-N, 보N-N, 거N-N 패턴
_CHUNK_ID_PATTERN = re.compile(r"^(차\d+-\d+|보\d+-\d+|거\d+-\d+)\s*$", re.MULTILINE)


def chunk_text(text: str) -> list[dict]:
    """
    텍스트를 case ID(차N-N, 보N-N, 거N-N) 기준으로 분할하여 chunks.json 형식으로 반환
    """
    matches = list(_CHUNK_ID_PATTERN.finditer(text))
    if not matches:
        log.warning("청크 ID 패턴을 찾지 못했습니다. PDF 구조를 확인하세요.")
        return []

    chunks = []
    for i, match in enumerate(matches):
        chunk_id = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            chunks.append({"id": chunk_id, "content": content})

    log.info(f"청킹 완료: {len(chunks)}개 chunk 생성")
    return chunks


def update_chunks(text: str) -> int:
    """chunks.json 업데이트, 생성된 chunk 수 반환"""
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("청킹 결과가 비어 있습니다.")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info(f"chunks.json 업데이트 완료: {CHUNKS_PATH}")
    return len(chunks)


# ─────────────────────────────────────────────
# 5. 임베딩 및 Qdrant 재삽입
# ─────────────────────────────────────────────

def reinsert_qdrant() -> int:
    """parse_chunks + insert_qdrant 로직으로 Qdrant 재삽입"""
    log.info("Qdrant 재삽입 시작...")

    sys.path.insert(0, BASE_DIR)
    from parse_chunks import parse_all_chunks
    from insert_qdrant import insert_all

    total = insert_all()
    log.info(f"Qdrant 재삽입 완료: {total}건")
    return total


# ─────────────────────────────────────────────
# 6. 로그 기록
# ─────────────────────────────────────────────

def write_log(changed: bool, chunks: int = 0, qdrant_count: int = 0, error: str = "") -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "changed": changed,
        "chunks": chunks,
        "qdrant_count": qdrant_count,
        "error": error,
    }

    history = []
    if os.path.exists(UPDATE_LOG_PATH):
        try:
            with open(UPDATE_LOG_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.insert(0, entry)
    history = history[:50]  # 최근 50건만 유지

    with open(UPDATE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    log.info(f"로그 저장 완료: {UPDATE_LOG_PATH}")


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def run(force: bool = False) -> dict:
    """
    전체 업데이트 파이프라인 실행

    Args:
        force: True이면 해시 비교 없이 강제 업데이트

    Returns:
        실행 결과 dict
    """
    log.info("=" * 60)
    log.info("과실비율 인정기준 업데이트 파이프라인 시작")
    log.info("=" * 60)

    try:
        # 1. PDF 다운로드
        pdf_bytes = download_pdf()

        # 2. 해시 비교
        new_hash = sha256(pdf_bytes)
        stored_hash = load_stored_hash()

        if not force and new_hash == stored_hash:
            log.info("변경 없음 — 업데이트를 건너뜁니다.")
            write_log(changed=False)
            return {"changed": False, "message": "변경 없음"}

        log.info(f"변경 감지 (구: {stored_hash[:12]}... → 신: {new_hash[:12]}...)")

        # PDF 저장
        with open(PDF_PATH, "wb") as f:
            f.write(pdf_bytes)

        # 3. 텍스트 추출
        text = extract_text(pdf_bytes)

        # 4. 청킹
        chunk_count = update_chunks(text)

        # 5. Qdrant 재삽입
        qdrant_count = reinsert_qdrant()

        # 6. 해시 저장 및 로그
        save_hash(new_hash)
        write_log(changed=True, chunks=chunk_count, qdrant_count=qdrant_count)

        log.info("=" * 60)
        log.info(f"업데이트 완료 — chunk {chunk_count}건, Qdrant {qdrant_count}건")
        log.info("=" * 60)

        return {
            "changed": True,
            "chunks": chunk_count,
            "qdrant_count": qdrant_count,
            "message": "업데이트 완료",
        }

    except Exception as e:
        log.error(f"파이프라인 오류: {e}", exc_info=True)
        write_log(changed=False, error=str(e))
        raise


if __name__ == "__main__":
    force = "--force" in sys.argv
    result = run(force=force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
