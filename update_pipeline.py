import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent
PDF_URL = "https://www.knia.or.kr/file-manager/105814"
PDF_PATH = BASE_DIR / "docs" / "fault_rules_latest.pdf"
HASH_PATH = BASE_DIR / "docs" / "fault_rules_latest.sha256"
TEXT_PATH = BASE_DIR / "docs" / "fault_rules_latest.txt"
CHUNKS_PATH = BASE_DIR / "chunks.json"
STRUCTURED_PATH = BASE_DIR / "chunks_structured.json"
UPDATE_LOG_PATH = BASE_DIR / "update_log.json"
CRON_LINE = (
    "0 3 1 1,7 * /home/minsung0830/miniconda3/bin/python3 "
    "/home/minsung0830/innoflow_rag/update_pipeline.py "
    ">> /home/minsung0830/innoflow_rag/cron.log 2>&1"
)

RULE_ID_RE = re.compile(r"^(?:차|보|거)\d{1,2}-\d{1,2}$")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_text(path):
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def atomic_write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def download_pdf(url=PDF_URL):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    content = response.content
    if not content.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded file is not a PDF: content-type={content_type!r}")
    return content, content_type


def extract_pages_with_pdfplumber(pdf_path):
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            pages.append((page_number, text))
    return pages


def normalize_page_text(text):
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return lines


def chunk_pages(pages):
    chunks = []
    current_id = None
    current_lines = []

    def flush():
        if current_id is None:
            return
        content_lines = [current_id] + current_lines
        content = "\n".join(line for line in content_lines if line).strip()
        chunks.append({"id": current_id, "content": content})

    for page_number, text in pages:
        page_marker = f"=== {page_number}페이지 ==="
        for line in normalize_page_text(text):
            if RULE_ID_RE.fullmatch(line):
                flush()
                current_id = line
                current_lines = [page_marker]
                continue
            if current_id is not None:
                current_lines.append(line)

    flush()
    return chunks


def write_extracted_text(pages):
    parts = []
    for page_number, text in pages:
        parts.append(f"=== {page_number}페이지 ===\n{text.strip()}")
    TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_PATH.write_text("\n\n".join(parts), encoding="utf-8")


def write_structured_chunks():
    from parse_chunks import parse_all_chunks

    parsed = parse_all_chunks(str(CHUNKS_PATH))
    atomic_write_json(STRUCTURED_PATH, parsed)
    return parsed


def reinsert_qdrant():
    from insert_qdrant import insert_all

    return insert_all()


def write_update_log(entry):
    existing = []
    if UPDATE_LOG_PATH.exists():
        try:
            with UPDATE_LOG_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                existing = data
            elif isinstance(data, dict):
                existing = [data]
        except json.JSONDecodeError:
            backup_path = UPDATE_LOG_PATH.with_suffix(".json.bak")
            shutil.copy2(UPDATE_LOG_PATH, backup_path)
    existing.append(entry)
    atomic_write_json(UPDATE_LOG_PATH, existing[-30:])


def run_pipeline():
    log = {
        "started_at": now_iso(),
        "pdf_url": PDF_URL,
        "status": "running",
        "steps": {},
    }

    try:
        pdf_content, content_type = download_pdf()
        pdf_hash = sha256_bytes(pdf_content)
        previous_hash = read_text(HASH_PATH)
        hash_changed = previous_hash != pdf_hash
        atomic_write_bytes(PDF_PATH, pdf_content)
        HASH_PATH.write_text(pdf_hash + "\n", encoding="utf-8")
        log["steps"]["download"] = {
            "ok": True,
            "content_type": content_type,
            "bytes": len(pdf_content),
            "path": str(PDF_PATH),
        }
        log["steps"]["hash"] = {
            "ok": True,
            "previous_hash": previous_hash,
            "current_hash": pdf_hash,
            "changed": hash_changed,
        }
        print(f"PDF 다운로드 성공: {len(pdf_content)} bytes")
        print(f"해시 비교 완료: changed={hash_changed}")

        pages = extract_pages_with_pdfplumber(PDF_PATH)
        write_extracted_text(pages)
        total_text_chars = sum(len(text) for _, text in pages)
        log["steps"]["extract_text"] = {
            "ok": True,
            "pages": len(pages),
            "text_chars": total_text_chars,
            "path": str(TEXT_PATH),
            "extractor": "pdfplumber",
        }
        print(f"텍스트 추출 완료: pages={len(pages)}, chars={total_text_chars}")

        chunks = chunk_pages(pages)
        if not chunks:
            raise RuntimeError("Chunking produced 0 chunks")
        atomic_write_json(CHUNKS_PATH, chunks)
        log["steps"]["chunk"] = {
            "ok": True,
            "chunk_count": len(chunks),
            "path": str(CHUNKS_PATH),
            "sample_ids": [chunk["id"] for chunk in chunks[:10]],
        }
        print(f"청킹 완료: chunks={len(chunks)}")

        parsed = write_structured_chunks()
        rule_count = sum(1 for item in parsed if item.get("is_rule"))
        log["steps"]["structure"] = {
            "ok": True,
            "structured_count": len(parsed),
            "rule_count": rule_count,
            "path": str(STRUCTURED_PATH),
        }
        print(f"구조화 완료: structured={len(parsed)}, rules={rule_count}")

        inserted = reinsert_qdrant()
        log["steps"]["qdrant"] = {"ok": True, "inserted": inserted}
        print(f"Qdrant 재삽입 완료: inserted={inserted}")

        log["status"] = "success"
        return log
    except Exception as exc:
        log["status"] = "failed"
        log["error"] = repr(exc)
        raise
    finally:
        log["finished_at"] = now_iso()
        write_update_log(log)
        print(f"update_log 저장: {UPDATE_LOG_PATH}")


def main():
    try:
        run_pipeline()
    except Exception as exc:
        print(f"업데이트 파이프라인 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
