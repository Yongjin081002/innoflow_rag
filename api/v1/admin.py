import json
import os
import threading
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPDATE_LOG_PATH = os.path.join(BASE_DIR, "update_log.json")

router = APIRouter(prefix="/admin", tags=["admin"])

_update_status: dict[str, Any] = {"running": False, "last_result": None}
_update_lock = threading.Lock()

# searcher 재초기화 콜백 (main.py에서 주입)
_reload_searcher: Any = None


def set_reload_callback(callback) -> None:
    global _reload_searcher
    _reload_searcher = callback


def _run_update(force: bool) -> None:
    with _update_lock:
        _update_status["running"] = True
        _update_status["last_result"] = None
    try:
        from update_pipeline import run as pipeline_run
        result = pipeline_run(force=force)
        if result.get("changed") and _reload_searcher:
            _reload_searcher()
        _update_status["last_result"] = result
    except Exception as e:
        _update_status["last_result"] = {"error": str(e)}
    finally:
        _update_status["running"] = False


@router.post("/update-rag")
def update_rag(background_tasks: BackgroundTasks, force: bool = False):
    """
    과실비율 인정기준 업데이트 파이프라인 실행.

    - force=false (기본): PDF 변경 시에만 업데이트
    - force=true: 변경 여부 무관하게 강제 업데이트
    """
    if _update_status["running"]:
        raise HTTPException(status_code=409, detail="업데이트가 이미 진행 중입니다.")
    background_tasks.add_task(_run_update, force)
    return {"message": "업데이트 파이프라인 시작됨", "force": force}


@router.get("/update-status")
def update_status():
    """마지막 업데이트 이력 및 현재 실행 상태 반환"""
    history: list[dict] = []
    if os.path.exists(UPDATE_LOG_PATH):
        try:
            with open(UPDATE_LOG_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    return {
        "running": _update_status["running"],
        "last_result": _update_status["last_result"],
        "last_log": history[0] if history else None,
        "history_count": len(history),
    }
