import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QDRANT_HOST = os.getenv("QDRANT_HOST", "")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fault_rules")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_fault_data")

_model_env = os.getenv("MODEL_PATH", "")
if _model_env:
    MODEL_PATH = _model_env if os.path.isabs(_model_env) else os.path.join(BASE_DIR, _model_env)
else:
    for _name in ("boost_best_vlm_v7", "boost_best_vlm_v4", "boost_best_vlm_v3"):
        _candidate = os.path.join(BASE_DIR, _name)
        if os.path.isdir(_candidate):
            MODEL_PATH = _candidate
            break
    else:
        MODEL_PATH = os.path.join(BASE_DIR, "boost_best_vlm_v4")
