import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QDRANT_HOST = os.getenv("QDRANT_HOST", "")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fault_rules")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_fault_data")

def select_model_path(base_dir=BASE_DIR):
    candidates = []
    for name in os.listdir(base_dir):
        if not name.startswith("boost_best_vlm_v"):
            continue
        version = name.removeprefix("boost_best_vlm_v")
        if not version.isdigit():
            continue
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            candidates.append((int(version), path))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return os.path.join(base_dir, "boost_best_vlm_v4")


_model_env = os.getenv("MODEL_PATH", "")
if _model_env:
    MODEL_PATH = _model_env if os.path.isabs(_model_env) else os.path.join(BASE_DIR, _model_env)
else:
    MODEL_PATH = select_model_path(BASE_DIR)
