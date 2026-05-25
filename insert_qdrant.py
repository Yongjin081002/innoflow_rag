from infrastructure.rag.ingest_qdrant import get_model, insert_all, select_model_device

__all__ = ["get_model", "insert_all", "select_model_device"]


if __name__ == "__main__":
    insert_all()
