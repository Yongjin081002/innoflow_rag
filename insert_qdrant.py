from infrastructure.rag.ingest_qdrant import get_model, insert_all

__all__ = ["get_model", "insert_all"]


if __name__ == "__main__":
    insert_all()
