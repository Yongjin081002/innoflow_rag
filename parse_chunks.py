from infrastructure.rag.chunk_parser import parse_all_chunks, parse_chunk

__all__ = ["parse_all_chunks", "parse_chunk"]


if __name__ == "__main__":
    import runpy

    runpy.run_module("infrastructure.rag.chunk_parser", run_name="__main__")
