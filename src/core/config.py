from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "rag_docs"

    # Embedding
    embedding_url: str = "http://jina-embedding:18081"
    embedding_model: str = "jinaai/jina-embeddings-v3"
    embedding_dim: int = 1024

    # LLM
    vllm_url: str = "http://host.docker.internal/v1"
    vllm_model: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    vllm_api_key: str = "secret-key-change-me"

    # RAG
    top_k: int = 20
    top_k_rerank: int = 5
    similarity_threshold: float = 0.3
    chunk_size: int = 512
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


settings = Settings()
