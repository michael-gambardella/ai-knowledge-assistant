from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Anthropic
    anthropic_api_key: str = ""

    # LLM
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # Document registry
    document_registry_path: str = "./document_registry.json"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k_results: int = 5

    # App
    app_env: str = "development"
    log_level: str = "INFO"


# Single shared instance — imported everywhere
settings = Settings()
