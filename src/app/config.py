from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    database_url: str = "postgresql+psycopg://booksearch:booksearch@localhost:5432/booksearch"
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    pdf_directory: Path = Path("data/pdfs")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5
