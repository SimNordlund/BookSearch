from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    database_url: str = "postgresql+psycopg://booksearch:booksearch@localhost:5432/booksearch"
    chat_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"
    query_rewrite_enabled: bool = True
    query_rewrite_model: str = "gpt-4.1-mini"
    rerank_enabled: bool = True
    rerank_model: str = "gpt-4.1-mini"
    rerank_candidate_count: int = 20
    lexical_search_enabled: bool = True
    lexical_search_config: Literal["simple", "swedish", "english"] = "simple"
    embedding_model: str = "text-embedding-3-small"
    #text-embedding-3-large <-- Bättre för böcker
    pdf_directory: Path = Path("data/pdfs")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5
    
    
    ##Reranking
    ##Bättre chunking + metadata
    ##Hybridsökning (embedding + BM25) + RRF
    ##Query rewriting 
    
##1.  Eval-frågor + mätning
##2. Hybrid search
##3. Reranking
##4. Bättre chunking/kapitelmetadata
##5. text-embedding-3-large
##6. Parent-child retrieval eller HyDE
