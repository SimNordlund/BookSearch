import hashlib

from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.engine import Engine


class LexicalSearch:
    """PostgreSQL full-text search for book chunks, kept separate from pgvector."""

    def __init__(self, engine: Engine, search_config: str) -> None:
        self.engine = engine
        self.search_config = search_config

    def initialize(self) -> None:
        """Create the application-owned full-text search tables and indexes."""
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS lexical_chunks (
                        chunk_key TEXT PRIMARY KEY,
                        book TEXT NOT NULL,
                        page INTEGER,
                        content TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        search_vector TSVECTOR NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS lexical_ingestions (
                        filename TEXT PRIMARY KEY,
                        content_sha256 TEXT NOT NULL,
                        search_config TEXT NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS lexical_chunks_search_vector_idx
                    ON lexical_chunks USING GIN (search_vector)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS lexical_chunks_book_idx
                    ON lexical_chunks (book)
                    """
                )
            )

    def is_indexed(self, filename: str, content_sha256: str) -> bool:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT 1 FROM lexical_ingestions
                    WHERE filename = :filename
                      AND content_sha256 = :content_sha256
                      AND search_config = :search_config
                    """
                ),
                {
                    "filename": filename,
                    "content_sha256": content_sha256,
                    "search_config": self.search_config,
                },
            ).first()
        return row is not None

    def replace_book(
        self,
        book: str,
        chunks: list[Document],
        content_sha256: str,
    ) -> None:
        """Replace all lexical chunks for one book after it has been ingested."""
        records = [
            {
                "chunk_key": self._chunk_key(chunk),
                "book": book,
                "page": chunk.metadata.get("page"),
                "content": chunk.page_content,
                "content_sha256": content_sha256,
                "search_config": self.search_config,
            }
            for chunk in chunks
        ]
        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM lexical_chunks WHERE book = :book"),
                {"book": book},
            )
            if records:
                connection.execute(
                    text(
                        """
                        INSERT INTO lexical_chunks (
                            chunk_key, book, page, content, content_sha256, search_vector
                        )
                        VALUES (
                            :chunk_key, :book, :page, :content, :content_sha256,
                            to_tsvector(CAST(:search_config AS regconfig), :content)
                        )
                        """
                    ),
                    records,
                )
            connection.execute(
                text(
                    """
                    INSERT INTO lexical_ingestions (
                        filename, content_sha256, search_config, chunk_count
                    )
                    VALUES (:filename, :content_sha256, :search_config, :chunk_count)
                    ON CONFLICT (filename) DO UPDATE SET
                        content_sha256 = EXCLUDED.content_sha256,
                        search_config = EXCLUDED.search_config,
                        chunk_count = EXCLUDED.chunk_count,
                        indexed_at = NOW()
                    """
                ),
                {
                    "filename": book,
                    "content_sha256": content_sha256,
                    "search_config": self.search_config,
                    "chunk_count": len(records),
                },
            )

    def search(
        self,
        query: str,
        book: str | None,
        limit: int,
    ) -> list[tuple[Document, float]]:
        """Return lexical matches ranked by PostgreSQL full-text relevance."""
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    WITH search_query AS (
                        SELECT websearch_to_tsquery(
                            CAST(:search_config AS regconfig), :query
                        ) AS value
                    )
                    SELECT
                        chunk.book,
                        chunk.page,
                        chunk.content,
                        chunk.content_sha256,
                        ts_rank_cd(chunk.search_vector, search_query.value) AS score
                    FROM lexical_chunks AS chunk
                    CROSS JOIN search_query
                    WHERE chunk.search_vector @@ search_query.value
                      AND (:book IS NULL OR chunk.book = :book)
                    ORDER BY score DESC, chunk.chunk_key
                    LIMIT :limit
                    """
                ),
                {
                    "query": query,
                    "book": book,
                    "limit": limit,
                    "search_config": self.search_config,
                },
            ).mappings().all()

        return [
            (
                Document(
                    page_content=row["content"],
                    metadata={
                        "book": row["book"],
                        "page": row["page"],
                        "content_sha256": row["content_sha256"],
                    },
                ),
                float(row["score"]),
            )
            for row in rows
        ]

    @staticmethod
    def _chunk_key(chunk: Document) -> str:
        content = "|".join(
            [
                chunk.metadata["book"],
                str(chunk.metadata.get("page", "")),
                chunk.page_content,
            ]
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
