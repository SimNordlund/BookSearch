from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    book: str | None = Field(
        default=None,
        description="Optional PDF filename, for example 'my-book.pdf'.",
    )
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    book: str
    page: int | None
    excerpt: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class Book(BaseModel):
    filename: str
    chunk_count: int
    indexed_at: str
