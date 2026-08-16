from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    book: str | None = Field(
        default=None,
        description="Optional PDF filename, for example 'my-book.pdf'.",
    )
    top_k: int | None = Field(default=None, ge=1, le=10)
    evaluate: bool = Field(
        default=False,
        description="Run an additional AI-as-judge evaluation against the retrieved passages.",
    )


class SourceChunk(BaseModel):
    book: str
    page: int | None
    excerpt: str
    relevance_score: float


class CriterionEvaluation(BaseModel):
    score: int = Field(ge=0, le=4)
    reason: str = Field(min_length=1, max_length=300)


class JudgeAssessment(BaseModel):
    """The structured assessment returned by the judge model."""

    groundedness: CriterionEvaluation = Field(
        description="Whether factual claims are supported by the retrieved passages."
    )
    relevance: CriterionEvaluation = Field(
        description="Whether the answer directly answers the user's question."
    )
    citation_support: CriterionEvaluation = Field(
        description="Whether claims have useful, accurate book and page citations."
    )
    clarity: CriterionEvaluation = Field(
        description="Whether the answer is clear, concise, and easy to understand."
    )


class AnswerEvaluation(JudgeAssessment):
    score: float = Field(ge=0, le=100)
    verdict: Literal["pass", "review", "fail"]
    summary: str = Field(min_length=1, max_length=500)


class JudgeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    answer: str = Field(min_length=1, max_length=12_000)
    sources: list[SourceChunk] = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    evaluation: AnswerEvaluation | None = None


class Book(BaseModel):
    filename: str
    chunk_count: int
    indexed_at: str
