import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.config import Settings
from app.rag import RAGService
from app.schemas import AnswerEvaluation, Book, ChatRequest, ChatResponse, JudgeRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RAGService(Settings())
    await asyncio.to_thread(service.initialize)
    app.state.rag_service = service
    yield
    service.engine.dispose()

app = FastAPI(
    title="BookSearch RAG API",
    description="Ask questions about PDFs placed in data/pdfs.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/books", response_model=list[Book])
def list_books(request: Request) -> list[Book]:
    return request.app.state.rag_service.list_books()

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        return request.app.state.rag_service.answer(
            question=payload.question,
            book=payload.book,
            top_k=payload.top_k,
            evaluate=payload.evaluate,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail="Unable to answer the question.") from error


@app.post("/evaluate", response_model=AnswerEvaluation)
def evaluate(payload: JudgeRequest, request: Request) -> AnswerEvaluation:
    """Evaluate a supplied RAG answer against its retrieved source passages."""
    try:
        return request.app.state.rag_service.judge.evaluate(
            question=payload.question,
            answer=payload.answer,
            sources=payload.sources,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail="Unable to evaluate the answer.") from error
