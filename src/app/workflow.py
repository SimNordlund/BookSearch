"""LangGraph orchestration for the book-question RAG flow."""

from typing import TYPE_CHECKING, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.schemas import AnswerEvaluation, ChatResponse, SourceChunk

if TYPE_CHECKING:
    from app.rag import RAGService


class RAGWorkflowState(TypedDict, total=False):
    """Data passed between the RAG workflow's nodes."""

    question: str
    book: str | None
    top_k: int | None
    evaluate: bool
    documents_with_scores: list[tuple[Document, float]]
    answer: str
    sources: list[SourceChunk]
    evaluation: AnswerEvaluation | None


class RAGWorkflow:
    """Make retrieval, answer generation, and optional judging explicit nodes."""

    def __init__(self, service: "RAGService") -> None:
        self.service = service
        self.graph = self._build_graph()

    def invoke(
        self,
        question: str,
        book: str | None,
        top_k: int | None,
        evaluate: bool,
    ) -> ChatResponse:
        result = self.graph.invoke(
            {
                "question": question,
                "book": book,
                "top_k": top_k,
                "evaluate": evaluate,
            }
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            evaluation=result.get("evaluation"),
        )

    def _build_graph(self):
        workflow = StateGraph(RAGWorkflowState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("judge", self._judge)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_conditional_edges(
            "generate_answer",
            self._should_judge,
            {"judge": "judge", "end": END},
        )
        workflow.add_edge("judge", END)
        return workflow.compile()

    def _retrieve(self, state: RAGWorkflowState) -> RAGWorkflowState:
        return {
            "documents_with_scores": self.service._retrieve(
                state["question"],
                state.get("book"),
                state.get("top_k"),
            )
        }

    def _generate_answer(self, state: RAGWorkflowState) -> RAGWorkflowState:
        answer, sources = self.service._generate_answer(
            state["question"], state["documents_with_scores"], state.get("book")
        )
        return {"answer": answer, "sources": sources}

    def _should_judge(self, state: RAGWorkflowState) -> str:
        if state.get("evaluate") and state.get("sources"):
            return "judge"
        return "end"

    def _judge(self, state: RAGWorkflowState) -> RAGWorkflowState:
        return {
            "evaluation": self.service.judge.evaluate(
                state["question"], state["answer"], state["sources"]
            )
        }
