from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Bước 1: Truy xuất (Retrieve) các đoạn văn bản liên quan nhất từ Store
        results = self.store.search(question, top_k=top_k)
        
        # Bước 2: Trích xuất nội dung văn bản từ kết quả tìm kiếm
        # (Dựa trên cấu trúc dict trả về từ EmbeddingStore.search ở bước trước)
        context_chunks = [res["content"] for res in results]
        context_text = "\n\n".join(context_chunks)

        # Bước 3: Xây dựng Prompt (Prompt Engineering)
        # Chúng ta nhúng "ngữ cảnh" tìm được vào để ép LLM trả lời dựa trên đó
        prompt = self._build_prompt(question, context_text)

        # Bước 4: Gọi LLM (Generate) để lấy câu trả lời cuối cùng
        response = self.llm_fn(prompt)
        
        return response

    def _build_prompt(self, question: str, context: str) -> str:
        """Tạo cấu trúc prompt chuyên nghiệp cho LLM."""
        return f"""Bạn là một trợ lý thông minh. Hãy sử dụng phần NGỮ CẢNH dưới đây để trả lời CÂU HỎI của người dùng. 
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không biết, đừng tự bịa ra câu trả lời.

NGỮ CẢNH:
{context}

CÂU HỎI: 
{question}

TRẢ LỜI:"""
