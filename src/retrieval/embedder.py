
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from schemas.schemas import VisualChunk, TextChunk


class Embedder:
    """
    Sadece embedding text üretimi ve vektöre çevirme işini yapar.
    Hiçbir veritabanı/collection bilgisi taşımaz.
    """

    def __init__(self, model_name="BAAI/bge-m3", device="mps", batch_size: int = 32):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
            model_kwargs={"device": device}
        )

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def visual_chunk_to_embedding_text(self, chunk: VisualChunk) -> str:
        return f"{chunk.section_path} | {chunk.source_caption} | {chunk.generated_text}".strip(" |")

    def text_chunk_to_embedding_text(self, chunk: TextChunk) -> str:
        if chunk.section_path and chunk.section_path != "unknown":
            return f"{chunk.section_path}\n\n{chunk.text}"
        return chunk.text

    async def aembed_query(self, query: str) -> list[float]:
        """embed_query'nin asenkron wrapper'ı."""
        return await asyncio.to_thread(self.embed_query, query)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """embed_documents'ın asenkron wrapper'ı."""
        return await asyncio.to_thread(self.embed_documents, texts)
