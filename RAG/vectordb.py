import os
import json
import pickle
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import voyageai


class _BaseFaissDB:
    """Shared Faiss helpers (normalisation, save/load, etc.)."""

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        *,
        emb_key: str = "emb_ada",
        use_cosine: bool = True,
    ) -> None:
        self.name = name
        self.emb_key = emb_key
        self.use_cosine = use_cosine  # cosine(=IP) vs L2

        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)

        # data holders
        self.index: Optional[faiss.Index] = None
        self.metadatas: List[Dict[str, Any]] = []
        self.query_cache: Dict[str, List[float]] = {}
        self.dim: Optional[int] = None

        # file paths
        self.dir_path = f"./data/{name}"
        os.makedirs(self.dir_path, exist_ok=True)
        self.index_path = os.path.join(self.dir_path, "faiss.index")
        self.meta_path = os.path.join(self.dir_path, "meta.pkl")

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        """Unit‑length normalisation for cosine/IP search."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """Call VoyageAI once for a batch of texts."""
        return self.client.embed(texts, model="voyage-2").embeddings

    def _embed_query(self, query: str) -> np.ndarray:
        if query not in self.query_cache:
            self.query_cache[query] = self._embed_text([query])[0]
        return np.asarray(self.query_cache[query], dtype="float32")

    # ---------------------------------------------------------------------
    # persistence
    # ---------------------------------------------------------------------
    def save_db(self) -> None:
        if self.index is None:
            raise ValueError("Index未建立，無法保存。")
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({
                "metadatas": self.metadatas,
                "query_cache": self.query_cache,
                "use_cosine": self.use_cosine,
            }, f)

    def load_db(self) -> None:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            raise FileNotFoundError("找不到已儲存的索引，請先 load_data() 建立。")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
        self.metadatas = data["metadatas"]
        self.query_cache = data.get("query_cache", {})
        self.use_cosine = data.get("use_cosine", True)
        self.dim = self.index.d

    # ---------------------------------------------------------------------
    # main public APIs – 每個子類覆寫 load_data / search 如有需要
    # ---------------------------------------------------------------------
    def load_data(self, raw_data: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int = 5, similarity_threshold: float = 0.4):
        raise NotImplementedError


class VectorDBFaiss(_BaseFaissDB):
    """替代原本 VectorDB，僅使用「正文」向量。"""

    def load_data(self, raw_data: List[Dict[str, Any]]) -> None:
        # 若已存在存檔，直接讀取
        if os.path.exists(self.index_path):
            print("載入現有索引…")
            self.load_db()
            return

        texts, metadatas, vecs = [], [], []
        for item in raw_data:
            text = f"Heading: {item.get('chunk_heading', 'Doc ' + str(item.get('id', '')))}\n\n{item.get('text', '')}"
            v = item.get(self.emb_key)
            if v is None:
                texts.append(text)
                metadatas.append(item)
            else:
                vecs.append(v)
                metadatas.append(item)

        # 先把已有向量的放進去
        vecs_np = np.asarray(vecs, dtype="float32")
        if self.use_cosine and len(vecs_np):
            vecs_np = self._normalize(vecs_np)

        if texts:
            generated = self._embed_text(texts)
            vecs_np = np.vstack([vecs_np, np.asarray(generated, dtype="float32")]) if len(vecs_np) else np.asarray(generated, dtype="float32")

        # 建立或更新 faiss index
        self.dim = vecs_np.shape[1]
        self.index = faiss.IndexFlatIP(self.dim) if self.use_cosine else faiss.IndexFlatL2(self.dim)
        self.index.add(vecs_np)

        self.metadatas = metadatas
        self.save_db()
        print("Faiss 索引建立完成，共", len(self.metadatas), "條紀錄")

    def search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if not hasattr(self, 'index') or self.index is None:
            raise ValueError("Index not loaded. Call load_data() first.")
        
        # Get query embedding
        q_emb = self._embed_query(query)
        
        # Convert to numpy array and ensure correct shape
        q_emb = np.array(q_emb, dtype=np.float32)
        
        # Ensure it's 2D: (1, embedding_dimension)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        elif q_emb.ndim > 2:
            q_emb = q_emb.reshape(1, -1)
        
        # Search in Faiss index
        D, I = self.index.search(q_emb, min(k * 3, len(self.data)))
        
        # Return results
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.data):
                results.append(self.data[idx])
        
        return results[:k]


class SummaryIndexedVectorDBFaiss(_BaseFaissDB):
    """對應 SummaryIndexedVectorDB：把 heading + text + summary 都餵進向量。"""

    def load_data(self, raw_data: List[Dict[str, Any]]) -> None:
        if os.path.exists(self.index_path):
            print("載入現有索引…")
            self.load_db()
            return

        texts, vecs, metadatas = [], [], []
        for item in raw_data:
            joined = (
                f"{item.get('chunk_heading', 'Doc ' + str(item.get('id', '')))}\n\n"
                f"{item.get('text', '')}\n\n"
                f"{item.get('summary', item.get('text', '')[:120])}"
            )
            v = item.get(self.emb_key)
            if v is None:
                texts.append(joined)
                metadatas.append(item)
            else:
                vecs.append(v)
                metadatas.append(item)

        vecs_np = np.asarray(vecs, dtype="float32")
        if self.use_cosine and len(vecs_np):
            vecs_np = self._normalize(vecs_np)

        if texts:
            generated = self._embed_text(texts)
            vecs_np = np.vstack([vecs_np, np.asarray(generated, dtype="float32")]) if len(vecs_np) else np.asarray(generated, dtype="float32")

        self.dim = vecs_np.shape[1]
        self.index = faiss.IndexFlatIP(self.dim) if self.use_cosine else faiss.IndexFlatL2(self.dim)
        self.index.add(vecs_np)

        self.metadatas = metadatas
        self.save_db()
        print("Faiss 索引建立完成，共", len(self.metadatas), "條紀錄")

    def search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if not hasattr(self, 'index') or self.index is None:
            raise ValueError("Index not loaded. Call load_data() first.")
        
        # Get query embedding
        q_emb = self._embed_query(query)
        
        # Ensure q_emb is properly shaped for Faiss
        if len(q_emb.shape) == 1:
            # Convert 1D array to 2D array with shape (1, dimension)
            q_emb = q_emb.reshape(1, -1)
        
        # Ensure it's float32 and the right shape
        q_emb = np.asarray(q_emb, dtype="float32")
        
        # Search in Faiss index
        D, I = self.index.search(q_emb, k * 3)
        
        # Rest of the method remains the same
        results = []
        for idx in I[0]:  # I[0] because we searched with 1 query
            if idx < len(self.data):
                results.append(self.data[idx])
        
        return results[:k]


__all__ = [
    "VectorDBFaiss",
    "SummaryIndexedVectorDBFaiss",
]
