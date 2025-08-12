import os, json, math
from typing import List, Dict, Any, Iterable, Tuple
import numpy as np

DEFAULT_PATH = "./storage"
INDEX_FILE = "index.json"
EMB_FILE = "embeddings.npy"

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

class TinyVectorStore:
    def __init__(self, storage_dir: str = DEFAULT_PATH):
        self.storage_dir = storage_dir
        _ensure_dir(self.storage_dir)
        self.index_path = os.path.join(self.storage_dir, INDEX_FILE)
        self.emb_path = os.path.join(self.storage_dir, EMB_FILE)
        self.meta: List[Dict[str, Any]] = []
        self.emb: np.ndarray | None = None
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.emb_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.emb = np.load(self.emb_path)
        else:
            self.meta = []
            self.emb = np.zeros((0, 1536), dtype=np.float32)  # placeholder shape

    def _save(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        if self.emb is None:
            self.emb = np.zeros((0, 1536), dtype=np.float32)
        np.save(self.emb_path, self.emb)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        if self.emb is None or self.emb.size == 0:
            self.emb = embeddings.astype(np.float32)
        else:
            self.emb = np.vstack([self.emb, embeddings.astype(np.float32)])
        self.meta.extend(metadatas)
        self._save()

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if self.emb is None or len(self.meta) == 0:
            return []
        sims = np.dot(self.emb, query_emb) / (np.linalg.norm(self.emb, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.meta[i]) for i in idx]
