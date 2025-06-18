import os
import faiss
import pickle
import json
import numpy as np
from typing import List, Dict, Any

class FaissVectorDB:
    """
    A self-contained vector database using FAISS, designed to load and search
    PRE-COMPUTED embeddings from a given data source.

    This class does NOT connect to any external embedding APIs.
    """
    def __init__(self, name: str, embedding_key: str):
        """
        Initializes the database.

        Args:
            name (str): The name for the database, used for the storage directory.
            embedding_key (str): The key in the data source dictionary that holds
                                 the embedding vector to be used (e.g., "emb_ada").
        """
        if not embedding_key:
            raise ValueError("An 'embedding_key' must be provided.")
        
        self.name = name
        self.embedding_key = embedding_key
        
        # --- Define Paths ---
        self.db_dir = f"./data/{name}/"
        self.index_path = os.path.join(self.db_dir, "faiss.index")
        self.metadata_path = os.path.join(self.db_dir, "metadata.pkl")

        # --- Initialize State ---
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []

    def load_data(self, data: List[Dict[str, Any]]):
        """
        Loads pre-computed embeddings into the database. Idempotent.
        If a saved database exists on disk, it loads it. Otherwise, it processes
        and stores the new data.
        """
        # Check if data is already loaded in memory
        if self.index is not None and self.metadata:
            print("Vector database is already loaded in memory. Skipping data loading.")
            return

        # Check if a database exists on disk
        if os.path.exists(self.index_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        # If no data in memory and no DB on disk, process the new data
        print(f"No existing database found. Loading pre-computed embeddings using key: '{self.embedding_key}'...")
        self._load_precomputed_embeddings(data)
        print("Vector database loaded and saved.")

    def _load_precomputed_embeddings(self, data: List[Dict[str, Any]]):
        """
        Private method to extract embeddings and metadata from the source
        and load them into the FAISS index.
        """
        embeddings_to_load = []
        metadata_to_load = []

        for item in data:
            if self.embedding_key not in item:
                # Skip items that don't have the specified embedding
                print(f"Warning: Skipping item with id '{item.get('id')}' because it lacks the embedding key '{self.embedding_key}'.")
                continue
            
            # 1. Extract the embedding vector
            embeddings_to_load.append(item[self.embedding_key])
            
            # 2. Create the metadata by copying the item and removing all embedding keys
            meta = item.copy()
            # Find all keys starting with 'emb_' and remove them from the metadata dict
            emb_keys_to_pop = [k for k in meta.keys() if k.startswith('emb_')]
            for k in emb_keys_to_pop:
                meta.pop(k)
            metadata_to_load.append(meta)

        if not embeddings_to_load:
            raise ValueError(f"No valid embeddings found in the provided data using key '{self.embedding_key}'.")

        embeddings_np = np.array(embeddings_to_load, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity

        if self.index is None:
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatIP(dimension) # Index for Inner Product (Cosine Similarity)
            print(f"FAISS index created with dimension {dimension}.")

        self.index.add(embeddings_np)
        self.metadata.extend(metadata_to_load)
        
        self.save_db() # Save after initial creation

    def search(self, query_vector: List[float], k: int = 5, similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Searches the database using a pre-computed query vector.

        Args:
            query_vector (List[float]): The embedding vector of the query.
            k (int): The number of top results to return.
            similarity_threshold (float): The minimum similarity score for a result to be included.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No data loaded in the vector database. Use load_data() first.")

        query_np = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_np)  # Normalize the query vector

        scores, indices = self.index.search(query_np, k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            similarity_score = scores[0][i]

            if idx == -1 or similarity_score < similarity_threshold:
                continue

            results.append({
                "metadata": self.metadata[idx],
                "similarity": float(similarity_score),
            })

        return results

    def save_db(self):
        """
        Saves the FAISS index and metadata to the database directory.
        """
        os.makedirs(self.db_dir, exist_ok=True)

        if self.index:
            faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
            
        print(f"Database '{self.name}' saved to {self.db_dir}")

    def load_db(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        if not os.path.exists(self.db_dir):
            raise ValueError(f"Vector database directory not found at {self.db_dir}.")

        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Index or metadata file not found. The database is incomplete.")

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"Database '{self.name}' loaded successfully. Contains {self.index.ntotal} documents.")