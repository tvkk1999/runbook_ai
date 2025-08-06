# vector_store.py

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from config import EMBEDDINGS_DIR, VECTOR_STORE_COLLECTION_NAME, EMBEDDING_MODEL_NAME


class VectorStore:
    def __init__(self, persist_directory: str = EMBEDDINGS_DIR):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

    def create_collection(self, collection_name: str = VECTOR_STORE_COLLECTION_NAME):
        """Create or get collection for storing embeddings."""
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Runbook document embeddings"}
            )

    def add_documents(self, chunks_with_meta: list):
        """
        Add document chunks to the vector store.

        Each chunk_with_meta is a dict with keys:
        - "type": chunk type (text, image, table)
        - "content": actual chunk content (text or table data, empty string for images)
        - "metadata": metadata dict including captions, file refs, chunk_type, etc.

        Embeddings are generated from textual data for text chunks,
        and from captions for image/table chunks enabling semantic search.

        Tables are converted to markdown-like string for storage required by ChromaDB.
        """
        if not self.collection:
            self.create_collection()

        texts_to_embed = []
        metadatas = []
        ids = []
        chroma_docs = []  # documents as strings for ChromaDB

        for idx, chunk in enumerate(chunks_with_meta):
            chunk_type = chunk.get("type", "text")
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")

            # Prepare embedding text: content for text, captions for images/tables
            if chunk_type == "text":
                text_for_embedding = content
            elif chunk_type in ["image", "table"]:
                caption = metadata.get("caption", "No caption")
                text_for_embedding = f"[{chunk_type.upper()}] {caption}"
            else:
                text_for_embedding = content if isinstance(content, str) else str(content)

            texts_to_embed.append(text_for_embedding)

            # Enhance metadata with chunk info
            metadata_enhanced = metadata.copy()
            metadata_enhanced["chunk_type"] = chunk_type
            metadata_enhanced["chunk_index"] = idx
            
            def sanitize_metadata(md):
                md = dict(md)
                for k, v in md.items():
                    if not isinstance(v, (str, int, float, bool, type(None))):
                        md[k] = str(v)
                return md
            
            metadatas.append(sanitize_metadata(metadata_enhanced))

            ids.append(str(uuid.uuid4()))

            # Prepare document string for ChromaDB:
            # For tables, convert list-of-lists to markdown-like string
            if chunk_type == "table" and isinstance(content, list) and all(isinstance(row, list) for row in content):
                # Convert each row to pipe-separated line
                content_str = "\n".join([" | ".join(cell if cell is not None else "" for cell in row) for row in content])
                chroma_docs.append(content_str)
            else:
                # For images content may be empty string, for text ensure string
                chroma_docs.append(content if isinstance(content, str) else str(content))

        # Generate embeddings for all chunks
        embeddings = self.embedding_model.encode(texts_to_embed).tolist()

        # Add all data to vector store collection
        self.collection.add(
            embeddings=embeddings,
            documents=chroma_docs,      # all strings as required by ChromaDB
            metadatas=metadatas,
            ids=ids
        )
        print(self.collection)

    def search_similar(self, query: str, n_results: int = 5):
        """
        Perform semantic search on the vector store to find most relevant chunks.

        Returns dict with keys like "documents" and "metadatas", each a list of lists (for queries).
        """
        print(self.collection,"<<")
        if not self.collection:
            print("Hi")
            self.create_collection()

        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results

    def clear_collection(self):
        """Delete the current collection and reset."""
        if self.collection:
            self.chroma_client.delete_collection(self.collection.name)
            self.collection = None
