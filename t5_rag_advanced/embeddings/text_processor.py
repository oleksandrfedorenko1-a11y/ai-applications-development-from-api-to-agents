import os
from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE vectors")
            conn.commit()
        finally:
            conn.close()

    def process_text_file(
            self,
            file_name: str,
            chunk_size: int,
            overlap: int,
            dimensions: int,
            truncate: bool = False,
    ) -> None:
        if truncate:
            self._truncate_table()

        with open(file_name, "r") as f:
            content = f.read()

        chunks = chunk_text(content, chunk_size, overlap)
        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                document_name = os.path.basename(file_name)
                for i, chunk in enumerate(chunks):
                    embedding_str = str(embeddings[i])
                    cur.execute(
                        "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                        (document_name, chunk, embedding_str),
                    )
            conn.commit()
        finally:
            conn.close()

    def search(
            self,
            search_mode: SearchMode,
            query: str,
            top_k: int,
            min_score: float,
            dimensions: int,
    ) -> list[str]:
        query_embedding = self.embeddings_client.get_embeddings(query, dimensions)
        query_vector_str = str(query_embedding[0])

        op = "<->" if search_mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"

        sql = f"""
            SELECT text, embedding {op} %s::vector AS distance
            FROM vectors
            WHERE embedding {op} %s::vector <= %s
            ORDER BY distance
            LIMIT %s
        """

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (query_vector_str, query_vector_str, min_score, top_k))
                rows = cur.fetchall()
            return [row["text"] for row in rows]
        finally:
            conn.close()
