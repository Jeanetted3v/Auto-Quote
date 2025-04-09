import logging
import pandas as pd
import chromadb
from openai import OpenAI
import time
from typing import List, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)


class Embedder:
    """Generating embeddings of product names and storingin ChromaDB."""
    
    def __init__(
            self, 
            api_key: str, 
            model_name: str = "text-embedding-3-small", 
            collection_name: str = "product_embeddings",
            persist_dir: Optional[str] = "./data/embeddings"
        ):
        self.client_openai = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.client_chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client_chroma.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Product name embeddings"}
        )
    
    def load_csv(
        self,
        csv_path: str,
        product_name_column: str,
        id_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Load product data from a CSV file."""
        df = pd.read_csv(csv_path)
        if product_name_column not in df.columns:
            raise ValueError(f"Column '{product_name_column}' not found in CSV")
        if id_column is None:
            df["generated_id"] = [f"prod_{i}" for i in range(len(df))]
            id_column = "generated_id"
        elif id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in CSV")
        if df[id_column].duplicated().any():
            df["original_id"] = df[id_column]
            df[id_column] = [
                f"{row[id_column]}_{i}" 
                for i, row in df.reset_index().iterrows()
            ]
            logger.info(f"Found duplicate IDs in column '{id_column}'. "
                       "Generated unique IDs based on original values.")
        return df
    
    def generate_embeddings(
        self, texts: List[str], batch_size: int = 20
    ) -> List[List[float]]:
        """Generate embeddings for a list of product names
        
        Args:
            texts: List of product names to embed
            batch_size: Number of texts to process in each API call
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches to avoid API rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            clean_batch = []
            for text in batch:
                # Handle NaN, None, or empty strings
                if pd.isna(text) or text is None or text == "":
                    clean_batch.append("unknown product")
                else:
                    clean_batch.append(str(text).strip())
            try:
                response = self.client_openai.embeddings.create(
                    model=self.model_name,
                    input=clean_batch
                )
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                # Avoid rate limiting
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size}: {e}")
                raise
        return all_embeddings
    
    def add_to_chroma(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add embeddings to ChromaDB collection.
        
        Args:
            ids: List of unique IDs for each embedding
            embeddings: List of embedding vectors
            documents: List of original product names
            metadata: Optional list of metadata dictionaries for each product
        """
        if metadata is None:
            metadata = [{"type": "product_name"} for _ in ids]
        
        cleaned_documents = []
        for doc in documents:
            if pd.isna(doc) or doc is None or doc == "":
                cleaned_documents.append("unknown product")
            else:
                cleaned_documents.append(str(doc).strip())

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=cleaned_documents,
            metadatas=metadata
        )
    
    def embed_product_name(
        self, 
        csv_path: str, 
        product_name_column: str, 
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 20
    ) -> None:
        """Load a CSV file, generate embeddings, store in ChromaDB.
        
        Args:
            csv_path: Path to the CSV file
            product_name_column: Column name containing product names
            id_column: Column name to use as IDs (if None, will generate sequential IDs)
            metadata_columns: List of column names to include as metadata
            batch_size: Number of texts to process in each OpenAI API call
        """
        df = self.load_csv(csv_path, product_name_column, id_column)
        product_names = df[product_name_column].tolist()
        ids = df[id_column].astype(str).tolist()
        embeddings = self.generate_embeddings(product_names, batch_size)
        metadata = None
        if metadata_columns:
            metadata = []
            for _, row in df.iterrows():
                item_metadata = {col: row[col] for col in metadata_columns if col in df.columns}
                metadata.append(item_metadata)
        self.add_to_chroma(ids, embeddings, product_names, metadata)
    
        logger.info(f"Successfully added {len(product_names)} "
                    "product embeddings to ChromaDB collection")
    
    def query_similar(
        self, 
        query: str, 
        n_results: int = 1, 
        include_embeddings: bool = False,
        include_rerank: bool = False
    ) -> Dict:
        """Query the collection for products similar to the query.
        
        Args:
            query: The search query
            n_results: Number of similar products to return (default: 1)
            include_embeddings: Whether to include embeddings in results
            include_rerank: Whether to apply reranking (not implemented with base ChromaDB)
            
        Returns:
            Dictionary containing query results
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])[0]
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"] + 
                   (["embeddings"] if include_embeddings else [])
        )
        
        # Note: Basic ChromaDB doesn't support reranking natively.
        # If include_rerank is True, you would need to implement a custom reranker
        # or use a specialized library for this purpose.
        if include_rerank:
            print("Note: Reranking is not implemented in the base version. Results are ordered by embedding similarity.")
        
        return results
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection.
        
        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection.name,
            "count": count
        }


# Example usage:
if __name__ == "__main__":
    # Initialize embedder with your OpenAI API key
    embedder = Embedder(api_key="your-openai-api-key")
    
    # Process a CSV file
    embedder.process_csv(
        csv_path="products.csv",
        product_name_column="name",
        id_column="product_id",  # You can set this to None to auto-generate IDs
        metadata_columns=["category", "price"],
        batch_size=20  # Adjust based on API rate limits
    )
    
    # Query for the most similar product
    results = embedder.query_similar("smartphone", n_results=1)
    if results['documents'][0]:
        doc = results['documents'][0][0]
        dist = results['distances'][0][0]
        print(f"Most similar product: {doc} (distance: {dist:.4f})")