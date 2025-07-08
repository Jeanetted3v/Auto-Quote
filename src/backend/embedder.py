"""This is the main script to generate embeddings for product names
and store them in ChromaDB. It uses OpenAI's embedding model to
generate embeddings from product names in a CSV file, and then
stores these embeddings in a ChromaDB collection for efficient
retrieval and similarity search.

To run:
python -m src.backend.embedder
"""
import os
import logging
import pandas as pd
import chromadb
from openai import OpenAI
import time
from typing import List, Dict, Optional, Any
import hydra
from omegaconf import DictConfig
from src.backend.utils.logging import setup_logging
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()


class Embedder:
    """Generates embeddings of product names and stores them in ChromaDB."""
    
    def __init__(
            self, 
            api_key: str, 
            model_name: str = "text-embedding-3-small", 
            collection_name: str = "product_embeddings",
            persist_dir: Optional[str] = "./data/embeddings"
        ):
        os.makedirs(persist_dir, exist_ok=True) 
        self.client_openai = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.client_chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client_chroma.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Product name embeddings"}
        )
    
    def _load_csv(
        self,
        csv_path: str,
        product_name_column: str
    ) -> pd.DataFrame:
        """Load product data from a CSV file."""
        df = pd.read_csv(csv_path)
        if product_name_column not in df.columns:
            raise ValueError(f"Column '{product_name_column}' not found in CSV")
        # Generate unique internal IDs
        df["generated_id"] = [f"prod_{i}" for i in range(len(df))]
        return df
    
    def _generate_embeddings(
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
                logger.error(f"Error generating embeddings for batch "
                             f"starting at index {i}: {e}")
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
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 20
    ) -> None:
        """Entry point: load a CSV file, generate embeddings, store in ChromaDB.
        
        Args:
            csv_path: Path to the CSV file
            product_name_column: Column name containing product names
            metadata_columns: List of column names to include as metadata
            batch_size: Number of texts to process in each OpenAI API call
        """
        df = self._load_csv(csv_path, product_name_column)
        product_names = df[product_name_column].tolist()
        ids = df["generated_id"].tolist()
        embeddings = self._generate_embeddings(product_names, batch_size)
        metadata = None
        if metadata_columns:
            metadata = []
            for _, row in df.iterrows():
                item_metadata = {
                    col: row.get(col, "")
                    if isinstance(row.get(col), (str, int, float, bool)) and pd.notna(row.get(col))
                    else ""
                    for col in metadata_columns
                }
                metadata.append(item_metadata)
        self.add_to_chroma(ids, embeddings, product_names, metadata)
    
        logger.info(f"Successfully added {len(product_names)} "
                    "product embeddings to ChromaDB collection")
    
    def query_similar(
        self, 
        query: str, 
        n_results: int = 1, 
        include_embeddings: bool = False
    ) -> Dict:
        """Query the collection for products similar to the query.
        
        Args:
            query: The search query
            n_results: Number of similar products to return (default: 1)
            include_embeddings: Whether to include embeddings in results\
            
        Returns:
            Dictionary containing query results
        """
        # Generate embedding for query
        query_embedding = self._generate_embeddings([query])[0]
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"] + 
                   (["embeddings"] if include_embeddings else [])
        )
        return results


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point to load, chunk, embed documents."""
    logger.info("Starting the embedding generation process for products name.")
    embedder = Embedder(
        api_key=SETTINGS.OPENAI_API_KEY,
        model_name=cfg.embedder_model,
        collection_name=cfg.chroma_collection_name,
        persist_dir=cfg.chroma_persist_dir
    )
    embedder.embed_product_name(
        csv_path=cfg.price_list_path,
        product_name_column="Products Name",
        metadata_columns=[
            "Products Name",
            "Product Code",
            "MIN PRICE",
            "STOCK PRICE",
            "AD-HOC PRICE",
            "UOM",
            "PRODUCT COLOR"
        ],
        batch_size=50
    )
    logger.info("Generated embeddings successfully.")


if __name__ == "__main__":
    main()