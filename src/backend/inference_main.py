"""This is the main script to extract product items from RFQ emails,
Convert items into embeddings,  
To run:
python -m src.backend.main.inference_main
"""
import logging
import hydra
import asyncio
from omegaconf import DictConfig
from src.backend.utils.logging import setup_logging
from src.backend.extractor import Extractor
from src.backend.embedder import Embedder
from src.backend.quotes_generator import QuoteGenerator
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()

async def batch_extract(cfg: DictConfig) -> None:
    extractor = Extractor(cfg)
    batch_results = await extractor.batch_extract(cfg.email_dir, cfg.output_dir)
    logger.info(f"Processed {len(batch_results)} files")


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point to load, chunk, embed documents."""
    # logger.info("Starting the extraction from email.")
    # asyncio.run(batch_extract(cfg))
    # logger.info("Extraction completed successfully.")

    logger.info("Start matching process")
    embedder = Embedder(
        api_key=SETTINGS.OPENAI_API_KEY,
        model_name=cfg.embedder_model,
        collection_name=cfg.chroma_collection_name,
        persist_dir=cfg.chroma_persist_dir
    )
    quote_gen = QuoteGenerator(cfg, embedder)
    
    


if __name__ == "__main__":
    main()