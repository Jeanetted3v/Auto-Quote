"""To run:
python -m src.backend.main.data_ingest_main
"""
import logging
import hydra
import asyncio
from omegaconf import DictConfig
from src.backend.utils.logging import setup_logging
from src.backend.data_ingest.email_extractor import EmailExtractor
from src.backend.data_ingest.embedder import Embedder
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()

async def batch_extract(cfg: DictConfig) -> None:
    extractor = EmailExtractor(
        system_prompt=cfg.email_extractor_prompts.system_prompt
    )
    batch_results = await extractor.batch_extract(cfg.email_dir, cfg.output_dir)
    logger.info(f"Processed {len(batch_results)} files")


def create_embedding(cfg: DictConfig) -> None:
    embedder = Embedder(
        api_key=SETTINGS.OPENAI_API_KEY,
        model_name=cfg.embedder_model,
        collection_name=cfg.chroma_collection_name,
        persist_dir=cfg.chroma_persist_dir
    )
    embedder.embed_product_name(
        csv_path=cfg.price_list_path,
        product_name_column="Products Name",
        id_column="id",  # set this to None to auto-generate IDs
        metadata_columns=None,
        batch_size=50
    )
    logger.info("Generated embeddings successfully.")

@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="data_ingest")
def main(cfg: DictConfig) -> None:
    """Main entry point to load, chunk, embed documents."""
    logger.info("Starting the data ingestion process.")
    # asyncio.run(batch_extract(cfg))
    create_embedding(cfg)
    logger.info("Data ingestion process completed successfully.")


if __name__ == "__main__":
    main()