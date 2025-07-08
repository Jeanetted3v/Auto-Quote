import logging
import os
from typing import Optional, List
from pydantic import BaseModel
from pydantic_ai import Agent, ImageUrl
from pdf2image import convert_from_path
import base64
from io import BytesIO
from PIL import Image
import csv, json
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)

class ProductItem(BaseModel):
    name: Optional[str] = None
    impa_code: Optional[str] = None
    quantity: Optional[int] = None
    unit: Optional[str] = None

class ExtractedItems(BaseModel):
    sender_email: Optional[str] = None
    sender_company: Optional[str] = None
    products: List[ProductItem] = None
    

class Extractor:
    """This class is responsible for extracting data from pdf emails."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prompts = self.cfg.extractor_agent_prompts
        self.model = OpenAIModel(
            self.cfg.llm_model,
            provider=OpenRouterProvider(api_key=SETTINGS.OPENROUTER_API_KEY,),
        )
        self.extractor_agent = Agent(
            model=self.model,
            result_type=ExtractedItems,
            system_prompt=self.prompts.system_prompt
        )
 
    async def _extractor_vlm(self, pdf_path: str) -> ExtractedItems:
        """Extracts data from a PDF file using a vision language model."""
        try:
            images: List[Image.Image] = convert_from_path(pdf_path, dpi=200)
            if not images:
                raise ValueError("No images extracted from PDF.")
            
            # Convert PIL Images to base64 data URLs
            message_content = [self.prompts.user_prompt]
            for img in images:
                # Convert PIL Image to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                data_url = f"data:image/png;base64,{img_base64}"
                message_content.append(ImageUrl(url=data_url))

            result = await self.extractor_agent.run(message_content)
            logger.info(f"Extraction output: {result.data}")
            return result.data
        except Exception as e:
            raise RuntimeError(f"Failed to extract data from PDF: {str(e)}")

    async def batch_extract(
        self, pdf_dir: str, output_dir: str
    ) -> List[ExtractedItems]:
        """Processes multiple PDF files from a directory and savesto CSV."""
        if not os.path.isdir(pdf_dir):
            raise NotADirectoryError(f"Directory not found: {pdf_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        results = []
        csv_path = os.path.join(output_dir, "extracted_quotations.csv")

        json_dir = os.path.join(output_dir, "json_files")
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        all_json_data = []
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'filename', 'sender_email', 'sender_company', 'products_json'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for filename in os.listdir(pdf_dir):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(pdf_dir, filename)
                    try:
                        extracted_data = await self._extractor_vlm(file_path)
                        results.append(extracted_data)
                        row = {
                            'filename': filename,
                            'sender_email': extracted_data.sender_email,
                            'sender_company': extracted_data.sender_company,
                            'products_json': json.dumps([p.dict() for p in extracted_data.products])
                        }
                        writer.writerow(row)
                        json_data = {
                            'filename': filename,
                            'sender_email': extracted_data.sender_email,
                            'sender_company': extracted_data.sender_company,
                            'products': [p.dict() for p in extracted_data.products]
                        }
                    
                        individual_json_path = os.path.join(json_dir, f"{os.path.splitext(filename)[0]}.json")
                        with open(individual_json_path, 'w', encoding='utf-8') as json_file:
                            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

                        all_json_data.append(json_data)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}") 
        return results
