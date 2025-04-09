import logging
import os
from typing import Optional, List
from pydantic import BaseModel
from pydantic_ai import Agent
import PyPDF2
import csv, json
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)


class ProductItem(BaseModel):
    name: str
    code: Optional[str] = None
    quantity: Optional[int] = None
    unit: Optional[str] = None

class ExtractedItems(BaseModel):
    sender_email: str
    sender_company: str
    products: List[ProductItem]
    

class EmailExtractor:
    """This class is responsible for extracting data from pdf emails."""

    def __init__(self, system_prompt: str):
        self.agent = Agent(
            "openai:gpt-4o",   # using gpt-4o for multi-modal
            result_type=ExtractedItems,
            system_prompt=system_prompt
        )

    def data_ingest(self, pdf_path: str) -> str:
        """xtracts text content from a PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        text_content = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text()
            return text_content
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    async def extract(self, pdf_path: str) -> ExtractedItems:
        """Processes a PDF file and extracts structured information."""
        text_content = self.data_ingest(pdf_path)
        result = await self.agent.run(
            text_content)
        extracted_data = result.data
        print(f"Extracted Data: {extracted_data}")
        return extracted_data

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
                        extracted_data = await self.extract(file_path)
                        results.append(extracted_data)
                        row = {
                            'filename': filename,
                            'sender_email': extracted_data.sender_email,
                            'sender_company': extracted_data.sender_company,
                            'products_json': json.dumps([p.dict() for p in extracted_data.products])
                        }
                        writer.writerow(row)
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}") 
        logger.info(f"Extraction complete. Results saved to {csv_path}")
        return results
