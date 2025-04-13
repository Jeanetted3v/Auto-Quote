import logging
import os
from typing import Optional, List, Dict
from pydantic import BaseModel
from pydantic_ai import Agent
from pypdf import PdfReader
import tempfile
from pdf2image import convert_from_path
import csv, json
import base64
import pytesseract
from PIL import Image
import cv2
import numpy as np
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)

class ProductItem(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    quantity: Optional[int] = None
    unit: Optional[str] = None

class ExtractedItems(BaseModel):
    sender_email: Optional[str] = None
    sender_company: Optional[str] = None
    products: List[ProductItem] = None
    

class EmailExtractor:
    """This class is responsible for extracting data from pdf emails."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.email_parser_enabled = self.cfg.email_parser_enabled
        self.prompts = self.cfg.email_extractor_prompts
        self.agent = Agent(
            self.cfg.llm_model,
            result_type=ExtractedItems
        )

    def data_ingest(self, pdf_path: str) -> str:
        """xtracts text content from a PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        if self.email_parser_enabled:
            text_content = ""
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text()
                return text_content
            except Exception as e:
                raise Exception(f"Error extracting text from PDF: {str(e)}")
        else:
            try:
                return pdf_path
            except Exception as e:
                raise Exception(f"Error preparing PDF for image processing: {str(e)}")
    
    def pdf_to_jpeg(self, pdf_path: str) -> List[str]:
        """Converts a PDF file to a series of JPEG images, one per page."""
        temp_dir = tempfile.mkdtemp()
        images = convert_from_path(pdf_path, dpi=100)
        
        jpeg_paths = []
        for i, image in enumerate(images):
            jpeg_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            image.save(jpeg_path, "JPEG", quality=50)
            jpeg_paths.append(jpeg_path)
        return jpeg_paths

    def encode_images_to_base64(self, image_paths: List[str]) -> List[str]:
        encoded_images = []
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append(encoded_string)
        return encoded_images

    def check_pdf_type(pdf_path: str) -> bool:
        """Checks if a PDF contains extractable text or is primarily image-based."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            # Check if the PDF has any pages
            if len(pdf_reader.pages) == 0:
                return False
            
            # Check a sample of pages (first, middle, last)
            pages_to_check = [
                0,  # First page
                len(pdf_reader.pages) // 2,  # Middle page
                len(pdf_reader.pages) - 1  # Last page
            ]
            
            # Remove duplicates (for short PDFs)
            pages_to_check = list(set(pages_to_check))
            
            # Check each selected page
            total_text = 0
            for page_num in pages_to_check:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Count meaningful text (not just whitespace or a few characters)
                if text and len(text.strip()) > 50:  # At least 50 characters
                    total_text += len(text)
            
            # If we found a reasonable amount of text, consider it a text PDF
            avg_text_per_page = total_text / len(pages_to_check)
            return avg_text_per_page > 100  # Threshold of 100 chars per page

    async def extract(self, pdf_path: str) -> ExtractedItems:
        """Processes a PDF file and extracts structured information in batches."""
        original_filename = os.path.basename(pdf_path)
        
        if self.email_parser_enabled:
            text_content = self.data_ingest(pdf_path)
            result = await self.agent.run(text_content)
            return result.data if hasattr(result, 'data') else result
        
        # Convert PDF to JPEGs
        jpeg_paths = self.pdf_to_jpeg(pdf_path)
        all_text = ""
    
        for i, jpeg_path in enumerate(jpeg_paths):
            try:
                # Read image
                img = cv2.imread(jpeg_path)
                
                # Preprocess image for better OCR results
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                
                # Apply OCR
                page_text = pytesseract.image_to_string(gray)
                
                # Add page number for context
                all_text += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
                
                logger.info(f"OCR completed for page {i+1} of {len(jpeg_paths)}")
            except Exception as e:
                logger.error(f"Error performing OCR on page {i+1}: {str(e)}")
        
        result = await self.agent.run(self.prompts.user_prompt.format(text=all_text))
        extracted_data = result.data
        logger.info(f"Extracted data: {extracted_data}")
        
        logger.info(f"Completed extraction for {original_filename}")
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
                        extracted_data = await self.extract(file_path)
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
