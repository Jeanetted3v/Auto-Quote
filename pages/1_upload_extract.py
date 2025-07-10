"""To run:
PYTHONPATH=. streamlit run pages/1_upload_extract.py
"""
import os
import streamlit as st
import asyncio
from hydra import compose, initialize
from src.backend.extractor import Extractor


with initialize(config_path="../config", version_base=None):
    cfg = compose(config_name="config")

os.makedirs("data/rfq", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

st.title("Upload RFQ PDF")
uploaded_file = st.file_uploader(
    "Upload RFQ PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_file:
    for f in uploaded_file:
        save_path = f"data/rfq/{f.name}"
        with open(save_path, "wb") as out:
            out.write(f.read())

    if st.button("Extract All RFQs"):
        extractor = Extractor(cfg)
        # Async call for batch_extract or direct call to extract_single
        asyncio.run(extractor.batch_extract("data/rfq", "data/processed"))
        st.success("Extraction completed! Go to 'Review & Match' in the sidebar →")
