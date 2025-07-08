import streamlit as st
import asyncio
from src.backend.extractor import Extractor
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")

st.title("Upload RFQ PDF")
uploaded_file = st.file_uploader(
    "Upload RFQ PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_file:
    for f in uploaded_file:
        save_path = f".data/rfq/{f.name}"
        with open(save_path, "wb") as out:
            out.write(f.read())

    if st.button("Extract All RFQs"):
        extractor = Extractor(cfg)
        # Async call for batch_extract or direct call to extract_single
        asyncio.run(extractor.batch_extract(".data/rfq", ".data/processed"))
        st.success("Extraction completed!")
