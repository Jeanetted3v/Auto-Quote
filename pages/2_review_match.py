import streamlit as st
import os
import json
import pandas as pd
from omegaconf import OmegaConf
from src.backend.embedder import Embedder
from src.backend.quotes_generator import QuoteGenerator
from src.backend.utils.settings import SETTINGS

# --- Load config and setup backend ---
cfg = OmegaConf.load("config.yaml")
embedder = Embedder(
    api_key=SETTINGS.OPENAI_API_KEY,
    model_name=cfg.embedder_model,
    collection_name=cfg.chroma_collection_name,
    persist_dir=cfg.chroma_persist_dir
)
quote_gen = QuoteGenerator(cfg, embedder)

# --- UI: Select RFQ ---
st.title("🔍 Review & Match Extracted Items")
rfq_files = [f for f in os.listdir(".data/processed") if f.endswith(".json")]
selected_file = st.selectbox("Choose an RFQ to review:", rfq_files)

if selected_file:
    rfq_path = os.path.join(".data/processed", selected_file)
    with open(rfq_path, "r") as f:
        rfq_data = json.load(f)

    st.subheader(f"📄 RFQ: {rfq_data['filename']}")
    st.markdown("---")

    # --- Load PDF (left) and review table (right) ---
    col1, col2 = st.columns([1, 2])
    pdf_path = os.path.join(".data/rfq", rfq_data["filename"])
    if os.path.exists(pdf_path):
        with col1:
            st.markdown("#### PDF Preview")
            st.components.v1.iframe(f"file://{os.path.abspath(pdf_path)}", height=600)

    with col2:
        price_df = pd.read_csv(cfg.price_list_path)
        impa_columns = [col for col in price_df.columns if "IMPA" in col.upper()]

        reviewed_rows = []

        for i, item in enumerate(rfq_data["products"]):
            st.markdown(f"**Item {i+1}: {item['name']}**")
            suggestions = quote_gen.suggest_matches(item["name"], top_k=3)
            selected = st.selectbox(
                label="Select best match",
                options=suggestions,
                key=f"match_{i}",
                format_func=lambda x: f"{x.get('Products Name', 'Unnamed')} - ${x.get('STOCK PRICE', 'n/a')} ({x.get('match_confidence', '?')}%)"
            )

            enriched = quote_gen.enrich_match_with_request_context(selected, item)
            reviewed_rows.append(enriched)
            st.markdown("---")

        if st.button("✅ Submit & Generate Quote"):
            quote_df = pd.DataFrame(reviewed_rows)
            out_path = f".data/quotes/quote_{os.path.splitext(selected_file)[0]}.txt"
            os.makedirs(".data/quotes", exist_ok=True)
            with open(out_path, "w") as f:
                for row in quote_df.to_dict(orient="records"):
                    line = f"{row['requested_name']} - {row.get('Products Name', 'NO MATCH')} (${row.get('STOCK PRICE', '-')}) x {row['requested_qty']} {row['requested_unit']}\n"
                    f.write(line)
            st.success(f"Quote saved to {out_path}")