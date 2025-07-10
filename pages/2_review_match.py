import streamlit as st
import os
import json
import base64
import pandas as pd
from jinja2 import Template
from hydra import compose, initialize
from src.backend.embedder import Embedder
from src.backend.quotes_generator import QuoteGenerator
from src.backend.utils.settings import SETTINGS

with initialize(config_path="../config", version_base=None):
    cfg = compose(config_name="config")

st.set_page_config(layout="wide")

embedder = Embedder(
    api_key=SETTINGS.OPENAI_API_KEY,
    model_name=cfg.embedder_model,
    collection_name=cfg.chroma_collection_name,
    persist_dir=cfg.chroma_persist_dir
)
quote_gen = QuoteGenerator(cfg, embedder)

# --- UI: Select RFQ ---
st.title("🔍 Review & Match Extracted Items")
rfq_files = [f for f in os.listdir("data/processed/json_files") if f.endswith(".json")]
selected_file = st.selectbox("Choose an RFQ to review:", rfq_files)

if selected_file:
    rfq_path = os.path.join("data/processed/json_files", selected_file)
    with open(rfq_path, "r") as f:
        rfq_data = json.load(f)

    st.subheader(f"📄 RFQ: {rfq_data['filename']}")
    st.markdown("---")

    # --- Load PDF (left) and review table (right) ---
    col1, col2 = st.columns([4, 2], gap="large")
    pdf_path = os.path.join("data/rfq", rfq_data["filename"])

    # --- Left panel: PDF preview ---
    with col1:
        st.markdown("#### PDF Preview")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'''
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}"
                width="100%"
                height="1000"
                type="application/pdf">
            </iframe>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.warning("PDF not found.")

    # --- Right panel : Review matches ---
    with col2:
        price_df = pd.read_csv(cfg.price_list_path)
        impa_columns = [col for col in price_df.columns if "IMPA" in col.upper()]

        reviewed_rows = []

        for i, item in enumerate(rfq_data["products"]):
            st.markdown(f"**Item {i+1}: {item['name']}**")
            top1_match = quote_gen.match_product(item, price_df, impa_columns)
            suggestions = quote_gen.suggest_matches(item["name"], top_k=3)

            # Try to find top1_match in suggestions to preselect
            def find_index(match, candidates):
                for idx, cand in enumerate(candidates):
                    if cand.get("Products Name") == match.get("Products Name"):
                        return idx
                return 0  # fallback to first

            selected = st.selectbox(
                label="Select best match",
                options=suggestions,
                index=find_index(top1_match, suggestions),
                key=f"match_{i}",
                format_func=lambda x: (
                    f"{x.get('Products Name', 'Unnamed')}\n"
                    f"Price: ${x.get('STOCK PRICE', 'n/a')}\n"
                    f"Confidence: {x.get('match_confidence', '?')}%"
                )
            )

            enriched = quote_gen.enrich_match_with_request_context(selected, item)
            reviewed_rows.append(enriched)
            st.markdown("---")

        if st.button("✅ Submit & Generate Quote"):
            # 1. Enrich with totals
            for row in reviewed_rows:
                try:
                    price = float(row.get("STOCK PRICE", 0))
                    qty = float(row.get("requested_qty", 0))
                    row["total"] = round(price * qty, 2)
                except:
                    row["total"] = "n/a"

            # 2. Load & render email template
            with open("config/email_template.j2") as f:
                template = Template(f.read())

            email_body = template.render(
                recipient_name="Customer",
                rfq_filename=rfq_data["filename"],
                quote_rows=reviewed_rows
            )

            # 3. Save and show output
            output_path = f"data/quotes/quote_{os.path.splitext(selected_file)[0]}.txt"
            os.makedirs("data/quotes", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(email_body)

            st.success(f"Quote saved to {output_path}")
            st.markdown("### ✉️ Email Preview")
            st.markdown(email_body)