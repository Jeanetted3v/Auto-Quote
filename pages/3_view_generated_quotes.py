# pages/3_email_preview.py
import streamlit as st
import os
import base64
import json
import pandas as pd
from jinja2 import Template
from hydra import initialize, compose

st.set_page_config(layout="wide")

# --- Load config ---
with initialize(config_path="../config", version_base=None):
    cfg = compose(config_name="config")

# --- UI layout ---
st.title("📧 Email Preview with Quotation")
rfq_files = [f for f in os.listdir("data/processed/json_files") if f.endswith(".json")]
selected_file = st.selectbox("Choose an RFQ to preview:", rfq_files)

if selected_file:
    rfq_path = os.path.join("data/processed/json_files", selected_file)
    with open(rfq_path, "r") as f:
        rfq_data = json.load(f)

    reviewed_path = f"data/quotes/reviewed_{os.path.splitext(selected_file)[0]}.json"
    if not os.path.exists(reviewed_path):
        st.warning("No reviewed quote data found. Please complete matching in Page 2 first.")
        st.stop()

    with open(reviewed_path, "r") as f:
        reviewed_rows = json.load(f)

    for row in reviewed_rows:
        try:
            price = float(row.get("STOCK PRICE", 0))
            qty = float(row.get("requested_qty", 0))
            row["total"] = f"${price * qty:.2f}"
        except:
            row["total"] = "$n/a"

    # --- Layout: PDF left, email right ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### PDF Preview")
        pdf_path = os.path.join("data/rfq", rfq_data["filename"])
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}#zoom=150" 
                    width="100%" height="1000" style="border: none;"></iframe>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.warning("PDF not found.")

    with col2:
        st.markdown("#### ✉️ Email Preview")

        # Load and render template
        with open("config/email_template.j2") as f:
            template = Template(f.read())

        email_body = template.render(
            recipient_name="Customer",
            rfq_filename=rfq_data["filename"],
            quote_rows=reviewed_rows
        )

        st.markdown(email_body)

        # Optional: download button
        st.download_button("📥 Download Email Content", email_body, file_name="quote_email.txt")