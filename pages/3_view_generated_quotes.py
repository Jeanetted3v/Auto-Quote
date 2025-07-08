import streamlit as st
import os

st.title("📄 View Generated Quote")
quote_files = [f for f in os.listdir(".data/quotes") if f.endswith(".txt")]
selected_quote = st.selectbox("Select a quote to view:", quote_files)

if selected_quote:
    quote_path = os.path.join(".data/quotes", selected_quote)
    with open(quote_path, "r") as f:
        content = f.read()

    st.subheader("📝 Quote Preview")
    st.text_area("Generated Quote", content, height=300)

    st.download_button(
        label="⬇️ Download Quote",
        data=content,
        file_name=selected_quote,
        mime="text/plain"
    )