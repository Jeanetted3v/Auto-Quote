# Auto Quotation System
A proof-of-concept (POC) for automating quotation generation process, starting with extracting inquiry, matching with items in price list, finally generating quotation. 

## Quick Set-up
1. Create a new environment and activate it:
```bash
conda create -n autoquote python=3.12
conda activate autoquote
```

2. Install the packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
PYTHONPATH=. streamlit run app.py
```