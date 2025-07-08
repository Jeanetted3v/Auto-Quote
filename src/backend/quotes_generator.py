import json
import pandas as pd
from typing import Dict, Any, List

class QuoteGenerator:
    """A class to generate quotes based on price list embeddings."""

    def __init__(self, cfg, embedder):
        self.cfg = cfg
        self.embedder = embedder

    def _match_product(
        self,
        item: Dict[str, Any],
        price_df: pd.DataFrame,
        impa_columns: List[str]
    ) -> Dict[str, Any]:
        """Match a single product from RFQ to price list using IMPA code or semantic similarity."""
        # Check if the item has an IMPA code 
        rfq_impa_code = item.get("impa_code")
        rfq_product_name = item.get("name", "").strip()

        # 1. Try matching against all IMPA code columns
        if pd.notna(rfq_impa_code):
            for col in impa_columns:
                if col in price_df.columns:
                    matched = price_df[price_df[col] == rfq_impa_code]
                    if not matched.empty:
                        match = matched.iloc[0].to_dict() # returning the entire row
                        match["requested_name"] = rfq_product_name
                        match["requested_impa_code"] = rfq_impa_code
                        match["matched_via"] = "impa"
                        return match

        # 2. Fallback: semantic match by name
        results = self.embedder.query_similar(
            rfq_product_name,
            n_results=self.cfg.n_results
        )
        if results and results["metadatas"]:
            top_match = results["metadatas"][0]
            top_match["match_confidence"] = int((1 - results["distances"][0]) * 100)  # convert to confidence in percentage
            top_match["match_distance"] = round(results["distances"][0], 4)        # raw distance
            top_match["matched_via"] = "semantic"
            return top_match

        # 3. No match found
        return {
            "match_status": "not_found",
            "query_name": rfq_product_name,
            "matched_via": "none",
        }
    
    def suggest_matches(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Return top-k product match candidates for a given product name using semantic search.
        """
        results = self.embedder.query_similar(query, n_results=top_k)
        candidates = []

        for metadata, distance in zip(results["metadatas"], results["distances"]):
            candidates.append({
                **metadata,
                "match_confidence": int((1 - distance) * 100),
                "match_distance": round(distance, 4),
                "matched_via": "semantic"
            })

        return candidates
    
    def _enrich_match_with_request_context(
        self,
        match: Dict[str, Any],
        item: Dict[str, Any]
    ) -> Dict[str, Any]:
        match["requested_name"] = item.get("name", "")
        match["requested_qty"] = item.get("quantity", "")
        match["requested_unit"] = item.get("unit", "")
        return match

    
    def process_rfq(
        self,
        rfq_data: Dict[str, Any],
        price_df: pd.DataFrame,
        impa_columns: List[str]
    ) -> List[Dict[str, Any]]:
        matched_items = []
        for item in rfq_data["products"]:
            match = self._match_product(item, price_df, impa_columns)
            enriched = self._enrich_match_with_request_context(match, item)
            matched_items.append(enriched)
        return matched_items


    def generate_quote(self, prompt):
        """Entry point to match products and generate quotes."""
        price_df = pd.read_csv("sample_price_list.csv")
        impa_columns = [col for col in price_df.columns if "IMPA" in col.upper()]
        for item in rfq_data["products"]:
        match = quote_gen.match_product(item, price_df, embedder, impa_columns)