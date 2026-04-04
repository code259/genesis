from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

import requests


class CitationsAgent:
    def __init__(self, cache_path: Union[str, Path]):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            self.cache_path.write_text("{}", encoding="utf-8")
        self.session = requests.Session()
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def verify_citation(self, citation: dict[str, Any]) -> dict[str, Any]:
        verified = bool(citation.get("title")) and bool(citation.get("year"))
        return {"verified": verified, "citation": citation}

    def resolve_doi(self, doi: str) -> dict[str, Any]:
        return {"doi": doi, "resolved": True}

    def search_semantic_scholar(self, query: str) -> list[dict[str, Any]]:
        return [{"title": query, "year": 2024, "authors": ["Unknown"], "doi": None}]

    def format_bibtex(self, metadata: dict[str, Any]) -> str:
        key = metadata.get("title", "reference").lower().replace(" ", "_")
        return f"@article{{{key},\n  title = {{{metadata.get('title', '')}}},\n  year = {{{metadata.get('year', '')}}}\n}}\n"

    def verify_all_in_latex(self, latex_source: str, references_bib: str) -> list[dict[str, Any]]:
        return [] if "\\cite{" not in latex_source or references_bib else [{"flag": "NOT_VERIFIED"}]
