from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union

from genesis.scholarly import ScholarlyClient


class CitationsAgent:
    def __init__(self, cache_path: Union[str, Path]):
        self.cache_path = Path(cache_path)
        self.client = ScholarlyClient(cache_path=self.cache_path)

    def verify_citation(self, citation: dict[str, Any]) -> dict[str, Any]:
        evidence: list[dict[str, Any]] = []
        title = (citation.get("title") or "").strip()
        year = citation.get("year")
        doi = (citation.get("doi") or citation.get("DOI") or "").strip()

        if doi:
            crossref = self.resolve_doi(doi)
            if crossref:
                evidence.append({"source": "crossref", "match": True, "metadata": crossref})
        semantic_candidates = self.search_semantic_scholar(title or doi)
        if semantic_candidates:
            best = semantic_candidates[0]
            evidence.append({"source": "semantic_scholar", "match": self._titles_match(title, best.get("title", "")), "metadata": best})

        verified = bool(evidence) and any(item.get("match") for item in evidence)
        if year is not None and evidence:
            verified = verified and any(item["metadata"].get("year") in {None, year} for item in evidence if "metadata" in item)
        return {"verified": verified, "citation": citation, "evidence": evidence}

    def resolve_doi(self, doi: str) -> dict[str, Any]:
        return self.client.resolve_crossref_doi(doi)

    def search_semantic_scholar(self, query: str) -> list[dict[str, Any]]:
        return self.client.search_semantic_scholar(query)

    def format_bibtex(self, metadata: dict[str, Any]) -> str:
        title = metadata.get("title", "")
        authors = metadata.get("authors", [])
        if authors and isinstance(authors[0], dict):
            author_field = " and ".join(author.get("name", "") for author in authors if author.get("name"))
        else:
            author_field = " and ".join(str(author) for author in authors)
        key_root = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "reference"
        fields = {
            "title": title,
            "author": author_field,
            "year": metadata.get("year", ""),
            "journal": metadata.get("venue") or metadata.get("container-title", [""])[0] if isinstance(metadata.get("container-title"), list) else metadata.get("container-title", ""),
            "doi": metadata.get("doi") or metadata.get("DOI") or metadata.get("externalIds", {}).get("DOI", ""),
            "url": metadata.get("url", ""),
        }
        body = "\n".join(f"  {key} = {{{value}}}," for key, value in fields.items() if value)
        return f"@article{{{key_root},\n{body}\n}}\n"

    def verify_all_in_latex(self, latex_source: str, references_bib: str) -> list[dict[str, Any]]:
        citation_keys = {
            key.strip()
            for match in re.finditer(r"\\cite\{([^}]+)\}", latex_source)
            for key in match.group(1).split(",")
            if key.strip()
        }
        if not citation_keys:
            return []
        available_keys = {
            match.group(1).strip()
            for match in re.finditer(r"@[\w]+\{([^,]+),", references_bib)
        }
        missing = sorted(citation_keys - available_keys)
        return [{"flag": "NOT_VERIFIED", "citation_key": key} for key in missing]

    def search_crossref(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.client.search_crossref(query, limit=limit)

    def search_arxiv(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.client.search_arxiv(query, limit=limit)

    def _titles_match(self, left: str, right: str) -> bool:
        normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        left_norm = normalize(left)
        right_norm = normalize(right)
        return bool(left_norm) and (left_norm == right_norm or left_norm in right_norm or right_norm in left_norm)
