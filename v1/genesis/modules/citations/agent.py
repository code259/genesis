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
        title = (citation.get("title") or "").strip()
        year = citation.get("year")
        doi = self._normalize_doi(citation.get("doi") or citation.get("DOI") or citation.get("externalIds", {}).get("DOI", ""))
        evidence: list[dict[str, Any]] = []

        if not title and not doi:
            return {"verified": False, "citation": citation, "evidence": [{"source": "input", "flag": "missing_title_and_doi"}]}

        if doi:
            crossref = self.resolve_doi(doi)
            if crossref:
                evidence.append({"source": "crossref", "match": self._normalize_doi(crossref.get("DOI", "")) == doi, "metadata": crossref})

        search_query = title or doi
        semantic_candidates = self.search_semantic_scholar(search_query)
        if semantic_candidates:
            best = semantic_candidates[0]
            evidence.append(
                {
                    "source": "semantic_scholar",
                    "match": self._titles_match(title, best.get("title", "")) or self._normalize_doi(best.get("externalIds", {}).get("DOI", "")) == doi,
                    "metadata": best,
                }
            )
        elif title:
            crossref_candidates = self.search_crossref(title, limit=3)
            for candidate in crossref_candidates:
                candidate_title = candidate.get("title", [""])[0] if isinstance(candidate.get("title"), list) else candidate.get("title", "")
                evidence.append({"source": "crossref_search", "match": self._titles_match(title, candidate_title), "metadata": candidate})
                if evidence[-1]["match"]:
                    break

        verified = any(item.get("match") for item in evidence)
        if year is not None and verified:
            candidate_years = {item.get("metadata", {}).get("year") for item in evidence if item.get("match")}
            verified = year in candidate_years or None in candidate_years
            if not verified:
                evidence.append({"source": "year_check", "flag": "year_mismatch", "expected_year": year})
        return {"verified": verified, "citation": citation, "evidence": evidence}

    def resolve_doi(self, doi: str) -> dict[str, Any]:
        return self.client.resolve_crossref_doi(doi)

    def search_semantic_scholar(self, query: str) -> list[dict[str, Any]]:
        return self.client.search_semantic_scholar(query)

    def search_title(self, title: str) -> list[dict[str, Any]]:
        return self.client.search_title(title)

    def format_bibtex(self, metadata: dict[str, Any]) -> str:
        title = metadata.get("title", "")
        if isinstance(title, list):
            title = title[0] if title else ""
        authors = metadata.get("authors", [])
        if authors and isinstance(authors[0], dict):
            author_field = " and ".join(author.get("name", "") for author in authors if author.get("name"))
            first_author = re.sub(r"[^a-z0-9]+", "_", authors[0].get("name", "reference").lower()).strip("_")
        else:
            author_field = " and ".join(str(author) for author in authors)
            first_author = re.sub(r"[^a-z0-9]+", "_", str(authors[0]).lower()).strip("_") if authors else "reference"
        year = metadata.get("year", "")
        journal = metadata.get("venue")
        if not journal and isinstance(metadata.get("container-title"), list):
            journal = metadata.get("container-title", [""])[0]
        elif not journal:
            journal = metadata.get("container-title", "")
        key_root = re.sub(r"[^a-z0-9]+", "_", f"{first_author}_{year}_{str(title).lower()}").strip("_") or "reference"
        fields = {
            "title": title,
            "author": author_field,
            "year": year,
            "journal": journal,
            "doi": metadata.get("doi") or metadata.get("DOI") or metadata.get("externalIds", {}).get("DOI", ""),
            "url": metadata.get("url", ""),
        }
        body = "\n".join(
            f"  {key} = {{{self._escape_bibtex(str(value))}}},"
            for key, value in fields.items()
            if value
        )
        return f"@article{{{key_root},\n{body}\n}}\n"

    def verify_all_in_latex(self, latex_source: str, references_bib: str) -> list[dict[str, Any]]:
        citation_keys = {
            key.strip()
            for match in re.finditer(r"\\cite\w*\{([^}]+)\}", latex_source)
            for key in match.group(1).split(",")
            if key.strip()
        }
        if not citation_keys:
            return []
        entries = self._parse_bibtex_entries(references_bib)
        flags = []
        for key in sorted(citation_keys):
            metadata = entries.get(key)
            if metadata is None:
                flags.append({"flag": "MISSING_REFERENCE_ENTRY", "citation_key": key})
                continue
            verification = self.verify_citation(metadata)
            if not verification["verified"]:
                flags.append({"flag": "NOT_VERIFIED", "citation_key": key, "evidence": verification["evidence"]})
        return flags

    def search_crossref(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.client.search_crossref(query, limit=limit)

    def search_arxiv(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.client.search_arxiv(query, limit=limit)

    def _normalize_doi(self, doi: Any) -> str:
        value = str(doi or "").strip().lower()
        return value.removeprefix("https://doi.org/").removeprefix("doi:")

    def _titles_match(self, left: str, right: str) -> bool:
        normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        left_norm = normalize(left)
        right_norm = normalize(right)
        return bool(left_norm) and (left_norm == right_norm or left_norm in right_norm or right_norm in left_norm)

    def _escape_bibtex(self, value: str) -> str:
        return value.replace("{", "\\{").replace("}", "\\}")

    def _parse_bibtex_entries(self, references_bib: str) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for raw_entry in re.split(r"(?=@\w+\{)", references_bib):
            raw_entry = raw_entry.strip()
            if not raw_entry:
                continue
            header_match = re.match(r"@\w+\{([^,]+),", raw_entry)
            if not header_match:
                continue
            key = header_match.group(1).strip()
            metadata: dict[str, Any] = {}
            for field, value in re.findall(r"(\w+)\s*=\s*\{([^}]*)\}", raw_entry):
                metadata[field] = value
            if "author" in metadata:
                metadata["authors"] = [{"name": author.strip()} for author in metadata["author"].split(" and ") if author.strip()]
            if "year" in metadata and str(metadata["year"]).isdigit():
                metadata["year"] = int(metadata["year"])
            entries[key] = metadata
        return entries
