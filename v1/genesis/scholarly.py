from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import requests


class ScholarlyClient:
    def __init__(
        self,
        *,
        cache_path: str | Path,
        session: Optional[requests.Session] = None,
        semantic_scholar_api_key: Optional[str] = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            self.cache_path.write_text("{}", encoding="utf-8")
        self.session = session or requests.Session()
        self.semantic_scholar_api_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self._last_semantic_scholar_request = 0.0

    def search_semantic_scholar(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        cache_key = self._cache_key("s2_search", {"query": query, "limit": limit})
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,year,authors,abstract,externalIds,citationCount,url",
        }
        payload = self._request_json(url, params=params, semantic_scholar=True)
        data = payload.get("data", []) if isinstance(payload, dict) else []
        self._set_cache(cache_key, data)
        return data

    def search_title(self, title: str) -> list[dict[str, Any]]:
        results = self.search_semantic_scholar(title, limit=5)
        if results:
            return results
        results = self.search_crossref(title, limit=5)
        if results:
            return results
        return self.search_arxiv(title, limit=5)

    def get_paper(self, paper_id: str) -> dict[str, Any]:
        cache_key = self._cache_key("s2_paper", {"paper_id": paper_id})
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        url = f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_id, safe='')}"
        params = {"fields": "title,year,authors,abstract,externalIds,citationCount,url,citations.title"}
        payload = self._request_json(url, params=params, semantic_scholar=True)
        self._set_cache(cache_key, payload)
        return payload if isinstance(payload, dict) else {}

    def resolve_crossref_doi(self, doi: str) -> dict[str, Any]:
        normalized = doi.lower().strip()
        cache_key = self._cache_key("crossref_doi", {"doi": normalized})
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        payload = self._request_json(f"https://api.crossref.org/works/{urllib.parse.quote(normalized)}")
        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        self._set_cache(cache_key, message)
        return message

    def search_crossref(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        cache_key = self._cache_key("crossref_search", {"query": query, "limit": limit})
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        payload = self._request_json(
            "https://api.crossref.org/works",
            params={"query.title": query, "rows": limit},
        )
        items = payload.get("message", {}).get("items", []) if isinstance(payload, dict) else []
        self._set_cache(cache_key, items)
        return items

    def search_arxiv(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        cache_key = self._cache_key("arxiv_search", {"query": query, "limit": limit})
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        url = (
            "https://export.arxiv.org/api/query?"
            + urllib.parse.urlencode(
                {
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                }
            )
        )
        xml_text = self._request_text(url)
        if not xml_text:
            return []
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        results: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", namespace):
            title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=namespace) or "").strip()
            arxiv_id = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()
            authors = [
                (author.findtext("atom:name", default="", namespaces=namespace) or "").strip()
                for author in entry.findall("atom:author", namespace)
            ]
            results.append(
                {
                    "title": title,
                    "abstract": summary,
                    "year": int(published[:4]) if published[:4].isdigit() else None,
                    "authors": [{"name": author} for author in authors if author],
                    "externalIds": {"ArXiv": arxiv_id.rsplit("/", 1)[-1] if arxiv_id else None},
                    "url": arxiv_id,
                }
            )
        self._set_cache(cache_key, results)
        return results

    def _request_json(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        semantic_scholar: bool = False,
    ) -> dict[str, Any]:
        text = self._request_text(url, params=params, semantic_scholar=semantic_scholar)
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _request_text(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        semantic_scholar: bool = False,
    ) -> str:
        headers: dict[str, str] = {}
        if semantic_scholar and self.semantic_scholar_api_key:
            now = time.time()
            elapsed = now - self._last_semantic_scholar_request
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            headers["x-api-key"] = self.semantic_scholar_api_key
            self._last_semantic_scholar_request = time.time()
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return ""

    def _load_cache(self) -> dict[str, Any]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_cache(self, payload: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(self.cache_path.parent),
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(payload, indent=2))
            temp_name = handle.name
        os.replace(temp_name, self.cache_path)

    def _get_cache(self, key: str) -> Any:
        return self._load_cache().get(key)

    def _set_cache(self, key: str, value: Any) -> None:
        payload = self._load_cache()
        payload[key] = value
        self._save_cache(payload)

    def _cache_key(self, prefix: str, payload: dict[str, Any]) -> str:
        return prefix + ":" + json.dumps(payload, sort_keys=True)
