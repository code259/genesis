import json

from genesis.scholarly import ScholarlyClient


class _Response:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class _Session:
    def get(self, url, params=None, headers=None, timeout=None):
        if "crossref.org/works" in url and params is not None:
            payload = {
                "message": {
                    "items": [
                        {
                            "title": ["Normalized CrossRef Title"],
                            "author": [{"given": "Ada", "family": "Lovelace"}],
                            "DOI": "10.1000/test",
                            "URL": "https://doi.org/10.1000/test",
                            "container-title": ["Journal of Tests"],
                            "issued": {"date-parts": [[2025, 1, 1]]},
                        }
                    ]
                }
            }
            return _Response(json.dumps(payload))
        if "semanticscholar.org" in url:
            payload = {
                "data": [
                    {
                        "paperId": "paper-1",
                        "title": "Semantic Scholar Title",
                        "authors": [{"name": "Grace Hopper"}],
                        "year": 2024,
                        "abstract": "Abstract text",
                        "externalIds": {"DOI": "10.1000/semantic"},
                        "citationCount": 7,
                        "url": "https://example.com/paper-1",
                    }
                ]
            }
            return _Response(json.dumps(payload))
        return _Response("")


def test_crossref_results_are_normalized(tmp_path):
    client = ScholarlyClient(cache_path=tmp_path / "cache.json", session=_Session())
    results = client.search_crossref("normalized title", limit=1)
    assert results[0]["title"] == "Normalized CrossRef Title"
    assert results[0]["authors"] == [{"name": "Ada Lovelace"}]
    assert results[0]["year"] == 2025
    assert results[0]["doi"] == "10.1000/test"


def test_semantic_scholar_results_are_normalized(tmp_path):
    client = ScholarlyClient(cache_path=tmp_path / "cache.json", session=_Session())
    results = client.search_semantic_scholar("semantic title", limit=1)
    assert results[0]["paper_id"] == "paper-1"
    assert results[0]["title"] == "Semantic Scholar Title"
    assert results[0]["authors"] == [{"name": "Grace Hopper"}]


def test_arxiv_parse_errors_return_empty_results(tmp_path):
    class _BrokenSession:
        def get(self, url, params=None, headers=None, timeout=None):
            return _Response("<not-xml")

    client = ScholarlyClient(cache_path=tmp_path / "cache.json", session=_BrokenSession())
    assert client.search_arxiv("bad xml", limit=1) == []
