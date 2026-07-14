"""SSRF regression tests for SitemapLoader (issue #806).

A sitemap body is attacker-controllable: a malicious/​compromised sitemap can
list ``<loc>`` entries or nested sitemap-index URLs pointing at internal hosts
(e.g. the cloud metadata endpoint). These tests prove such URLs are validated
and never fetched.

No mocks: hand-written fake httpx + deterministic DNS resolver via monkeypatch.
"""

from __future__ import annotations

import socket
import sys

import pytest

from synapsekit.loaders.sitemap import SitemapLoader

pytest.importorskip("bs4")
pytest.importorskip("lxml")


# --------------------------------------------------------------------------
# Deterministic resolver
# --------------------------------------------------------------------------


def _install_resolver(monkeypatch, mapping):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror(socket.EAI_NONAME, "unknown host")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (mapping[host], 0))
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# --------------------------------------------------------------------------
# Fake httpx (records every fetched URL)
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, url, body="", status=200):
        self.text = body
        self.status_code = status
        self.is_redirect = False
        self.headers: dict[str, str] = {}
        self.next_request = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    def __init__(self, routes, log, **kwargs):
        self._routes = routes
        self._log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kwargs):
        self._log.append(url)
        if url not in self._routes:
            return _FakeResponse(url, status=404)
        return _FakeResponse(url, self._routes[url])


class _FakeHttpx:
    def __init__(self, routes, log):
        self._routes = routes
        self._log = log

    def AsyncClient(self, **kwargs):  # noqa: N802 (mirrors httpx.AsyncClient API)
        return _FakeAsyncClient(self._routes, self._log, **kwargs)


@pytest.fixture
def run_loader(monkeypatch):
    def _run(sitemap_url, routes, resolver_map):
        _install_resolver(monkeypatch, resolver_map)
        log: list[str] = []
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx(routes, log))
        docs = SitemapLoader(url=sitemap_url).load()
        return docs, log

    return _run


PAGE_HTML = "<html><body><p>public content</p></body></html>"


# --------------------------------------------------------------------------
# Init-time validation
# --------------------------------------------------------------------------


def test_init_rejects_private_sitemap_url():
    with pytest.raises(ValueError, match="private/internal"):
        SitemapLoader(url="http://169.254.169.254/sitemap.xml")


# --------------------------------------------------------------------------
# Regression: a <loc> pointing at a private/loopback address is NOT fetched
# --------------------------------------------------------------------------


def test_private_loc_url_not_fetched(run_loader):
    sitemap = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://example.com/public</loc></url>"
        "<url><loc>http://169.254.169.254/latest/meta-data/</loc></url>"
        "<url><loc>http://127.0.0.1/admin</loc></url>"
        "</urlset>"
    )
    routes = {
        "https://example.com/sitemap.xml": sitemap,
        "https://example.com/public": PAGE_HTML,
        # These are deliberately routed so that IF the guard failed, they'd
        # return content — making a regression loudly visible.
        "http://169.254.169.254/latest/meta-data/": "IAM-CREDENTIALS-LEAK",
        "http://127.0.0.1/admin": "INTERNAL-ADMIN",
    }
    resolver = {"example.com": "93.184.216.34"}

    docs, log = run_loader("https://example.com/sitemap.xml", routes, resolver)

    fetched_urls = {d.metadata["url"] for d in docs}
    assert fetched_urls == {"https://example.com/public"}
    # The internal endpoints must never have been requested.
    assert "http://169.254.169.254/latest/meta-data/" not in log
    assert "http://127.0.0.1/admin" not in log
    # And their content must never surface in any document.
    assert all("LEAK" not in d.text and "ADMIN" not in d.text for d in docs)


# --------------------------------------------------------------------------
# Regression: a nested sitemap-index URL pointing internal is NOT fetched
# --------------------------------------------------------------------------


def test_private_nested_sitemap_index_not_fetched(run_loader):
    index = (
        '<?xml version="1.0"?>'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<sitemap><loc>https://example.com/child.xml</loc></sitemap>"
        "<sitemap><loc>http://169.254.169.254/internal.xml</loc></sitemap>"
        "</sitemapindex>"
    )
    child = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://example.com/ok</loc></url>"
        "</urlset>"
    )
    routes = {
        "https://example.com/sitemap.xml": index,
        "https://example.com/child.xml": child,
        "https://example.com/ok": PAGE_HTML,
        "http://169.254.169.254/internal.xml": "<urlset/>",
    }
    resolver = {"example.com": "93.184.216.34"}

    docs, log = run_loader("https://example.com/sitemap.xml", routes, resolver)

    assert {d.metadata["url"] for d in docs} == {"https://example.com/ok"}
    assert "http://169.254.169.254/internal.xml" not in log


# --------------------------------------------------------------------------
# Sanity: an entirely public sitemap still loads normally
# --------------------------------------------------------------------------


def test_public_sitemap_loads(run_loader):
    sitemap = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://example.com/a</loc></url>"
        "<url><loc>https://example.com/b</loc></url>"
        "</urlset>"
    )
    routes = {
        "https://example.com/sitemap.xml": sitemap,
        "https://example.com/a": PAGE_HTML,
        "https://example.com/b": PAGE_HTML,
    }
    resolver = {"example.com": "93.184.216.34"}

    docs, _log = run_loader("https://example.com/sitemap.xml", routes, resolver)
    assert {d.metadata["url"] for d in docs} == {
        "https://example.com/a",
        "https://example.com/b",
    }
