"""SSRF regression tests for WebLoader (issue #807).

No mocks: a hand-written fake httpx module records which URLs are fetched and
DNS is redirected through a deterministic in-test resolver via monkeypatch.
Asserting on the fake's request log proves blocked URLs are never fetched.
"""

from __future__ import annotations

import socket
import sys

import pytest

from synapsekit.loaders._url_guard import SSRFValidationError
from synapsekit.loaders.web import WebLoader, _validate_url

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
# Fake httpx (records fetched URLs; supports redirect chains)
# --------------------------------------------------------------------------


class _FakeURL:
    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value


class _FakeRequest:
    def __init__(self, url: str) -> None:
        self.url = _FakeURL(url)


class _FakeResponse:
    def __init__(self, url, body="", *, status=200, redirect_to=None):
        self.text = body
        self.status_code = 302 if redirect_to else status
        self.is_redirect = redirect_to is not None
        self.headers = {"location": redirect_to} if redirect_to else {}
        self.next_request = _FakeRequest(redirect_to) if redirect_to else None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSyncClient:
    """Fake sync httpx client. ``routes`` maps URL -> _FakeResponse."""

    def __init__(self, routes, log, **kwargs):
        self._routes = routes
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        self._log.append(url)
        if url not in self._routes:
            raise RuntimeError(f"unexpected fetch: {url}")
        return self._routes[url]


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
            raise RuntimeError(f"unexpected fetch: {url}")
        return self._routes[url]


class _FakeHttpx:
    def __init__(self, routes, log):
        self._routes = routes
        self._log = log

    def Client(self, **kwargs):  # noqa: N802 (mirrors httpx.Client API)
        return _FakeSyncClient(self._routes, self._log, **kwargs)

    def AsyncClient(self, **kwargs):  # noqa: N802 (mirrors httpx.AsyncClient API)
        return _FakeAsyncClient(self._routes, self._log, **kwargs)


@pytest.fixture
def fake_httpx(monkeypatch):
    def _install(routes, log):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx(routes, log))

    return _install


@pytest.fixture(autouse=True)
def _stub_bs4(monkeypatch):
    # WebLoader._parse imports bs4; provide a trivial text extractor so the
    # tests don't depend on the optional dependency being installed.
    import types

    mod = types.ModuleType("bs4")

    class _Soup:
        def __init__(self, html, parser):
            self._html = html

        def get_text(self, separator="\n", strip=True):
            return self._html

    mod.BeautifulSoup = _Soup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bs4", mod)


# --------------------------------------------------------------------------
# __init__ validation
# --------------------------------------------------------------------------


def test_init_rejects_private_ip():
    with pytest.raises(SSRFValidationError, match="private/internal"):
        WebLoader("http://169.254.169.254/latest/meta-data/")


def test_init_fails_closed_on_unresolvable(monkeypatch):
    _install_resolver(monkeypatch, {})
    with pytest.raises(SSRFValidationError, match="Could not resolve"):
        WebLoader("http://nope.invalid/")


def test_validate_url_helper_still_raises_valueerror():
    # Backwards compat: SSRFValidationError subclasses ValueError.
    with pytest.raises(ValueError):
        _validate_url("http://127.0.0.1/")


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_load_sync_public(monkeypatch, fake_httpx):
    _install_resolver(monkeypatch, {"example.com": "93.184.216.34"})
    log: list[str] = []
    routes = {"http://example.com/": _FakeResponse("http://example.com/", "hello")}
    fake_httpx(routes, log)
    docs = WebLoader("http://example.com/").load_sync()
    assert docs[0].text == "hello"
    assert log == ["http://example.com/"]


async def test_load_async_public(monkeypatch, fake_httpx):
    _install_resolver(monkeypatch, {"example.com": "93.184.216.34"})
    log: list[str] = []
    routes = {"http://example.com/": _FakeResponse("http://example.com/", "hi")}
    fake_httpx(routes, log)
    docs = await WebLoader("http://example.com/").load()
    assert docs[0].text == "hi"


# --------------------------------------------------------------------------
# Redirect to a private address must be blocked mid-chain (never fetched)
# --------------------------------------------------------------------------


def test_redirect_to_private_is_blocked(monkeypatch, fake_httpx):
    # example.com (public) 302 -> metadata endpoint; the second hop must be
    # rejected by the guard and never appear in the fetch log.
    _install_resolver(
        monkeypatch,
        {"example.com": "93.184.216.34", "meta.internal": "169.254.169.254"},
    )
    log: list[str] = []
    routes = {
        "http://example.com/": _FakeResponse(
            "http://example.com/", redirect_to="http://meta.internal/latest/"
        ),
        "http://meta.internal/latest/": _FakeResponse("http://meta.internal/latest/", "SECRET"),
    }
    fake_httpx(routes, log)
    with pytest.raises(SSRFValidationError, match="private/internal"):
        WebLoader("http://example.com/").load_sync()
    # The internal hop must never have been fetched.
    assert "http://meta.internal/latest/" not in log
