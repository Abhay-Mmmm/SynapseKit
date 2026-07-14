from __future__ import annotations

from typing import Any

from ._url_guard import assert_response_url_public, redirect_target, validate_public_url
from .base import Document


def _redirect_target(response: Any) -> str | None:
    return redirect_target(response)

_MAX_REDIRECTS = 10


def _validate_url(url: str) -> None:
    """Reject non-http(s) schemes and URLs resolving to private addresses.

    Thin wrapper kept for backwards compatibility; delegates to the shared
    SSRF guard which fails closed on resolution errors (unlike the previous
    fail-open behaviour, which allowed a DNS-rebinding / TOCTOU bypass).
    """
    validate_public_url(url)


class WebLoader:
    """Fetch a URL and return its text content as a Document."""

    def __init__(self, url: str) -> None:
        _validate_url(url)
        self._url = url

    def _parse(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 required: pip install synapsekit[web]") from None
        soup = BeautifulSoup(html, "html.parser")
        return str(soup.get_text(separator="\n", strip=True))

    async def load(self) -> list[Document]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        # Disable httpx auto-redirects and follow them manually so every hop is
        # re-validated against the SSRF guard. Re-validating immediately before
        # the request also narrows the DNS-rebinding window between __init__ and
        # the actual connection (see _url_guard for the residual risk note).
        url = self._url
        async with httpx.AsyncClient(follow_redirects=False) as client:
            response = await self._get_validated(client.get, url)

        text = self._parse(response.text)
        return [Document(text=text, metadata={"source": self._url})]

    def load_sync(self) -> list[Document]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        with httpx.Client(follow_redirects=False) as client:
            response = self._get_validated_sync(client.get)

        text = self._parse(response.text)
        return [Document(text=text, metadata={"source": self._url})]

    async def _get_validated(self, getter: Any, url: str) -> Any:
        """Issue GET(s) following redirects manually, validating each hop."""
        for _ in range(_MAX_REDIRECTS + 1):
            assert_response_url_public(url)
            response = await getter(url)
            next_url = _redirect_target(response)
            if next_url is not None:
                url = next_url
                continue
            response.raise_for_status()
            return response
        raise ValueError(f"Too many redirects while fetching {self._url!r}")

    def _get_validated_sync(self, getter: Any) -> Any:
        url = self._url
        for _ in range(_MAX_REDIRECTS + 1):
            assert_response_url_public(url)
            response = getter(url)
            next_url = _redirect_target(response)
            if next_url is not None:
                url = next_url
                continue
            response.raise_for_status()
            return response
        raise ValueError(f"Too many redirects while fetching {self._url!r}")
