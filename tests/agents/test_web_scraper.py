import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.tools.web_scraper import WebScraperTool

bs4 = pytest.importorskip("bs4")

# Create mock httpx module for environments where httpx isn't installed
mock_httpx = MagicMock()
mock_httpx.HTTPError = type("HTTPError", (Exception,), {})


def _make_httpx_client(mock_response=None, side_effect=None):
    """Helper to create a mock httpx async client."""
    mock_client = MagicMock()
    if side_effect:
        mock_client.get = AsyncMock(side_effect=side_effect)
    else:
        mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return mock_client


def _make_response(html):
    """Helper to create a mock HTTP response."""
    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()
    return mock_response


class TestWebScraperTool:
    def test_init_defaults(self):
        tool = WebScraperTool()
        assert tool.timeout == 30
        assert tool.name == "web_scraper"

    def test_init_custom_timeout(self):
        tool = WebScraperTool(timeout=60)
        assert tool.timeout == 60

    @pytest.mark.asyncio
    async def test_run_success(self):
        tool = WebScraperTool()
        resp = _make_response("<html><body><p>Hello World</p></body></html>")
        _make_httpx_client(mock_response=resp)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await tool.run(url="https://example.com")

        assert not result.is_error
        assert "Hello World" in result.output

    @pytest.mark.asyncio
    async def test_run_strips_unwanted_elements(self):
        tool = WebScraperTool()
        html = """
        <html><body>
            <script>alert('x')</script>
            <style>.a{color:red}</style>
            <nav>Nav</nav>
            <header>Header</header>
            <footer>Footer</footer>
            <aside>Sidebar</aside>
            <p>Main content</p>
        </body></html>
        """
        resp = _make_response(html)
        _make_httpx_client(mock_response=resp)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await tool.run(url="https://example.com")

        assert "Main content" in result.output
        for unwanted in ("alert", "Nav", "Header", "Footer", "Sidebar"):
            assert unwanted not in result.output

    @pytest.mark.asyncio
    async def test_run_with_css_selector(self):
        tool = WebScraperTool()
        html = "<html><body><article><p>Article text</p></article><div>Other</div></body></html>"
        resp = _make_response(html)
        _make_httpx_client(mock_response=resp)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await tool.run(url="https://example.com", css_selector="article")

        assert "Article text" in result.output
        assert "Other" not in result.output

    @pytest.mark.asyncio
    async def test_run_css_selector_no_match(self):
        tool = WebScraperTool()
        resp = _make_response("<html><body><p>Hello</p></body></html>")
        _make_httpx_client(mock_response=resp)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await tool.run(url="https://example.com", css_selector=".nonexistent")

        assert result.output == ""

    @pytest.mark.asyncio
    async def test_run_no_url(self):
        tool = WebScraperTool()

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await tool.run()

        assert result.is_error
        assert "No URL provided" in result.error

    @pytest.mark.asyncio
    async def test_run_http_error(self):
        tool = WebScraperTool()
        _make_httpx_client(side_effect=Exception("Connection failed"))

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await tool.run(url="https://example.com")

        assert result.is_error
        assert "Scraping failed" in result.error

    @pytest.mark.asyncio
    async def test_run_url_from_kwargs(self):
        tool = WebScraperTool()
        resp = _make_response("<html><body><p>Content</p></body></html>")
        _make_httpx_client(mock_response=resp)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await tool.run(input="https://example.com")

        assert not result.is_error
        assert "Content" in result.output

    def test_import_from_synapsekit(self):
        from synapsekit import WebScraperTool as WST

        assert WST is WebScraperTool
