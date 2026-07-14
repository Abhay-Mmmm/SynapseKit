"""Tests for the shared SSRF URL guard (issues #806 / #807).

No mocks: DNS resolution is redirected to a deterministic in-test resolver via
monkeypatch, exercising the real guard logic against real ipaddress checks.
"""

from __future__ import annotations

import socket

import pytest

from synapsekit.loaders._url_guard import (
    SSRFValidationError,
    resolve_public_addresses,
    validate_public_url,
)


def _fake_getaddrinfo_factory(mapping: dict[str, list[str]]):
    """Return a getaddrinfo replacement resolving hosts from *mapping*.

    An unmapped host raises socket.gaierror, mirroring a real NXDOMAIN.
    """

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        infos = []
        for ip in mapping[host]:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, 0, 0, 0) if family == socket.AF_INET6 else (ip, 0)
            infos.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr))
        return infos

    return fake_getaddrinfo


@pytest.fixture
def resolver(monkeypatch):
    def _install(mapping: dict[str, list[str]]) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_factory(mapping))

    return _install


# --------------------------------------------------------------------------
# Scheme validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["ftp://example.com", "file:///etc/passwd", "gopher://x"])
def test_rejects_non_http_scheme(url):
    with pytest.raises(SSRFValidationError, match="not allowed"):
        validate_public_url(url)


def test_rejects_missing_hostname():
    with pytest.raises(SSRFValidationError, match="no hostname"):
        validate_public_url("http:///path")


# --------------------------------------------------------------------------
# Public host — allowed
# --------------------------------------------------------------------------


def test_public_host_allowed(resolver):
    resolver({"example.com": ["93.184.216.34"]})
    validate_public_url("https://example.com/page")  # no raise
    assert resolve_public_addresses("example.com") == ["93.184.216.34"]


def test_public_ip_literal_allowed():
    validate_public_url("http://93.184.216.34/")  # no DNS needed


# --------------------------------------------------------------------------
# Private / loopback / link-local / reserved — rejected (adversarial)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "169.254.169.254",  # cloud metadata (the classic SSRF target)
        "10.1.2.3",  # RFC1918
        "172.16.5.5",  # RFC1918
        "192.168.1.1",  # RFC1918
        "0.0.0.0",  # this-host
        "100.64.0.1",  # CGNAT
        "::1",  # IPv6 loopback
        "fe80::1",  # IPv6 link-local
        "fc00::1234",  # IPv6 ULA
    ],
)
def test_private_ip_literal_rejected(ip):
    with pytest.raises(SSRFValidationError, match="private/internal"):
        validate_public_url(f"http://[{ip}]/" if ":" in ip else f"http://{ip}/")


def test_private_via_dns_rejected(resolver):
    """A public-looking hostname that resolves to a private IP must be blocked."""
    resolver({"evil.example.com": ["169.254.169.254"]})
    with pytest.raises(SSRFValidationError, match="private/internal"):
        validate_public_url("http://evil.example.com/latest/meta-data/")


def test_mixed_resolution_rejected_if_any_private(resolver):
    """If a host resolves to both a public and a private address, reject it."""
    resolver({"sneaky.example.com": ["93.184.216.34", "127.0.0.1"]})
    with pytest.raises(SSRFValidationError, match="private/internal"):
        validate_public_url("http://sneaky.example.com/")


def test_ipv4_mapped_ipv6_loopback_rejected():
    """::ffff:127.0.0.1 must be normalised and blocked, not slip through."""
    with pytest.raises(SSRFValidationError, match="private/internal"):
        validate_public_url("http://[::ffff:127.0.0.1]/")


# --------------------------------------------------------------------------
# Fail CLOSED on resolution failure (regression for #807 fail-open bug)
# --------------------------------------------------------------------------


def test_unresolvable_host_fails_closed(resolver):
    resolver({})  # every lookup raises gaierror
    with pytest.raises(SSRFValidationError, match="Could not resolve"):
        validate_public_url("http://does-not-exist.invalid/")
