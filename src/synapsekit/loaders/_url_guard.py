"""Shared SSRF guard for loaders that fetch remote URLs.

Both :mod:`synapsekit.loaders.web` and :mod:`synapsekit.loaders.sitemap` fetch
arbitrary, potentially attacker-controlled URLs. Without validation these are a
classic Server-Side Request Forgery (SSRF) vector: an attacker can point a URL
at ``http://169.254.169.254/`` (cloud metadata), ``http://127.0.0.1/`` or an
internal ``10.0.0.0/8`` service and exfiltrate credentials or reach internal
APIs.

This module centralises the guard so every fetch path uses identical rules:

* only ``http`` / ``https`` schemes are permitted;
* the hostname is resolved and **every** resolved address must be public;
* resolution failure fails **closed** (raises), never open.

.. warning::

   ``validate_public_url`` resolves DNS itself, but an HTTP client re-resolves
   the host at connection time. A malicious authoritative server can answer the
   validation lookup with a public IP and the connection lookup with a private
   one (DNS rebinding / TOCTOU). Callers should therefore validate again
   *immediately* before issuing the request and, where possible, on every
   redirect hop (see :func:`assert_response_url_public`). This narrows, but does
   not fully close, the rebinding window; pinning the socket to the validated IP
   is the only complete mitigation and is intentionally left out here to keep
   the guard client-agnostic.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

__all__ = [
    "SSRFValidationError",
    "assert_response_url_public",
    "redirect_target",
    "resolve_public_addresses",
    "validate_public_url",
]

# HTTP status codes that carry a Location redirect.
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


def redirect_target(response: object) -> str | None:
    """Return the absolute redirect URL for *response*, or ``None``.

    Robust against duck-typed/fake responses: only treats the response as a
    redirect when the status code is a genuine 3xx redirect **and** a non-empty
    string ``Location`` is present. Used to follow redirects manually so every
    hop can be re-validated by the SSRF guard.
    """
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or status not in _REDIRECT_STATUS:
        return None
    headers = getattr(response, "headers", None)
    location = headers.get("location") if headers is not None else None
    if not isinstance(location, str) or not location:
        return None
    # Prefer httpx's resolved absolute next_request.url when available.
    next_request = getattr(response, "next_request", None)
    next_url = getattr(next_request, "url", None) if next_request is not None else None
    if next_url is not None:
        return str(next_url)
    return location

# Networks that must never be reached from a user-supplied URL. Covers loopback,
# RFC1918 private ranges, link-local (incl. the 169.254.169.254 cloud metadata
# endpoint), carrier-grade NAT, and their IPv6 equivalents / mapped forms.
_BLOCKED_NETS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class SSRFValidationError(ValueError):
    """Raised when a URL is rejected by the SSRF guard.

    Subclasses :class:`ValueError` so existing ``except ValueError`` call sites
    keep working.
    """


def _is_blocked_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # Normalise IPv4-mapped IPv6 (::ffff:127.0.0.1) to its IPv4 form so a mapped
    # loopback/private address can't slip past the IPv4 network checks.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if not addr.is_global:
        return True
    return any(addr in net for net in _BLOCKED_NETS)


def resolve_public_addresses(host: str) -> list[str]:
    """Resolve *host* and return its public IP addresses as strings.

    Fails **closed**: a resolution error or any private/loopback/link-local/
    reserved address raises :class:`SSRFValidationError`.
    """
    if not host:
        raise SSRFValidationError("URL has no hostname.")

    # A bare IP literal skips DNS entirely; validate it directly.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_address(literal):
            raise SSRFValidationError(
                f"Requests to private/internal addresses are not allowed: {host!r}"
            )
        return [str(literal)]

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        # Fail closed: an unresolvable host must be rejected, never fetched.
        raise SSRFValidationError(f"Could not resolve host {host!r}: {exc}") from exc

    resolved: list[str] = []
    for info in infos:
        # sockaddr is (host, port[, flowinfo, scope_id]); host is always a str.
        ip_str = str(info[4][0])
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise SSRFValidationError(f"Invalid resolved address {ip_str!r}") from exc
        if _is_blocked_address(addr):
            raise SSRFValidationError(
                f"Requests to private/internal addresses are not allowed: "
                f"{host!r} resolves to {ip_str}"
            )
        resolved.append(ip_str)

    if not resolved:
        raise SSRFValidationError(f"Host {host!r} did not resolve to any address.")
    return resolved


def validate_public_url(url: str) -> None:
    """Raise :class:`SSRFValidationError` unless *url* is safe to fetch.

    Checks the scheme, requires a hostname, resolves it, and rejects the URL if
    any resolved address is non-public. Fails closed on resolution errors.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFValidationError(
            f"URL scheme {parsed.scheme!r} is not allowed; use http or https."
        )
    resolve_public_addresses(parsed.hostname or "")


def assert_response_url_public(url: str) -> None:
    """Re-validate a (possibly redirected) URL before/after a request.

    Alias of :func:`validate_public_url` with a name that documents intent at
    redirect-handling call sites. Use it on every hop when following redirects
    manually so a redirect to an internal address is rejected.
    """
    validate_public_url(url)
