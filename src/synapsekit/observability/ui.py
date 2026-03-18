"""Local tracing UI — lightweight web dashboard for viewing traces."""

from __future__ import annotations

import json
from typing import Any

from .otel import OTelExporter

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>SynapseKit Traces</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }}
.span {{ border-left: 3px solid #30363d; margin: 8px 0; padding: 8px 16px; background: #161b22; border-radius: 4px; }}
.span-name {{ font-weight: bold; color: #58a6ff; }}
.span-duration {{ color: #8b949e; font-size: 0.9em; }}
.span-attr {{ color: #8b949e; font-size: 0.85em; margin-top: 4px; }}
.status-ok {{ border-left-color: #3fb950; }}
.status-error {{ border-left-color: #f85149; }}
.children {{ margin-left: 24px; }}
.stats {{ display: flex; gap: 20px; margin: 16px 0; }}
.stat {{ background: #161b22; padding: 12px 20px; border-radius: 8px; }}
.stat-value {{ font-size: 1.5em; color: #58a6ff; font-weight: bold; }}
.stat-label {{ color: #8b949e; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>SynapseKit Traces</h1>
<div class="stats">
<div class="stat"><div class="stat-value">{total_spans}</div><div class="stat-label">Total Spans</div></div>
<div class="stat"><div class="stat-value">{total_duration:.0f}ms</div><div class="stat-label">Total Duration</div></div>
<div class="stat"><div class="stat-value">{error_count}</div><div class="stat-label">Errors</div></div>
</div>
{spans_html}
</body>
</html>"""


def _render_span(span: dict[str, Any], depth: int = 0) -> str:
    status_class = "status-error" if span.get("status") == "error" else "status-ok"
    duration = span.get("duration_ms", 0)
    attrs = span.get("attributes", {})
    attrs_html = ""
    if attrs:
        attrs_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
        attrs_html = f'<div class="span-attr">{attrs_str}</div>'

    children_html = ""
    if span.get("children"):
        children_parts = [_render_span(c, depth + 1) for c in span["children"]]
        children_html = f'<div class="children">{"".join(children_parts)}</div>'

    return (
        f'<div class="span {status_class}">'
        f'<span class="span-name">{span["name"]}</span> '
        f'<span class="span-duration">{duration:.1f}ms</span>'
        f"{attrs_html}"
        f"{children_html}"
        f"</div>"
    )


def _count_spans(spans: list[dict[str, Any]]) -> tuple[int, int, float]:
    total = 0
    errors = 0
    duration = 0.0
    for span in spans:
        total += 1
        if span.get("status") == "error":
            errors += 1
        duration += span.get("duration_ms", 0)
        child_total, child_errors, _child_duration = _count_spans(span.get("children", []))
        total += child_total
        errors += child_errors
    return total, errors, duration


class TracingUI:
    """Serve a local web UI for viewing SynapseKit traces.

    Usage::
        exporter = OTelExporter()
        ui = TracingUI(exporter)

        # Generate HTML report
        html = ui.render_html()

        # Save to file
        ui.save_html("traces.html")

        # Serve on localhost (requires no extra deps)
        ui.serve(port=8080)  # opens http://localhost:8080
    """

    def __init__(self, exporter: OTelExporter) -> None:
        self._exporter = exporter

    def render_html(self) -> str:
        """Render traces as an HTML page."""
        spans = self._exporter.export()
        total, errors, duration = _count_spans(spans)

        spans_html = "".join(_render_span(s) for s in spans)

        return _HTML_TEMPLATE.format(
            total_spans=total,
            total_duration=duration,
            error_count=errors,
            spans_html=spans_html,
        )

    def save_html(self, path: str) -> None:
        """Save trace report as HTML file."""
        html = self.render_html()
        with open(path, "w") as f:
            f.write(html)

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Serve traces via a local HTTP server (stdlib only)."""
        import http.server
        import webbrowser

        ui = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(ui.render_html().encode())

            def log_message(self, format, *args):
                pass  # suppress logs

        server = http.server.HTTPServer((host, port), Handler)
        print(f"SynapseKit Traces UI: http://{host}:{port}")
        webbrowser.open(f"http://{host}:{port}")
        server.serve_forever()

    def get_json(self) -> str:
        """Get traces as JSON string."""
        return json.dumps(self._exporter.export(), indent=2)
