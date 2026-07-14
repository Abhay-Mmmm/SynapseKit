from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..base import BaseTool, ToolResult

# Maximum number of bytes a single read will return, mirroring the mesh loader
# precedent (synapsekit.mesh.loaders.DEFAULT_MAX_FILE_BYTES).
DEFAULT_MAX_FILE_BYTES = 2_000_000


class FileReadTool(BaseTool):
    """Read the contents of a local file."""

    name = "file_read"
    description = "Read the contents of a file from disk. Input: an absolute or relative file path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to read",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8)",
                "default": "utf-8",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self, base_dir: str | None = None, max_bytes: int = DEFAULT_MAX_FILE_BYTES
    ) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir else None
        self._max_bytes = max_bytes

    def _read_blocking(self, resolved: Path, encoding: str) -> str:
        """Read the file synchronously, enforcing the byte cap. Runs in a thread."""
        size = resolved.stat().st_size
        if size > self._max_bytes:
            raise ValueError(
                f"File is too large: {size} bytes exceeds the limit of {self._max_bytes} bytes."
            )
        with open(resolved, encoding=encoding) as f:
            return f.read()

    async def run(self, path: str = "", encoding: str = "utf-8", **kwargs: Any) -> ToolResult:
        file_path = path or kwargs.get("input", "")
        if not file_path:
            return ToolResult(output="", error="No file path provided.")
        try:
            resolved = Path(file_path).resolve()
            if self._base_dir is not None and not str(resolved).startswith(str(self._base_dir)):
                return ToolResult(
                    output="", error="Access denied: path is outside the allowed directory."
                )
            # open()/read()/stat() are blocking; run them off the event loop.
            content = await asyncio.to_thread(self._read_blocking, resolved, encoding)
            return ToolResult(output=content)
        except ValueError as e:
            return ToolResult(output="", error=str(e))
        except FileNotFoundError:
            return ToolResult(output="", error=f"File not found: {file_path!r}")
        except PermissionError:
            return ToolResult(output="", error=f"Permission denied: {file_path!r}")
        except Exception as e:
            return ToolResult(output="", error=f"Error reading file: {e}")
