from __future__ import annotations

import ast
import math
from typing import Any

from ..base import BaseTool, ToolResult

# Cap exponents to prevent DoS via astronomically large integers (e.g. 9**9**9).
_MAX_EXPONENT = 1000

# AST node types permitted in a calculator expression. Anything outside this
# allowlist (attribute access, subscripting, comprehensions, lambda, etc.) is
# rejected before evaluation, closing the ``().__class__.__bases__`` RCE path.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Tuple,
    # binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # unary operators
    ast.UAdd,
    ast.USub,
    # boolean / comparison operators
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "divmod": divmod,
    "int": int,
    "float": float,
    # math module
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "inf": math.inf,
}


_ALLOWED_NAMES = frozenset(k for k in _SAFE_GLOBALS if k != "__builtins__")


def _validate_expression(expr: str) -> ast.Expression:
    """Parse *expr* and reject anything outside the safe node/name allowlist.

    Raises ``ValueError`` on any disallowed construct: attribute access, calls
    to non-whitelisted names, subscripting, comprehensions, huge exponents, etc.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"Unknown name: {node.id!r}")
        if isinstance(node, ast.Call):
            # Only direct calls to whitelisted names, e.g. sqrt(16). No
            # attribute calls (obj.method()) and no calls of call results.
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError("Only calls to whitelisted math functions are allowed.")
            if any(kw.arg is None for kw in node.keywords):
                raise ValueError("Argument unpacking is not allowed.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            _reject_large_exponent(node.right)
    return tree


def _reject_large_exponent(exponent: ast.AST) -> None:
    """Reject unsafe exponents to prevent DoS (e.g. ``9**9**9``).

    The exponent must be a plain numeric literal (optionally unary-signed) that
    is <= ``_MAX_EXPONENT``. Anything more complex — a nested power, a call, a
    name — is rejected, because its value cannot be bounded cheaply and could
    evaluate to an astronomically large integer.
    """
    value: Any = None
    if isinstance(exponent, ast.Constant):
        value = exponent.value
    elif isinstance(exponent, ast.UnaryOp) and isinstance(exponent.operand, ast.Constant):
        value = exponent.operand.value
    else:
        raise ValueError("Exponent must be a simple numeric literal.")
    if not isinstance(value, (int, float)):
        raise ValueError("Exponent must be numeric.")
    if abs(value) > _MAX_EXPONENT:
        raise ValueError(f"Exponent too large (> {_MAX_EXPONENT}).")


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely."""

    name = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Input: a math expression string, e.g. '2 + 2 * 3' or 'sqrt(144)'. "
        "Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, pi, e, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate, e.g. '2 ** 10'",
            }
        },
        "required": ["expression"],
    }

    async def run(self, expression: str = "", **kwargs: Any) -> ToolResult:
        expr = expression or kwargs.get("input", "")
        if not expr:
            return ToolResult(output="", error="No expression provided.")
        try:
            tree = _validate_expression(expr)
        except SyntaxError as e:
            return ToolResult(output="", error=f"Could not evaluate expression: {e}")
        except ValueError as e:
            return ToolResult(output="", error=f"Unsafe or unsupported expression: {e}")
        try:
            result = eval(
                compile(tree, "<calculator>", "eval"), _SAFE_GLOBALS, {}
            )
            return ToolResult(output=str(result))
        except ZeroDivisionError:
            return ToolResult(output="", error="Division by zero.")
        except Exception as e:
            return ToolResult(output="", error=f"Could not evaluate expression: {e}")
