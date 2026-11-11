"""Tools available to the agents: web search and a safe calculator."""
from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool


def ddg_search(query: str, max_results: int = 4) -> list[dict]:
    """Free web search through DuckDuckGo (no API key needed)."""
    from ddgs import DDGS

    try:
        hits = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:  # network errors, rate limits...
        print(f"[web_search] failed: {exc}")
        return []
    return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")} for h in hits]


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {name: getattr(math, name) for name in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "floor", "ceil")}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '(1250 - 980) / 980 * 100'.
    Use this for any percentage, growth, ratio or other numeric calculation."""
    try:
        result = _eval(ast.parse(expression.replace("^", "**"), mode="eval"))
        return str(round(result, 6) if isinstance(result, float) else result)
    except Exception as exc:
        return f"Error: {exc}"
