#!/usr/bin/env python3
"""Shared browser-demo registry for the demo audit tools.

Keep demo identity, base paths, default ports, and lightweight API markers in one
place so static docs checks, local HTTP smoke tests, and hosted-link probes fail
together when a browser demo moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Demo:
    name: str
    base_path: str
    api_path: str
    page_marker: str
    extra_args: tuple[str, ...] = ()
    default_port: int = 0
    hosted_path: str = ""
    hosted_port: int = 0
    hosted_api_keys: tuple[str, ...] = ()


DEMOS: tuple[Demo, ...] = (
    Demo("guarddemo", "/guarddemo", "api/scenarios", "safety floor", ("-jobs", "1"), 8151),
    Demo(
        "turntaxdemo", "/turntax", "api/suites", "turn-tax demo", ("-jobs", "1"), 8150,
        hosted_path="/", hosted_port=8150, hosted_api_keys=("suites",),
    ),
    Demo(
        "ctxdemo", "/ctxdemo", "api/scenarios", "multi-agent context demo", ("-jobs", "1"), 8153,
        hosted_path="/", hosted_port=8153, hosted_api_keys=("models", "scenarios"),
    ),
    Demo(
        "demorace", "/demorace", "api/ladder", "reuse demo", ("-jobs", "1"), 8147,
        hosted_path="/demorace/", hosted_api_keys=("models", "prefill_tok_ratio_a_over_c"),
    ),
    Demo("dropindemo", "/dropin", "api/gallery", "fak guard drop-in gallery", default_port=8154),
    Demo("unseedemo", "/unsee", "api/events", "Un-See It", default_port=8156),
    Demo("timewolfdemo", "/timewolf", "api/scenarios", "what time is it, Mr. Wolf?", default_port=8155),
    Demo("trychatdemo", "/trychat", "api/suggestions", "try-it agentic chat", default_port=8157),
)


def demo_map(demos: tuple[Demo, ...] = DEMOS) -> dict[str, Demo]:
    return {d.name: d for d in demos}


def discover_browser_demo_names(workspace: Path) -> list[str]:
    cmd_dir = workspace / "cmd"
    return sorted(p.parent.name for p in cmd_dir.glob("*/page.html"))


def registry_defects(workspace: Path, demos: tuple[Demo, ...] = DEMOS) -> list[str]:
    discovered = set(discover_browser_demo_names(workspace))
    registered = {d.name for d in demos}
    defects: list[str] = []
    for name in sorted(discovered - registered):
        defects.append(f"browser demo has page.html but is not in DEMOS registry: cmd/{name}")
    for name in sorted(registered - discovered):
        defects.append(f"DEMOS registry names a browser demo without page.html: cmd/{name}")
    return defects


def select_demos(names: list[str] | None, demos: tuple[Demo, ...] = DEMOS) -> tuple[list[Demo], list[str]]:
    if not names:
        return list(demos), []
    by_name = demo_map(demos)
    unknown = [name for name in names if name not in by_name]
    return [by_name[name] for name in names if name in by_name], unknown


def normalize_base(base_path: str) -> str:
    p = "/" + base_path.strip("/")
    return "" if p == "/" else p


def hosted_url(host: str, demo: Demo) -> str:
    if not demo.hosted_path:
        return ""
    port = f":{demo.hosted_port}" if demo.hosted_port else ""
    path = "/" if normalize_base(demo.hosted_path) == "" else f"{normalize_base(demo.hosted_path)}/"
    return f"http://{host}{port}{path}"


def hosted_demo_urls(host: str, demos: tuple[Demo, ...] = DEMOS) -> dict[str, str]:
    return {demo.name: url for demo in demos if (url := hosted_url(host, demo))}


def demo_url(port: int, base_path: str, rel: str = "") -> str:
    base = normalize_base(base_path)
    suffix = (rel or "").lstrip("/")
    if suffix:
        return f"http://127.0.0.1:{port}{base}/{suffix}"
    return f"http://127.0.0.1:{port}{base}/"
