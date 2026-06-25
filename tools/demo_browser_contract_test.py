#!/usr/bin/env python3
"""Tests for demo_browser_contract.py."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_browser_contract as dbc  # noqa: E402
import demo_registry as dr  # noqa: E402


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def browser_main(port: int, base_path: str) -> str:
    return f'''
package main

const defaultAddr = "127.0.0.1:{port}"

func main() {{
    basePath := demoui.BasePathFlag(flag.CommandLine, "{base_path}")
    app := http.NewServeMux()
    mux := http.NewServeMux()
    base := demoui.MountWithBasePath(mux, *basePath, app)
    bind := demoui.ListenAddr(*addr, defaultAddr)
    _ = demoui.LocalURL(bind, base)
}}
'''


def test_source_default_addr_extracts_port() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write(root, "cmd/guarddemo/main.go", 'package main\nconst defaultAddr = "127.0.0.1:8151"\n')
        demo = dr.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety", default_port=8151)
        assert dbc.source_default_addr(root, demo) == ("127.0.0.1:8151", 8151)


def test_demo_contract_accepts_matching_doc_and_readme() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        demo = dr.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety", default_port=8151)
        write(root, "cmd/guarddemo/main.go", browser_main(8151, "/guarddemo"))
        write(root, "cmd/guarddemo/README.md", "FAK_DEMO_BASE_PATH=/guarddemo go run ./cmd/guarddemo\n")
        run_doc = """
go run ./cmd/guarddemo # -> http://127.0.0.1:8151
PORT=8151 ./guarddemo
FAK_DEMO_BASE_PATH=/guarddemo PORT=8151 ./guarddemo
location /guarddemo/ {
  proxy_pass http://127.0.0.1:8151;
}
"""
        public_doc = "go run ./cmd/guarddemo # -> http://127.0.0.1:8151\n"
        assert dbc.demo_contract_defects(root, demo, run_doc, public_doc) == []


def test_demo_contract_flags_mismatched_port_and_missing_readme_command() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        demo = dr.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety", default_port=8151)
        write(root, "cmd/guarddemo/main.go", 'const defaultAddr = "127.0.0.1:9000"\n')
        write(root, "cmd/guarddemo/README.md", "go run ./cmd/guarddemo\n")
        defects = dbc.demo_contract_defects(root, demo, "", "")
        assert "guarddemo: main.go defaultAddr 127.0.0.1:9000, want 127.0.0.1:8151" in defects
        assert any("README missing base-path env command" in d for d in defects)
        assert any("missing local loopback URL" in d for d in defects)


def test_demo_contract_flags_ad_hoc_base_path_helpers() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        demo = dr.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety", default_port=8151)
        write(
            root,
            "cmd/guarddemo/main.go",
            '''
const defaultAddr = "127.0.0.1:8151"
func main() {
    _ = flag.String("base-path", os.Getenv(demoui.DemoBasePathEnv), "path")
}
func listenAddr(addr string) string { return addr }
''',
        )
        defects = dbc.demo_contract_defects(root, demo, "", "")
        assert any("missing shared base-path flag helper" in d for d in defects), defects
        assert any("defines base-path flag directly" in d for d in defects), defects
        assert any("defines local listenAddr" in d for d in defects), defects
        assert any("reads guarddemo base-path env directly" in d for d in defects), defects


def test_demo_contract_flags_public_page_drift() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        demo = dr.Demo("demorace", "/demorace", "api/run", "race", default_port=8147)
        write(root, "cmd/demorace/main.go", browser_main(8147, "/demorace"))
        write(root, "cmd/demorace/README.md", "FAK_DEMO_BASE_PATH=/demorace go run ./cmd/demorace\n")
        run_doc = """
go run ./cmd/demorace # -> http://127.0.0.1:8147
PORT=8147 ./demorace
FAK_DEMO_BASE_PATH=/demorace PORT=8147 ./demorace
location /demorace/ {
  proxy_pass http://127.0.0.1:8147;
}
"""
        public_doc = "<code>./cmd/demorace</code>\n"
        defects = dbc.demo_contract_defects(root, demo, run_doc, public_doc)
        assert any("docs/demos.html missing public local go-run command" in d for d in defects)
        assert any("docs/demos.html missing public local loopback URL" in d for d in defects)
        assert any("docs/demos.html has bare command path" in d for d in defects)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("demo_browser_contract_test: OK")
