#!/usr/bin/env python3
"""Tests for demo_command_audit.py.

Run: `python tools/demo_command_audit_test.py`, or
`python -m pytest tools/demo_command_audit_test.py -q`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_command_audit as dca  # noqa: E402
import demo_registry as dr  # noqa: E402


def write(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fixture() -> tempfile.TemporaryDirectory[str]:
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    write(root, "cmd/good/main.go", "package main\nfunc main() {}\n")
    write(root, "cmd/good/main_test.go", "package main\nimport \"testing\"\nfunc TestGood(t *testing.T) {}\n")
    write(root, "tools/run_good.sh", "#!/usr/bin/env bash\n")
    write(root, "tools/good_tool.py", "print('ok')\n")
    write(root, "Makefile", "demo-smoke:\n\tpython tools/demo_http_smoke.py\n")
    return td


def test_collect_accepts_valid_go_and_tool_commands() -> None:
    with fixture() as td:
        root = Path(td)
        write(
            root,
            "docs/run-the-demos.md",
            f"""
`go run ./cmd/good`
FAK_DEMO_BASE_PATH=/good go run ./cmd/good
go build -trimpath -o out/good ./cmd/good
go -C {root.name} test ./cmd/good/ -run TestGood -v
go test ./cmd/good
bash tools/run_good.sh -q
python tools/good_tool.py --json
make demo-smoke
""",
        )
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert payload["ok"], payload
        assert payload["command_count"] == 8, payload
        assert payload["counts"] == {
            "go-build": 1,
            "go-run": 2,
            "go-test": 2,
            "make-target": 1,
            "python-tool": 1,
            "shell-script": 1,
        }, payload


def test_collect_rejects_missing_go_run_target() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "docs/run-the-demos.md", "go run ./cmd/ghost\n")
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert not payload["ok"], payload
        assert any("go run target missing: cmd/ghost" in d for d in payload["defects"]), payload


def test_collect_rejects_missing_go_build_target() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "docs/run-the-demos.md", 'RUN go build -trimpath -ldflags "-s -w" -o /out/ghost ./cmd/ghost\n')
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert not payload["ok"], payload
        assert any("go build target missing: cmd/ghost" in d for d in payload["defects"]), payload


def test_collect_rejects_missing_script_and_python_tool() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "docs/run-the-demos.md", "bash tools/missing.sh\npython tools/missing.py\n")
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert not payload["ok"], payload
        assert any("shell-script target missing: tools/missing.sh" in d for d in payload["defects"]), payload
        assert any("python-tool target missing: tools/missing.py" in d for d in payload["defects"]), payload


def test_collect_rejects_go_test_target_without_test_file() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "cmd/notests/main.go", "package main\nfunc main() {}\n")
        write(root, "docs/run-the-demos.md", "go test ./cmd/notests\n")
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert not payload["ok"], payload
        assert any("go test target has no *_test.go: cmd/notests" in d for d in payload["defects"]), payload


def test_collect_rejects_missing_make_target() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "docs/run-the-demos.md", "make missing-target\n")
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert not payload["ok"], payload
        assert any("make target missing from Makefile: missing-target" in d for d in payload["defects"]), payload


def test_collect_rejects_unsupported_go_c_directory() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "docs/run-the-demos.md", "go -C elsewhere test ./cmd/good/\n")
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert not payload["ok"], payload
        assert any("unsupported go -C directory in demo command: elsewhere" in d for d in payload["defects"]), payload


def test_collect_rejects_bare_inline_cmd_path() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "docs/run-the-demos.md", "`./cmd/good` and <code>./cmd/good</code>\n")
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert not payload["ok"], payload
        assert len([d for d in payload["defects"] if "bare cmd path in inline code" in d]) == 2, payload


def test_browser_demo_coverage_defects_flags_registry_demo_without_go_run_doc() -> None:
    refs = [
        dca.CommandRef("docs/run-the-demos.md", 1, "go-run", "cmd/guarddemo", "go run ./cmd/guarddemo"),
    ]
    demos = (
        dr.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety", default_port=8151),
        dr.Demo("newdemo", "/newdemo", "api/state", "new", default_port=8199),
    )
    defects = dca.browser_demo_coverage_defects(refs, demos=demos)
    assert defects == ["browser demo registry entry is not documented with a go run command: cmd/newdemo"], defects


def test_collect_explicit_sources_skips_repo_browser_coverage_gate() -> None:
    with fixture() as td:
        root = Path(td)
        write(root, "docs/run-the-demos.md", "go run ./cmd/good\n")
        payload = dca.collect(root, sources=["docs/run-the-demos.md"])
        assert payload["ok"], payload


def test_collect_real_demo_docs_are_clean() -> None:
    payload = dca.collect(dca.repo_root())
    assert payload["ok"], payload


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("demo_command_audit_test: OK")
