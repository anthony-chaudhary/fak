package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// gate_democommand_test.go — parity for the DEMO_COMMAND gate. The Python checker
// (tools/demo_command_audit.py) is workspace-scoped (no --audit-staged / --root), so the staged
// runParity harness does not cover it. Instead we (1) replay the golden fixtures from
// tools/demo_command_audit_test.py — the oracle's own accept/reject samples — against the ported
// collect/extract logic, (2) pin the hardcoded browser-demo registry against the live Python DEMOS,
// (3) prove the live tracked tree is clean (the twin of test_collect_real_demo_docs_are_clean), and
// (4) run a verdict-level differential against the live Python over the real tree.

// demoWrite creates root/rel with content, making parent dirs — the Go twin of the Python test write().
func demoWrite(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", p, err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
}

// demoFixture builds the same on-disk fixture as demo_command_audit_test.fixture(): a runnable
// cmd/good package (main.go + a test), a shell + python tool under tools/, and a Makefile with one
// real target.
func demoFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	demoWrite(t, root, "cmd/good/main.go", "package main\nfunc main() {}\n")
	demoWrite(t, root, "cmd/good/main_test.go", "package main\nimport \"testing\"\nfunc TestGood(t *testing.T) {}\n")
	demoWrite(t, root, "tools/run_good.sh", "#!/usr/bin/env bash\n")
	demoWrite(t, root, "tools/good_tool.py", "print('ok')\n")
	demoWrite(t, root, "Makefile", "demo-smoke:\n\tpython tools/demo_http_smoke.py\n")
	return root
}

// TestDemoCommand_AcceptsValidCommands is the twin of test_collect_accepts_valid_go_and_tool_commands:
// every documented go/tool/make command resolves, and the extracted per-kind counts match.
func TestDemoCommand_AcceptsValidCommands(t *testing.T) {
	root := demoFixture(t)
	doc := "\n" +
		"`go run ./cmd/good`\n" +
		"FAK_DEMO_BASE_PATH=/good go run ./cmd/good\n" +
		"go build -trimpath -o out/good ./cmd/good\n" +
		"go -C " + filepath.Base(root) + " test ./cmd/good/ -run TestGood -v\n" +
		"go test ./cmd/good\n" +
		"bash tools/run_good.sh -q\n" +
		"python tools/good_tool.py --json\n" +
		"make demo-smoke\n"
	demoWrite(t, root, "docs/run-the-demos.md", doc)

	defects := demoCommandDefects(root, []string{"docs/run-the-demos.md"})
	if len(defects) != 0 {
		t.Fatalf("expected clean, got defects: %+v", defects)
	}

	counts := map[string]int{}
	for _, r := range extractDemoCommandRefs("docs/run-the-demos.md", doc) {
		counts[r.kind]++
	}
	want := map[string]int{"go-build": 1, "go-run": 2, "go-test": 2, "make-target": 1, "python-tool": 1, "shell-script": 1}
	if len(counts) != len(want) {
		t.Fatalf("counts = %+v, want %+v", counts, want)
	}
	for k, v := range want {
		if counts[k] != v {
			t.Fatalf("counts[%q] = %d, want %d (all=%+v)", k, counts[k], v, counts)
		}
	}
}

// TestDemoCommand_RejectsMissingTargets is the twin of the reject-* golden tests: each stale reference
// produces its specific defect.
func TestDemoCommand_RejectsMissingTargets(t *testing.T) {
	cases := []struct {
		name string
		doc  string
		want string
	}{
		{"missing go run", "go run ./cmd/ghost\n", "go run target missing: cmd/ghost"},
		{"missing go build", `RUN go build -trimpath -ldflags "-s -w" -o /out/ghost ./cmd/ghost` + "\n", "go build target missing: cmd/ghost"},
		{"missing shell script", "bash tools/missing.sh\n", "shell-script target missing: tools/missing.sh"},
		{"missing python tool", "python tools/missing.py\n", "python-tool target missing: tools/missing.py"},
		{"missing make target", "make missing-target\n", "make target missing from Makefile: missing-target"},
		{"unsupported go -C dir", "go -C elsewhere test ./cmd/good/\n", "unsupported go -C directory in demo command: elsewhere"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := demoFixture(t)
			demoWrite(t, root, "docs/run-the-demos.md", c.doc)
			defects := demoCommandDefects(root, []string{"docs/run-the-demos.md"})
			if !demoAnyContains(defects, c.want) {
				t.Fatalf("want a defect containing %q, got %+v", c.want, defects)
			}
		})
	}
}

// TestDemoCommand_RejectsGoTestWithoutTestFile is the twin of test_collect_rejects_go_test_target_without_test_file.
func TestDemoCommand_RejectsGoTestWithoutTestFile(t *testing.T) {
	root := demoFixture(t)
	demoWrite(t, root, "cmd/notests/main.go", "package main\nfunc main() {}\n")
	demoWrite(t, root, "docs/run-the-demos.md", "go test ./cmd/notests\n")
	defects := demoCommandDefects(root, []string{"docs/run-the-demos.md"})
	if !demoAnyContains(defects, "go test target has no *_test.go: cmd/notests") {
		t.Fatalf("want no-test-file defect, got %+v", defects)
	}
}

// TestDemoCommand_RejectsBareInlineCmd is the twin of test_collect_rejects_bare_inline_cmd_path: both
// the backtick and <code> forms on one line are flagged.
func TestDemoCommand_RejectsBareInlineCmd(t *testing.T) {
	root := demoFixture(t)
	demoWrite(t, root, "docs/run-the-demos.md", "`./cmd/good` and <code>./cmd/good</code>\n")
	defects := demoCommandDefects(root, []string{"docs/run-the-demos.md"})
	n := 0
	for _, d := range defects {
		if strings.Contains(d, "bare cmd path in inline code") {
			n++
		}
	}
	if n != 2 {
		t.Fatalf("want 2 bare-cmd defects, got %d (all=%+v)", n, defects)
	}
}

// TestDemoCommand_ExplicitSourcesSkipCoverage is the twin of
// test_collect_explicit_sources_skips_repo_browser_coverage_gate: an explicit --source disables the
// registry-coverage gate, so a doc with one valid command is clean even though no browser demo is documented.
func TestDemoCommand_ExplicitSourcesSkipCoverage(t *testing.T) {
	root := demoFixture(t)
	demoWrite(t, root, "docs/run-the-demos.md", "go run ./cmd/good\n")
	if defects := demoCommandDefects(root, []string{"docs/run-the-demos.md"}); len(defects) != 0 {
		t.Fatalf("explicit sources should skip coverage gate; got %+v", defects)
	}
}

// TestDemoCommand_CoverageFlagsUndocumentedDemo is the twin of
// test_browser_demo_coverage_defects_flags_registry_demo_without_go_run_doc: a registered demo with no
// `go run` doc is flagged (using the real registry list, minus one documented name).
func TestDemoCommand_CoverageFlagsUndocumentedDemo(t *testing.T) {
	// Document only the first registry demo; every other registered demo must be flagged.
	refs := []demoCommandRef{{source: "docs/run-the-demos.md", line: 1, kind: "go-run", target: "cmd/" + demoBrowserNames[0], command: "go run ./cmd/" + demoBrowserNames[0]}}
	defects := demoBrowserCoverageDefects(refs)
	if len(defects) != len(demoBrowserNames)-1 {
		t.Fatalf("want %d coverage defects, got %d: %+v", len(demoBrowserNames)-1, len(defects), defects)
	}
	for _, name := range demoBrowserNames[1:] {
		if !demoAnyContains(defects, "not documented with a go run command: cmd/"+name) {
			t.Fatalf("expected a coverage defect for cmd/%s, got %+v", name, defects)
		}
	}
}

// TestDemoCommand_RegistryMatchesPython pins demoBrowserNames against the live tools/demo_registry.py
// DEMOS. A registry add/remove that skips this file reds `go test` here, so the hardcoded coverage
// list can never silently drift from the oracle. Skipped under -short or when python/git is absent.
func TestDemoCommand_RegistryMatchesPython(t *testing.T) {
	if testing.Short() {
		t.Skip("python registry pin skipped under -short")
	}
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	clone := repoRoot(t)
	args := append(append([]string{}, pyArgs...), "-c",
		"import sys; sys.path.insert(0, 'tools'); import demo_registry as dr; print('\\n'.join(d.name for d in dr.DEMOS))")
	cmd := exec.Command(py, args...)
	cmd.Dir = clone
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("python demo_registry dump failed: %v", err)
	}
	var pyNames []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if s := strings.TrimSpace(line); s != "" {
			pyNames = append(pyNames, s)
		}
	}
	goNames := append([]string{}, demoBrowserNames...)
	sort.Strings(goNames)
	sort.Strings(pyNames)
	if strings.Join(goNames, ",") != strings.Join(pyNames, ",") {
		t.Fatalf("demoBrowserNames drifted from demo_registry.DEMOS:\n  go:     %v\n  python: %v", goNames, pyNames)
	}
}

// TestDemoCommand_LiveTreeClean asserts the real tracked tree carries no stale demo-command reference
// — the Go twin of test_collect_real_demo_docs_are_clean. Skipped outside a git checkout.
func TestDemoCommand_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateDemoCommandTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Fatalf("stale demo-command references on the tracked tree: %+v", findings)
	}
}

// TestDemoCommand_PythonParity is the verdict-level differential: the ported gate and the live Python
// checker must agree (clean vs. defect) over the SAME real workspace. Both default to the repo root and
// audit the same default sources. Skipped under -short or when python/git is absent.
func TestDemoCommand_PythonParity(t *testing.T) {
	if testing.Short() {
		t.Skip("python parity skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	clone := repoRoot(t)

	tree, err := ReadTrackedTree(clone)
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateDemoCommandTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	goBad := len(findings) > 0

	args := append(append([]string{}, pyArgs...), "tools/demo_command_audit.py")
	cmd := exec.Command(py, args...)
	cmd.Dir = clone
	out, _ := cmd.CombinedOutput()
	pyBad := cmd.ProcessState.ExitCode() == 1

	if goBad != pyBad {
		t.Fatalf("VERDICT MISMATCH: go bad=%v (%d findings) vs python bad=%v\npython said: %s\ngo findings: %+v",
			goBad, len(findings), pyBad, out, findings)
	}
}

func demoAnyContains(xs []string, sub string) bool {
	for _, x := range xs {
		if strings.Contains(x, sub) {
			return true
		}
	}
	return false
}
