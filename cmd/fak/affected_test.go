package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
)

// goListObj mirrors the JSON shape `go list -json` emits for the fields parseGoList
// reads. Marshalling real structs (rather than hand-writing JSON) keeps the fixture's
// Windows backslash paths correctly escaped.
type goListObj struct {
	ImportPath   string
	Dir          string
	Module       *struct{ Path, Dir string }
	GoFiles      []string `json:",omitempty"`
	TestGoFiles  []string `json:",omitempty"`
	EmbedFiles   []string `json:",omitempty"`
	Imports      []string `json:",omitempty"`
	TestImports  []string `json:",omitempty"`
	XTestImports []string `json:",omitempty"`
}

func TestParseGoListAndSelectEndToEnd(t *testing.T) {
	modDir := filepath.FromSlash("/work/m")
	mod := &struct{ Path, Dir string }{Path: "example.com/m", Dir: modDir}
	objs := []goListObj{
		{ImportPath: "example.com/m", Dir: modDir, Module: mod,
			GoFiles: []string{"main.go"},
			Imports: []string{"example.com/m/internal/foo"}},
		{ImportPath: "example.com/m/internal/foo", Dir: filepath.Join(modDir, "internal", "foo"), Module: mod,
			GoFiles: []string{"foo.go"},
			Imports: []string{"fmt"}}, // stdlib import must be filtered out of edges
		{ImportPath: "example.com/m/internal/bar", Dir: filepath.Join(modDir, "internal", "bar"), Module: mod,
			GoFiles:     []string{"bar.go"},
			TestGoFiles: []string{"bar_test.go"},
			TestImports: []string{"example.com/m/internal/foo"}}, // bar's TEST imports foo
	}
	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	for _, o := range objs {
		if err := enc.Encode(o); err != nil {
			t.Fatal(err)
		}
	}

	fileToPkg, edges, total, err := parseGoList(strings.NewReader(sb.String()))
	if err != nil {
		t.Fatal(err)
	}
	if total != 3 {
		t.Fatalf("total = %d, want 3", total)
	}
	wantFiles := map[string]string{
		"main.go":                  "example.com/m",
		"internal/foo/foo.go":      "example.com/m/internal/foo",
		"internal/bar/bar.go":      "example.com/m/internal/bar",
		"internal/bar/bar_test.go": "example.com/m/internal/bar",
	}
	if !reflect.DeepEqual(fileToPkg, wantFiles) {
		t.Fatalf("fileToPkg = %v, want %v", fileToPkg, wantFiles)
	}
	// foo's only import was stdlib (fmt), so it has no intra-module edge.
	if _, ok := edges["example.com/m/internal/foo"]; ok {
		t.Errorf("foo should have no intra-module edges, got %v", edges["example.com/m/internal/foo"])
	}
	// bar's TEST import of foo must be recorded as an edge (the test-import correctness case).
	if got := edges["example.com/m/internal/bar"]; !reflect.DeepEqual(got, []string{"example.com/m/internal/foo"}) {
		t.Errorf("bar edges = %v, want [foo]", got)
	}

	// End to end: change a file in foo -> foo + everything importing it (root m imports
	// foo; bar's test imports foo).
	changed := affectedtests.ChangedPackages(fileToPkg, []string{"internal/foo/foo.go"})
	if !reflect.DeepEqual(changed, []string{"example.com/m/internal/foo"}) {
		t.Fatalf("changed = %v, want [foo]", changed)
	}
	selected := affectedtests.Select(edges, changed)
	want := []string{"example.com/m", "example.com/m/internal/bar", "example.com/m/internal/foo"}
	if !reflect.DeepEqual(selected, want) {
		t.Fatalf("selected = %v, want %v", selected, want)
	}

	// A top-level non-source change (Makefile / go.mod) selects nothing -- the root-package
	// over-selection guard. main.go (a real root source file) selects just the root.
	if got := affectedtests.ChangedPackages(fileToPkg, []string{"Makefile", "go.mod"}); len(got) != 0 {
		t.Fatalf("non-source change selected %v, want empty", got)
	}
	if got := affectedtests.ChangedPackages(fileToPkg, []string{"main.go"}); !reflect.DeepEqual(got, []string{"example.com/m"}) {
		t.Fatalf("root source change selected %v, want [example.com/m]", got)
	}

	// A docs-only change selects nothing end to end.
	docChanged := affectedtests.ChangedPackages(fileToPkg, []string{"docs/x.md"})
	if len(docChanged) != 0 {
		t.Fatalf("docs-only change selected %v, want empty", docChanged)
	}
	if got := affectedtests.Select(edges, docChanged); len(got) != 0 {
		t.Fatalf("docs-only selection = %v, want empty", got)
	}
}

// TestAffectedBlameAttribution drives the #2138 rung through runAffected with every
// impure seam injected: a red run's FAIL lines are parsed, the failing packages are
// attributed against the clean-baseline rerun and the --mine closure, the report
// carries the blame rows, and the exit code reflects ONLY 'mine' reds — green for the
// caller's diff when every red is a peer's.
func TestAffectedBlameAttribution(t *testing.T) {
	origCF, origLG, origRT, origBR := affectedChangedFiles, affectedListGraph, affectedRunGoTest, affectedBaselineRed
	defer func() {
		affectedChangedFiles, affectedListGraph, affectedRunGoTest, affectedBaselineRed = origCF, origLG, origRT, origBR
	}()

	affectedChangedFiles = func(root, base string) ([]string, error) {
		return []string{"a/a.go", "b/b.go", "c/c.go"}, nil
	}
	affectedListGraph = func(root string) (map[string]string, map[string][]string, int, error) {
		return map[string]string{"a/a.go": "m/a", "b/b.go": "m/b", "c/c.go": "m/c"},
			map[string][]string{}, 3, nil
	}
	affectedRunGoTest = func(root string, args []string, stdout, stderr io.Writer) (int, error) {
		fmt.Fprintln(stdout, "FAIL\tm/a\t0.10s")
		fmt.Fprintln(stdout, "ok  \tm/c\t0.01s")
		fmt.Fprintln(stdout, "FAIL\tm/b\t0.10s")
		return 1, nil
	}
	var baselineAsked []string
	affectedBaselineRed = func(root, ref string, pkgs []string) (map[string]bool, map[string]bool, error) {
		if ref != "HEAD" {
			t.Errorf("baseline ref = %q, want HEAD (no --base given)", ref)
		}
		baselineAsked = append([]string(nil), pkgs...)
		// m/a was red before any working-tree change; both produced a baseline verdict.
		return map[string]bool{"m/a": true}, map[string]bool{"m/a": true, "m/b": true}, nil
	}

	// Every red is a peer's: m/a is red at clean HEAD (peer-preexisting), m/b is outside
	// the closure of the caller's declared c/c.go (peer-wip) -> exit 0, PEER_RED_ONLY.
	report := filepath.Join(t.TempDir(), "report.json")
	var out, errb bytes.Buffer
	code := runAffected(&out, &errb, []string{"--mine", "c/c.go", "--report", report})
	if code != 0 {
		t.Fatalf("exonerated run exit = %d, want 0\nstderr=%s", code, errb.String())
	}
	if s := errb.String(); !strings.Contains(s, "peer-preexisting") || !strings.Contains(s, "peer-wip") || !strings.Contains(s, "green for your diff") {
		t.Fatalf("blame narration missing from stderr:\n%s", s)
	}
	if !reflect.DeepEqual(baselineAsked, []string{"m/a", "m/b"}) {
		t.Fatalf("baseline asked for %v, want the failing packages [m/a m/b]", baselineAsked)
	}
	raw, err := os.ReadFile(report)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	var rep affectedRunReport
	if err := json.Unmarshal(raw, &rep); err != nil {
		t.Fatalf("report JSON: %v\n%s", err, raw)
	}
	if rep.Verdict != "PEER_RED_ONLY" || rep.BaselineRef != "HEAD" || len(rep.Blame) != 2 {
		t.Fatalf("report = verdict %q baseline %q blame %+v, want PEER_RED_ONLY/HEAD/2 rows", rep.Verdict, rep.BaselineRef, rep.Blame)
	}
	classes := map[string]string{}
	for _, b := range rep.Blame {
		classes[b.Package] = b.Class
	}
	if classes["m/a"] != affectedtests.BlamePeerPreexisting || classes["m/b"] != affectedtests.BlamePeerWIP {
		t.Fatalf("blame classes = %v, want m/a peer-preexisting, m/b peer-wip", classes)
	}

	// The caller's own red keeps the failing exit: declaring b/b.go puts m/b in the mine
	// closure, and m/b is green at baseline -> mine -> exit stays 1.
	out.Reset()
	errb.Reset()
	if code := runAffected(&out, &errb, []string{"--blame", "--mine", "b/b.go"}); code != 1 {
		t.Fatalf("mine-red run exit = %d, want 1\nstderr=%s", code, errb.String())
	}
	if s := errb.String(); !strings.Contains(s, "m/b — mine") {
		t.Fatalf("mine attribution missing from stderr:\n%s", s)
	}

	// A --mine file that is not among the changed files (a typo) must void the closure
	// rung entirely — otherwise a mistyped path would shrink the closure and exonerate
	// every red as peer-wip. m/b stays mine (baseline-green) -> exit 1, with the warn.
	out.Reset()
	errb.Reset()
	if code := runAffected(&out, &errb, []string{"--blame", "--mine", "typo/nope.go"}); code != 1 {
		t.Fatalf("typo'd --mine run exit = %d, want 1 (closure rung voided)\nstderr=%s", code, errb.String())
	}
	if s := errb.String(); !strings.Contains(s, "not among the changed files") || !strings.Contains(s, "m/b — mine") {
		t.Fatalf("typo'd --mine narration missing from stderr:\n%s", s)
	}

	// Baseline unavailable: nothing is exonerated by that rung (fail-closed), but the
	// closure rung still works; with a red inside the closure the exit stays 1.
	affectedBaselineRed = func(root, ref string, pkgs []string) (map[string]bool, map[string]bool, error) {
		return nil, nil, fmt.Errorf("git worktree unavailable")
	}
	out.Reset()
	errb.Reset()
	if code := runAffected(&out, &errb, []string{"--blame", "--mine", "a/a.go"}); code != 1 {
		t.Fatalf("baseline-unavailable run exit = %d, want 1 (fail-closed)\nstderr=%s", code, errb.String())
	}
	if s := errb.String(); !strings.Contains(s, "fail-closed") {
		t.Fatalf("fail-closed narration missing from stderr:\n%s", s)
	}
}

// TestBaselineHarnessFailure pins the exoneration guard: output carrying a
// binary-could-not-run marker names the marker, clean test output does not match.
func TestBaselineHarnessFailure(t *testing.T) {
	blocked := "--- FAIL: TestX\nfork/exec C:\\tmp\\pkg.test.exe: Access is denied.\nFAIL\tm/a\t0.00s\n"
	if m := baselineHarnessFailure(blocked); m == "" {
		t.Fatalf("blocked-binary output not detected as a harness failure")
	}
	clean := "--- FAIL: TestX (0.00s)\n    x_test.go:10: want 1, got 2\nFAIL\tm/a\t0.42s\nok  \tm/b\t0.01s\n"
	if m := baselineHarnessFailure(clean); m != "" {
		t.Fatalf("clean red output misread as harness failure (marker %q)", m)
	}
}

// TestAffectedTestCommandRouting pins the host routing: on Windows the fast inner-loop
// gate must route `go test` through test.ps1 -> WSL (native go test is OS-policy-blocked
// there), the SAME bridge `fak test` uses; on every other host it runs go test directly.
func TestAffectedTestCommandRouting(t *testing.T) {
	args := []string{"test", "-short", "github.com/x/y/internal/foo"}

	// Non-Windows: a direct `go test ...`, unchanged.
	name, cmdArgs := affectedTestCommand("linux", args)
	if name != "go" || !reflect.DeepEqual(cmdArgs, args) {
		t.Fatalf("linux routing = %s %v, want go %v", name, cmdArgs, args)
	}

	// Windows: powershell -> test.ps1, with the leading "test" verb dropped (test.ps1
	// re-adds it before `go`), so the OS-blocked native go test is never exec'd.
	name, cmdArgs = affectedTestCommand("windows", args)
	want := []string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "test.ps1", "-short", "github.com/x/y/internal/foo"}
	if name != "powershell" || !reflect.DeepEqual(cmdArgs, want) {
		t.Fatalf("windows routing = %s %v, want powershell %v", name, cmdArgs, want)
	}
	// The forwarded args must NOT contain the native "test" verb test.ps1 re-adds.
	for _, a := range cmdArgs {
		if a == "test" {
			t.Errorf("windows routing must not pass the 'test' verb to test.ps1; got %v", cmdArgs)
		}
	}
}
