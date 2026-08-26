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
	CgoFiles     []string `json:",omitempty"`
	CFiles       []string `json:",omitempty"`
	CXXFiles     []string `json:",omitempty"`
	MFiles       []string `json:",omitempty"`
	HFiles       []string `json:",omitempty"`
	FFiles       []string `json:",omitempty"`
	SFiles       []string `json:",omitempty"`
	SwigFiles    []string `json:",omitempty"`
	SwigCXXFiles []string `json:",omitempty"`
	SysoFiles    []string `json:",omitempty"`
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
			GoFiles:      []string{"foo.go"},
			CgoFiles:     []string{"cgo.go"},
			CFiles:       []string{"bridge.c"},
			CXXFiles:     []string{"bridge.cc"},
			MFiles:       []string{"bridge.m"},
			HFiles:       []string{"bridge.h"},
			FFiles:       []string{"bridge.f"},
			SFiles:       []string{"bridge.s"},
			SwigFiles:    []string{"bridge.swig"},
			SwigCXXFiles: []string{"bridge.swigcxx"},
			SysoFiles:    []string{"bridge.syso"},
			Imports:      []string{"fmt"}}, // stdlib import must be filtered out of edges
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
		"main.go":                     "example.com/m",
		"internal/foo/foo.go":         "example.com/m/internal/foo",
		"internal/foo/cgo.go":         "example.com/m/internal/foo",
		"internal/foo/bridge.c":       "example.com/m/internal/foo",
		"internal/foo/bridge.cc":      "example.com/m/internal/foo",
		"internal/foo/bridge.m":       "example.com/m/internal/foo",
		"internal/foo/bridge.h":       "example.com/m/internal/foo",
		"internal/foo/bridge.f":       "example.com/m/internal/foo",
		"internal/foo/bridge.s":       "example.com/m/internal/foo",
		"internal/foo/bridge.swig":    "example.com/m/internal/foo",
		"internal/foo/bridge.swigcxx": "example.com/m/internal/foo",
		"internal/foo/bridge.syso":    "example.com/m/internal/foo",
		"internal/bar/bar.go":         "example.com/m/internal/bar",
		"internal/bar/bar_test.go":    "example.com/m/internal/bar",
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

// TestAffectedRerunFailFlaky pins the --rerun-fail inner-loop flake gate: a red package
// that passes on a same-tree rerun is FLAKY_PASSED_ON_RETRY (exit 0, recorded), a real
// red that stays red keeps the failing exit, a rerun that is unavailable exonerates
// nothing (fail-closed), and the classifier composes with --blame on the survivors.
func TestAffectedRerunFailFlaky(t *testing.T) {
	origCF, origLG, origRT, origBR, origRR := affectedChangedFiles, affectedListGraph, affectedRunGoTest, affectedBaselineRed, affectedRerunFailed
	defer func() {
		affectedChangedFiles, affectedListGraph, affectedRunGoTest, affectedBaselineRed, affectedRerunFailed = origCF, origLG, origRT, origBR, origRR
	}()

	affectedChangedFiles = func(root, base string) ([]string, error) {
		return []string{"a/a.go", "b/b.go"}, nil
	}
	affectedListGraph = func(root string) (map[string]string, map[string][]string, int, error) {
		return map[string]string{"a/a.go": "m/a", "b/b.go": "m/b"}, map[string][]string{}, 2, nil
	}
	affectedRunGoTest = func(root string, args []string, stdout, stderr io.Writer) (int, error) {
		fmt.Fprintln(stdout, "FAIL\tm/a\t0.10s")
		fmt.Fprintln(stdout, "FAIL\tm/b\t0.10s")
		return 1, nil
	}

	// Scenario 1 — both failing packages pass on rerun: all-flaky, exit 0, and the report
	// records the rerun budget and the flaky set for a peer to audit.
	var rerunAsked []string
	var rerunFlags []string
	var rerunAttempts int
	affectedRerunFailed = func(root string, flags, pkgs []string, attempts int) (map[string]bool, error) {
		rerunAsked = append([]string(nil), pkgs...)
		rerunFlags = append([]string(nil), flags...)
		rerunAttempts = attempts
		return map[string]bool{"m/a": true, "m/b": true}, nil
	}
	report := filepath.Join(t.TempDir(), "report.json")
	var out, errb bytes.Buffer
	if code := runAffected(&out, &errb, []string{"--rerun-fail", "2", "--report", report}); code != 0 {
		t.Fatalf("all-flaky run exit = %d, want 0\nstderr=%s", code, errb.String())
	}
	if s := errb.String(); !strings.Contains(s, affectedtests.FlakyPassedOnRetry) || !strings.Contains(s, "passed on rerun") {
		t.Fatalf("flaky narration missing from stderr:\n%s", s)
	}
	if !reflect.DeepEqual(rerunAsked, []string{"m/a", "m/b"}) || rerunAttempts != 2 || len(rerunFlags) != 0 {
		t.Fatalf("rerun asked pkgs=%v attempts=%d flags=%v, want [m/a m/b] 2 []", rerunAsked, rerunAttempts, rerunFlags)
	}
	raw, err := os.ReadFile(report)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	var rep affectedRunReport
	if err := json.Unmarshal(raw, &rep); err != nil {
		t.Fatalf("report JSON: %v\n%s", err, raw)
	}
	if rep.Verdict != affectedtests.FlakyPassedOnRetry || rep.RerunAttempts != 2 || !reflect.DeepEqual(rep.FlakyPackages, []string{"m/a", "m/b"}) {
		t.Fatalf("report = verdict %q attempts %d flaky %v, want FLAKY_PASSED_ON_RETRY/2/[m/a m/b]", rep.Verdict, rep.RerunAttempts, rep.FlakyPackages)
	}

	// Scenario 2 — m/a flakes green, m/b stays red: a real regression keeps the red exit,
	// but m/a is still named flaky so the agent does not chase it.
	affectedRerunFailed = func(root string, flags, pkgs []string, attempts int) (map[string]bool, error) {
		return map[string]bool{"m/a": true}, nil
	}
	out.Reset()
	errb.Reset()
	if code := runAffected(&out, &errb, []string{"--rerun-fail", "3"}); code != 1 {
		t.Fatalf("partial-flaky run exit = %d, want 1 (m/b genuinely red)\nstderr=%s", code, errb.String())
	}
	if s := errb.String(); !strings.Contains(s, "m/a — "+affectedtests.FlakyPassedOnRetry) {
		t.Fatalf("m/a flaky line missing from stderr:\n%s", s)
	}

	// Scenario 3 — the rerun itself is unavailable (harness error): exonerate nothing, the
	// red exit stands, fail-closed.
	affectedRerunFailed = func(root string, flags, pkgs []string, attempts int) (map[string]bool, error) {
		return nil, fmt.Errorf("worktree busy")
	}
	out.Reset()
	errb.Reset()
	if code := runAffected(&out, &errb, []string{"--rerun-fail", "2"}); code != 1 {
		t.Fatalf("rerun-unavailable run exit = %d, want 1 (fail-closed)\nstderr=%s", code, errb.String())
	}
	if s := errb.String(); !strings.Contains(s, "rerun unavailable") || !strings.Contains(s, "fail-closed") {
		t.Fatalf("fail-closed narration missing from stderr:\n%s", s)
	}

	// Scenario 4 — composition with --blame: m/a flakes green, m/b stays red and is
	// peer-preexisting at baseline, so the survivor is attributed to a peer -> exit 0.
	affectedRerunFailed = func(root string, flags, pkgs []string, attempts int) (map[string]bool, error) {
		return map[string]bool{"m/a": true}, nil
	}
	var baselineAsked []string
	affectedBaselineRed = func(root, ref string, pkgs []string) (map[string]bool, map[string]bool, error) {
		baselineAsked = append([]string(nil), pkgs...)
		return map[string]bool{"m/b": true}, map[string]bool{"m/b": true}, nil
	}
	out.Reset()
	errb.Reset()
	if code := runAffected(&out, &errb, []string{"--rerun-fail", "2", "--blame"}); code != 0 {
		t.Fatalf("flaky+peer run exit = %d, want 0 (m/a flaky, m/b peer-preexisting)\nstderr=%s", code, errb.String())
	}
	if !reflect.DeepEqual(baselineAsked, []string{"m/b"}) {
		t.Fatalf("baseline asked for %v, want only the rerun survivor [m/b]", baselineAsked)
	}
	if s := errb.String(); !strings.Contains(s, "m/a — "+affectedtests.FlakyPassedOnRetry) || !strings.Contains(s, "peer-preexisting") || !strings.Contains(s, "attributed to peers") {
		t.Fatalf("flaky+blame narration missing from stderr:\n%s", s)
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
