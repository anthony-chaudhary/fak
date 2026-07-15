package safecommit

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// writeChecker writes content at the repo-relative slash path rel under root, creating parent
// directories. It returns the same rel so call sites read as declarations.
func writeChecker(t *testing.T, root, rel, content string) string {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir for %s: %v", rel, err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
	return rel
}

func mustPin(t *testing.T, root string, paths ...string) CheckerBaseline {
	t.Helper()
	base, err := PinCheckers(root, paths)
	if err != nil {
		t.Fatalf("PinCheckers(%v): %v", paths, err)
	}
	return base
}

// TestGuardCheckerPin_UnchangedPassesThrough is the "unchanged checker passes through untouched"
// half of the DoD: a checker whose bytes are identical at grade-time is not drift, even if the file
// was rewritten with the same content (fingerprint is content-addressed, not mtime).
func TestGuardCheckerPin_UnchangedPassesThrough(t *testing.T) {
	root := t.TempDir()
	rel := writeChecker(t, root, "checker_test.go", "package p\nfunc want() int { return 42 }\n")
	base := mustPin(t, root, rel)

	// Rewrite with byte-identical content: a touch, not a tamper.
	writeChecker(t, root, rel, "package p\nfunc want() int { return 42 }\n")

	if reason, refused := GuardCheckerPin(root, base); refused {
		t.Fatalf("unchanged checker refused with %q; want pass-through", reason)
	}
	if d := VerifyCheckers(root, base); d.Tampered {
		t.Fatalf("unchanged checker reported drift: %+v", d)
	}
}

// TestGuardCheckerPin_ChangedBytesRefused is the load-bearing half of the DoD: a declared checker
// whose bytes changed between declare-time and grade-time is refused with CHECKER_TAMPERED, and the
// exact path is named under Changed.
func TestGuardCheckerPin_ChangedBytesRefused(t *testing.T) {
	root := t.TempDir()
	rel := writeChecker(t, root, "checker_test.go", "package p\nfunc want() int { return 42 }\n")
	base := mustPin(t, root, rel)

	// The classic attack: flip the checker so a wrong answer grades green.
	writeChecker(t, root, rel, "package p\nfunc want() int { return 0 }\n")

	reason, refused := GuardCheckerPin(root, base)
	if !refused {
		t.Fatal("mutated checker was not refused")
	}
	if reason != ReasonCheckerTampered {
		t.Fatalf("reason = %q; want %q", reason, ReasonCheckerTampered)
	}
	d := VerifyCheckers(root, base)
	if !d.Tampered || d.Reason != ReasonCheckerTampered {
		t.Fatalf("verdict = %+v; want Tampered with %q", d, ReasonCheckerTampered)
	}
	if !reflect.DeepEqual(d.Changed, []string{"checker_test.go"}) {
		t.Fatalf("Changed = %v; want [checker_test.go]", d.Changed)
	}
	if len(d.Missing) != 0 || len(d.Appeared) != 0 {
		t.Fatalf("expected only a Changed drift, got %+v", d)
	}
}

// TestVerifyCheckers_DeletedIsMissing: a checker present at pin time but gone at grade-time is drift,
// classified as Missing (deleting the grader is as untrustworthy as mutating it).
func TestVerifyCheckers_DeletedIsMissing(t *testing.T) {
	root := t.TempDir()
	rel := writeChecker(t, root, "checker_test.go", "assert x == 1\n")
	base := mustPin(t, root, rel)

	if err := os.Remove(filepath.Join(root, filepath.FromSlash(rel))); err != nil {
		t.Fatalf("remove checker: %v", err)
	}

	d := VerifyCheckers(root, base)
	if !d.Tampered || !reflect.DeepEqual(d.Missing, []string{"checker_test.go"}) {
		t.Fatalf("verdict = %+v; want Missing=[checker_test.go], Tampered", d)
	}
	if len(d.Changed) != 0 || len(d.Appeared) != 0 {
		t.Fatalf("expected only a Missing drift, got %+v", d)
	}
}

// TestVerifyCheckers_AppearedIsInjected: a checker path that did not exist at pin time (pinned
// absent) but is present at grade-time is drift, classified as Appeared — an injected grader.
func TestVerifyCheckers_AppearedIsInjected(t *testing.T) {
	root := t.TempDir()
	// Declared but not yet written: PinCheckers records it absent.
	base := mustPin(t, root, "checker_test.go")
	if got := base["checker_test.go"]; got != checkerAbsent {
		t.Fatalf("pinned fingerprint = %q; want absent sentinel", got)
	}

	writeChecker(t, root, "checker_test.go", "assert always_pass\n")

	d := VerifyCheckers(root, base)
	if !d.Tampered || !reflect.DeepEqual(d.Appeared, []string{"checker_test.go"}) {
		t.Fatalf("verdict = %+v; want Appeared=[checker_test.go], Tampered", d)
	}
}

// TestVerifyCheckers_AbsentAtBothIsClean: a declared checker absent at both pin and grade is not
// drift — there was no grader to tamper with, so the run passes through.
func TestVerifyCheckers_AbsentAtBothIsClean(t *testing.T) {
	root := t.TempDir()
	base := mustPin(t, root, "never_written_test.go")
	if d := VerifyCheckers(root, base); d.Tampered {
		t.Fatalf("absent-at-both reported drift: %+v", d)
	}
}

// TestVerifyCheckers_EmptyBaselinePasses: nil and empty baselines pin nothing, so nothing can drift.
func TestVerifyCheckers_EmptyBaselinePasses(t *testing.T) {
	root := t.TempDir()
	for name, base := range map[string]CheckerBaseline{"nil": nil, "empty": {}} {
		if d := VerifyCheckers(root, base); d.Tampered {
			t.Fatalf("%s baseline reported drift: %+v", name, d)
		}
		if reason, refused := GuardCheckerPin(root, base); refused {
			t.Fatalf("%s baseline refused with %q", name, reason)
		}
	}
}

// TestVerifyCheckers_MultiCheckerDeterministic pins several checkers, drifts a subset in two ways,
// and asserts the verdict is stable and its slices sorted — the property a close arm relies on to
// render a reproducible refusal.
func TestVerifyCheckers_MultiCheckerDeterministic(t *testing.T) {
	root := t.TempDir()
	a := writeChecker(t, root, "a_test.go", "a=1\n")
	b := writeChecker(t, root, "sub/b_test.go", "b=2\n")
	c := writeChecker(t, root, "c_test.go", "c=3\n")
	base := mustPin(t, root, c, a, b) // pin order intentionally unsorted

	writeChecker(t, root, b, "b=999\n")                                 // change
	if err := os.Remove(filepath.Join(root, "a_test.go")); err != nil { // delete
		t.Fatalf("remove a: %v", err)
	}
	// c untouched.

	d1 := VerifyCheckers(root, base)
	d2 := VerifyCheckers(root, base)
	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("non-deterministic verdict:\n d1=%+v\n d2=%+v", d1, d2)
	}
	if !reflect.DeepEqual(d1.Changed, []string{"sub/b_test.go"}) {
		t.Fatalf("Changed = %v; want [sub/b_test.go] (slash-normalized)", d1.Changed)
	}
	if !reflect.DeepEqual(d1.Missing, []string{"a_test.go"}) {
		t.Fatalf("Missing = %v; want [a_test.go]", d1.Missing)
	}
	if !d1.Tampered {
		t.Fatal("want Tampered")
	}
}

// TestPinCheckers_SlashNormalizedKeys: a nested path declared with forward slashes is keyed with
// forward slashes regardless of host OS, so a baseline round-trips across platforms.
func TestPinCheckers_SlashNormalizedKeys(t *testing.T) {
	root := t.TempDir()
	rel := writeChecker(t, root, "internal/grader/rules_test.go", "rule\n")
	base := mustPin(t, root, rel)
	if _, ok := base["internal/grader/rules_test.go"]; !ok {
		t.Fatalf("baseline key not slash-normalized: keys=%v", keysOf(base))
	}
	// And it verifies clean against the same tree.
	if d := VerifyCheckers(root, base); d.Tampered {
		t.Fatalf("freshly pinned nested checker reported drift: %+v", d)
	}
}

// TestPinCheckers_UnreadableErrors: a declared path that exists but cannot be read as a file (here a
// directory) is a bad declaration, surfaced as an error at pin time rather than mis-graded later.
func TestPinCheckers_UnreadableErrors(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "adir"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if _, err := PinCheckers(root, []string{"adir"}); err == nil {
		t.Fatal("PinCheckers of a directory did not error")
	}
}

// TestVerifyCheckers_UnreadableAtGradeIsDrift is the fail-closed property: a checker readable at pin
// time but unreadable at grade-time (replaced by a directory) is drift, never a silent pass.
func TestVerifyCheckers_UnreadableAtGradeIsDrift(t *testing.T) {
	root := t.TempDir()
	rel := writeChecker(t, root, "checker_test.go", "ok\n")
	base := mustPin(t, root, rel)

	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.Remove(p); err != nil {
		t.Fatalf("remove file: %v", err)
	}
	if err := os.MkdirAll(p, 0o755); err != nil {
		t.Fatalf("mkdir over checker: %v", err)
	}

	d := VerifyCheckers(root, base)
	if !d.Tampered || !reflect.DeepEqual(d.Changed, []string{"checker_test.go"}) {
		t.Fatalf("verdict = %+v; want Changed=[checker_test.go], Tampered (fail-closed)", d)
	}
}

// TestCheckerTamperedDeclaredInDosToml binds the code reason to its dos.toml registration: the whole
// point of ReasonCheckerTampered is to be a first-class structured refusal, which requires a
// [reasons.CHECKER_TAMPERED] table declaring refusal = true so `dos_check_reason` resolves it and
// `dos_refuse_reasons` lists it. This fails loudly if the const is shipped without the declaration
// (the UNCLASSIFIED-drift trap the closed vocabulary exists to prevent).
func TestCheckerTamperedDeclaredInDosToml(t *testing.T) {
	content := readRepoDosTomlForChecker(t)
	header := "[reasons." + ReasonCheckerTampered + "]"
	i := strings.Index(content, header)
	if i < 0 {
		t.Fatalf("%s not declared in dos.toml — dos_check_reason %s would return known=false", header, ReasonCheckerTampered)
	}
	// Scope to this reason's block: header up to the next top-level [section] or EOF.
	block := content[i:]
	if j := strings.Index(content[i+len(header):], "\n["); j >= 0 {
		block = content[i : i+len(header)+j]
	}
	if !dosReasonFieldTrue(block, "refusal") {
		t.Errorf("%s is registered but not marked refusal = true", header)
	}
}

// dosReasonFieldTrue reports whether block has a `field = true` line, tolerant of the aligned
// whitespace the dos.toml [reasons] tables use (e.g. "refusal  = true").
func dosReasonFieldTrue(block, field string) bool {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, field))
		if strings.HasPrefix(rest, "=") && strings.TrimSpace(rest[1:]) == "true" {
			return true
		}
	}
	return false
}

// readRepoDosTomlForChecker reads the repo-root dos.toml relative to this test's own source path
// (internal/safecommit → ../..), so the lookup is independent of the working directory.
func readRepoDosTomlForChecker(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}

func keysOf(m CheckerBaseline) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}

func TestCommitWithRefusesTamperedDeclaredCheckerBeforeGit(t *testing.T) {
	root := t.TempDir()
	checker := writeChecker(t, root, "checker_test.go", "original\n")
	base := mustPin(t, root, checker)
	writeChecker(t, root, checker, "weakened\n")

	g := &fakeGit{reply: onTrunkBase()}
	opts := baseOpts()
	opts.Dir = root
	opts.CheckerBaseline = base
	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("CommitWith error: %v", err)
	}
	if res.Reason != ReasonCheckerTampered {
		t.Fatalf("reason=%q, want %q", res.Reason, ReasonCheckerTampered)
	}
	if len(g.calls) != 0 {
		t.Fatalf("checker drift must refuse before git effects; calls=%v", g.calls)
	}
}

func TestCommitWithUnchangedDeclaredCheckerPassesThrough(t *testing.T) {
	root := t.TempDir()
	checker := writeChecker(t, root, "checker_test.go", "original\n")
	g := &fakeGit{reply: onTrunkBase()}
	opts := baseOpts()
	opts.Dir = root
	opts.CheckerBaseline = mustPin(t, root, checker)
	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("CommitWith error: %v", err)
	}
	if res.Reason == ReasonCheckerTampered {
		t.Fatalf("unchanged checker refused: %+v", res)
	}
	if !g.sawSubcommand("commit") {
		t.Fatalf("unchanged checker did not reach commit; calls=%v result=%+v", g.calls, res)
	}
}
