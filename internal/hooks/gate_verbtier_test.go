package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// TestVerbTierGate_LiveTreeClean: on the real tracked tree the pre-push gate must find ZERO
// untiered verbs — the same end-to-end witness devindex.TestVerbTierCoverageIsTotal provides,
// one boundary earlier. If this reds, the trunk's fast CI subset is about to red too (exactly
// the failure mode this gate exists to prevent).
func TestVerbTierGate_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateVerbTierTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Fatalf("untiered dispatch verb(s) on the tracked tree (CI is about to red): %+v", findings)
	}
}

// TestVerbTierGate_AgreesWithRatchet is the anti-rival-authority witness: the pre-push gate and
// the CI ratchet it fronts must compute the SAME untiered-verb set on the live tree. The set is
// derived here exactly as devindex.TestVerbTierCoverageIsTotal derives it — devindex.DispatchVerbs
// over the real main.go, then devindex.TierOf on each token — and must equal the verbs the gate
// flags. If they diverge, one side has drifted from the shared parser/classifier.
func TestVerbTierGate_AgreesWithRatchet(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := repoRoot(t)
	tree, err := ReadTrackedTree(root)
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	body, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skipf("read main.go: %v", err)
	}
	// The ratchet's own computation (mirrors TestVerbTierCoverageIsTotal).
	ratchetMissing := map[string]bool{}
	for _, tok := range devindex.DispatchVerbs(body) {
		if _, ok := devindex.TierOf(tok); !ok {
			ratchetMissing[tok] = true
		}
	}
	// The gate's computation.
	findings, gerr := gateVerbTierTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != len(ratchetMissing) {
		t.Fatalf("gate/ratchet disagree on the live tree: gate=%d untiered, ratchet=%d untiered",
			len(findings), len(ratchetMissing))
	}
	for _, f := range findings {
		if f.Gate != reasonVerbUntiered {
			t.Fatalf("finding has wrong reason class %q, want %q", f.Gate, reasonVerbUntiered)
		}
		var named bool
		for tok := range ratchetMissing {
			if strings.Contains(f.Detail, "verb "+tok+" ") {
				named = true
				break
			}
		}
		if !named {
			t.Fatalf("gate flagged a verb the ratchet did not: %q", f.Detail)
		}
	}
}

// TestVerbTierGate_FiresOnNewVerb: a synthetic main.go dispatching a verb absent from the tier
// table must produce exactly one VERB_UNTIERED finding naming it; a switch of only-classified
// verbs is clean. This is the half of the witness that proves the gate bites.
func TestVerbTierGate_FiresOnNewVerb(t *testing.T) {
	mainGo := func(includeNew bool) string {
		var b strings.Builder
		b.WriteString("package main\n\nfunc dispatch() {\n\tswitch os.Args[1] {\n")
		b.WriteString("\tcase \"guard\":\n\t\trunGuard()\n") // frontdoor — classified
		b.WriteString("\tcase \"sweep\":\n\t\trunSweep()\n") // dev — classified
		if includeNew {
			// An unclassified verb with a brace-bearing body, to also exercise that the
			// shared parser does not truncate at the first nested `}`.
			b.WriteString("\tcase \"totally-new-verb\":\n\t\tif err := run(); err != nil {\n\t\t\treturn\n\t\t}\n")
		}
		b.WriteString("\tdefault:\n\t\tusage()\n\t}\n}\n")
		return b.String()
	}

	build := func(includeNew bool) *TrackedTree {
		root := t.TempDir()
		p := filepath.Join(root, filepath.FromSlash(mainGoFile))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(mainGo(includeNew)), 0o644); err != nil {
			t.Fatal(err)
		}
		return &TrackedTree{Root: root, Paths: []string{mainGoFile}, fileCache: map[string]fileEntry{}}
	}

	// New unclassified verb -> exactly one finding naming it.
	findings, err := gateVerbTierTree(build(true))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want 1 VERB_UNTIERED finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].Gate != reasonVerbUntiered || findings[0].File != mainGoFile ||
		!strings.Contains(findings[0].Detail, "totally-new-verb") {
		t.Fatalf("finding wrong: %+v", findings[0])
	}

	// Only-classified verbs -> clean.
	findings, err = gateVerbTierTree(build(false))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("classified-only switch should be clean, got %+v", findings)
	}
}

// TestVerbTierGate_FailsOpen: with no main.go on the tree, and with a main.go that holds no
// dispatch switch (empty parse), the gate returns ErrCouldNotRun (fail open) rather than
// flagging anything — the devindex TEST stays the backstop, and the gate never emits a false
// VERB_UNTIERED against an unreadable or shape-changed source.
func TestVerbTierGate_FailsOpen(t *testing.T) {
	// Missing main.go.
	tree := &TrackedTree{Root: t.TempDir(), Paths: []string{"cmd/fak/other.go"}, fileCache: map[string]fileEntry{}}
	if _, err := gateVerbTierTree(tree); err != ErrCouldNotRun {
		t.Fatalf("want ErrCouldNotRun on a missing main.go, got %v", err)
	}

	// Present but no dispatch switch -> zero tokens parsed -> fail open.
	root := t.TempDir()
	p := filepath.Join(root, filepath.FromSlash(mainGoFile))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tree = &TrackedTree{Root: root, Paths: []string{mainGoFile}, fileCache: map[string]fileEntry{}}
	if _, err := gateVerbTierTree(tree); err != ErrCouldNotRun {
		t.Fatalf("want ErrCouldNotRun on a switch-less main.go, got %v", err)
	}
}
