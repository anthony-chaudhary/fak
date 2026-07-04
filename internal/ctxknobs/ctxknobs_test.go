package ctxknobs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

const fixtureRoot = "testdata/repo"

// TestScanFixture pins the walker against a known fixture tree: it finds exactly
// the two context flag/env knobs and the one user-required context skill, and
// ignores the non-context flag, env, and skill.
func TestScanFixture(t *testing.T) {
	inv, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if inv.UserRequired != 1 {
		t.Errorf("UserRequired = %d, want 1", inv.UserRequired)
	}
	if inv.OperatorDebug != 2 {
		t.Errorf("OperatorDebug = %d, want 2", inv.OperatorDebug)
	}
	if len(inv.Knobs) != 3 {
		t.Fatalf("len(Knobs) = %d, want 3: %+v", len(inv.Knobs), inv.Knobs)
	}

	got := map[string]Knob{}
	for _, k := range inv.Knobs {
		got[k.Key()] = k
	}
	for _, want := range []struct {
		key   string
		class Class
	}{
		{"flag:ctx-view-budget", OperatorDebug},
		{"env:FAK_CONTEXT_TOKENS", OperatorDebug},
		{"skill:ctx-overlay", UserRequired},
	} {
		k, ok := got[want.key]
		if !ok {
			t.Errorf("missing expected knob %q", want.key)
			continue
		}
		if k.Class != want.class {
			t.Errorf("%q class = %q, want %q", want.key, k.Class, want.class)
		}
		if k.File == "" || k.Line == 0 {
			t.Errorf("%q missing file:line provenance: %+v", want.key, k)
		}
	}
	// The over-match guards: none of these may be classified as a knob.
	for _, bad := range []string{"flag:verbose", "env:HOME", "skill:quality-report"} {
		if _, ok := got[bad]; ok {
			t.Errorf("walker over-matched a non-context knob: %q", bad)
		}
	}
}

// TestScanDeterministic is the "run the verb twice → identical output" witness
// at the walker level: two scans of the same tree marshal to byte-identical JSON.
func TestScanDeterministic(t *testing.T) {
	a, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan a: %v", err)
	}
	b, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan b: %v", err)
	}
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Errorf("scan not deterministic:\n a=%s\n b=%s", ja, jb)
	}
}

// TestRatchetCore verifies the verifier on synthetic inventories: a user-required
// knob absent from the baseline is an offense; the same knob baselined is clean;
// an operator-debug knob is never an offense.
func TestRatchetCore(t *testing.T) {
	inv := Inventory{Knobs: []Knob{
		{Kind: KindSkill, Name: "x", Class: UserRequired},
		{Kind: KindFlag, Name: "ctx-budget", Class: OperatorDebug},
	}}

	if off := RatchetOffenses(inv, nil); len(off) != 1 || off[0].Key() != "skill:x" {
		t.Fatalf("empty baseline: want 1 offense skill:x, got %+v", off)
	}
	if off := RatchetOffenses(inv, []string{"skill:x"}); len(off) != 0 {
		t.Errorf("baselined overlay should be clean, got %+v", off)
	}
	// An inventory of only operator-debug knobs is never an offense, whatever the baseline.
	opOnly := Inventory{Knobs: []Knob{{Kind: KindFlag, Name: "ctx-budget", Class: OperatorDebug}}}
	if off := RatchetOffenses(opOnly, nil); len(off) != 0 {
		t.Errorf("operator-debug knob must never be an offense, got %+v", off)
	}
}

// TestFixtureUserRequiredRedsAgainstEmptyBaseline is the #2199 witness end to
// end: the fixture carries a dummy user-required overlay, so scanning it and
// ratcheting against an EMPTY baseline reds (naming the overlay); baselining its
// key makes it green. This is "add a dummy user-required flag to the walker's
// test fixture → ratchet test reds".
func TestFixtureUserRequiredRedsAgainstEmptyBaseline(t *testing.T) {
	inv, err := Scan(fixtureRoot)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	off := RatchetOffenses(inv, nil)
	if len(off) != 1 || off[0].Key() != "skill:ctx-overlay" {
		t.Fatalf("ratchet did not red on the fixture overlay; got %+v", off)
	}
	if off2 := RatchetOffenses(inv, []string{"skill:ctx-overlay"}); len(off2) != 0 {
		t.Errorf("baselined fixture overlay should be green, got %+v", off2)
	}
}

// TestNoNewUserRequiredKnobs is the LIVE trunk guard (`make ci`): scanning the
// real repo against the frozen baseline must yield ZERO offenses. The day a new
// user-required context overlay lands without a baseline update, this reds the
// trunk with its file:line.
func TestNoNewUserRequiredKnobs(t *testing.T) {
	root := repoRoot(t)
	inv, err := Scan(root)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	off := RatchetOffenses(inv, BaselineUserRequired)
	if len(off) > 0 {
		t.Errorf("%d NEW user-required context knob(s) not in the baseline (%s):", len(off), ReasonNewUserRequiredKnob)
		for _, k := range off {
			t.Errorf("  %s  (%s:%d) — %s", k.Key(), k.File, k.Line, k.Evidence)
		}
		t.Errorf("fix: retire the overlay behind an automatic default, OR (if it is genuinely " +
			"required) add its key to internal/ctxknobs.BaselineUserRequired in the SAME commit.")
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
