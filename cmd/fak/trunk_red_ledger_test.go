package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// pinTrunkRedClock freezes the witness clock so rows are deterministic.
func pinTrunkRedClock(t *testing.T) {
	t.Helper()
	prev := trunkRedNow
	fixed := time.Date(2026, 7, 11, 12, 0, 0, 0, time.UTC)
	trunkRedNow = func() time.Time { return fixed }
	t.Cleanup(func() { trunkRedNow = prev })
}

// isolateTrunkRedEnv points the ledger at a temp file and clears mode/session so a test
// starts from a known state. Returns the ledger path.
func isolateTrunkRedEnv(t *testing.T) string {
	t.Helper()
	ledger := filepath.Join(t.TempDir(), "trunk-red.jsonl")
	t.Setenv(trunkRedLedgerEnv, ledger)
	t.Setenv(trunkRedModeEnv, "")
	t.Setenv("FAK_SESSION_ID", "")
	t.Setenv("CLAUDE_SESSION_ID", "")
	return ledger
}

func TestTrunkRedClassIsOrderIndependentAndBaseScoped(t *testing.T) {
	a := trunkRedClass("abc123", []string{"pkg/b", "pkg/a"})
	b := trunkRedClass("abc123", []string{"pkg/a", "pkg/b"})
	if a != b {
		t.Fatalf("class must be independent of package order:\n a=%q\n b=%q", a, b)
	}
	if same := trunkRedClass("def456", []string{"pkg/a", "pkg/b"}); same == a {
		t.Fatalf("a different base must be a different class; got %q for both", same)
	}
	if got := trunkRedClass("", []string{"pkg/a"}); !strings.HasPrefix(got, "unknown ") {
		t.Fatalf("empty base must fall back to unknown; got %q", got)
	}
}

func TestEmitTrunkRedWitnessFailOpen(t *testing.T) {
	pinTrunkRedClock(t)

	t.Run("mode off writes nothing", func(t *testing.T) {
		ledger := isolateTrunkRedEnv(t)
		t.Setenv(trunkRedModeEnv, "off")
		var stderr bytes.Buffer
		w := emitTrunkRedWitness(&stderr, "commit", "base1", []string{"pkg/a"}, "Foo")
		if w.Witnessed {
			t.Fatalf("mode=off must not witness")
		}
		if _, err := os.Stat(ledger); !os.IsNotExist(err) {
			t.Fatalf("mode=off must not create the ledger, stat err=%v", err)
		}
	})

	t.Run("no packages is a no-op", func(t *testing.T) {
		isolateTrunkRedEnv(t)
		var stderr bytes.Buffer
		w := emitTrunkRedWitness(&stderr, "commit", "base1", nil, "")
		if w.Witnessed {
			t.Fatalf("a witness with no nameable break must not record")
		}
	})

	t.Run("writes a row and reports occurrence 1", func(t *testing.T) {
		ledger := isolateTrunkRedEnv(t)
		t.Setenv("FAK_SESSION_ID", "sess-1")
		var stderr bytes.Buffer
		w := emitTrunkRedWitness(&stderr, "commit", "base1", []string{"pkg/a", "pkg/b"}, "Foo")
		if !w.Witnessed {
			t.Fatalf("expected a witness; stderr=%q", stderr.String())
		}
		if w.Occurrences != 1 || w.Sessions != 1 {
			t.Fatalf("first witness should be occ=1 sess=1; got occ=%d sess=%d", w.Occurrences, w.Sessions)
		}
		b, err := os.ReadFile(ledger)
		if err != nil {
			t.Fatalf("read ledger: %v", err)
		}
		line := strings.TrimSpace(string(b))
		if strings.Count(line, "\n") != 0 {
			t.Fatalf("expected exactly one row, got:\n%s", line)
		}
		for _, want := range []string{`"schema":"fak.trunk-red.v1"`, `"gate":"commit"`, `"base_sha":"base1"`, `"pkg/a"`, `"session":"sess-1"`, `"2026-07-11T12:00:00Z"`} {
			if !strings.Contains(line, want) {
				t.Fatalf("row missing %q:\n%s", want, line)
			}
		}
	})
}

func TestEmitTrunkRedWitnessConvergesAcrossSessions(t *testing.T) {
	pinTrunkRedClock(t)
	isolateTrunkRedEnv(t)
	var stderr bytes.Buffer

	// Same clone (same session) re-hits the same break: occurrences climb, sessions stay 1.
	t.Setenv("FAK_SESSION_ID", "sess-1")
	_ = emitTrunkRedWitness(&stderr, "commit", "base1", []string{"pkg/a"}, "Foo")
	w2 := emitTrunkRedWitness(&stderr, "pre-push", "base1", []string{"pkg/a"}, "Foo")
	if w2.Occurrences != 2 || w2.Sessions != 1 {
		t.Fatalf("same session re-hit: want occ=2 sess=1; got occ=%d sess=%d", w2.Occurrences, w2.Sessions)
	}

	// A different clone hits the SAME class: sessions climbs — the convergence signal.
	t.Setenv("FAK_SESSION_ID", "sess-2")
	w3 := emitTrunkRedWitness(&stderr, "commit", "base1", []string{"pkg/a"}, "Foo")
	if w3.Occurrences != 3 || w3.Sessions != 2 {
		t.Fatalf("second clone same break: want occ=3 sess=2; got occ=%d sess=%d", w3.Occurrences, w3.Sessions)
	}

	// A genuinely different break (different package) is its own class.
	w4 := emitTrunkRedWitness(&stderr, "commit", "base1", []string{"pkg/other"}, "Bar")
	if w4.Occurrences != 1 {
		t.Fatalf("distinct break must be its own class; got occ=%d", w4.Occurrences)
	}
}

func TestTrunkRedWitnessNoteHonesty(t *testing.T) {
	if note := trunkRedWitnessNote(trunkRedWitness{Witnessed: false}); note != "" {
		t.Fatalf("a note must never claim a witness that did not happen; got %q", note)
	}
	multi := trunkRedWitnessNote(trunkRedWitness{Witnessed: true, Occurrences: 4, Sessions: 3})
	if !strings.Contains(multi, "4 clone(s) across 3 session(s)") {
		t.Fatalf("multi-session note should surface the spread; got %q", multi)
	}
	if !strings.Contains(multi, "fak trunk-red") {
		t.Fatalf("note should point at the fold command; got %q", multi)
	}
	solo := trunkRedWitnessNote(trunkRedWitness{Witnessed: true, Occurrences: 1, Sessions: 1})
	if !strings.Contains(solo, "recorded once for the fleet") {
		t.Fatalf("solo note should read as a single record; got %q", solo)
	}
}

func TestSummarizeTrunkRedFoldsAndOrders(t *testing.T) {
	pinTrunkRedClock(t)
	isolateTrunkRedEnv(t)
	var stderr bytes.Buffer

	// Class X: two clones stuck (2 sessions). Class Y: one clone.
	t.Setenv("FAK_SESSION_ID", "sess-1")
	_ = emitTrunkRedWitness(&stderr, "commit", "baseX", []string{"pkg/x"}, "Xsym")
	t.Setenv("FAK_SESSION_ID", "sess-2")
	_ = emitTrunkRedWitness(&stderr, "pre-push", "baseX", []string{"pkg/x"}, "Xsym")
	t.Setenv("FAK_SESSION_ID", "sess-3")
	_ = emitTrunkRedWitness(&stderr, "commit", "baseY", []string{"pkg/y"}, "Ysym")

	content, err := os.ReadFile(os.Getenv(trunkRedLedgerEnv))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	sum := summarizeTrunkRed(string(content))
	if sum.Total != 3 {
		t.Fatalf("want 3 total rows, got %d", sum.Total)
	}
	if len(sum.Classes) != 2 {
		t.Fatalf("want 2 distinct classes, got %d", len(sum.Classes))
	}
	// Worst (most sessions stuck) first: class X (2 sessions) before Y (1).
	top := sum.Classes[0]
	if len(top.Packages) != 1 || top.Packages[0] != "pkg/x" {
		t.Fatalf("worst class should be pkg/x; got %+v", top.Packages)
	}
	if top.Sessions != 2 || top.Rows != 2 {
		t.Fatalf("class x should have 2 sessions / 2 rows; got sess=%d rows=%d", top.Sessions, top.Rows)
	}
	if strings.Join(top.Gates, ",") != "commit,pre-push" {
		t.Fatalf("class x should record both gates; got %v", top.Gates)
	}
}

func TestRunTrunkRedReaderPaths(t *testing.T) {
	t.Run("missing ledger is an empty view, exit 0", func(t *testing.T) {
		ledger := filepath.Join(t.TempDir(), "absent.jsonl")
		var stdout, stderr bytes.Buffer
		if code := runTrunkRed(&stdout, &stderr, []string{"--ledger", ledger}); code != 0 {
			t.Fatalf("missing ledger should exit 0; got %d stderr=%q", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "no pre-existing trunk-red admissions recorded") {
			t.Fatalf("expected empty-view text; got %q", stdout.String())
		}
	})

	t.Run("folds a populated ledger", func(t *testing.T) {
		pinTrunkRedClock(t)
		ledger := isolateTrunkRedEnv(t)
		var stderr bytes.Buffer
		t.Setenv("FAK_SESSION_ID", "sess-1")
		_ = emitTrunkRedWitness(&stderr, "commit", "baseX", []string{"pkg/x"}, "Xsym")

		var stdout, rerr bytes.Buffer
		if code := runTrunkRed(&stdout, &rerr, []string{"--ledger", ledger}); code != 0 {
			t.Fatalf("exit 0 expected; got %d stderr=%q", code, rerr.String())
		}
		out := stdout.String()
		for _, want := range []string{"pkg/x", "clone(s)", "first break: undefined: Xsym"} {
			if !strings.Contains(out, want) {
				t.Fatalf("render missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("json mode emits the summary", func(t *testing.T) {
		pinTrunkRedClock(t)
		ledger := isolateTrunkRedEnv(t)
		var stderr bytes.Buffer
		_ = emitTrunkRedWitness(&stderr, "commit", "baseX", []string{"pkg/x"}, "Xsym")

		var stdout, rerr bytes.Buffer
		if code := runTrunkRed(&stdout, &rerr, []string{"--ledger", ledger, "--json"}); code != 0 {
			t.Fatalf("exit 0 expected; got %d", code)
		}
		if !strings.Contains(stdout.String(), `"classes"`) {
			t.Fatalf("json mode should emit the summary object; got %q", stdout.String())
		}
	})
}
