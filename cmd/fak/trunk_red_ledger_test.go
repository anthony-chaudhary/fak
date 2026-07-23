package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
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
	sum := summarizeTrunkRed(string(content), nil)
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

// trunkRedTestLedger marshals synthetic records into JSONL content, so the
// resolve-filter tests never depend on the live ledger or the writer path.
func trunkRedTestLedger(t *testing.T, recs ...trunkRedRecord) string {
	t.Helper()
	var b strings.Builder
	for _, rec := range recs {
		rec.Schema = trunkRedRecordSchema
		line, err := json.Marshal(rec)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		b.Write(line)
		b.WriteString("\n")
	}
	return b.String()
}

func TestSummarizeTrunkRedResolveFilter(t *testing.T) {
	// Synthetic ledger: a class whose base resolves (2 rows), a class that does
	// not (1 row), and a class with NO base sha (never provably resolved).
	content := trunkRedTestLedger(t,
		trunkRedRecord{Gate: "commit", BaseSha: "deadbase", Packages: []string{"pkg/dead"}, FirstBreak: "Gone", Session: "sess-1"},
		trunkRedRecord{Gate: "pre-push", BaseSha: "deadbase", Packages: []string{"pkg/dead"}, FirstBreak: "Gone", Session: "sess-2"},
		trunkRedRecord{Gate: "commit", BaseSha: "livebase", Packages: []string{"pkg/live"}, FirstBreak: "Live", Session: "sess-3"},
		trunkRedRecord{Gate: "commit", BaseSha: "", Packages: []string{"pkg/unknown"}, FirstBreak: "NoBase", Session: "sess-4"},
	)

	classPkgs := func(sum trunkRedSummary) []string {
		var got []string
		for _, c := range sum.Classes {
			got = append(got, strings.Join(c.Packages, " "))
		}
		sort.Strings(got)
		return got
	}

	cases := []struct {
		name                string
		resolved            func(baseSha string) bool
		wantPkgs            []string
		wantTotal           int
		wantResolvedClasses int
		wantResolvedRows    int
	}{
		{
			name:                "resolved base folds out, unresolved surfaces",
			resolved:            func(base string) bool { return base == "deadbase" },
			wantPkgs:            []string{"pkg/live", "pkg/unknown"},
			wantTotal:           2,
			wantResolvedClasses: 1,
			wantResolvedRows:    2,
		},
		{
			name:     "nil resolver keeps everything",
			resolved: nil,
			wantPkgs: []string{"pkg/dead", "pkg/live", "pkg/unknown"},
			// All 4 rows stay live.
			wantTotal: 4,
		},
		{
			// The production resolver maps EVERY git error / missing origin/main
			// to false — the erroring/unknown case must KEEP the class.
			name:      "erroring resolver (always unknown) keeps every class",
			resolved:  func(string) bool { return false },
			wantPkgs:  []string{"pkg/dead", "pkg/live", "pkg/unknown"},
			wantTotal: 4,
		},
		{
			// KEEP-SIDE invariant: an empty base can never be PROVABLY resolved,
			// even against a resolver that claims everything is.
			name:                "empty base survives an always-true resolver",
			resolved:            func(string) bool { return true },
			wantPkgs:            []string{"pkg/unknown"},
			wantTotal:           1,
			wantResolvedClasses: 2,
			wantResolvedRows:    3,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sum := summarizeTrunkRed(content, tc.resolved)
			if got := classPkgs(sum); strings.Join(got, "|") != strings.Join(tc.wantPkgs, "|") {
				t.Fatalf("surfaced classes = %v, want %v", got, tc.wantPkgs)
			}
			if sum.Total != tc.wantTotal {
				t.Fatalf("Total (live rows) = %d, want %d", sum.Total, tc.wantTotal)
			}
			if sum.ResolvedClasses != tc.wantResolvedClasses {
				t.Fatalf("ResolvedClasses = %d, want %d", sum.ResolvedClasses, tc.wantResolvedClasses)
			}
			if sum.ResolvedRows != tc.wantResolvedRows {
				t.Fatalf("ResolvedRows = %d, want %d", sum.ResolvedRows, tc.wantResolvedRows)
			}
		})
	}
}

func TestRenderTrunkRedResolvedNote(t *testing.T) {
	content := trunkRedTestLedger(t,
		trunkRedRecord{Gate: "commit", BaseSha: "deadbase", Packages: []string{"pkg/dead"}, Session: "sess-1"},
		trunkRedRecord{Gate: "commit", BaseSha: "livebase", Packages: []string{"pkg/live"}, Session: "sess-2"},
	)

	t.Run("live view notes the folded-out classes", func(t *testing.T) {
		sum := summarizeTrunkRed(content, func(base string) bool { return base == "deadbase" })
		out := renderTrunkRed(sum)
		if !strings.Contains(out, "pkg/live") || strings.Contains(out, "pkg/dead") {
			t.Fatalf("live view must show pkg/live and fold pkg/dead out:\n%s", out)
		}
		if !strings.Contains(out, "1 resolved class(es) across 1 row(s) folded out") {
			t.Fatalf("live view should note the folded-out resolved classes:\n%s", out)
		}
	})

	t.Run("all-resolved view is not the empty view", func(t *testing.T) {
		sum := summarizeTrunkRed(content, func(string) bool { return true })
		out := renderTrunkRed(sum)
		if strings.Contains(out, "no pre-existing trunk-red admissions recorded") {
			t.Fatalf("an all-resolved ledger is not an empty ledger:\n%s", out)
		}
		if !strings.Contains(out, "no LIVE shared breaks") || !strings.Contains(out, "2 resolved class(es) across 2 witness row(s)") {
			t.Fatalf("all-resolved view should say every class folded out:\n%s", out)
		}
	})
}

func TestTrunkRedGitResolverKeepSide(t *testing.T) {
	// No repo root / empty base: never provably resolved, no git shelled.
	if trunkRedGitResolver("")("abc123") {
		t.Fatalf("an empty root must never prove a base resolved")
	}
	if trunkRedGitResolver(t.TempDir())("") {
		t.Fatalf("an empty base must never prove resolved")
	}
	// A non-repo dir makes every git call error: fail-open KEEP (false).
	if trunkRedGitResolver(t.TempDir())("abc123") {
		t.Fatalf("git errors must report NOT resolved (keep-side)")
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
