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

// TestTrunkRedWitnessFilesInTheGatedRepo pins the write-side scoping: a gate run
// against ANOTHER checkout records that checkout's break in ITS ledger, never in
// the repo the process happens to sit in. Before this, the build-check gate's own
// tests — which drive it against a throwaway repo — filed their synthetic
// `buildcheck.test/p undefined: neverDefined` break into the developer's real
// ledger, where the temp repo's base sha resolves against nothing and the row can
// never fold out. `fak trunk-red` reported 28 fleet-wide "shared breaks" that were
// all that one fixture, burying the single real break it exists to surface.
func TestTrunkRedWitnessFilesInTheGatedRepo(t *testing.T) {
	pinTrunkRedClock(t)
	// No env override: the gated root alone must decide the destination.
	t.Setenv(trunkRedLedgerEnv, "")
	t.Setenv(trunkRedModeEnv, "")
	t.Setenv("FAK_SESSION_ID", "")
	t.Setenv("CLAUDE_SESSION_ID", "")

	gated := t.TempDir()
	var stderr bytes.Buffer
	w := emitTrunkRedWitness(&stderr, gated, "commit", "base1", []string{"buildcheck.test/p"}, "neverDefined")
	if !w.Witnessed {
		t.Fatalf("expected a witness in the gated repo; stderr=%q", stderr.String())
	}
	want := filepath.Join(gated, ".fak", "trunk-red.jsonl")
	if w.Ledger != want {
		t.Fatalf("witness filed at %q, want the GATED repo's ledger %q", w.Ledger, want)
	}
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("gated repo's ledger not written: %v", err)
	}
	// The repo the process is in must be untouched.
	if here := trunkRedLedgerDefault(); here != "" && here == w.Ledger {
		t.Fatalf("witness landed in the process's own repo ledger %q", here)
	}

	// A blank gated root is unrecordable, not a silent fallback to the cwd's repo.
	w2 := emitTrunkRedWitness(&stderr, "", "commit", "base1", []string{"pkg/a"}, "Foo")
	if w2.Witnessed {
		t.Fatalf("a witness with no gated root must not record; landed at %q", w2.Ledger)
	}
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
		w := emitTrunkRedWitness(&stderr, "", "commit", "base1", []string{"pkg/a"}, "Foo")
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
		w := emitTrunkRedWitness(&stderr, "", "commit", "base1", nil, "")
		if w.Witnessed {
			t.Fatalf("a witness with no nameable break must not record")
		}
	})

	t.Run("writes a row and reports occurrence 1", func(t *testing.T) {
		ledger := isolateTrunkRedEnv(t)
		t.Setenv("FAK_SESSION_ID", "sess-1")
		var stderr bytes.Buffer
		w := emitTrunkRedWitness(&stderr, "", "commit", "base1", []string{"pkg/a", "pkg/b"}, "Foo")
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
	_ = emitTrunkRedWitness(&stderr, "", "commit", "base1", []string{"pkg/a"}, "Foo")
	w2 := emitTrunkRedWitness(&stderr, "", "pre-push", "base1", []string{"pkg/a"}, "Foo")
	if w2.Occurrences != 2 || w2.Sessions != 1 {
		t.Fatalf("same session re-hit: want occ=2 sess=1; got occ=%d sess=%d", w2.Occurrences, w2.Sessions)
	}

	// A different clone hits the SAME class: sessions climbs — the convergence signal.
	t.Setenv("FAK_SESSION_ID", "sess-2")
	w3 := emitTrunkRedWitness(&stderr, "", "commit", "base1", []string{"pkg/a"}, "Foo")
	if w3.Occurrences != 3 || w3.Sessions != 2 {
		t.Fatalf("second clone same break: want occ=3 sess=2; got occ=%d sess=%d", w3.Occurrences, w3.Sessions)
	}

	// A genuinely different break (different package) is its own class.
	w4 := emitTrunkRedWitness(&stderr, "", "commit", "base1", []string{"pkg/other"}, "Bar")
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
	_ = emitTrunkRedWitness(&stderr, "", "commit", "baseX", []string{"pkg/x"}, "Xsym")
	t.Setenv("FAK_SESSION_ID", "sess-2")
	_ = emitTrunkRedWitness(&stderr, "", "pre-push", "baseX", []string{"pkg/x"}, "Xsym")
	t.Setenv("FAK_SESSION_ID", "sess-3")
	_ = emitTrunkRedWitness(&stderr, "", "commit", "baseY", []string{"pkg/y"}, "Ysym")

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
		resolved            func(trunkRedBreak) bool
		wantPkgs            []string
		wantTotal           int
		wantResolvedClasses int
		wantResolvedRows    int
	}{
		{
			name:                "resolved base folds out, unresolved surfaces",
			resolved:            func(b trunkRedBreak) bool { return b.BaseSha == "deadbase" },
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
			resolved:  func(trunkRedBreak) bool { return false },
			wantPkgs:  []string{"pkg/dead", "pkg/live", "pkg/unknown"},
			wantTotal: 4,
		},
		{
			// KEEP-SIDE invariant: an empty base can never be PROVABLY resolved,
			// even against a resolver that claims everything is.
			name:                "empty base survives an always-true resolver",
			resolved:            func(trunkRedBreak) bool { return true },
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
	// Both rows carry a first_break: a class with no symbol to look up can never be
	// PROVEN resolved, so the fold would keep it structurally (see
	// TestSummarizeTrunkRedKeepsUnprovableClasses) and the render note would be empty.
	content := trunkRedTestLedger(t,
		trunkRedRecord{Gate: "commit", BaseSha: "deadbase", Packages: []string{"pkg/dead"}, FirstBreak: "Gone", Session: "sess-1"},
		trunkRedRecord{Gate: "commit", BaseSha: "livebase", Packages: []string{"pkg/live"}, FirstBreak: "Live", Session: "sess-2"},
	)

	t.Run("live view notes the folded-out classes", func(t *testing.T) {
		sum := summarizeTrunkRed(content, func(b trunkRedBreak) bool { return b.BaseSha == "deadbase" })
		out := renderTrunkRed(sum)
		if !strings.Contains(out, "pkg/live") || strings.Contains(out, "pkg/dead") {
			t.Fatalf("live view must show pkg/live and fold pkg/dead out:\n%s", out)
		}
		if !strings.Contains(out, "1 resolved class(es) across 1 row(s) folded out") {
			t.Fatalf("live view should note the folded-out resolved classes:\n%s", out)
		}
	})

	t.Run("all-resolved view is not the empty view", func(t *testing.T) {
		sum := summarizeTrunkRed(content, func(trunkRedBreak) bool { return true })
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
	brk := trunkRedBreak{
		BaseSha:    "abc123",
		FirstBreak: "someSymbol",
		Packages:   []string{"github.com/anthony-chaudhary/fak/cmd/fak"},
	}
	// No repo root / empty base / empty symbol: never provably resolved, no git shelled.
	if trunkRedGitResolver("")(brk) {
		t.Fatalf("an empty root must never prove a break resolved")
	}
	noBase := brk
	noBase.BaseSha = ""
	if trunkRedGitResolver(t.TempDir())(noBase) {
		t.Fatalf("an empty base must never prove resolved")
	}
	noSym := brk
	noSym.FirstBreak = ""
	if trunkRedGitResolver(t.TempDir())(noSym) {
		t.Fatalf("an empty first-break symbol must never prove resolved")
	}
	// A non-repo dir has no go.mod and makes every git call error: KEEP (false).
	if trunkRedGitResolver(t.TempDir())(brk) {
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
		_ = emitTrunkRedWitness(&stderr, "", "commit", "baseX", []string{"pkg/x"}, "Xsym")

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
		_ = emitTrunkRedWitness(&stderr, "", "commit", "baseX", []string{"pkg/x"}, "Xsym")

		var stdout, rerr bytes.Buffer
		if code := runTrunkRed(&stdout, &rerr, []string{"--ledger", ledger, "--json"}); code != 0 {
			t.Fatalf("exit 0 expected; got %d", code)
		}
		if !strings.Contains(stdout.String(), `"classes"`) {
			t.Fatalf("json mode should emit the summary object; got %q", stdout.String())
		}
	})
}

// TestSummarizeTrunkRedKeepsUnprovableClasses is the acceptance witness for the
// keep-side contract: over a SYNTHETIC ledger written to t.TempDir() (never the
// operator's real machine-local .fak/trunk-red.jsonl) mixing a LIVE class, a
// PROVABLY-RESOLVED class, and several UNPROVABLE ones, only the provably-resolved
// class may be folded out. Every unprovable class is asserted against an ALWAYS-TRUE
// resolver, so the test pins the fold's own structural guard rather than the
// correctness of whatever predicate is passed in — a resolve check that errors, or
// one that lies, must still surface the row.
func TestSummarizeTrunkRedKeepsUnprovableClasses(t *testing.T) {
	const modPkg = "github.com/anthony-chaudhary/fak/internal/example"

	// (b) provably resolved: has BOTH a base to date and a symbol to look up, and the
	// predicate proves it. (a) live: same shape, predicate says no. (c) the unprovable
	// family: each is missing exactly one thing the two conjuncts need.
	content := trunkRedTestLedger(t,
		trunkRedRecord{Gate: "commit", BaseSha: "resolvedbase", Packages: []string{modPkg}, FirstBreak: "wasFixed", Session: "s1"},
		trunkRedRecord{Gate: "pre-push", BaseSha: "resolvedbase", Packages: []string{modPkg}, FirstBreak: "wasFixed", Session: "s2"},
		trunkRedRecord{Gate: "commit", BaseSha: "livebase", Packages: []string{modPkg}, FirstBreak: "stillBroken", Session: "s3"},
		trunkRedRecord{Gate: "commit", BaseSha: "", Packages: []string{modPkg}, FirstBreak: "noBaseToDate", Session: "s4"},
		trunkRedRecord{Gate: "commit", BaseSha: "nosymbase", Packages: []string{modPkg}, FirstBreak: "", Session: "s5"},
	)

	ledger := filepath.Join(t.TempDir(), "trunk-red.jsonl")
	if err := os.WriteFile(ledger, []byte(content), 0o644); err != nil {
		t.Fatalf("write synthetic ledger: %v", err)
	}
	read, err := readTrunkRedLedger(ledger)
	if err != nil {
		t.Fatalf("read synthetic ledger: %v", err)
	}

	// "resolvedbase" is the ONLY thing this predicate is willing to prove. It is
	// deliberately also always-true for the no-base / no-symbol rows' bases, so those
	// classes survive only because the FOLD refuses to ask about them.
	resolveOnlyTheFixed := func(b trunkRedBreak) bool { return b.BaseSha != "livebase" }

	cases := []struct {
		name     string
		resolved func(trunkRedBreak) bool
		wantLive []string // first_break of every class that must SURFACE
	}{
		{
			name:     "only the provably-resolved class folds out",
			resolved: resolveOnlyTheFixed,
			wantLive: []string{"noBaseToDate", "stillBroken", ""},
		},
		{
			name:     "a resolver that errors on everything keeps every class",
			resolved: func(trunkRedBreak) bool { return false },
			wantLive: []string{"noBaseToDate", "stillBroken", "wasFixed", ""},
		},
		{
			name:     "a nil resolver keeps every class",
			resolved: nil,
			wantLive: []string{"noBaseToDate", "stillBroken", "wasFixed", ""},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sum := summarizeTrunkRed(read, tc.resolved)
			var live []string
			for _, c := range sum.Classes {
				live = append(live, c.FirstBreak)
			}
			sort.Strings(live)
			want := append([]string(nil), tc.wantLive...)
			sort.Strings(want)
			if strings.Join(live, "|") != strings.Join(want, "|") {
				t.Fatalf("surfaced first_breaks = %v, want %v", live, want)
			}
		})
	}

	t.Run("resolved rows are counted, never silently dropped", func(t *testing.T) {
		sum := summarizeTrunkRed(read, resolveOnlyTheFixed)
		if sum.ResolvedClasses != 1 || sum.ResolvedRows != 2 {
			t.Fatalf("want 1 resolved class / 2 resolved rows; got %d / %d", sum.ResolvedClasses, sum.ResolvedRows)
		}
		if sum.Total != 3 {
			t.Fatalf("want 3 LIVE rows remaining; got %d", sum.Total)
		}
	})

	// The keep-side guard has to hold against a predicate that claims EVERYTHING is
	// resolved. Only the two classes carrying both a base and a symbol may go.
	t.Run("an always-true resolver cannot drop an unprovable class", func(t *testing.T) {
		sum := summarizeTrunkRed(read, func(trunkRedBreak) bool { return true })
		var live []string
		for _, c := range sum.Classes {
			live = append(live, c.FirstBreak)
		}
		sort.Strings(live)
		if strings.Join(live, "|") != "|noBaseToDate" {
			t.Fatalf("no-base and no-symbol classes must survive an always-true resolver; got %v", live)
		}
	})
}

// TestTrunkRedPackageDirsKeepSide pins the import-path -> directory mapping that scopes
// the symbol search. Anything not provably inside THIS module yields no directories,
// which makes the class unresolvable and therefore kept.
func TestTrunkRedPackageDirsKeepSide(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/m\n\ngo 1.22\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	cases := []struct {
		name string
		pkgs []string
		want []string
	}{
		{"in-module packages map to dirs", []string{"example.com/m/internal/b", "example.com/m/cmd/a"}, []string{"cmd/a", "internal/b"}},
		{"duplicate packages fold", []string{"example.com/m/cmd/a", "example.com/m/cmd/a"}, []string{"cmd/a"}},
		{"a synthetic fixture path poisons the whole class", []string{"example.com/m/cmd/a", "buildcheck.test/p"}, nil},
		{"a stdlib path poisons the whole class", []string{"time"}, nil},
		{"the bare module path is not searchable", []string{"example.com/m"}, nil},
		{"no packages", nil, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := trunkRedPackageDirs(root, tc.pkgs)
			if strings.Join(got, "|") != strings.Join(tc.want, "|") {
				t.Fatalf("trunkRedPackageDirs(%v) = %v, want %v", tc.pkgs, got, tc.want)
			}
		})
	}
	t.Run("no go.mod means nothing is mappable", func(t *testing.T) {
		if dirs := trunkRedPackageDirs(t.TempDir(), []string{"example.com/m/cmd/a"}); dirs != nil {
			t.Fatalf("an unreadable go.mod must map nothing; got %v", dirs)
		}
	})
}

// TestTrunkRedPlainIdent pins which first-break strings are even searchable. A
// qualified symbol names another package's declaration, which the failing class's own
// directories cannot witness — unresolvable, so kept.
func TestTrunkRedPlainIdent(t *testing.T) {
	for _, sym := range []string{"rawCostUSD", "cmdAuditExport", "_x", "X9"} {
		if !trunkRedPlainIdent(sym) {
			t.Fatalf("%q is a bare Go identifier and must be searchable", sym)
		}
	}
	for _, sym := range []string{"", "metrics.AnchorRefusalMonitor", "9lives", "a b", "a|b", "a.*"} {
		if trunkRedPlainIdent(sym) {
			t.Fatalf("%q must NOT be treated as a searchable identifier", sym)
		}
	}
}

// TestTrunkRedSymbolDefinedAtHead witnesses resolve conjunct 2 against REAL git, using
// this repo as its own fixture: a symbol that is committed at HEAD in cmd/fak reads as
// defined, and one that exists nowhere reads as NOT defined (KEEP). It reads HEAD, not
// the working tree, which is what makes it immune to the peer WIP always present in
// this shared checkout.
func TestTrunkRedSymbolDefinedAtHead(t *testing.T) {
	root := repoRoot()
	if strings.TrimSpace(root) == "" {
		t.Skip("no repo root resolvable; conjunct 2 needs real git")
	}
	dirs := []string{"cmd/fak"}
	// summarizeTrunkRed is declared at column 0 in this very file and is committed at
	// HEAD, so it is a stable self-referential fixture.
	if !trunkRedSymbolDefinedAtHead(root, dirs, "summarizeTrunkRed") {
		t.Fatalf("summarizeTrunkRed is defined at HEAD in cmd/fak but read as undefined")
	}
	if trunkRedSymbolDefinedAtHead(root, dirs, "trunkRedSymbolThatIsDefinedNowhere5356") {
		t.Fatalf("a symbol defined nowhere must read as NOT defined (keep-side)")
	}
	// An indented match is not a package-level declaration: the pattern must anchor.
	if !strings.HasPrefix(trunkRedDefinitionPattern("x"), "^(func|type|var|const)") {
		t.Fatalf("the definition pattern must anchor at column 0")
	}
}

// TestTrunkRedGitResolverSecondConjunct is the witness that ancestry ALONE cannot drop a
// class. Both breaks below share one base sha that really is a strict ancestor of the
// remote trunk, so conjunct 1 is constant and true for both; the ONLY difference is
// whether the first-break symbol is defined at HEAD. Deleting conjunct 2 reds the second
// assertion, which is the whole point of this issue: a base becomes an ancestor the
// moment any unrelated peer commit lands, so ancestry alone would fold out a live break.
func TestTrunkRedGitResolverSecondConjunct(t *testing.T) {
	root := repoRoot()
	if strings.TrimSpace(root) == "" {
		t.Skip("no repo root resolvable; this witness needs real git")
	}
	// A commit the remote trunk has provably moved PAST.
	ancestor, err := gitOut(root, "rev-parse", "--verify", trunkRedTrunkRef(root)+"~1^{commit}")
	if err != nil {
		t.Skipf("no remote trunk ancestor available in this clone: %v", err)
	}
	base := strings.TrimSpace(ancestor)
	if !trunkRedBaseMergedPast(root, base) {
		t.Skipf("%s is not a strict ancestor of the remote trunk here", base)
	}

	resolve := trunkRedGitResolver(root)
	pkgs := []string{"github.com/anthony-chaudhary/fak/cmd/fak"}

	fixed := trunkRedBreak{BaseSha: base, FirstBreak: "summarizeTrunkRed", Packages: pkgs}
	if !resolve(fixed) {
		t.Fatalf("both conjuncts hold (ancestor base + symbol defined at HEAD): must resolve")
	}
	stillBroken := trunkRedBreak{BaseSha: base, FirstBreak: "trunkRedSymbolThatIsDefinedNowhere5356", Packages: pkgs}
	if resolve(stillBroken) {
		t.Fatalf("ancestry alone must NOT resolve a break whose symbol is still undefined at HEAD")
	}
	// A synthetic package path leaves nothing to search: unresolvable, so kept.
	foreign := trunkRedBreak{BaseSha: base, FirstBreak: "summarizeTrunkRed", Packages: []string{"buildcheck.test/p"}}
	if resolve(foreign) {
		t.Fatalf("a break in an out-of-module package must never be provably resolved")
	}
}
