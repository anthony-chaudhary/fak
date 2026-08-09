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

// trunkRedProbeResolving is a synthetic probe that proves EXACTLY the named bases fixed
// and says nothing about any other class — the shape a correct production probe has.
func trunkRedProbeResolving(bases ...string) func(trunkRedBreak) trunkRedVerdict {
	set := map[string]struct{}{}
	for _, b := range bases {
		set[b] = struct{}{}
	}
	return func(b trunkRedBreak) trunkRedVerdict {
		if _, ok := set[b.BaseSha]; ok {
			return trunkRedVerdict{Status: trunkRedStatusResolved}
		}
		return trunkRedUnprovable("not in the fixture's resolved set")
	}
}

// trunkRedProbeAlwaysResolved is the ADVERSARY: a probe that claims every break it is
// shown is fixed. The fold's structural guard, not the probe, is what must keep the
// unprovable classes surfaced against it.
func trunkRedProbeAlwaysResolved() func(trunkRedBreak) trunkRedVerdict {
	return func(trunkRedBreak) trunkRedVerdict { return trunkRedVerdict{Status: trunkRedStatusResolved} }
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
		probe               func(trunkRedBreak) trunkRedVerdict
		wantPkgs            []string
		wantTotal           int
		wantResolvedClasses int
		wantResolvedRows    int
	}{
		{
			name:                "resolved base folds out, unresolved surfaces",
			probe:               trunkRedProbeResolving("deadbase"),
			wantPkgs:            []string{"pkg/live", "pkg/unknown"},
			wantTotal:           2,
			wantResolvedClasses: 1,
			wantResolvedRows:    2,
		},
		{
			name:     "nil probe keeps everything",
			probe:    nil,
			wantPkgs: []string{"pkg/dead", "pkg/live", "pkg/unknown"},
			// All 4 rows stay surfaced.
			wantTotal: 4,
		},
		{
			// The production probe maps EVERY git error / missing origin/main to
			// unprovable — the erroring/unknown case must KEEP the class.
			name:      "erroring probe (always unprovable) keeps every class",
			probe:     func(trunkRedBreak) trunkRedVerdict { return trunkRedUnprovable("git said nothing") },
			wantPkgs:  []string{"pkg/dead", "pkg/live", "pkg/unknown"},
			wantTotal: 4,
		},
		{
			// KEEP-SIDE invariant: an empty base can never be PROVABLY resolved,
			// even against a probe that claims everything is.
			name:                "empty base survives an always-resolved probe",
			probe:               trunkRedProbeAlwaysResolved(),
			wantPkgs:            []string{"pkg/unknown"},
			wantTotal:           1,
			wantResolvedClasses: 2,
			wantResolvedRows:    3,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sum := summarizeTrunkRed(content, tc.probe)
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
		sum := summarizeTrunkRed(content, trunkRedProbeResolving("deadbase"))
		out := renderTrunkRed(sum)
		if !strings.Contains(out, "pkg/live") || strings.Contains(out, "pkg/dead") {
			t.Fatalf("live view must show pkg/live and fold pkg/dead out:\n%s", out)
		}
		if !strings.Contains(out, "1 resolved class(es) across 1 row(s) folded out") {
			t.Fatalf("live view should note the folded-out resolved classes:\n%s", out)
		}
	})

	t.Run("all-resolved view is not the empty view", func(t *testing.T) {
		sum := summarizeTrunkRed(content, trunkRedProbeAlwaysResolved())
		out := renderTrunkRed(sum)
		if strings.Contains(out, "no pre-existing trunk-red admissions recorded") {
			t.Fatalf("an all-resolved ledger is not an empty ledger:\n%s", out)
		}
		if !strings.Contains(out, "no LIVE shared breaks") || !strings.Contains(out, "2 resolved class(es) across 2 witness row(s)") {
			t.Fatalf("all-resolved view should say every class folded out:\n%s", out)
		}
	})
}

// TestTrunkRedGitProbeKeepSide pins that the production probe proves NOTHING — neither a
// drop nor a liveness claim — from any input it cannot check, and that it always names
// which witness was missing.
func TestTrunkRedGitProbeKeepSide(t *testing.T) {
	brk := trunkRedBreak{
		BaseSha:    "abc123",
		FirstBreak: "someSymbol",
		Packages:   []string{"github.com/anthony-chaudhary/fak/cmd/fak"},
	}
	noBase, noSym, qualified := brk, brk, brk
	noBase.BaseSha = ""
	noSym.FirstBreak = ""
	qualified.FirstBreak = "metrics.AnchorRefusalMonitor"

	cases := []struct {
		name       string
		root       string
		brk        trunkRedBreak
		wantReason string
	}{
		{"no repo root", "", brk, trunkRedReasonNoRoot},
		{"no base sha", t.TempDir(), noBase, trunkRedReasonNoBase},
		{"no first-break symbol", t.TempDir(), noSym, trunkRedReasonNoSymbol},
		{"qualified symbol names another package", t.TempDir(), qualified, trunkRedReasonQualifiedSymbol},
		// A non-repo dir has no go.mod, so no package maps into a module here.
		{"unmappable packages", t.TempDir(), brk, trunkRedReasonOutOfModule},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := trunkRedGitProbe(tc.root)(tc.brk)
			if got.Status != trunkRedStatusUnprovable {
				t.Fatalf("status = %q, want unprovable — nothing here is checkable", got.Status)
			}
			if got.Reason != tc.wantReason {
				t.Fatalf("reason = %q, want %q", got.Reason, tc.wantReason)
			}
		})
	}
}

// TestTrunkRedSymbolStateAtHead witnesses the four-valued HEAD read against a REAL git
// repo built in a temp dir, so every state is produced by git rather than asserted. The
// grouped-declaration case is the one that matters most: a `const (...)` member is a real
// declaration the column-0 pattern cannot see, so it must read as UNKNOWN — never as the
// positive "still undeclared" proof that would let the view call a fixed break live.
func TestTrunkRedSymbolStateAtHead(t *testing.T) {
	root := t.TempDir()
	runGitFixture(t, root, "init", "-q", ".")
	runGitFixture(t, root, "config", "user.email", "fixture@example.test")
	runGitFixture(t, root, "config", "user.name", "Fixture")
	if err := os.MkdirAll(filepath.Join(root, "pkg"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	src := strings.Join([]string{
		"package pkg",
		"",
		"func declaredAtTopLevel() int { return groupedMember }",
		"",
		"const (",
		"\tgroupedMember = 1",
		")",
		"",
		"func caller() int { return referencedButUndeclared() + declaredAtTopLevel() }",
		"",
	}, "\n")
	if err := os.WriteFile(filepath.Join(root, "pkg", "a.go"), []byte(src), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	runGitFixture(t, root, "add", "pkg/a.go")
	runGitFixture(t, root, "commit", "-q", "-m", "fixture")

	dirs := []string{"pkg"}
	cases := []struct {
		name string
		sym  string
		want trunkRedSymbolState
	}{
		{"package-level declaration", "declaredAtTopLevel", trunkRedSymbolDeclared},
		{"referenced but declared nowhere", "referencedButUndeclared", trunkRedSymbolUndeclared},
		{"neither declared nor referenced", "notInThisTreeAtAll", trunkRedSymbolAbsent},
		// The keep-side heart of the four-valued read.
		{"a grouped const member is NOT a proof of absence", "groupedMember", trunkRedSymbolUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := trunkRedSymbolStateAtHead(root, dirs, tc.sym); got != tc.want {
				t.Fatalf("state(%q) = %v, want %v", tc.sym, got, tc.want)
			}
		})
	}

	t.Run("only a real declaration reads as defined", func(t *testing.T) {
		if !trunkRedSymbolDefinedAtHead(root, dirs, "declaredAtTopLevel") {
			t.Fatalf("a package-level func must read as defined")
		}
		for _, sym := range []string{"referencedButUndeclared", "notInThisTreeAtAll", "groupedMember"} {
			if trunkRedSymbolDefinedAtHead(root, dirs, sym) {
				t.Fatalf("%q is not a proven package-level declaration here", sym)
			}
		}
	})

	t.Run("git failures are not evidence of absence", func(t *testing.T) {
		// A directory with no git repo at all: every grep fails outright, and the
		// difference between "git could not answer" and "no match" is the whole reason
		// the read is four-valued.
		if got := trunkRedSymbolStateAtHead(t.TempDir(), dirs, "declaredAtTopLevel"); got != trunkRedSymbolUnknown {
			t.Fatalf("a failing git must read as UNKNOWN, got %v", got)
		}
		if _, ok := trunkRedGrepAtHead(t.TempDir(), dirs, "-F", "-e", "x"); ok {
			t.Fatalf("a failing git must report that it could not answer")
		}
		if hit, ok := trunkRedGrepAtHead(root, dirs, "-F", "-e", "nothingMatchesThis"); hit || !ok {
			t.Fatalf("a clean no-match must be (hit=false, ok=true); got hit=%v ok=%v", hit, ok)
		}
	})
}

// TestTrunkRedGitProbeGradesAgainstAFixtureRepo drives the WHOLE probe against a real
// repo with a real remote trunk, so the three grades are produced end to end rather than
// stubbed: one class whose symbol came back (resolved, folded out), one whose symbol is
// still referenced and declared nowhere (proven still undefined at HEAD), and one whose
// base the trunk has not moved past (unprovable, surfaced).
func TestTrunkRedGitProbeGradesAgainstAFixtureRepo(t *testing.T) {
	upstream := t.TempDir()
	runGitFixture(t, upstream, "init", "-q", "--bare", ".")

	root := t.TempDir()
	runGitFixture(t, root, "init", "-q", "-b", "main", ".")
	runGitFixture(t, root, "config", "user.email", "fixture@example.test")
	runGitFixture(t, root, "config", "user.name", "Fixture")
	runGitFixture(t, root, "remote", "add", "origin", upstream)
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/m\n\ngo 1.22\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "pkg"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	writeFixturePkg := func(body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, "pkg", "a.go"), []byte(body), 0o644); err != nil {
			t.Fatalf("write pkg: %v", err)
		}
	}
	// The BASE commit: wasFixed is referenced and undeclared (that is the recorded red),
	// and so is stillBroken.
	writeFixturePkg("package pkg\n\nfunc caller() int { return wasFixed() + stillBroken() }\n")
	runGitFixture(t, root, "add", ".")
	runGitFixture(t, root, "commit", "-q", "-m", "base")
	baseOut, err := gitOut(root, "rev-parse", "--verify", "HEAD")
	if err != nil {
		t.Fatalf("rev-parse base: %v", err)
	}
	base := strings.TrimSpace(baseOut)

	// HEAD: wasFixed is declared again; stillBroken is still referenced and declared
	// nowhere.
	writeFixturePkg("package pkg\n\nfunc wasFixed() int { return 1 }\n\nfunc caller() int { return wasFixed() + stillBroken() }\n")
	runGitFixture(t, root, "add", ".")
	runGitFixture(t, root, "commit", "-q", "-m", "repair")
	runGitFixture(t, root, "push", "-q", "origin", "main")

	pkgs := []string{"example.com/m/pkg"}
	probe := trunkRedGitProbe(root)

	t.Run("symbol came back and the trunk moved past the base: RESOLVED", func(t *testing.T) {
		got := probe(trunkRedBreak{BaseSha: base, FirstBreak: "wasFixed", Packages: pkgs})
		if got.Status != trunkRedStatusResolved {
			t.Fatalf("status = %q (%s), want resolved", got.Status, got.Reason)
		}
	})

	t.Run("symbol still referenced and declared nowhere: LIVE", func(t *testing.T) {
		got := probe(trunkRedBreak{BaseSha: base, FirstBreak: "stillBroken", Packages: pkgs})
		if got.Status != trunkRedStatusLive {
			t.Fatalf("status = %q (%s), want still-undefined-at-head", got.Status, got.Reason)
		}
	})

	t.Run("the trunk has not moved past the tip itself: UNPROVABLE", func(t *testing.T) {
		tipOut, err := gitOut(root, "rev-parse", "--verify", "origin/main")
		if err != nil {
			t.Fatalf("rev-parse tip: %v", err)
		}
		got := trunkRedGitProbe(root)(trunkRedBreak{
			BaseSha:    strings.TrimSpace(tipOut),
			FirstBreak: "wasFixed",
			Packages:   pkgs,
		})
		if got.Status != trunkRedStatusUnprovable || got.Reason != trunkRedReasonBaseNotMergedPast {
			t.Fatalf("status = %q reason = %q, want unprovable/%q", got.Status, got.Reason, trunkRedReasonBaseNotMergedPast)
		}
	})

	t.Run("a symbol HEAD no longer even mentions is not a proven repair", func(t *testing.T) {
		got := probe(trunkRedBreak{BaseSha: base, FirstBreak: "goneEntirely", Packages: pkgs})
		if got.Status != trunkRedStatusUnprovable || got.Reason != trunkRedReasonSymbolAbsent {
			t.Fatalf("status = %q reason = %q, want unprovable/%q", got.Status, got.Reason, trunkRedReasonSymbolAbsent)
		}
	})
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

	// "resolvedbase" is the ONLY thing this probe is willing to prove. It is deliberately
	// also resolve-true for the no-base / no-symbol rows' bases, so those classes survive
	// only because the FOLD refuses to ask about them.
	resolveOnlyTheFixed := func(b trunkRedBreak) trunkRedVerdict {
		if b.BaseSha == "livebase" {
			return trunkRedUnprovable("the fixture keeps this one")
		}
		return trunkRedVerdict{Status: trunkRedStatusResolved}
	}

	cases := []struct {
		name     string
		probe    func(trunkRedBreak) trunkRedVerdict
		wantLive []string // first_break of every class that must SURFACE
	}{
		{
			name:     "only the provably-resolved class folds out",
			probe:    resolveOnlyTheFixed,
			wantLive: []string{"noBaseToDate", "stillBroken", ""},
		},
		{
			name:     "a probe that errors on everything keeps every class",
			probe:    func(trunkRedBreak) trunkRedVerdict { return trunkRedUnprovable("git errored") },
			wantLive: []string{"noBaseToDate", "stillBroken", "wasFixed", ""},
		},
		{
			name:     "a nil probe keeps every class",
			probe:    nil,
			wantLive: []string{"noBaseToDate", "stillBroken", "wasFixed", ""},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sum := summarizeTrunkRed(read, tc.probe)
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

	// The keep-side guard has to hold against a probe that claims EVERYTHING is resolved.
	// Only the two classes carrying both a base and a symbol may go.
	t.Run("an always-resolved probe cannot drop an unprovable class", func(t *testing.T) {
		sum := summarizeTrunkRed(read, trunkRedProbeAlwaysResolved())
		var live []string
		for _, c := range sum.Classes {
			live = append(live, c.FirstBreak)
		}
		sort.Strings(live)
		if strings.Join(live, "|") != "|noBaseToDate" {
			t.Fatalf("no-base and no-symbol classes must survive an always-resolved probe; got %v", live)
		}
	})

	// Surfacing is not the same as asserting. Every class the fold could not prove must
	// carry a NAMED reason, and a probe that answers unprovable without one must be given
	// one — an unexplained line in this view is the wall of undifferentiated "shared
	// breaks" the whole resolve path exists to stop printing.
	t.Run("every surfaced class carries a named reason", func(t *testing.T) {
		for _, probe := range []func(trunkRedBreak) trunkRedVerdict{
			nil,
			resolveOnlyTheFixed,
			trunkRedProbeAlwaysResolved(),
			func(trunkRedBreak) trunkRedVerdict { return trunkRedVerdict{Status: trunkRedStatusUnprovable} },
		} {
			sum := summarizeTrunkRed(read, probe)
			if len(sum.Classes) == 0 {
				t.Fatalf("fixture must always surface something")
			}
			for _, c := range sum.Classes {
				if c.Status != trunkRedStatusUnprovable {
					t.Fatalf("class %q: no synthetic probe here proves a break live; got status %q", c.Class, c.Status)
				}
				if strings.TrimSpace(c.KeepReason) == "" {
					t.Fatalf("class %q surfaced with no reason", c.Class)
				}
				if !strings.Contains(trunkRedStatusLabel(c), c.KeepReason) {
					t.Fatalf("class %q label %q must show its reason %q", c.Class, trunkRedStatusLabel(c), c.KeepReason)
				}
			}
		}
	})

	// The grade must be usable as an ordering: a class PROVEN still undefined at HEAD is
	// the only one a reader can act on without re-deriving the evidence, so it goes first
	// even when a bigger unprovable class would otherwise outrank it on session spread.
	t.Run("a proven-live class outranks a larger unprovable one", func(t *testing.T) {
		probe := func(b trunkRedBreak) trunkRedVerdict {
			if b.FirstBreak == "stillBroken" {
				return trunkRedVerdict{Status: trunkRedStatusLive}
			}
			return trunkRedUnprovable("fixture keeps the rest")
		}
		sum := summarizeTrunkRed(read, probe)
		if len(sum.Classes) == 0 || sum.Classes[0].FirstBreak != "stillBroken" {
			t.Fatalf("the proven-live class must sort first; got %+v", sum.Classes)
		}
		if sum.Classes[0].Status != trunkRedStatusLive || sum.Classes[0].KeepReason != "" {
			t.Fatalf("a proven-live class carries no keep reason; got %+v", sum.Classes[0])
		}
		if sum.LiveClasses != 1 || sum.LiveRows != 1 {
			t.Fatalf("want 1 live class / 1 live row; got %d / %d", sum.LiveClasses, sum.LiveRows)
		}
		if sum.UnprovableClasses+sum.LiveClasses != len(sum.Classes) {
			t.Fatalf("every surfaced class must be graded exactly once; got live=%d unprovable=%d of %d",
				sum.LiveClasses, sum.UnprovableClasses, len(sum.Classes))
		}
		if sum.UnprovableRows+sum.LiveRows != sum.Total {
			t.Fatalf("every surfaced row must be graded exactly once; got live=%d unprovable=%d of %d",
				sum.LiveRows, sum.UnprovableRows, sum.Total)
		}
	})
}

// TestRenderTrunkRedNeverAssertsAnUncheckedClassIsRed is the anti-regression witness for
// the mistake this view kept making: printing "Each break below is ALREADY red on the
// trunk" over a list nothing had checked against HEAD. A class the fold could not prove
// must render as unprovable WITH its reason, and the blanket claim must appear only over
// classes that earned it.
func TestRenderTrunkRedNeverAssertsAnUncheckedClassIsRed(t *testing.T) {
	content := trunkRedTestLedger(t,
		trunkRedRecord{Gate: "commit", BaseSha: "b1", Packages: []string{"pkg/unknowable"}, FirstBreak: "Sym", Session: "s1"},
	)

	t.Run("an unprovable-only view makes no liveness claim", func(t *testing.T) {
		sum := summarizeTrunkRed(content, func(trunkRedBreak) trunkRedVerdict {
			return trunkRedUnprovable(trunkRedReasonOutOfModule)
		})
		out := renderTrunkRed(sum)
		if !strings.Contains(out, "unprovable: "+trunkRedReasonOutOfModule) {
			t.Fatalf("the reason must be on the class line:\n%s", out)
		}
		if strings.Contains(out, "still undefined at HEAD]") {
			t.Fatalf("nothing here was proven still undefined:\n%s", out)
		}
		if strings.Contains(out, "actually biting") {
			t.Fatalf("the live-break banner must not print with zero live classes:\n%s", out)
		}
		if !strings.Contains(out, "0 still undefined at HEAD, 1 unprovable") {
			t.Fatalf("the headline must split proven from unprovable:\n%s", out)
		}
	})

	t.Run("a proven-live view says so, and says why", func(t *testing.T) {
		sum := summarizeTrunkRed(content, func(trunkRedBreak) trunkRedVerdict {
			return trunkRedVerdict{Status: trunkRedStatusLive}
		})
		out := renderTrunkRed(sum)
		if !strings.Contains(out, "[still undefined at HEAD] pkg/unknowable") {
			t.Fatalf("a proven class must be labelled as such:\n%s", out)
		}
		if !strings.Contains(out, "1 still undefined at HEAD, 0 unprovable") {
			t.Fatalf("the headline must count the proven class:\n%s", out)
		}
		if strings.Contains(out, "unprovable]") {
			t.Fatalf("no unprovable banner belongs in an all-proven view:\n%s", out)
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

// TestTrunkRedGitProbeSecondConjunct is the witness that ancestry ALONE cannot drop a
// class. Both breaks below share one base sha that really is a strict ancestor of the
// remote trunk, so conjunct 1 is constant and true for both; the ONLY difference is
// whether the first-break symbol is defined at HEAD. Deleting conjunct 2 reds the second
// assertion, which is the whole point of this issue: a base becomes an ancestor the
// moment any unrelated peer commit lands, so ancestry alone would fold out a live break.
func TestTrunkRedGitProbeSecondConjunct(t *testing.T) {
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

	probe := trunkRedGitProbe(root)
	pkgs := []string{"github.com/anthony-chaudhary/fak/cmd/fak"}

	fixed := trunkRedBreak{BaseSha: base, FirstBreak: "summarizeTrunkRed", Packages: pkgs}
	if got := probe(fixed); got.Status != trunkRedStatusResolved {
		t.Fatalf("both conjuncts hold (ancestor base + symbol defined at HEAD): must resolve; got %q (%s)", got.Status, got.Reason)
	}
	stillBroken := trunkRedBreak{BaseSha: base, FirstBreak: "trunkRedSymbolThatIsDefinedNowhere5356", Packages: pkgs}
	if got := probe(stillBroken); got.Status == trunkRedStatusResolved {
		t.Fatalf("ancestry alone must NOT resolve a break whose symbol is still undefined at HEAD")
	}
	// A synthetic package path leaves nothing to search: unresolvable, so kept — and the
	// view has to SAY that is why, not imply the break is live.
	foreign := trunkRedBreak{BaseSha: base, FirstBreak: "summarizeTrunkRed", Packages: []string{"buildcheck.test/p"}}
	got := probe(foreign)
	if got.Status != trunkRedStatusUnprovable || got.Reason != trunkRedReasonOutOfModule {
		t.Fatalf("an out-of-module break must be unprovable/%q; got %q (%s)", trunkRedReasonOutOfModule, got.Status, got.Reason)
	}
}

// TestTrunkRedWitnessStampsGatedModule pins the WRITE-side half of provenance (#5540):
// the row records the module of the repo the gate was actually run against, read from
// that repo's own go.mod at the moment it is written.
//
// This is the fact that has to be captured at creation or not at all. `c1fbc87ed` routed
// a gate's own throwaway-repo witness into the throwaway repo's ledger, which stopped the
// bleeding, but the row it writes is still anonymous: on disk, a synthetic fixture break
// and a real trunk break are the same shape. The 28 legacy `buildcheck.test/p` rows in the
// real ledger are exactly that residue, and nothing but their package TEXT distinguishes
// them — which is why the stamp goes on the writer rather than a matcher on the reader.
func TestTrunkRedWitnessStampsGatedModule(t *testing.T) {
	pinTrunkRedClock(t)
	// No env override: the gated root alone decides both destination and provenance.
	t.Setenv(trunkRedLedgerEnv, "")
	t.Setenv(trunkRedModeEnv, "")
	t.Setenv("FAK_SESSION_ID", "")
	t.Setenv("CLAUDE_SESSION_ID", "")

	readBack := func(t *testing.T, ledger string) trunkRedRecord {
		t.Helper()
		b, err := os.ReadFile(ledger)
		if err != nil {
			t.Fatalf("read ledger: %v", err)
		}
		line := strings.TrimSpace(string(b))
		var rec trunkRedRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("unmarshal %q: %v", line, err)
		}
		return rec
	}

	t.Run("the gated repo's module is recorded", func(t *testing.T) {
		gated := t.TempDir()
		if err := os.WriteFile(filepath.Join(gated, "go.mod"), []byte("module buildcheck.test\n\ngo 1.22\n"), 0o644); err != nil {
			t.Fatalf("write go.mod: %v", err)
		}
		var stderr bytes.Buffer
		w := emitTrunkRedWitness(&stderr, gated, "commit", "base1", []string{"buildcheck.test/p"}, "neverDefined")
		if !w.Witnessed {
			t.Fatalf("expected a witness; stderr=%q", stderr.String())
		}
		rec := readBack(t, w.Ledger)
		if rec.Module != "buildcheck.test" {
			t.Fatalf("row must carry the GATED repo's module as provenance; got %q, want %q", rec.Module, "buildcheck.test")
		}
		// The stamp is additive: the schema must stay readable, or every already-written
		// row in a real ledger would silently vanish from the fold.
		if rec.Schema != trunkRedRecordSchema {
			t.Fatalf("an additive field must not bump the schema; got %q want %q", rec.Schema, trunkRedRecordSchema)
		}
	})

	t.Run("an unreadable go.mod records no provenance rather than a guess", func(t *testing.T) {
		gated := t.TempDir() // no go.mod at all
		var stderr bytes.Buffer
		w := emitTrunkRedWitness(&stderr, gated, "commit", "base1", []string{"buildcheck.test/p"}, "neverDefined")
		if !w.Witnessed {
			t.Fatalf("expected a witness; stderr=%q", stderr.String())
		}
		// The package name says "buildcheck.test" plainly, and the writer still must not
		// infer the module from it: an unknown provenance is recorded as unknown.
		if rec := readBack(t, w.Ledger); rec.Module != "" {
			t.Fatalf("no go.mod means unknown provenance, not an inferred one; got %q", rec.Module)
		}
	})
}

// TestTrunkRedForeignProvenanceIsStructuralNotTextual is the witness that the foreign
// verdict is read off the writer's stamp and NOT pattern-matched out of the row's text
// (#5540).
//
// The whole test rests on one deliberate choice: every break below names the SAME
// in-module package, `example.com/m/p`. So `trunkRedPackageDirs` maps all of them, the
// out-of-module inference cannot fire for any of them, and the only thing that differs
// between the foreign row and the control row is the recorded module. A matcher on the
// package string could not tell these apart at all — which is the point. If this test
// passes while a heuristic-on-content implementation is in place, it is passing for the
// wrong reason, so the fixture is built so that no such implementation can pass it.
func TestTrunkRedForeignProvenanceIsStructuralNotTextual(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/m\n\ngo 1.22\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	// Guard the premise: the shared package really is mappable inside this module, so a
	// foreign verdict below cannot be the out-of-module answer wearing a new name.
	pkgs := []string{"example.com/m/p"}
	if dirs := trunkRedPackageDirs(root, pkgs); len(dirs) == 0 {
		t.Fatalf("fixture premise broken: %v must map inside example.com/m, else this test cannot isolate provenance", pkgs)
	}

	probe := trunkRedGitProbe(root)

	t.Run("a row stamped with another module is graded foreign", func(t *testing.T) {
		got := probe(trunkRedBreak{BaseSha: "base1", FirstBreak: "neverDefined", Packages: pkgs, Module: "buildcheck.test"})
		if got.Status != trunkRedStatusUnprovable || got.Reason != trunkRedReasonForeignModule {
			t.Fatalf("a row recorded against another module must be unprovable/%q; got %q (%s)", trunkRedReasonForeignModule, got.Status, got.Reason)
		}
	})

	t.Run("control: an identically-worded row stamped with THIS module is not foreign", func(t *testing.T) {
		got := probe(trunkRedBreak{BaseSha: "base2", FirstBreak: "neverDefined", Packages: pkgs, Module: "example.com/m"})
		if got.Reason == trunkRedReasonForeignModule {
			t.Fatalf("a row recorded against THIS module must never be graded foreign; got %q (%s)", got.Status, got.Reason)
		}
	})

	t.Run("control: an unstamped legacy row is unknown provenance, never foreign", func(t *testing.T) {
		got := probe(trunkRedBreak{BaseSha: "base3", FirstBreak: "neverDefined", Packages: pkgs, Module: ""})
		if got.Reason == trunkRedReasonForeignModule {
			t.Fatalf("an absent field is not evidence: a row predating the stamp must not read as foreign; got %q (%s)", got.Status, got.Reason)
		}
	})

	// The provenance has to survive the fold, not just the probe: it travels row ->
	// rollup -> break, and a class keeps its stamp even when older unstamped rows share it.
	t.Run("provenance reaches the fold and drops nothing", func(t *testing.T) {
		rows := []trunkRedRecord{
			{Schema: trunkRedRecordSchema, Gate: "commit", BaseSha: "base1", Packages: pkgs, FirstBreak: "neverDefined", Module: "buildcheck.test"},
			{Schema: trunkRedRecordSchema, Gate: "commit", BaseSha: "base2", Packages: pkgs, FirstBreak: "neverDefined", Module: "example.com/m"},
			{Schema: trunkRedRecordSchema, Gate: "commit", BaseSha: "base3", Packages: pkgs, FirstBreak: "neverDefined"},
		}
		var content strings.Builder
		for _, r := range rows {
			b, err := json.Marshal(r)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			content.Write(b)
			content.WriteString("\n")
		}
		sum := summarizeTrunkRed(content.String(), probe)

		// Nothing is filtered away. A foreign row is SURFACED with its reason, exactly
		// like every other unprovable class — suppressing it would be the metric
		// laundering this change exists to avoid.
		if sum.ResolvedRows != 0 {
			t.Fatalf("provenance must not drop rows; %d row(s) folded out", sum.ResolvedRows)
		}
		if sum.Total != len(rows) || len(sum.Classes) != len(rows) {
			t.Fatalf("all %d rows must stay surfaced as %d classes; got total=%d classes=%d", len(rows), len(rows), sum.Total, len(sum.Classes))
		}

		byBase := map[string]trunkRedClassRollup{}
		for _, c := range sum.Classes {
			byBase[c.BaseSha] = c
		}
		foreign, ok := byBase["base1"]
		if !ok {
			t.Fatalf("the foreign class disappeared from the fold: %+v", sum.Classes)
		}
		if foreign.Module != "buildcheck.test" {
			t.Fatalf("the rollup must carry the recorded provenance; got %q", foreign.Module)
		}
		if foreign.KeepReason != trunkRedReasonForeignModule {
			t.Fatalf("the foreign class must be surfaced with reason %q; got %q", trunkRedReasonForeignModule, foreign.KeepReason)
		}
		// And the reader SEES it: the reason has to reach the rendered line, or the fold
		// knows something the operator does not.
		if out := renderTrunkRed(sum); !strings.Contains(out, trunkRedReasonForeignModule) {
			t.Fatalf("the rendered view must name the provenance reason:\n%s", out)
		}
		for _, base := range []string{"base2", "base3"} {
			if c := byBase[base]; c.KeepReason == trunkRedReasonForeignModule {
				t.Fatalf("class %s is not foreign but was graded %q", base, c.KeepReason)
			}
		}
	})
}
