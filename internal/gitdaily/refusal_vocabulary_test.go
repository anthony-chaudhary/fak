package gitdaily

// refusal_vocabulary_test.go — bind gitdaily's emitted refusal codes to the closed vocabulary
// they are supposed to speak (#5589, a follow-on of the spine in #5577).
//
// #5589 wired every RefusalCode in this package into dos.toml's [reasons.*] table so that
// `dos man wedge TICK_BUSY --explain` resolves instead of classifying fak's own structured refusal as
// UNCLASSIFIED prose drift. That wiring is a snapshot: it was true the day it landed and nothing
// keeps it true. Add a seventh RefusalCode below without a matching dos.toml row and the tick
// starts emitting a token the kernel cannot route on — silently, because the other six still
// resolve and the table still reads as complete.
//
// internal/architest/reason_vocabulary_test.go already guards the OPPOSITE direction (a declared
// row with no production emitter). This is the missing half: an emitter with no declared row.
// It is scoped to this package's own codes deliberately — the general source-to-table binding is
// a wider change than this leaf owns.
//
// The checks are split into pure functions over source text so the guard itself is testable.
// A binding test that only ever runs against a passing tree cannot show that it would bite; the
// synthetic cases at the bottom are the failing-before evidence for the real ones above.

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// vocabRow is one parsed [reasons.*] block: key -> raw (still-quoted) value.
type vocabRow map[string]string

// requiredReasonKeys are the fields a row needs to be useful. A row that resolves but carries no
// summary or fix is the same dead end as no row at all: the operator learns a name, not a cure.
var requiredReasonKeys = []string{"category", "refusal", "summary", "fix", "see_also"}

// codeDeclRE matches the RefusalCode constant block in refusal.go. Reading the SOURCE rather
// than listing the constants here is the whole point: a hand-maintained list in the test would
// need the same edit as the dos.toml row, so it would drift in lockstep and catch nothing.
var codeDeclRE = regexp.MustCompile(`RefusalCode\s*=\s*"([A-Z0-9_]+)"`)

// parseRefusalCodes returns every RefusalCode literal declared in the given Go source, plus any
// token declared more than once.
func parseRefusalCodes(src string) (codes, dups []string) {
	seen := map[string]bool{}
	for _, m := range codeDeclRE.FindAllStringSubmatch(src, -1) {
		if seen[m[1]] {
			dups = append(dups, m[1])
			continue
		}
		seen[m[1]] = true
		codes = append(codes, m[1])
	}
	sort.Strings(codes)
	return codes, dups
}

// parseReasonRows parses dos.toml's [reasons.*] blocks. A hand parse keeps the test
// dependency-free and is sufficient: the rows are flat key = value pairs, and a malformed table
// fails `dos doctor` long before it reaches here.
func parseReasonRows(raw string) map[string]vocabRow {
	rows := map[string]vocabRow{}
	var current string
	for _, line := range strings.Split(raw, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "[") {
			current = ""
			if rest, ok := strings.CutPrefix(trimmed, "[reasons."); ok {
				if token, ok := strings.CutSuffix(rest, "]"); ok {
					current = token
					rows[token] = vocabRow{}
				}
			}
			continue
		}
		if current == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		key, value, ok := strings.Cut(trimmed, "=")
		if !ok {
			continue
		}
		rows[current][strings.TrimSpace(key)] = strings.TrimSpace(value)
	}
	return rows
}

// declarationProblems reports every emitted code that is undeclared or declared incompletely.
func declarationProblems(codes []string, rows map[string]vocabRow) []string {
	var problems []string
	for _, code := range codes {
		row, declared := rows[code]
		if !declared {
			problems = append(problems, "gitdaily emits RefusalCode "+code+" but dos.toml declares no [reasons."+code+"]: "+
				"`dos man wedge "+code+" --explain` classifies the tick's own structured refusal as UNCLASSIFIED prose drift, "+
				"so a fleet cannot route or recover from it mechanically. Declare the row with category/refusal/summary/fix/see_also (#5589).")
			continue
		}
		for _, key := range requiredReasonKeys {
			if value := strings.Trim(row[key], `"`); strings.TrimSpace(value) == "" || value == "[]" {
				problems = append(problems, "dos.toml [reasons."+code+"] lacks a non-empty "+key+
					" — the token must carry its own class, explanation, and cure so `dos man wedge "+code+" --explain` teaches an operator rather than just naming the failure (#5589).")
			}
		}
	}
	return problems
}

// polarityProblems reports rows whose refusal flag disagrees with whether the outcome is
// advisory. Polarity is not cosmetic: it decides whether the kernel routes the tick to replan.
func polarityProblems(codes []string, rows map[string]vocabRow, advisory map[string]bool) []string {
	var problems []string
	for _, code := range codes {
		row, declared := rows[code]
		if !declared {
			continue // reported by declarationProblems
		}
		want := "true"
		if advisory[code] {
			want = "false"
		}
		if got := row["refusal"]; got != want {
			problems = append(problems, "dos.toml [reasons."+code+"] has refusal = "+got+", want "+want+
				": TICK_BUSY is advisory because an overlapping fire mutates nothing, while every other gitdaily outcome needs operator recovery before the scheduler retries (#5589).")
		}
	}
	return problems
}

// advisoryCodes is the set of outcomes that must NOT block. TICK_BUSY means a live tick already
// owns the mutator and this fire was skipped WITHOUT mutating the clone — normal scheduler
// overlap, not an incident. Declaring it refusal = true would make routine overlap page an
// operator and route a healthy clone to replan.
var advisoryCodes = map[string]bool{string(RefusalTickBusy): true}

// repoRoot walks up from the package directory to the clone root (the directory holding
// dos.toml). The test reads a repo-level config, so it cannot assume a fixed relative depth.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("resolve package dir: %v", err)
	}
	for {
		if _, statErr := os.Stat(filepath.Join(dir, "dos.toml")); statErr == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("walked to the filesystem root without finding dos.toml; the refusal vocabulary cannot be checked")
		}
		dir = parent
	}
}

// liveVocabulary loads this package's real codes and the repo's real reason table.
func liveVocabulary(t *testing.T) (codes []string, rows map[string]vocabRow) {
	t.Helper()
	src, err := os.ReadFile("refusal.go")
	if err != nil {
		t.Fatalf("read refusal.go: %v", err)
	}
	codes, dups := parseRefusalCodes(string(src))
	for _, d := range dups {
		t.Errorf("refusal.go declares RefusalCode %q twice", d)
	}
	if len(codes) == 0 {
		t.Fatal("refusal.go declares no RefusalCode constants; the vocabulary binding would pass vacuously")
	}

	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	rows = parseReasonRows(string(raw))
	if len(rows) == 0 {
		t.Fatal("dos.toml declares no [reasons.*] rows; the vocabulary binding would pass vacuously")
	}
	return codes, rows
}

func TestEveryRefusalCodeIsDeclaredInVocabulary(t *testing.T) {
	codes, rows := liveVocabulary(t)
	for _, p := range declarationProblems(codes, rows) {
		t.Error(p)
	}
}

func TestTickBusyIsAdvisoryAndTheRestRefuse(t *testing.T) {
	codes, rows := liveVocabulary(t)
	for _, p := range polarityProblems(codes, rows, advisoryCodes) {
		t.Error(p)
	}
}

// --- the guard's own failing-before evidence -------------------------------------------------
//
// Each case below is the exact drift the live tests above are meant to catch, fed in
// synthetically. If one of these stops reporting a problem, the live test has gone vacuous.

const probeTable = `
[reasons.DECLARED_FULLY]
category = "OPERATOR_GATE"
refusal  = true
summary  = "s"
fix      = "f"
see_also = ["x"]

[reasons.DECLARED_THINLY]
category = "OPERATOR_GATE"
refusal  = true
summary  = ""
fix      = "f"
see_also = []

[reasons.WRONG_POLARITY]
category = "OPERATOR_GATE"
refusal  = true
summary  = "s"
fix      = "f"
see_also = ["x"]
`

func TestGuardCatchesAnUndeclaredCode(t *testing.T) {
	codes, dups := parseRefusalCodes(`const ( A RefusalCode = "DECLARED_FULLY"
		B RefusalCode = "NEVER_DECLARED" )`)
	if len(dups) != 0 {
		t.Fatalf("unexpected duplicates: %v", dups)
	}
	problems := declarationProblems(codes, parseReasonRows(probeTable))
	if len(problems) != 1 || !strings.Contains(problems[0], "NEVER_DECLARED") {
		t.Fatalf("guard did not flag the undeclared code; got %d problem(s): %v", len(problems), problems)
	}
}

func TestGuardCatchesAThinlyDeclaredRow(t *testing.T) {
	codes, _ := parseRefusalCodes(`A RefusalCode = "DECLARED_THINLY"`)
	problems := declarationProblems(codes, parseReasonRows(probeTable))
	// Both the empty summary and the empty see_also must be named, not just the first.
	if len(problems) != 2 {
		t.Fatalf("want the empty summary AND see_also flagged, got %d: %v", len(problems), problems)
	}
	joined := strings.Join(problems, "\n")
	for _, key := range []string{"summary", "see_also"} {
		if !strings.Contains(joined, key) {
			t.Errorf("guard did not name the empty %q field: %v", key, problems)
		}
	}
}

func TestGuardCatchesInvertedPolarity(t *testing.T) {
	codes, _ := parseRefusalCodes(`A RefusalCode = "WRONG_POLARITY"`)
	rows := parseReasonRows(probeTable)

	// Declared refusal = true while the outcome is advisory: routine overlap would page an operator.
	problems := polarityProblems(codes, rows, map[string]bool{"WRONG_POLARITY": true})
	if len(problems) != 1 || !strings.Contains(problems[0], "want false") {
		t.Fatalf("guard did not flag a blocking row that should be advisory; got: %v", problems)
	}
	// Same row, correctly classified as blocking: no problem.
	if problems := polarityProblems(codes, rows, nil); len(problems) != 0 {
		t.Fatalf("guard flagged a correctly-blocking row: %v", problems)
	}
}

func TestGuardCatchesDuplicateCodeDeclarations(t *testing.T) {
	_, dups := parseRefusalCodes(`A RefusalCode = "TWICE"
		B RefusalCode = "TWICE"`)
	if len(dups) != 1 || dups[0] != "TWICE" {
		t.Fatalf("want the duplicated token reported once, got %v", dups)
	}
}
