package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memvaluescore"
)

// fixtureMemoryStore writes a markdown memory store whose three notes exercise
// the three read-time verdicts against the REAL DefaultArtifactVerifier run in
// this checkout: a note naming a path that exists (fresh), a note naming a
// deleted package + unresolvable commit (stale → withheld), and a prose note
// with nothing checkable (unverified, rendered hedged).
func fixtureMemoryStore(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"MEMORY.md": "# Memory index\n\n" +
			"- [Gate helper](fresh.md) — still true\n" +
			"- [Moved fix](stale.md) — names a gone artifact\n" +
			"- [Preference](prose.md) — prose only\n",
		"fresh.md": "---\nname: gate-helper\ndescription: where the algebra lives\nmetadata:\n  type: feedback\n---\n\nThe memory algebra executor lives in internal/memq/exec.go.\n",
		"stale.md": "---\nname: moved-fix\ndescription: an old location\nmetadata:\n  type: project\n---\n\nThe fix lives in internal/gonepkg/gone.go.\n",
		"prose.md": "---\nname: preference\ndescription: terse answers\nmetadata:\n  type: user\n---\n\nThe user prefers the outcome stated first.\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// #2347 done-condition at the CLI: `fak memory recall` renders the still-true
// note tagged fresh, renders the prose note hedged, and WITHHOLDS the stale note
// with the failing claim named — the orientation block a loop turn injects.
func TestMemoryRecall_verifiedOrientationBlock(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "memory algebra gate"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "fresh.md) [fresh]") {
		t.Errorf("still-true note must render tagged [fresh]; got:\n%s", s)
	}
	if !strings.Contains(s, "prose.md) [unverified]") {
		t.Errorf("prose-only note must render hedged [unverified]; got:\n%s", s)
	}
	if strings.Contains(s, "internal/gonepkg/gone.go.\n") || strings.Contains(s, "stale.md) [") {
		t.Errorf("stale note body must never render; got:\n%s", s)
	}
	if !strings.Contains(s, "stale.md [withheld:stale_recall_artifact]") {
		t.Errorf("stale note must be withheld with the reason named; got:\n%s", s)
	}
	if !strings.Contains(s, "internal/gonepkg/gone.go") {
		t.Errorf("the withheld line must name the failing claim as evidence; got:\n%s", s)
	}
}

// The JSON envelope parses and carries the same verdicts (the machine surface a
// loop harness consumes).
func TestMemoryRecall_jsonEnvelope(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	if code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "memory algebra gate", "--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var env recallEnvelope
	if err := json.Unmarshal([]byte(out.String()), &env); err != nil {
		t.Fatalf("envelope must parse: %v\n%s", err, out.String())
	}
	verdicts := map[string]string{}
	for _, n := range env.Rendered {
		verdicts[n.ID] = n.Verdict
		if n.Body == "" {
			t.Errorf("rendered note %s must carry its body", n.ID)
		}
	}
	for _, n := range env.Withheld {
		verdicts[n.ID] = n.Verdict
		if n.Body != "" {
			t.Errorf("withheld note %s must never carry a body", n.ID)
		}
	}
	if verdicts["fresh.md"] != "fresh" || verdicts["prose.md"] != "unverified" || verdicts["stale.md"] != "withheld:stale_recall_artifact" {
		t.Fatalf("verdicts = %+v", verdicts)
	}
}

// A missing store yields an empty block and exit 0 — the loop's recall step must
// fail open on a fresh node, never refuse the turn.
func TestMemoryRecall_missingStoreFailsOpen(t *testing.T) {
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", filepath.Join(t.TempDir(), "nope"), "--intent", "anything"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "no notes rendered") {
		t.Errorf("empty store must say so; got:\n%s", out.String())
	}
}

// --list-formats prints the registered memview formats and exits, independent of
// any store — this is the "what can I surface this in" discovery seam.
func TestMemoryRecall_listFormats(t *testing.T) {
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--list-formats"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"markdown", "json", "toon"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("--list-formats must list %q; got:\n%s", want, out.String())
		}
	}
}

// --format toon surfaces the SAME verdicts the markdown mode renders, just under
// the compact TOON encoding — the "consumed under a chosen format at the right
// time" half of the goal. A withheld note's body must still never appear.
func TestMemoryRecall_formatTOON(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "memory algebra gate", "--format", "toon"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.HasPrefix(s, "verified_memory_recall[3]{") {
		t.Fatalf("expected a TOON header for 3 rows; got:\n%s", s)
	}
	if !strings.Contains(s, "fresh") || !strings.Contains(s, "unverified") || !strings.Contains(s, "withheld:stale_recall_artifact") {
		t.Errorf("TOON surface must carry all three verdicts; got:\n%s", s)
	}
	// The withheld row may name the failing claim as evidence (same as markdown
	// mode's withheld line) — but never the note's actual prose body.
	if strings.Contains(s, "The fix lives in internal/gonepkg/gone.go.") {
		t.Errorf("withheld note's body must never surface under any format; got:\n%s", s)
	}
}

// An unknown --format name fails closed, naming the bad value — never silently
// falls back to markdown.
func TestMemoryRecall_formatUnknownFailsClosed(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "x", "--format", "yaml"})
	if code == 0 {
		t.Fatalf("expected a non-zero exit for an unknown format; stdout:\n%s", out.String())
	}
	if !strings.Contains(errb.String(), "yaml") {
		t.Errorf("stderr must name the unknown format; got:\n%s", errb.String())
	}
}

// --ledger appends one fak-memory-value-ledger/1 row per recall that witnesses
// events, and memvaluescore.FoldLedger folds it — the frontier's event feed:
// this fixture witnesses 1 fresh render (×2) + 1 stale withholding (×8).
func TestMemoryRecall_appendsMemoryValueLedgerRow(t *testing.T) {
	dir := fixtureMemoryStore(t)
	ledger := filepath.Join(t.TempDir(), "memory-value.jsonl")
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "memory algebra gate", "--ledger", ledger})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("ledger row must be appended: %v", err)
	}
	var row memoryValueRow
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &row); err != nil {
		t.Fatalf("row must parse: %v\n%s", err, raw)
	}
	if row.Schema != memvaluescore.LedgerSchema {
		t.Errorf("schema = %q, want %q", row.Schema, memvaluescore.LedgerSchema)
	}
	// The prose note renders hedged [unverified] — it must NOT count as fresh.
	if row.Fresh != 1 || row.WithheldStale != 1 || row.Lessons != 0 {
		t.Errorf("row events = fresh=%d withheld_stale=%d lessons=%d, want 1/1/0", row.Fresh, row.WithheldStale, row.Lessons)
	}
	fold := memvaluescore.FoldLedger(ledger)
	if !fold.Present || fold.Rows != 1 || fold.SkippedRows != 0 {
		t.Errorf("FoldLedger must accept the row: %+v", fold)
	}
	frontier, _ := memvaluescore.Frontier(fold.Events)
	if frontier != 8+2 {
		t.Errorf("frontier = %d, want 10 (stale_withheld×8 + fresh_rendered×2)", frontier)
	}
}

// A recall that witnesses nothing (empty store) must NOT append a row — a
// zero-event row would flip recall_value_witnessed without any value.
func TestMemoryRecall_zeroEventRecallAppendsNothing(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "memory-value.jsonl")
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", filepath.Join(t.TempDir(), "nope"), "--intent", "anything", "--ledger", ledger})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Errorf("zero-event recall must not create the ledger; stat err = %v", err)
	}
}

// The default-path rule: an explicit --store never ledgers into the repo P&L
// unless --ledger names a path, and "off" always disables — only a default-store
// recall inherits the committed docs/nightrun ledger.
func TestMemoryRecall_ledgerPathResolution(t *testing.T) {
	if got := recallLedgerPath("", "/tmp/fixture-store"); got != "" {
		t.Errorf("explicit store without --ledger must not append; got %q", got)
	}
	if got := recallLedgerPath("off", ""); got != "" {
		t.Errorf("--ledger off must disable; got %q", got)
	}
	if got := recallLedgerPath("/x/custom.jsonl", "/tmp/fixture-store"); got == "" {
		t.Error("an explicit --ledger path must win even with an explicit store")
	}
	if got := recallLedgerPath("", ""); got != "" && !strings.HasSuffix(filepath.ToSlash(got), memvaluescore.DefaultLedgerRel) {
		t.Errorf("default-store recall must resolve to the committed ledger; got %q", got)
	}
}

// --ablate-formats measures the SAME note set's cost under every named format
// instead of rendering it — the ablation half of the goal: same content, vary
// only the encoding, read the byte/token delta off one table.
func TestMemoryRecall_ablateFormats(t *testing.T) {
	dir := fixtureMemoryStore(t)
	var out, errb strings.Builder
	code := runMemoryRecall(&out, &errb, []string{"--store", dir, "--intent", "memory algebra gate", "--ablate-formats", "json,toon"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "json") || !strings.Contains(s, "toon") {
		t.Fatalf("ablation table must list both requested formats; got:\n%s", s)
	}
	if strings.Contains(s, "markdown") {
		t.Errorf("ablation table must only list the requested formats, not the full registry; got:\n%s", s)
	}
}
