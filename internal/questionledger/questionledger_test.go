package questionledger

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// good returns a clean baseline row; tests clone and mutate it.
func good() map[string]interface{} {
	return map[string]interface{}{
		"id":       "Q-20260708-001",
		"ts":       "2026-07-08T00:00:00Z",
		"category": "CONTRARIAN",
		"target":   "the witness thesis",
		"question": "Is the witness layer itself witnessed?",
		"why":      "trust may have just moved up a level",
		"status":   "open",
		"ticket":   nil,
	}
}

// with clones good() and applies the overrides.
func with(overrides map[string]interface{}) map[string]interface{} {
	m := good()
	for k, v := range overrides {
		m[k] = v
	}
	return m
}

// rows turns dicts (or raw JSON strings) into ParseRows-shaped rows.
func rows(objs ...interface{}) []Row {
	var lines []string
	for _, o := range objs {
		if s, ok := o.(string); ok {
			lines = append(lines, s)
			continue
		}
		b, err := json.Marshal(o)
		if err != nil {
			panic(err)
		}
		lines = append(lines, string(b))
	}
	return ParseRows(strings.Join(lines, "\n"))
}

func hasProblem(probs []string, substr string) bool {
	for _, p := range probs {
		if strings.Contains(p, substr) {
			return true
		}
	}
	return false
}

func TestGoodRowIsClean(t *testing.T) {
	if probs := LintRows(rows(good())); len(probs) != 0 {
		t.Fatalf("expected clean, got %v", probs)
	}
}

func TestBadIDRejected(t *testing.T) {
	if !hasProblem(LintRows(rows(with(map[string]interface{}{"id": "Q-2026-07-08-1"}))), "Q-YYYYMMDD-NNN") {
		t.Fatal("bad id not rejected")
	}
}

func TestDuplicateIDRejected(t *testing.T) {
	if !hasProblem(LintRows(rows(good(), good())), "duplicate id") {
		t.Fatal("duplicate id not rejected")
	}
}

func TestUnknownCategoryRejected(t *testing.T) {
	if !hasProblem(LintRows(rows(with(map[string]interface{}{"category": "SPICY"}))), "category") {
		t.Fatal("unknown category not rejected")
	}
}

func TestUnknownStatusRejected(t *testing.T) {
	if !hasProblem(LintRows(rows(with(map[string]interface{}{"status": "parked"}))), "status") {
		t.Fatal("unknown status not rejected")
	}
}

func TestTicketedWithoutNumberRejected(t *testing.T) {
	bad := with(map[string]interface{}{"status": "ticketed", "ticket": nil})
	if !hasProblem(LintRows(rows(bad)), "requires a positive int ticket") {
		t.Fatal("ticketed w/o number not rejected")
	}
}

func TestOpenWithTicketRejected(t *testing.T) {
	bad := with(map[string]interface{}{"status": "open", "ticket": 42})
	if !hasProblem(LintRows(rows(bad)), "must be null") {
		t.Fatal("open with ticket not rejected")
	}
}

func TestTicketedWithNumberOK(t *testing.T) {
	ok := with(map[string]interface{}{"status": "ticketed", "ticket": 42})
	if probs := LintRows(rows(ok)); len(probs) != 0 {
		t.Fatalf("ticketed+number should be clean, got %v", probs)
	}
}

func TestBooleanTicketRejected(t *testing.T) {
	// A JSON true decodes to bool, not float64 — the guard must reject it.
	bad := with(map[string]interface{}{"status": "ticketed", "ticket": true})
	if !hasProblem(LintRows(rows(bad)), "positive int ticket") {
		t.Fatal("boolean ticket not rejected")
	}
}

func TestExtraKeyRejected(t *testing.T) {
	if !hasProblem(LintRows(rows(with(map[string]interface{}{"severity": "high"}))), "unexpected key") {
		t.Fatal("extra key not rejected")
	}
}

func TestMissingKeyRejected(t *testing.T) {
	bad := good()
	delete(bad, "why")
	if !hasProblem(LintRows(rows(bad)), "missing key") {
		t.Fatal("missing key not rejected")
	}
}

func TestNonQuestionRejected(t *testing.T) {
	bad := with(map[string]interface{}{"question": "This is a statement, not a question."})
	if !hasProblem(LintRows(rows(bad)), "must end with '?'") {
		t.Fatal("non-question not rejected")
	}
}

func TestLeakAbsolutePathRejected(t *testing.T) {
	bad := with(map[string]interface{}{"question": `Why does C:\work\fak leak here?`})
	if !hasProblem(LintRows(rows(bad)), "leaks") {
		t.Fatal("absolute path leak not rejected")
	}
}

func TestLeakEmailRejected(t *testing.T) {
	bad := with(map[string]interface{}{"why": "raised by someone@example.com"})
	if !hasProblem(LintRows(rows(bad)), "leaks") {
		t.Fatal("email leak not rejected")
	}
}

func TestBadJSONLineRejected(t *testing.T) {
	if !hasProblem(LintRows(rows("{not json")), "not valid JSON") {
		t.Fatal("bad JSON line not rejected")
	}
}

func TestNextIDIncrements(t *testing.T) {
	r := rows(
		with(map[string]interface{}{"id": "Q-20260708-001"}),
		with(map[string]interface{}{"id": "Q-20260708-005"}),
	)
	got, err := NextID(r, "20260708")
	if err != nil || got != "Q-20260708-006" {
		t.Fatalf("next-id = %q, %v; want Q-20260708-006", got, err)
	}
}

func TestNextIDFirstOfDay(t *testing.T) {
	r := rows(with(map[string]interface{}{"id": "Q-20260708-009"}))
	got, err := NextID(r, "20260709")
	if err != nil || got != "Q-20260709-001" {
		t.Fatalf("next-id = %q, %v; want Q-20260709-001", got, err)
	}
}

func TestDedupeDetectsIdentical(t *testing.T) {
	hit := DedupeMatch(rows(good()), "Is the witness layer itself witnessed?", "")
	if hit == nil || hit["id"] != "Q-20260708-001" {
		t.Fatalf("dedupe miss: %v", hit)
	}
}

func TestDedupePassesNovel(t *testing.T) {
	if hit := DedupeMatch(rows(good()), "What breaks if the nightly target is halved?", ""); hit != nil {
		t.Fatalf("novel question wrongly matched: %v", hit)
	}
}

func TestStatsCounts(t *testing.T) {
	r := rows(
		with(map[string]interface{}{"id": "Q-20260708-001", "category": "CONTRARIAN", "status": "open"}),
		with(map[string]interface{}{"id": "Q-20260708-002", "category": "AFRAID", "status": "ticketed", "ticket": 7}),
	)
	s := Stats(r)
	if s.Total != 2 || s.ByCategory["CONTRARIAN"] != 1 || s.ByStatus["ticketed"] != 1 {
		t.Fatalf("stats wrong: %+v", s)
	}
}

func TestLintCLIOnRealLedger(t *testing.T) {
	real := filepath.Join("..", "..", "docs", "questions", "asked.jsonl")
	if _, err := os.Stat(real); err != nil {
		t.Skipf("no seed ledger at %s", real)
	}
	var out, errb bytes.Buffer
	if rc := Run(&out, &errb, []string{"lint", "--ledger", real}, "20260708", nil); rc != 0 {
		t.Fatalf("real ledger not clean: rc=%d out=%s err=%s", rc, out.String(), errb.String())
	}
}

func TestAbsentLedgerIsEmptyNotError(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "nope.jsonl")
	var out, errb bytes.Buffer
	if rc := Run(&out, &errb, []string{"lint", "--ledger", missing}, "20260708", nil); rc != 0 {
		t.Fatalf("absent ledger lint rc=%d", rc)
	}
	out.Reset()
	errb.Reset()
	if rc := Run(&out, &errb, []string{"next-id", "--ledger", missing, "--date", "20260708"}, "20260708", nil); rc != 0 {
		t.Fatalf("absent ledger next-id rc=%d", rc)
	}
	if got := strings.TrimSpace(out.String()); got != "Q-20260708-001" {
		t.Fatalf("next-id on empty = %q; want Q-20260708-001", got)
	}
}

func TestDedupeCheckExitCode(t *testing.T) {
	// Round-trips through Run so the exit-3 dedupe contract is covered.
	dir := t.TempDir()
	ledger := filepath.Join(dir, "asked.jsonl")
	b, _ := json.Marshal(good())
	if err := os.WriteFile(ledger, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	rc := Run(&out, &errb, []string{"dedupe-check", "--ledger", ledger, "--question", "Is the witness layer itself witnessed?"}, "20260708", nil)
	if rc != 3 {
		t.Fatalf("dedupe hit rc=%d; want 3 (out=%s)", rc, out.String())
	}
	out.Reset()
	rc = Run(&out, &errb, []string{"dedupe-check", "--ledger", ledger, "--question", "An entirely unrelated novel question?"}, "20260708", nil)
	if rc != 0 {
		t.Fatalf("novel dedupe rc=%d; want 0", rc)
	}
}

func TestEnsureLabelPrints(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := Run(&out, &errb, []string{"ensure-label"}, "20260708", nil); rc != 0 {
		t.Fatalf("ensure-label rc=%d", rc)
	}
	if !strings.Contains(out.String(), "gh label create "+Label) {
		t.Fatalf("ensure-label print = %q", out.String())
	}
}
