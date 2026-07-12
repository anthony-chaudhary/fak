package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// metaRow is a terse constructor for an "improve" journal row in the meta CLI tests —
// the same shape internal/rsiloop's own meta tests use.
func metaRow(decision string, kept, truthClean, suiteGreen bool) rsiloop.Row {
	return rsiloop.Row{Mode: "improve", Decision: decision, Kept: kept, TruthClean: truthClean, SuiteGreen: suiteGreen}
}

// writeJournal writes rows as a JSONL rsiloop journal under dir and returns its path.
func writeJournal(t *testing.T, dir, name string, rows []rsiloop.Row) string {
	t.Helper()
	path := filepath.Join(dir, name)
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			t.Fatalf("encode row: %v", err)
		}
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("write journal %s: %v", path, err)
	}
	return path
}

// clusteredJournal is a 10-cycle improve journal with 2 clustered ESCALATEs (enough to
// trip DefaultMetaConfig's MinEscalations) and a truth-clean keep-rate of 0.2 (2/10) —
// the low baseline the witnessed-KEEP/REVERT cases measure a witness journal against.
func clusteredJournal() []rsiloop.Row {
	return []rsiloop.Row{
		metaRow("KEEP", true, true, true),
		metaRow("KEEP", true, true, true),
		metaRow("REVERT", false, true, true),
		metaRow("ESCALATE", false, false, true),
		metaRow("REVERT", false, true, true),
		metaRow("ESCALATE", false, false, true),
		metaRow("REVERT", false, true, true),
		metaRow("REVERT", false, true, true),
		metaRow("REVERT", false, true, true),
		metaRow("REVERT", false, true, true),
	}
}

// decodeMeta parses runMeta's single stdout JSON object.
func decodeMeta(t *testing.T, out *bytes.Buffer) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(out.Bytes(), &m); err != nil {
		t.Fatalf("runMeta emitted non-JSON stdout %q: %v", out.String(), err)
	}
	return m
}

// TestRunMeta_ProposeOnlyEmitsProposalJSON is acceptance bullet 1: -mode meta folds a
// bounded proposal and emits it as JSON, and NOTHING is applied without --apply.
func TestRunMeta_ProposeOnlyEmitsProposalJSON(t *testing.T) {
	dir := t.TempDir()
	journal := writeJournal(t, dir, "loop.jsonl", clusteredJournal())

	var out, errOut bytes.Buffer
	code := runMeta(metaOptions{
		journalPath: journal,
		cur:         rsiloop.KeepPolicy{GainThreshold: 0.10},
		cfg:         rsiloop.DefaultMetaConfig(),
	}, &out, &errOut)

	if code != 0 {
		t.Fatalf("propose-only exit = %d, want 0 (stderr: %s)", code, errOut.String())
	}
	m := decodeMeta(t, &out)
	if m["has_proposal"] != true {
		t.Fatalf("clustered journal produced no proposal: %v", m)
	}
	if m["witnessed"] != false || m["applied"] != false {
		t.Fatalf("propose-only must not witness or apply: %v", m)
	}
	prop, ok := m["proposal"].(map[string]any)
	if !ok {
		t.Fatalf("proposal object missing: %v", m)
	}
	if prop["knob"] != "gain_threshold" {
		t.Errorf("proposal knob = %v, want gain_threshold", prop["knob"])
	}
	before, after := prop["before"].(float64), prop["after"].(float64)
	if !(after > before) {
		t.Errorf("proposal did not tighten: before=%v after=%v", before, after)
	}
}

// TestRunMeta_NoProposalOnQuietJournal: a journal without clustered escalation proposes
// nothing and still emits a stable JSON object at exit 0 (mutating nothing).
func TestRunMeta_NoProposalOnQuietJournal(t *testing.T) {
	dir := t.TempDir()
	quiet := []rsiloop.Row{
		metaRow("KEEP", true, true, true),
		metaRow("ESCALATE", false, false, true),
		metaRow("KEEP", true, true, true),
	}
	journal := writeJournal(t, dir, "quiet.jsonl", quiet)

	var out, errOut bytes.Buffer
	code := runMeta(metaOptions{journalPath: journal, cfg: rsiloop.DefaultMetaConfig()}, &out, &errOut)
	if code != 0 {
		t.Fatalf("no-proposal exit = %d, want 0", code)
	}
	if m := decodeMeta(t, &out); m["has_proposal"] != false {
		t.Fatalf("a single escalation tripped a proposal: %v", m)
	}
}

// TestRunMeta_ApplyWithoutWitnessRefuses is acceptance bullet 2: --apply with no witness
// journal surfaces the library's own refusal and exits with the usage code, not KEEP.
func TestRunMeta_ApplyWithoutWitnessRefuses(t *testing.T) {
	dir := t.TempDir()
	journal := writeJournal(t, dir, "loop.jsonl", clusteredJournal())

	var out, errOut bytes.Buffer
	code := runMeta(metaOptions{
		journalPath: journal,
		apply:       true, // but witnessJournal is empty
		cur:         rsiloop.KeepPolicy{GainThreshold: 0.10},
		cfg:         rsiloop.DefaultMetaConfig(),
	}, &out, &errOut)

	if code != 2 {
		t.Fatalf("--apply without a witness exit = %d, want 2 (stderr: %s)", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "witness") {
		t.Fatalf("refusal did not name the missing witness: %q", errOut.String())
	}
}

// TestRunMeta_WitnessedKeepApplies is acceptance bullet 4 (KEEP): a witness journal whose
// truth-clean keep-rate strictly rises over the observed journal is KEPT, and with --apply
// the policy lands. The keep-bit is the library's non-forgeable shipgate.Evaluate.
func TestRunMeta_WitnessedKeepApplies(t *testing.T) {
	dir := t.TempDir()
	journal := writeJournal(t, dir, "loop.jsonl", clusteredJournal()) // rate 0.2
	witness := writeJournal(t, dir, "witness.jsonl", []rsiloop.Row{   // rate 0.667
		metaRow("KEEP", true, true, true),
		metaRow("KEEP", true, true, true),
		metaRow("KEEP", true, true, true),
		metaRow("KEEP", true, true, true),
		metaRow("REVERT", false, true, true),
		metaRow("REVERT", false, true, true),
	})

	var out, errOut bytes.Buffer
	code := runMeta(metaOptions{
		journalPath:    journal,
		witnessJournal: witness,
		apply:          true,
		cur:            rsiloop.KeepPolicy{GainThreshold: 0.10},
		cfg:            rsiloop.DefaultMetaConfig(),
	}, &out, &errOut)

	if code != 0 {
		t.Fatalf("witnessed KEEP exit = %d, want 0 (stderr: %s)", code, errOut.String())
	}
	m := decodeMeta(t, &out)
	if m["decision"] != "KEEP" || m["applied"] != true {
		t.Fatalf("witnessed KEEP did not apply: %v", m)
	}
	policy := m["policy"].(map[string]any)
	if policy["gain_threshold"].(float64) <= 0.10 {
		t.Fatalf("applied policy did not raise the gain bar past 0.10: %v", policy)
	}
}

// TestRunMeta_WitnessedRevertLeavesPolicyUntouched is acceptance bullet 3 + 4 (REVERT): a
// witness journal that does NOT beat the observed truth-clean keep-rate is REVERTED; the
// policy stays at the current bar and the run exits with the defined REVERT code (3).
func TestRunMeta_WitnessedRevertLeavesPolicyUntouched(t *testing.T) {
	dir := t.TempDir()
	journal := writeJournal(t, dir, "loop.jsonl", clusteredJournal()) // rate 0.2
	revert := make([]rsiloop.Row, 0, 10)                              // rate 0.1
	revert = append(revert, metaRow("KEEP", true, true, true))
	for i := 0; i < 9; i++ {
		revert = append(revert, metaRow("REVERT", false, true, true))
	}
	witness := writeJournal(t, dir, "witness.jsonl", revert)

	var out, errOut bytes.Buffer
	code := runMeta(metaOptions{
		journalPath:    journal,
		witnessJournal: witness,
		apply:          true,
		cur:            rsiloop.KeepPolicy{GainThreshold: 0.10},
		cfg:            rsiloop.DefaultMetaConfig(),
	}, &out, &errOut)

	if code != 3 {
		t.Fatalf("witnessed REVERT exit = %d, want 3 (stderr: %s)", code, errOut.String())
	}
	m := decodeMeta(t, &out)
	if m["decision"] != "REVERT" || m["applied"] != false {
		t.Fatalf("witnessed REVERT should not apply: %v", m)
	}
	policy := m["policy"].(map[string]any)
	if policy["gain_threshold"].(float64) != 0.10 {
		t.Fatalf("REVERT changed the policy off 0.10: %v", policy)
	}
}
