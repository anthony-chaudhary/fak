package bench

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

// coarseClock reports whether this host's monotonic clock cannot resolve
// sub-microsecond intervals (Windows interrupt-tick hosts step ~0.5 ms,
// quantizing every µs-band duration to 0 or a ~520 µs tick). Duration and
// distribution assertions are gated on it; structural assertions are not.
// Artifact writers refuse on a coarse clock — a quantized tail is not a
// measurement (run them under Linux, e.g. WSL, instead).
func coarseClock() bool {
	for i := 0; i < 1000; i++ {
		t0 := time.Now()
		if d := time.Since(t0); d > 0 && d < time.Microsecond {
			return false
		}
	}
	return true
}

// #2223: the composed run stamps all four bands, the causal parent chain is
// walkable B6→B4→B2→B0, and every DENY span carries an observed downstream
// consequence class — with all three classes witnessed in one run.
func TestRunFullSpan_FourBandsAndDenyClasses(t *testing.T) {
	tr, err := RunFullSpan(context.Background())
	if err != nil {
		t.Fatalf("RunFullSpan: %v", err)
	}
	if tr.Schema != FullSpanSchema {
		t.Errorf("schema = %q, want %q", tr.Schema, FullSpanSchema)
	}
	if !strings.Contains(tr.Fence, "NEVER multiplied") {
		t.Errorf("fence text must travel with the artifact; got %q", tr.Fence)
	}

	byBand := map[string][]Span{}
	for _, s := range tr.Spans {
		byBand[s.Band] = append(byBand[s.Band], s)
	}
	for _, band := range []string{"B6", "B4", "B2", "B0"} {
		if len(byBand[band]) == 0 {
			t.Fatalf("band %s has no spans; bands present: %v", band, keysOf(byBand))
		}
	}

	// The causal chain: B6 is the root, B4 follows from B6, the turn is
	// admitted by B4, and every B0 decide is contained in the turn.
	b6, b4 := byBand["B6"][0], byBand["B4"][0]
	if b6.Parent != -1 {
		t.Errorf("B6 parent = %d, want -1 (root)", b6.Parent)
	}
	if b4.Parent != b6.ID {
		t.Errorf("B4 parent = %d, want B6 id %d", b4.Parent, b6.ID)
	}
	if b4.Verdict != "SPAWN_OK" {
		t.Errorf("B4 verdict = %q, want SPAWN_OK", b4.Verdict)
	}
	var turn *Span
	for i := range byBand["B2"] {
		if byBand["B2"][i].Name == "agent turn (scripted call set)" {
			turn = &byBand["B2"][i]
			break
		}
	}
	if turn == nil {
		t.Fatal("no turn span found in band B2")
	}
	if turn.Parent != b4.ID {
		t.Errorf("turn parent = %d, want B4 id %d", turn.Parent, b4.ID)
	}
	fineClock := !coarseClock()
	for _, s := range byBand["B0"] {
		if s.Parent != turn.ID {
			t.Errorf("B0 span %d (%s) parent = %d, want turn id %d", s.ID, s.Tool, s.Parent, turn.ID)
		}
		if fineClock && s.DurNS <= 0 {
			t.Errorf("B0 span %d has non-positive duration %d", s.ID, s.DurNS)
		}
	}
	if !fineClock {
		t.Log("coarse monotonic clock: span-duration positivity not asserted on this host")
	}

	// The scripted turn's deterministic verdicts.
	verdicts := map[string]string{}
	denies := 0
	for _, s := range byBand["B0"] {
		verdicts[s.Tool] = s.Verdict
		if s.Verdict == "DENY" {
			denies++
			if s.DenyConsequence == "" {
				t.Errorf("DENY span %d (%s) carries no consequence class", s.ID, s.Tool)
			}
		}
	}
	if verdicts["get_user_details"] != "ALLOW" {
		t.Errorf("get_user_details verdict = %q, want ALLOW", verdicts["get_user_details"])
	}
	if verdicts["write_ledger_entry"] != "DENY" {
		t.Errorf("write_ledger_entry verdict = %q, want DENY", verdicts["write_ledger_entry"])
	}
	if verdicts["shell_rm_rf"] != "DENY" {
		t.Errorf("shell_rm_rf verdict = %q, want DENY", verdicts["shell_rm_rf"])
	}
	if denies != 3 {
		t.Errorf("denies = %d, want 3", denies)
	}

	// All three consequence classes observed in one run (R5 as a rate).
	classes := map[string]bool{}
	for _, da := range tr.Denies {
		classes[da.Consequence] = true
	}
	for _, want := range []string{ConsequenceRetryTurn, ConsequenceForkedOutcome, ConsequenceCleanStop} {
		if !classes[want] {
			t.Errorf("consequence class %s not observed; got %+v", want, tr.Denies)
		}
	}

	// Per-band attribution has one row per band and no cross-band ratio field —
	// the artifact's shape IS the fence (verified structurally on the JSON).
	if len(tr.PerBand) != 4 {
		t.Errorf("per-band rows = %d, want 4", len(tr.PerBand))
	}
	raw, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	assertNoForbiddenKeys(t, doc, "")
}

// The classifier is a pure fold: verify each class on a synthetic span list.
func TestAttributeDenyConsequences_Classifier(t *testing.T) {
	spans := []Span{
		{ID: 0, Parent: -1, Band: "B2", Name: "turn"},
		{ID: 1, Parent: 0, Band: "B0", Tool: "a", Verdict: "DENY", Reason: "DEFAULT_DENY", DurNS: 10},
		{ID: 2, Parent: 0, Band: "B0", Tool: "a", Verdict: "DENY", Reason: "DEFAULT_DENY", DurNS: 20},
		{ID: 3, Parent: 0, Band: "B0", Tool: "b", Verdict: "ALLOW", DurNS: 30},
		{ID: 4, Parent: 0, Band: "B2", Tool: "b", Name: "dispatch (Submit+Reap)", StartNS: 5, DurNS: 100},
		{ID: 5, Parent: 0, Band: "B0", Tool: "c", Verdict: "DENY", Reason: "POLICY_BLOCK", DurNS: 40},
	}
	got := AttributeDenyConsequences(spans)
	if len(got) != 3 {
		t.Fatalf("attributions = %d, want 3: %+v", len(got), got)
	}
	if got[0].Consequence != ConsequenceRetryTurn || got[0].InducedNS != 20 {
		t.Errorf("deny 1 = %+v, want retry_turn induced 20", got[0])
	}
	// Deny 2's follow-up is tool b: a fork, inducing b's decide AND dispatch.
	if got[1].Consequence != ConsequenceForkedOutcome || got[1].InducedNS != 130 || got[1].InducedSpans != 2 {
		t.Errorf("deny 2 = %+v, want forked_outcome induced 130 over 2 spans", got[1])
	}
	if got[2].Consequence != ConsequenceCleanStop || got[2].InducedNS != 0 {
		t.Errorf("deny 3 = %+v, want clean_stop induced 0", got[2])
	}
}

// Artifact writer, env-gated: FAK_FULLSPAN_OUT=<path> writes the trace JSON.
// Skipped in normal CI; the committed artifacts under docs/benchmarks/fullspan/
// are produced by exactly this test (see the results sheet for the command).
func TestWriteFullSpanArtifact(t *testing.T) {
	out := os.Getenv("FAK_FULLSPAN_OUT")
	if out == "" {
		t.Skip("set FAK_FULLSPAN_OUT=<path> to write the full-span trace artifact")
	}
	if coarseClock() {
		t.Skip("coarse monotonic clock (~0.5 ms tick): refusing to write a quantized artifact; run under Linux (e.g. WSL)")
	}
	tr, err := RunFullSpan(context.Background())
	if err != nil {
		t.Fatalf("RunFullSpan: %v", err)
	}
	raw, err := json.MarshalIndent(tr, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(out, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", out, err)
	}
	t.Logf("full-span trace written to %s (%d spans)", out, len(tr.Spans))
}

// assertNoForbiddenKeys walks the artifact's JSON KEYS (not values — the fence
// prose legitimately names the forbidden concept) and fails on any field that
// would carry a cross-band multiplication.
func assertNoForbiddenKeys(t *testing.T, v any, path string) {
	t.Helper()
	switch node := v.(type) {
	case map[string]any:
		for k, child := range node {
			for _, forbidden := range []string{"end_to_end", "ratio", "speedup"} {
				if strings.Contains(k, forbidden) {
					t.Errorf("artifact field %s/%s — the three-clocks fence forbids a cross-band %q field", path, k, forbidden)
				}
			}
			assertNoForbiddenKeys(t, child, path+"/"+k)
		}
	case []any:
		for _, child := range node {
			assertNoForbiddenKeys(t, child, path+"[]")
		}
	}
}

func keysOf(m map[string][]Span) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}
