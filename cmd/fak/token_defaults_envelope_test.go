package main

// token_defaults_envelope_test.go — the acceptance lock for #2947: the token-defaults scorecard
// must report the effective-context envelope's provenance (hard cap, min evidence floor, target,
// effective ceiling, witnessed/observed/modeled/fallback) with ZERO raw-window-target debt, and
// both front doors (guard/serve) must stay wired to the same envelope-derived resident defaults so
// one cannot silently drift away from the other. All assertions read the REAL scorecard corpus and
// the entrypoint source, never a roster copy.

import (
	"strings"
	"testing"
)

// TestTokenDefaults_EnvelopeProvenanceReported pins the envelope/provenance fields into the
// scorecard JSON corpus: every default context envelope exposes hard cap, min evidence floor,
// target, effective ceiling, and a provenance label, and none is a raw-window target.
func TestTokenDefaults_EnvelopeProvenanceReported(t *testing.T) {
	c := collectTokenDefaultsScorecard("../..")["corpus"].(map[string]any)

	if got := c["raw_window_target_debt"].(int); got != 0 {
		t.Errorf("raw_window_target_debt = %d, want 0 (no default may target the raw provider window without a witness)", got)
	}

	env, ok := c["context_envelope"].([]map[string]any)
	if !ok || len(env) == 0 {
		t.Fatalf("corpus.context_envelope must be a non-empty slice of envelope rows, got %T (len depends on ctxplan.DefaultEnvelopes)", c["context_envelope"])
	}

	valid := map[string]bool{"WITNESSED": true, "OBSERVED": true, "MODELED": true, "FALLBACK": true}
	for i, row := range env {
		for _, k := range []string{"hard_context_cap", "min_viable_evidence_tokens", "target_resident_tokens", "max_effective_tokens", "safe_cap"} {
			if _, has := row[k]; !has {
				t.Errorf("envelope row %d missing required field %q", i, k)
			}
		}
		prov, _ := row["provenance"].(string)
		if !valid[prov] {
			t.Errorf("envelope row %d has invalid provenance %q (want witnessed/observed/modeled/fallback)", i, prov)
		}
		if row["raw_window_target"].(bool) {
			t.Errorf("envelope row %d (%v) targets the raw provider window without a witness", i, row["task_class"])
		}
		// The doctrine's core rule: the target resident budget is held below the advertised cap.
		if row["target_resident_tokens"].(int) >= row["hard_context_cap"].(int) {
			t.Errorf("envelope row %d target %d must stay below the hard cap %d", i, row["target_resident_tokens"], row["hard_context_cap"])
		}
	}
}

// TestTokenDefaults_FrontDoorsShareEnvelopeDefaults locks each front door to its own
// envelope-governed posture const so neither door can drift (the issue's explicit
// "one cannot drift away from the other" acceptance). The ctxview budget (the planned
// resident view) and the compaction budget are the two envelope-governed sizes. The
// 2026-11-10 hosting-simplification flip split the posture: guard keeps the armed
// library consts (agent.DefaultCtxViewBudget / gateway.DefaultCompactHistoryBudget),
// serve resolves its own raw serving-router consts (gateway.DefaultServe*). Each door
// must wire its OWN const.
func TestTokenDefaults_FrontDoorsShareEnvelopeDefaults(t *testing.T) {
	guard := readEntrypoint(t, "guard.go")
	serve := readEntrypoint(t, "serve.go")
	for _, wire := range []struct {
		guard, serve string
	}{
		{guard: `fs.Int("ctx-view-budget", agent.DefaultCtxViewBudget`, serve: `fs.Int("ctx-view-budget", gateway.DefaultServeCtxViewBudget`},
		{guard: `fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget`, serve: `fs.Int("compact-history-budget", gateway.DefaultServeCompactHistoryBudget`},
	} {
		if !strings.Contains(guard, wire.guard) {
			t.Errorf("guard.go must wire the envelope-governed armed default: %s", wire.guard)
		}
		if !strings.Contains(serve, wire.serve) {
			t.Errorf("serve.go must wire the envelope-governed raw serving-router default (2026-11-10 hosting-simplification flip): %s", wire.serve)
		}
	}
}
