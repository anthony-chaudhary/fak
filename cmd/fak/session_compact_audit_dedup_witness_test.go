package main

// session_compact_audit_dedup_witness_test.go — the CAUSAL witness for #5254: does the
// shipped cross-turn dedup actually move the metric the issue is scored on?
//
// #5254's DoD item 3 asks for a field re-run of `fak session compact-audit` showing
// `REPEATED_TOOL_RESULT` window share and `tool_result/*` dup_bytes materially reduced.
// Two facts make that field number unreachable today, both measured rather than assumed
// (docs/notes/COMPACTION-REPEATED-TOOL-RESULT-DEDUP-2026-07-19.md and
// internal/session/testdata/compactaudit/guarded-cohort-witness-2026-08-06.md):
//
//  1. Only ~3.8% of the audited Codex rollout corpus ever crossed fak's wire, so an
//     unscoped before/after moves for reasons unrelated to the port.
//  2. Scoped to the fak-routed cohort, there are 2 post-port rollouts carrying 0
//     compaction fires — no post-port window exists to measure.
//
// So the field number is blocked on POPULATION. What was never witnessed at all is the
// causal link the DoD's number is a proxy for: that folding repeated tool-result spans
// reduces the very quantity `compact-audit` reports. That link spans two packages that
// cannot see each other — internal/session is tier 1 and internal/agent is tier 4, so
// session may not import the elider — which is why this witness lives in cmd/fak, the
// tier-4 shell that already depends on both.
//
// The test drives the REAL objects end to end: it renders a Codex-shaped rollout whose
// post-fire window carries byte-identical shell tool-result spans under distinct call
// ids, scans it with the REAL session.ScanCompactRollout, then folds the same bodies
// with the REAL agent.ElideMessages at the REAL default threshold and re-scans. The
// before/after delta is the proof.
//
// Scope note, stated so this witness is not over-read. It measures the transform at the
// point fak can act: the decoded []Message wire the gateway forwards upstream. It is NOT
// a claim about the on-disk rollout corpus. A Codex rollout is the CLIENT's own
// append-only transcript, written before fak ever sees the turn and never written back
// to by the gateway, so a gateway-side fold cannot retroactively shrink rollout rows.
// That is a property of where the two artifacts sit, not a shortcoming of the port, and
// it is the reason DoD item 3's instrument reads a population fak mostly cannot touch.

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// dedupWitnessToolResults is how many byte-identical shell results the post-fire window
// carries. It must exceed the elider's protected recent working set (which is deliberately
// left verbatim) by enough that the eligible band dominates, so the residual after folding
// is visibly the protected tail rather than a partial transform.
const dedupWitnessToolResults = 20

// dedupWitnessShellBody is one shell tool result: comfortably over
// session.RegrowthDupMinBytes (so a repeat of it registers as duplication at all) and well
// over the elider's line/byte span floors (so it is foldable). Shaped like the
// tool_result/shell_command rows the 2026-07-18 attribution witness measured.
func dedupWitnessShellBody() string {
	var b strings.Builder
	b.WriteString("Exit code: 0\nOutput:\n")
	for i := 0; i < 48; i++ {
		fmt.Fprintf(&b, "internal/gateway/responses_%02d.go:%d: handleResponses decodes the turn and forwards it\n", i, 100+i)
	}
	return b.String()
}

// dedupWitnessRollout renders a Codex rollout whose single compaction fire is followed by
// one function_call / function_call_output pair per body. Row shapes match the fixtures in
// internal/session/testdata/compactaudit (the audit's own corpus), so the scanner classifies
// these exactly as it classifies the field corpus.
func dedupWitnessRollout(t *testing.T, bodies []string) string {
	t.Helper()
	var lines []string
	tick := 0
	row := func(typ string, payload map[string]any) {
		tick++
		raw, err := json.Marshal(map[string]any{
			"timestamp": fmt.Sprintf("2026-08-06T10:%02d:%02d.000Z", tick/60, tick%60),
			"type":      typ,
			"payload":   payload,
		})
		if err != nil {
			t.Fatalf("marshal %s row: %v", typ, err)
		}
		lines = append(lines, string(raw))
	}
	tokens := func(resident, cumulative int) {
		row("event_msg", map[string]any{
			"type": "token_count",
			"info": map[string]any{
				"total_token_usage":    map[string]any{"input_tokens": cumulative},
				"last_token_usage":     map[string]any{"input_tokens": resident},
				"model_context_window": 258400,
			},
		})
	}

	row("session_meta", map[string]any{"id": "sess-5254-witness", "cwd": "C:\\work\\fak"})
	row("event_msg", map[string]any{"type": "task_started"})
	tokens(250000, 250000) // pre-fire: the window is full
	row("compacted", map[string]any{"message": ""})
	row("event_msg", map[string]any{"type": "context_compacted"})
	tokens(25000, 275000) // post-fire: the fire shed, and the regrowth window opens here

	// The regrowth the issue is about: the same output re-entering the window it was just
	// compacted out of, under a DIFFERENT call id each time.
	for i, body := range bodies {
		callID := fmt.Sprintf("call_shell_%02d", i)
		row("response_item", map[string]any{
			"type":      "function_call",
			"name":      "shell_command",
			"call_id":   callID,
			"arguments": fmt.Sprintf(`{"command":["rg","handleResponses","--glob","*_%02d.go"]}`, i),
		})
		row("response_item", map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  body,
		})
	}

	tokens(120000, 395000) // the window refilled out of those rows
	return strings.Join(lines, "\n") + "\n"
}

// scanDedupWitness scans a rendered rollout and returns the single fire's regrowth block.
func scanDedupWitness(t *testing.T, rollout string) *session.CompactRegrowth {
	t.Helper()
	rep, err := session.ScanCompactRollout(strings.NewReader(rollout), "mem", int64(len(rollout)))
	if err != nil {
		t.Fatalf("scan rollout: %v", err)
	}
	if len(rep.Fires) != 1 {
		t.Fatalf("fires = %d, want exactly 1 (the compacted/context_compacted pair)", len(rep.Fires))
	}
	reg := rep.Fires[0].Regrowth
	if reg == nil {
		t.Fatal("fire carried no regrowth block — the post-fire telemetry did not bind")
	}
	return reg
}

func hasDedupWitnessAnomaly(list []string, want string) bool {
	for _, a := range list {
		if a == want {
			return true
		}
	}
	return false
}

// TestCompactAuditDedupCutsRepeatedToolResultClass is the causal line of #5254: the shipped
// decoded-wire fold reduces the exact class `compact-audit` scores the issue on.
//
// BEFORE is the defect the issue reports — a post-fire window whose shell results are
// byte-identical under distinct call ids, which the audit flags REPEATED_TOOL_RESULT.
// AFTER runs those same bodies through agent.ElideMessages at gateway.DefaultElideResultBytes
// (the shipped default, armed from cmd/fak/serve.go and cmd/fak/guard.go) and re-renders.
// Every body is UNDER that 16 KB threshold, so head+tail elision is structurally blind to
// them: any reduction seen here is attributable to the #5254 cross-turn dedup alone.
func TestCompactAuditDedupCutsRepeatedToolResultClass(t *testing.T) {
	body := dedupWitnessShellBody()
	if len(body) < session.RegrowthDupMinBytes {
		t.Fatalf("witness body is %d bytes, under the audit's %d-byte duplication floor — it could never register as a repeat", len(body), session.RegrowthDupMinBytes)
	}
	if len(body) > gateway.DefaultElideResultBytes {
		t.Fatalf("witness body is %d bytes, OVER the %d-byte head+tail threshold — the reduction would not be attributable to dedup", len(body), gateway.DefaultElideResultBytes)
	}

	bodies := make([]string, dedupWitnessToolResults)
	for i := range bodies {
		bodies[i] = body
	}

	const class = session.RegrowClassToolResPrefix + "shell_command"

	before := scanDedupWitness(t, dedupWitnessRollout(t, bodies))
	beforeStat := before.Classes[class]
	if beforeStat == nil {
		t.Fatalf("no %s class in the pre-fold window; classes = %v", class, before.Classes)
	}
	if !hasDedupWitnessAnomaly(before.Anomalies, session.AnomalyRepeatedToolResult) {
		t.Fatalf("pre-fold anomalies = %v, want %s — the witness does not reproduce the defect it is meant to measure", before.Anomalies, session.AnomalyRepeatedToolResult)
	}
	if beforeStat.DupRows != dedupWitnessToolResults-1 {
		t.Errorf("pre-fold dup_rows = %d, want %d (every repeat after the first)", beforeStat.DupRows, dedupWitnessToolResults-1)
	}

	// The transform under test, called exactly as the gateway calls it.
	msgs := make([]agent.Message, 0, len(bodies))
	for i, b := range bodies {
		msgs = append(msgs, agent.Message{Role: "tool", ToolCallID: fmt.Sprintf("call_shell_%02d", i), Content: b})
	}
	folded, outcome := agent.ElideMessages(msgs, gateway.DefaultElideResultBytes)
	if outcome.Elided == 0 {
		t.Fatalf("ElideMessages folded nothing (outcome %+v) — sub-threshold repeats are exactly the #5254 class", outcome)
	}
	if len(folded) != len(msgs) {
		t.Fatalf("fold changed the message count: %d -> %d; elision must never drop a turn", len(msgs), len(folded))
	}

	foldedBodies := make([]string, len(folded))
	for i, m := range folded {
		foldedBodies[i] = m.Content
	}
	after := scanDedupWitness(t, dedupWitnessRollout(t, foldedBodies))
	afterStat := after.Classes[class]
	if afterStat == nil {
		t.Fatalf("no %s class in the post-fold window; classes = %v", class, after.Classes)
	}

	// The DoD's two quantities, measured by the audit itself.
	if afterStat.DupBytes >= beforeStat.DupBytes {
		t.Fatalf("dup_bytes did not fall: %d -> %d", beforeStat.DupBytes, afterStat.DupBytes)
	}
	cut := 1 - float64(afterStat.DupBytes)/float64(beforeStat.DupBytes)
	if cut < 0.70 {
		t.Errorf("dup_bytes cut = %.1f%% (%d -> %d), want a material reduction (>=70%%)", cut*100, beforeStat.DupBytes, afterStat.DupBytes)
	}
	if afterStat.Bytes >= beforeStat.Bytes {
		t.Errorf("class bytes did not fall: %d -> %d", beforeStat.Bytes, afterStat.Bytes)
	}
	t.Logf("#5254 causal witness: %s dup_bytes %d -> %d (-%.1f%%), dup_rows %d -> %d, class bytes %d -> %d",
		class, beforeStat.DupBytes, afterStat.DupBytes, cut*100,
		beforeStat.DupRows, afterStat.DupRows, beforeStat.Bytes, afterStat.Bytes)
}

// TestCompactAuditDedupResidualIsTheProtectedWorkingSet names what survives the fold, so the
// reduction above is never read as elimination. agent.ElideMessages deliberately leaves the
// most recent messages byte-for-byte intact — the model is still reasoning over them — so a
// window whose tail repeats a body keeps those repeats by design. Pinning the residual keeps
// a future tightening of the fold from silently eating the live working set.
func TestCompactAuditDedupResidualIsTheProtectedWorkingSet(t *testing.T) {
	body := dedupWitnessShellBody()
	bodies := make([]string, dedupWitnessToolResults)
	for i := range bodies {
		bodies[i] = body
	}

	msgs := make([]agent.Message, 0, len(bodies))
	for i, b := range bodies {
		msgs = append(msgs, agent.Message{Role: "tool", ToolCallID: fmt.Sprintf("call_shell_%02d", i), Content: b})
	}
	folded, _ := agent.ElideMessages(msgs, gateway.DefaultElideResultBytes)

	// The control that makes the cut ATTRIBUTABLE: the identical body, appearing once with
	// nothing to match against, is left alone. It is under the head+tail threshold, so the
	// pre-#5254 elider saw exactly this and did nothing — which is why a size gate is
	// structurally blind to a class whose defining property is repetition.
	lone := make([]agent.Message, len(msgs))
	copy(lone, msgs)
	for i := range lone {
		var b strings.Builder
		for line := 0; line < 48; line++ {
			fmt.Fprintf(&b, "run %02d line %02d: no other result shares this text\n", i, line)
		}
		lone[i].Content = b.String()
		if len(lone[i].Content) < session.RegrowthDupMinBytes {
			t.Fatalf("control body %d is only %d bytes — too small to be a like-for-like control", i, len(lone[i].Content))
		}
	}
	if _, loneOutcome := agent.ElideMessages(lone, gateway.DefaultElideResultBytes); loneOutcome.Elided != 0 {
		t.Errorf("non-repeating sub-threshold bodies were shrunk (%+v) — the reduction would not be attributable to dedup", loneOutcome)
	}

	// The earliest occurrence is never folded: the bytes stay reachable in-band, which is
	// what makes this dedup lossless-by-relocation rather than a deletion.
	if folded[0].Content != body {
		t.Errorf("the earliest occurrence was folded — the pointer target must stay verbatim")
	}

	verbatimTail := 0
	for i := len(folded) - 1; i > 0 && folded[i].Content == body; i-- {
		verbatimTail++
	}
	if verbatimTail == 0 {
		t.Fatal("no verbatim tail survived — the protected recent working set was folded away")
	}
	foldedCount := 0
	for _, m := range folded {
		if m.Content != body {
			foldedCount++
		}
	}
	if foldedCount == 0 {
		t.Fatal("nothing was folded at all")
	}
	if foldedCount <= verbatimTail {
		t.Errorf("folded %d bodies but left %d verbatim — the eligible band should dominate the protected tail", foldedCount, verbatimTail)
	}
	t.Logf("#5254 fold shape: %d/%d bodies folded, %d verbatim in the protected recent window, source kept at index 0",
		foldedCount, len(folded), verbatimTail)
}
