package sessionobs

import (
	"strings"
	"testing"
)

// jsonl joins transcript lines into one JSONL stream.
func jsonl(lines ...string) string { return strings.Join(lines, "\n") + "\n" }

func TestFoldTranscriptShipped(t *testing.T) {
	tr := jsonl(
		// one billed assistant turn: Edit (mutating) + Bash + Read (read-only)
		`{"type":"assistant","message":{"id":"m1","role":"assistant","model":"claude-opus","usage":{"output_tokens":120},"content":[{"type":"tool_use","name":"Edit","input":{}},{"type":"tool_use","name":"Bash","input":{}},{"type":"tool_use","name":"Read","input":{}}]}}`,
		// a DUPLICATE serialization of the same billed turn -> must be de-duped
		`{"type":"assistant","message":{"id":"m1","role":"assistant","usage":{"output_tokens":120},"content":[]}}`,
		// the git-commit success marker lands verbatim in a tool_result
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"[main 809c5339] feat(x): do the thing\n 3 files changed, 10 insertions(+)","is_error":false}]}}`,
		// a /goal directive
		`{"type":"user","message":{"role":"user","content":"<command-name>/goal</command-name> ship the thing"}}`,
		// a second turn that was interrupted
		`{"type":"assistant","message":{"id":"m2","role":"assistant","usage":{"output_tokens":5},"stop_reason":"interrupted","content":[]}}`,
		// a torn / malformed final line must be skipped, not abort the fold
		`not json {`,
	)
	rec, ev := FoldTranscript(strings.NewReader(tr), FoldMeta{SessionID: "s1", Namespace: "C--work-fak", Account: "day24"})

	if rec.AssistantTurns != 2 {
		t.Errorf("AssistantTurns: dup must be de-duped, want 2 got %d", rec.AssistantTurns)
	}
	if rec.ToolCalls != 3 {
		t.Errorf("ToolCalls want 3 got %d", rec.ToolCalls)
	}
	if rec.ReadOnlyCalls != 1 {
		t.Errorf("ReadOnlyCalls want 1 (Read) got %d", rec.ReadOnlyCalls)
	}
	if rec.OutputTokens != 125 {
		t.Errorf("OutputTokens want 125 got %d", rec.OutputTokens)
	}
	if !ev.Mutated {
		t.Errorf("Mutated should be true (Edit ran)")
	}
	if len(ev.CommitSHAs) != 1 || ev.CommitSHAs[0] != "809c5339" {
		t.Errorf("CommitSHAs want [809c5339] got %v", ev.CommitSHAs)
	}
	if ev.GoalEvents != 1 {
		t.Errorf("GoalEvents want 1 got %d", ev.GoalEvents)
	}
	if ev.Interrupts != 1 {
		t.Errorf("Interrupts want 1 (stop_reason interrupted) got %d", ev.Interrupts)
	}
	if rec.Signals.Commits != 1 || rec.Signals.GoalEvents != 1 || rec.Signals.Interrupts != 1 {
		t.Errorf("Signals not promoted from evidence: %+v", rec.Signals)
	}
	// committed + in-history -> Shipped; committed + not-in-history -> Claimed.
	if got := Classify(ev, true); got != OutcomeShipped {
		t.Errorf("Classify(witnessed=true) want shipped got %v", got)
	}
	if got := Classify(ev, false); got != OutcomeClaimed {
		t.Errorf("Classify(witnessed=false) want claimed got %v", got)
	}
}

func TestFoldTranscriptNoOp(t *testing.T) {
	tr := jsonl(
		`{"type":"assistant","message":{"id":"a","usage":{"output_tokens":40},"content":[{"type":"tool_use","name":"Read","input":{}},{"type":"tool_use","name":"Grep","input":{}}]}}`,
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"some file contents","is_error":false}]}}`,
	)
	rec, ev := FoldTranscript(strings.NewReader(tr), FoldMeta{SessionID: "s2"})
	if ev.Mutated || ev.committed() || ev.stopped() {
		t.Fatalf("read-only session should have no mutation/commit/stop evidence: %+v", ev)
	}
	if rec.ReadOnlyCalls != 2 {
		t.Errorf("ReadOnlyCalls want 2 got %d", rec.ReadOnlyCalls)
	}
	if got := Classify(ev, false); got != OutcomeNoOp {
		t.Errorf("read-only session should classify NoOp, got %v", got)
	}
}

func TestFoldTranscriptStopped(t *testing.T) {
	tr := jsonl(
		`{"type":"assistant","message":{"id":"a","usage":{"output_tokens":80},"content":[{"type":"tool_use","name":"Write","input":{}}]}}`,
		// a blocked Stop hook reminder, no commit ever landed -> waste side
		`{"type":"user","message":{"role":"user","content":"A Stop hook is active and will block stopping until the condition holds."}}`,
	)
	_, ev := FoldTranscript(strings.NewReader(tr), FoldMeta{SessionID: "s3"})
	if !ev.Mutated {
		t.Errorf("Write should mark Mutated")
	}
	if ev.committed() {
		t.Errorf("no commit marker present; committed() must be false")
	}
	if !ev.stopped() {
		t.Errorf("a blocked-Stop reminder should set a stop mark")
	}
	if got := Classify(ev, false); got != OutcomeStopped {
		t.Errorf("mutated-but-no-commit + stop should classify Stopped, got %v", got)
	}
}

func TestFoldTranscriptUnknownWhenMutatedNoSignal(t *testing.T) {
	// Mutated, no commit, no stop signal -> honestly Unknown, never guessed waste.
	tr := jsonl(
		`{"type":"assistant","message":{"id":"a","usage":{"output_tokens":50},"content":[{"type":"tool_use","name":"Edit","input":{}}]}}`,
	)
	_, ev := FoldTranscript(strings.NewReader(tr), FoldMeta{SessionID: "s4"})
	if got := Classify(ev, false); got != OutcomeUnknown {
		t.Errorf("mutated with no terminal signal should stay Unknown, got %v", got)
	}
}

func TestFoldCountsToolErrors(t *testing.T) {
	tr := jsonl(
		`{"type":"assistant","message":{"id":"a","usage":{"output_tokens":10},"content":[{"type":"tool_use","name":"Bash","input":{}}]}}`,
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"command failed: exit 1","is_error":true}]}}`,
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"ok","is_error":false}]}}`,
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"another failure","is_error":true}]}}`,
	)
	rec, ev := FoldTranscript(strings.NewReader(tr), FoldMeta{SessionID: "te"})
	if ev.ToolErrors != 2 {
		t.Errorf("ToolErrors want 2 (two is_error results) got %d", ev.ToolErrors)
	}
	if rec.Signals.ToolErrors != 2 {
		t.Errorf("Signals.ToolErrors must be promoted from evidence, want 2 got %d", rec.Signals.ToolErrors)
	}
}

func TestFoldCountsGuardRefusals(t *testing.T) {
	tr := jsonl(
		// the canonical Anthropic-wire denial note, prepended by the gateway as assistant text
		`{"type":"assistant","message":{"id":"g1","role":"assistant","usage":{"output_tokens":10},"content":[{"type":"text","text":"[fak] refused 1 tool call(s): Write (DEFAULT_DENY/TERMINAL). Do not re-propose a refused call unchanged; choose an allowed alternative."},{"type":"tool_use","name":"Edit","input":{}}]}}`,
		// the fak-unaware wire all-denied summary (no [fak] prefix)
		`{"type":"assistant","message":{"id":"g2","role":"assistant","usage":{"output_tokens":8},"content":[{"type":"text","text":"All proposed tool calls were refused by the fak kernel: Bash: DENY (POLICY_BLOCK/TERMINAL)"}]}}`,
		// QUARANTINE-EXCLUDED: a result page-out note. It is a real gateway banner but a
		// DIFFERENT phenomenon (an inbound result held out, often a false positive) that fires
		// per-result -- 150+ times in a single real session -- so it must NOT count as a
		// tool-call denial. guard_refusals tracks denials only; this turn contributes 0.
		`{"type":"assistant","message":{"id":"g3","role":"assistant","usage":{"output_tokens":6},"content":[{"type":"text","text":"[fak] Heads up: 1 inbound tool result was held out of context as a safety precaution (paged out, not lost)."}]}}`,
		// a benign assistant turn -- a non-refusal [fak] note must NOT count as friction
		`{"type":"assistant","message":{"id":"g4","role":"assistant","usage":{"output_tokens":4},"content":[{"type":"text","text":"[fak] compacted 12 turns. Looks good, shipping."}]}}`,
		// HARDENING PROOF: a META/ANALYSIS turn that casually mentions "[fak] refused" in its
		// own prose but never reproduces the full banner. The old short-prefix marker would
		// have counted this (and inflated to hundreds across an analysis session); the
		// distinctive-phrase marker correctly reads it as talk ABOUT friction, not friction.
		`{"type":"assistant","message":{"id":"g5","role":"assistant","usage":{"output_tokens":7},"content":[{"type":"text","text":"As discussed, the [fak] refused note appears when the kernel denies a call; nothing was actually denied here."}]}}`,
		// PER-TURN PROOF: one turn whose content carries the deny banner in TWO text blocks
		// still counts as ONE guard-friction event (the gateway emits one note per turn).
		`{"type":"assistant","message":{"id":"g6","role":"assistant","usage":{"output_tokens":9},"content":[{"type":"text","text":"refused 2 tool call(s): Bash (POLICY_BLOCK/TERMINAL). Do not re-propose a refused call unchanged; choose an allowed alternative."},{"type":"text","text":"Continuing. Do not re-propose a refused call unchanged."}]}}`,
		// a user tool_result that merely QUOTES the full banner (a Read of this file or a gateway
		// test fixture) lands in a user block -- it must NOT be mistaken for a real refusal.
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"// fixture: ... Do not re-propose a refused call unchanged; choose an allowed alternative.","is_error":false}]}}`,
	)
	rec, ev := FoldTranscript(strings.NewReader(tr), FoldMeta{SessionID: "gr"})
	// g1 deny + g2 all-denied + g6 two-blocks-one-turn = 3. g3 quarantine, g4 benign, g5
	// casual mention, and the quoting user block all correctly contribute 0.
	if ev.GuardRefusals != 3 {
		t.Errorf("GuardRefusals want 3 (deny + kernel-deny + one per-turn; quarantine excluded), got %d", ev.GuardRefusals)
	}
	if rec.Signals.GuardRefusals != 3 {
		t.Errorf("Signals.GuardRefusals must be promoted from evidence, want 3 got %d", rec.Signals.GuardRefusals)
	}
	// NON-CIRCULARITY: a guard refusal is a behavior FEATURE the contrast ranks, never a
	// cohort definer. Even with an Edit (Mutated) and refusals present but no stop/commit, the
	// session must NOT classify Stopped on the strength of the refusals alone -- it stays Unknown.
	if ev.stopped() {
		t.Errorf("guard refusals must not set stopped(); they are a feature, not a waste definition")
	}
	if got := Classify(ev, false); got == OutcomeStopped {
		t.Errorf("a guard refusal alone must not classify Stopped, got %v", got)
	}
}

func TestFoldExtractsMultipleSortedSHAs(t *testing.T) {
	tr := jsonl(
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"[main ffff111] one"}]}}`,
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"[main 0000aaa] two"}]}}`,
		// repeat of the first SHA must not duplicate
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"[main ffff111] one again"}]}}`,
	)
	_, ev := FoldTranscript(strings.NewReader(tr), FoldMeta{SessionID: "s5"})
	if len(ev.CommitSHAs) != 2 {
		t.Fatalf("want 2 distinct SHAs got %v", ev.CommitSHAs)
	}
	if ev.CommitSHAs[0] != "0000aaa" || ev.CommitSHAs[1] != "ffff111" {
		t.Errorf("SHAs must be sorted+deduped, got %v", ev.CommitSHAs)
	}
}
