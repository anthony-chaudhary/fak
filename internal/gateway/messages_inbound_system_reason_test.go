package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

// messages_inbound_system_reason_test.go — the #5446 half-B acceptance, driven through the
// REAL gateway consumer rather than the promptmmu helper. Before this, the system[] prune's
// entire output was discarded at its single call site, and the reader that fronts it
// answered a bare empty list from all four of its exits — so a turn that pruned nothing
// because there was no system[] array and a turn that pruned nothing because the blocks
// could not be read were indistinguishable from outside the process. A promptmmu-only unit
// test cannot discharge that; it is the consumer-less shape #5435 tracks.

// inboundSystemBodyWith builds a wire-shaped /v1/messages body carrying the given `system`
// value verbatim, so a test can hand the seam an ordinary body, a bare-string one, or one
// whose system[] is array-shaped and still unreadable.
func inboundSystemBodyWith(t *testing.T, system any) []byte {
	t.Helper()
	type obj map[string]any
	raw, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024, "stream": true,
		"system":   system,
		"messages": []obj{{"role": "user", "content": []obj{{"type": "text", "text": "hi"}}}},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// driveInboundSystemPrune runs the EXACT composition the serve path runs (the prune, then
// the emitter that consumes its result — see messages_transform.go) over a decoded wire
// body, and returns the record, the emitted log lines, and the possibly-rewritten request.
func driveInboundSystemPrune(t *testing.T, drop func(block, name string) bool, raw []byte) (inboundSystemPrune, []string, *agent.AnthropicMessagesRequest) {
	t.Helper()
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	var lines []string
	s := &Server{
		planner:         &agent.HTTPPlanner{Provider: agent.ProviderAnthropic},
		systemBlockDrop: drop,
		logf:            func(format string, args ...any) { lines = append(lines, fmt.Sprintf(format, args...)) },
	}
	rec := s.maybeCompactInboundSystem(req)
	s.logInboundSystemPrune(rec)
	return rec, lines, req
}

func dropStaleSkill(block, name string) bool {
	return block == promptmmu.BlockSkills && name == "old_skill"
}

// TestInboundSystemStructuralReadFailureIsNotBenignIdle is the acceptance: a body whose
// `system` value OPENS WITH `[` and still fails to read must reach the gateway's observable
// surface as a STRUCTURAL failure, distinct from the far more common "there was no system[]
// array" idle — which must stay silent, because the idle turn is the dominant shape and a
// log full of it would carry no signal.
func TestInboundSystemStructuralReadFailureIsNotBenignIdle(t *testing.T) {
	// STRUCTURAL: system IS an array — it opens with `[`, and its first element is a
	// perfectly good anchored block — but the second element is a bare number, not a typed
	// block. This is live-wire reachable, and it is where the system path's read actually
	// fails (the promptmmu spine below is fed an already-validated document).
	structuralRaw := inboundSystemBodyWith(t, []any{
		map[string]any{
			"type": "text", "name": "policy", "block": promptmmu.BlockSystem, "text": "resident policy",
			"cache_control": map[string]any{"type": "ephemeral"},
		},
		5,
	})
	// BENIGN: a bare-string `system`. A legitimate wire shape and the ordinary idle turn.
	benignRaw := inboundSystemBodyWith(t, "a plain string system")

	structural, structuralLog, structuralReq := driveInboundSystemPrune(t, dropStaleSkill, structuralRaw)
	benign, benignLog, benignReq := driveInboundSystemPrune(t, dropStaleSkill, benignRaw)

	if structural.SkipReason != promptmmu.SkipUndecodableSystem {
		t.Errorf("unreadable system[]: SkipReason = %q, want %q", structural.SkipReason, promptmmu.SkipUndecodableSystem)
	}
	if !structural.Structural {
		t.Errorf("unreadable system[]: Structural = false, want true — nothing about this turn is routine")
	}
	if benign.SkipReason != promptmmu.SkipNoSystem {
		t.Errorf("bare-string system: SkipReason = %q, want %q", benign.SkipReason, promptmmu.SkipNoSystem)
	}
	if benign.Structural {
		t.Errorf("bare-string system: Structural = true, want false — it is a legitimate wire shape")
	}

	// THE acceptance: the two must not share one observable outcome at the real call site.
	if structural.SkipReason == benign.SkipReason {
		t.Fatalf("an unreadable system[] and an absent one collapsed into one reason %q — "+
			"a reader regression would hide inside the expected-large benign bucket", structural.SkipReason)
	}

	// And the reason must actually REACH an operator, not merely be computed and dropped.
	if len(structuralLog) != 1 {
		t.Fatalf("unreadable system[]: emitted %d log lines, want exactly 1: %v", len(structuralLog), structuralLog)
	}
	line := structuralLog[0]
	if !strings.Contains(line, promptmmu.SkipUndecodableSystem) || !strings.Contains(line, "structural=true") {
		t.Errorf("log line does not name the structural failure: %q", line)
	}
	if len(benignLog) != 0 {
		t.Fatalf("the dominant idle turn must stay silent, got %v", benignLog)
	}

	// Both are fail-safe: neither outcome may rewrite the body.
	if !bytes.Equal(structuralReq.Raw, structuralRaw) {
		t.Error("an unreadable system[] must leave req.Raw byte-identical")
	}
	if !bytes.Equal(benignReq.Raw, benignRaw) {
		t.Error("a bare-string system must leave req.Raw byte-identical")
	}
}

// TestInboundSystemPruneReachesTheLog pins the other half of the same surface: a turn that
// really did prune names what it removed and reports no skip reason, so an operator can tell
// a working prune from a silent one. Before #5446 the pruned names were built and thrown
// away by the caller.
func TestInboundSystemPruneReachesTheLog(t *testing.T) {
	rec, lines, req := driveInboundSystemPrune(t, dropStaleSkill, inboundSystemBody(t))

	want := promptmmu.BlockSkills + ":old_skill"
	if len(rec.Pruned) != 1 || rec.Pruned[0] != want {
		t.Fatalf("Pruned = %v, want [%s]", rec.Pruned, want)
	}
	if rec.SkipReason != "" || rec.Structural {
		t.Errorf("a real prune must carry no skip reason, got %+v", rec)
	}
	if len(lines) != 1 {
		t.Fatalf("a real prune emitted %d log lines, want exactly 1: %v", len(lines), lines)
	}
	if !strings.Contains(lines[0], want) || !strings.Contains(lines[0], "structural=false") {
		t.Errorf("log line does not name the pruned block: %q", lines[0])
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("pruned body failed to re-decode: %v", err)
	}
}

// TestInboundSystemNothingDroppableIsBenignAndSilent: a well-formed system[] that simply
// holds nothing the floor denies is the expected-large shape. It must be named (never left
// blank) but must stay off the structural side and off the log.
func TestInboundSystemNothingDroppableIsBenignAndSilent(t *testing.T) {
	rec, lines, _ := driveInboundSystemPrune(t,
		func(block, name string) bool { return false }, inboundSystemBody(t))

	if rec.SkipReason != promptmmu.SkipEmptyPlan {
		t.Errorf("SkipReason = %q, want %q", rec.SkipReason, promptmmu.SkipEmptyPlan)
	}
	if rec.Structural {
		t.Error("nothing droppable is the ordinary idle turn, not a fak fault")
	}
	if len(lines) != 0 {
		t.Fatalf("the idle turn must stay silent, got %v", lines)
	}
}

// TestInboundSystemReaderNamesEveryExit: the reader that fronts the prune must name a
// closed-set reason on every empty-handed exit, and "" exactly when it produced work. A
// blank reason on any exit is how the false-benign reading came back the first time.
func TestInboundSystemReaderNamesEveryExit(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  []byte
		drop func(block, name string) bool
		want string
	}{
		{"nil-predicate", inboundSystemBody(t), nil, inboundPruneDisabled},
		{"not-json", []byte(`not json at all`), dropStaleSkill, promptmmu.SkipNotJSONObject},
		{"no-system", []byte(`{"model":"m","messages":[]}`), dropStaleSkill, promptmmu.SkipNoSystem},
		{"null-system", []byte(`{"model":"m","system":null}`), dropStaleSkill, promptmmu.SkipNoSystem},
		{"unreadable", inboundSystemBodyWith(t, []any{5}), dropStaleSkill, promptmmu.SkipUndecodableSystem},
		{"nothing-droppable", inboundSystemBody(t), func(string, string) bool { return false }, promptmmu.SkipEmptyPlan},
	} {
		plans, reason := inboundSystemBlockPlans(tc.raw, tc.drop)
		if len(plans) != 0 {
			t.Errorf("%s: got %d plans, want none", tc.name, len(plans))
		}
		if reason != tc.want {
			t.Errorf("%s: reason = %q, want %q", tc.name, reason, tc.want)
		}
	}
	plans, reason := inboundSystemBlockPlans(inboundSystemBody(t), dropStaleSkill)
	if len(plans) != 1 || reason != "" {
		t.Fatalf("a real plan set = %d plans reason %q, want 1 plan and no reason", len(plans), reason)
	}
}
