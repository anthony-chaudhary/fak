package gateway

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// probeSessionBodyAtStep builds a Claude-Code-shaped /v1/messages body for a session that has
// run `step` turn-pairs. It mirrors the real flagship shape the head anchor targets (#1407):
//   - a stable `system` head that marks its OWN cache_control breakpoint (the cached head),
//   - alternating user/assistant messages,
//   - a RECENT message cache_control breakpoint (Claude Code marks recent turns), so the
//     first-breakpoint anchor is starved and only the head anchor can shed the middle,
//   - a few EARLY oversized tool_result turns (big file reads) that are the elidable/compactible
//     middle a real coding session accumulates.
//
// bigSteps names the (user) turn indices that carry an oversized ~24 KB tool_result.
func probeSessionBodyAtStep(t *testing.T, step int, bigSteps map[int]bool) []byte {
	t.Helper()
	type block map[string]any
	nMsgs := step * 2
	recentBpIdx := nMsgs - 2 // a recent turn carries the only message breakpoint
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		var blk block
		switch {
		case role == "user" && bigSteps[i/2]:
			// A big tool_result (file read / command output) ~24 KB — over the 16 KB elide bar.
			blk = block{"type": "tool_result", "content": strings.Repeat("file contents line of a big read. ", 720)}
		case role == "user":
			blk = block{"type": "tool_result", "content": strings.Repeat("small tool result. ", 20)}
		default:
			blk = block{"type": "text", "text": strings.Repeat("assistant reasoning and a tool call. ", 40)}
		}
		if i == recentBpIdx {
			blk["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []block{blk}})
	}
	ordered := struct {
		Model     string           `json:"model"`
		MaxTokens int              `json:"max_tokens"`
		System    []block          `json:"system"`
		Messages  []map[string]any `json:"messages"`
	}{
		Model:     "claude-sonnet-4-6",
		MaxTokens: 1024,
		System: []block{
			{"type": "text", "text": "You are a coding agent."},
			{"type": "text", "text": strings.Repeat("standing policy and tool docs. ", 120), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		Messages: msgs,
	}
	raw, err := json.Marshal(ordered)
	if err != nil {
		t.Fatalf("marshal step %d: %v", step, err)
	}
	return raw
}

// TestProbeFakFiresEarlyPerStep MEASURES (does not assert) the step at which the default
// `fak guard -- claude` path first saves fak-authored value on a realistic warm session,
// sweeping the compaction budget to test whether a lower budget fires earlier AND profitably.
// Run: go test ./internal/gateway/ -run TestProbeFakFiresEarlyPerStep -v
func TestProbeFakFiresEarlyPerStep(t *testing.T) {
	// A realistic heavy-ish coding session: big file reads at turn-pairs 1, 3, 5, 8, 12.
	bigSteps := map[int]bool{1: true, 3: true, 5: true, 8: true, 12: true}

	for _, budget := range []int{DefaultCompactHistoryBudget} {
		trace := "probe-session"
		s := anthropicPassthroughServer(budget)
		s.compactAnchorHead = true                        // default-on
		s.assumeSessionTurns = DefaultAssumedSessionTurns // default-on (calibrated to ~p90 session length)
		s.elideResultBytes = DefaultElideResultBytes      // 16384, default-on
		now := time.Now()
		s.metrics = newGatewayMetrics(now)

		var prevShed uint64
		firstFire := 0
		t.Logf("========== budget=%d ==========", budget)
		for step := 2; step <= 40; step++ {
			body := probeSessionBodyAtStep(t, step, bigSteps)
			req, err := agent.DecodeAnthropicMessagesRequest(body)
			if err != nil {
				t.Fatalf("decode step %d: %v", step, err)
			}
			bodyTokens := len(body) / 4
			// Same order as the live path: compaction, then elision. Warm trace (no wired turnsLeft).
			compFired, compReason := s.compactAnthropicRawWithReason(req, 0, trace)
			s.maybeElideAnthropicRaw(req)

			sum := s.metrics.adjudicationSummary()
			stepShed := sum.CompactionShedTokens - prevShed
			prevShed = sum.CompactionShedTokens
			if stepShed > 0 && firstFire == 0 {
				firstFire = step
			}
			if step <= 15 || stepShed > 0 {
				t.Logf("step %2d | body~%6d tok | comp_fired=%-5v reason=%-18s | step_shed=%6d | cum_shed=%d",
					step, bodyTokens, compFired, compReason, stepShed, sum.CompactionShedTokens)
			}
			// Advance served-turn depth and keep the trace warm (fires the assumed-length prior).
			s.metrics.observeHarnessCoherence(trace, now, "", compFired, "", false, false, 0, 0, 0)
		}
		t.Logf("budget=%d → FIRST fak value at step: %d (0 = never in 40 steps)", budget, firstFire)
	}
}
