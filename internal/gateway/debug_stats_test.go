package gateway

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestFormatTurnDebugStats_SurfacesHealthState(t *testing.T) {
	// The cache column reports the rolling resetScore state verbatim; a healthy read-heavy
	// turn (read 80, no write) clears the verdict to ok. stale_prefix/decay drive degraded.
	for _, tc := range []struct {
		reason  ResetReason
		reset   bool
		verdict string
	}{
		{ResetReasonHealthy, false, "ok"},
		{ResetReasonCooldown, false, "ok"},
		{ResetReasonUnknown, false, "ok"},
		{ResetReasonDecay, false, "degraded"},
		{ResetReasonStalePrefix, true, "degraded"},
	} {
		d := ResetDecision{Reason: tc.reason, Score: 0.5, ShouldReset: tc.reset}
		line := formatTurnDebugStats("t1", "anthropic_messages", true, "end_turn", 20, 5, 80, 0, true, d, true)
		if !strings.Contains(line, "cache="+string(tc.reason)) {
			t.Fatalf("reason %q not surfaced in: %s", tc.reason, line)
		}
		if !strings.Contains(line, "fak-turn trace=t1 "+tc.verdict+" ") {
			t.Fatalf("reason %q want verdict %q in: %s", tc.reason, tc.verdict, line)
		}
	}
}

func TestFormatTurnDebugStats_NoResetHealthIsInert(t *testing.T) {
	// A first turn with no cache activity and no rolling health reads cold + saved=0 + cache n/a.
	line := formatTurnDebugStats("t1", "anthropic_messages", false, "", 100, 0, 0, 0, false, ResetDecision{}, false)
	for _, want := range []string{"fak-turn trace=t1 cold ", "saved=0 tok", "cache=n/a", "compact=none", "finish=unknown"} {
		if !strings.Contains(line, want) {
			t.Fatalf("missing %q in inert line: %s", want, line)
		}
	}
}

func TestFormatTurnDebugStats_LeadsWithVerdictAndNetSaving(t *testing.T) {
	// prompt=20 (uncached input), cacheRead=60, cacheCreate=20.
	// baseline = 20+60+20 = 100 token-equiv; actual = 20 + 60*0.1 + 20*1.25 = 51; saved = 49 (49%).
	// A proven net saving on a healthy session reads ok.
	fired := formatTurnDebugStats("t1", "w", true, "end_turn", 20, 5, 60, 20, true, ResetDecision{Reason: ResetReasonHealthy}, true)
	for _, want := range []string{"fak-turn trace=t1 ok ", "saved=49 tok (49% of prompt)", "compact=fired", "cache=healthy_cache"} {
		if !strings.Contains(fired, want) {
			t.Fatalf("want %q: %s", want, fired)
		}
	}
	// The raw provider counters must be GONE from the glanceable line (the fak-vs-SOTA noise).
	for _, gone := range []string{"request_tokens", "cache_read=", "cache_creation=", "cache_hit=", "cache_rebate_tokens", "reset_score", "recommend"} {
		if strings.Contains(fired, gone) {
			t.Fatalf("raw counter %q must not appear on the glanceable line: %s", gone, fired)
		}
	}
	// A cold turn (no read, no write) reads cold + saved=0.
	none := formatTurnDebugStats("t1", "w", false, "end_turn", 100, 5, 0, 0, false, ResetDecision{}, false)
	if !strings.Contains(none, "fak-turn trace=t1 cold ") || !strings.Contains(none, "saved=0 tok") {
		t.Fatalf("want cold + saved=0: %s", none)
	}
}

func TestFormatTurnDebugStats_ColdWriteIsWarmingWithNegativeSaving(t *testing.T) {
	// A cold write the reads have not yet repaid: prompt=20, cacheRead=0, cacheCreate=100.
	// baseline = 120; actual = 20 + 0 + 100*1.25 = 145; saved = -25 (REFUTED). This is the
	// honest write-premium-aware number the old read-only rebate would have HIDDEN.
	line := formatTurnDebugStats("t1", "w", true, "end_turn", 20, 5, 0, 100, false, ResetDecision{Reason: ResetReasonHealthy}, true)
	if !strings.Contains(line, "fak-turn trace=t1 warming ") {
		t.Fatalf("a cold write with no net saving must read warming: %s", line)
	}
	if !strings.Contains(line, "saved=-25 tok") {
		t.Fatalf("an unrepaid write premium must show a NEGATIVE saving, not a phantom rebate: %s", line)
	}
}

func TestFormatTurnDebugStats_FieldsAreFlattenedSingleLine(t *testing.T) {
	// trace/finish are kernel-minted tokens carrying no prompt content, but a stray
	// whitespace must never split the line into two rows or break key=val parsing.
	line := formatTurnDebugStats("trace one", "wire\two", true, "stop\nnow", 1, 1, 1, 0, true, ResetDecision{Reason: ResetReasonHealthy}, true)
	if strings.ContainsAny(line, "\n\t") {
		t.Fatalf("debug line must be a single flat row: %q", line)
	}
	if !strings.Contains(line, "trace=trace_one") || !strings.Contains(line, "finish=stop_now") {
		t.Fatalf("fields not flattened: %s", line)
	}
}

func TestRenderTurnDebugStats_GatedOffWhenSinkNil(t *testing.T) {
	s := newResetShadowServer() // debugStatsf nil, logf nil
	// Must be a byte-identical no-op: no panic, nothing emitted, no state minted.
	s.logInferenceTurn("t1", "anthropic_messages", true, agent.Usage{PromptTokens: 20, CacheReadInputTokens: 80}, "end_turn", time.Millisecond, true)
	if s.resetHealth != nil {
		t.Fatalf("a peek-free render path must not mint per-session health")
	}
}

func TestRenderTurnDebugStats_PeeksHealthAndIsIndependentOfLogf(t *testing.T) {
	s := newResetShadowServer() // logf stays nil — proves --debug-stats works with --log off
	var sb strings.Builder
	s.debugStatsf = func(format string, args ...any) {
		fmt.Fprintf(&sb, format, args...)
		sb.WriteByte('\n')
	}
	// Build healthy rolling health on a compacted-turn path (observeResetHealth = the #792 roll).
	for i := 0; i < DefaultMinObservedTurns; i++ {
		s.observeResetHealth("t1", 20, 80, 0)
	}
	// A served turn logs: even with logf nil, the debug line must fire and carry the peeked health.
	s.logInferenceTurn("t1", "anthropic_messages", true,
		agent.Usage{PromptTokens: 20, CompletionTokens: 5, CacheReadInputTokens: 80}, "end_turn", time.Millisecond, true)
	out := sb.String()
	if !strings.Contains(out, "fak-turn ") {
		t.Fatalf("debug line did not fire with logf nil: %q", out)
	}
	if !strings.Contains(out, "cache=healthy_cache") {
		t.Fatalf("debug line must peek the rolling reset health: %q", out)
	}
	if !strings.Contains(out, "compact=fired") {
		t.Fatalf("debug line must show the compaction action: %q", out)
	}
}

func TestRenderTurnDebugStats_UntrackedSessionReadsNA(t *testing.T) {
	s := newResetShadowServer()
	var sb strings.Builder
	s.debugStatsf = func(format string, args ...any) { fmt.Fprintf(&sb, format, args...); sb.WriteByte('\n') }
	// A session that was never compacted has no rolling health: render n/a, not a phantom verdict.
	s.logInferenceTurn("never-compacted", "anthropic_messages", false,
		agent.Usage{PromptTokens: 100}, "end_turn", time.Millisecond, false)
	out := sb.String()
	if !strings.Contains(out, "cache=n/a") {
		t.Fatalf("an untracked session must read cache=n/a: %q", out)
	}
	if s.resetHealth != nil {
		t.Fatalf("the read-only peek must not mint a record for an untracked session")
	}
}

func TestChatCompletionsDebugStatsEmitsOnePayloadFreeLine(t *testing.T) {
	s := newTestServer(t)
	s.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "model payload must stay off debug stats"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:             13,
			CompletionTokens:         2,
			CacheReadInputTokens:     7,
			CacheCreationInputTokens: 3,
		},
	}}
	var lines []string
	s.debugStatsf = func(format string, args ...any) {
		lines = append(lines, fmt.Sprintf(format, args...))
	}
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	const secretPrompt = "sk-live-prompt-secret"
	var resp ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "test-model",
		Messages: []agent.Message{{Role: "user", Content: "do not log " + secretPrompt}},
	}, &resp)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200", code)
	}
	if len(lines) != 1 {
		t.Fatalf("debug lines = %d (%q), want exactly one", len(lines), strings.Join(lines, "\n"))
	}
	line := lines[0]
	// prompt=13 (uncached input), cacheRead=7, cacheCreate=3.
	// baseline = 23; actual = 13 + 7*0.1 + 3*1.25 = 17.45; saved = 5.55 -> "6" rounded (24% of prompt).
	for _, want := range []string{
		"fak-turn trace=", "saved=6 tok", "cache=", "compact=", "finish=stop",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("debug line missing %q: %s", want, line)
		}
	}
	// The raw provider counters and the prompt itself must never reach the glanceable line.
	for _, gone := range []string{"cache_read=", "cache_creation=", "request_tokens", "cache_rebate_tokens"} {
		if strings.Contains(line, gone) {
			t.Fatalf("raw counter %q must not appear on the glanceable line: %s", gone, line)
		}
	}
	for _, leak := range []string{secretPrompt, "model payload"} {
		if strings.Contains(line, leak) {
			t.Fatalf("debug line leaked payload %q: %s", leak, line)
		}
	}
}
