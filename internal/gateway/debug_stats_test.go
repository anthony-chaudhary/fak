package gateway

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestFormatTurnDebugStats_DistinguishesAllFiveStates(t *testing.T) {
	for _, reason := range []ResetReason{
		ResetReasonHealthy, ResetReasonDecay, ResetReasonStalePrefix, ResetReasonCooldown, ResetReasonUnknown,
	} {
		d := ResetDecision{Reason: reason, Score: 0.5, ShouldReset: reason == ResetReasonStalePrefix}
		line := formatTurnDebugStats("t1", "anthropic_messages", true, "end_turn", 20, 5, 80, 0, true, d, true)
		if !strings.Contains(line, "health="+string(reason)) {
			t.Fatalf("reason %q not surfaced in: %s", reason, line)
		}
	}
}

func TestFormatTurnDebugStats_NoResetHealthIsInert(t *testing.T) {
	line := formatTurnDebugStats("t1", "anthropic_messages", false, "", 100, 0, 0, 0, false, ResetDecision{}, false)
	for _, want := range []string{"health=n/a", "reset_score=-", "recommend=-", "compact=none", "finish=unknown"} {
		if !strings.Contains(line, want) {
			t.Fatalf("missing %q in inert line: %s", want, line)
		}
	}
}

func TestFormatTurnDebugStats_CompactionAndCacheHit(t *testing.T) {
	fired := formatTurnDebugStats("t1", "w", true, "end_turn", 20, 5, 60, 20, true, ResetDecision{Reason: ResetReasonHealthy}, true)
	if !strings.Contains(fired, "compact=fired") {
		t.Fatalf("want compact=fired: %s", fired)
	}
	// cache_hit = 60/(60+20+20) = 0.60
	if !strings.Contains(fired, "cache_hit=0.60") {
		t.Fatalf("want cache_hit=0.60: %s", fired)
	}
	if !strings.Contains(fired, "recommend=no") {
		t.Fatalf("healthy verdict should render recommend=no: %s", fired)
	}
	none := formatTurnDebugStats("t1", "w", false, "end_turn", 100, 5, 0, 0, false, ResetDecision{}, false)
	if !strings.Contains(none, "compact=none") || !strings.Contains(none, "cache_hit=0.00") {
		t.Fatalf("want compact=none + cache_hit=0.00: %s", none)
	}
}

func TestFormatTurnDebugStats_FieldsAreFlattenedSingleLine(t *testing.T) {
	// trace/wire/finish are kernel-minted tokens carrying no prompt content, but a stray
	// whitespace must never split the line into two rows or break key=val parsing.
	line := formatTurnDebugStats("trace one", "wire\two", true, "stop\nnow", 1, 1, 1, 0, true, ResetDecision{Reason: ResetReasonHealthy}, true)
	if strings.ContainsAny(line, "\n\t") {
		t.Fatalf("debug line must be a single flat row: %q", line)
	}
	if !strings.Contains(line, "trace=trace_one") || !strings.Contains(line, "wire=wire_wo") {
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
	if !strings.Contains(out, "health=healthy_cache") {
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
	if !strings.Contains(out, "health=n/a") {
		t.Fatalf("an untracked session must read health=n/a: %q", out)
	}
	if s.resetHealth != nil {
		t.Fatalf("the read-only peek must not mint a record for an untracked session")
	}
}
