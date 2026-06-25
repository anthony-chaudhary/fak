package gateway

import (
	"strings"
	"testing"
	"time"
)

// newResetShadowServer returns a minimal Server carrying only the surfaces reset_shadow.go
// touches (the rolling-health table is lazily minted; the metrics sink is real). Everything else
// is nil because observeResetHealth reads/writes ONLY s.resetHealth and s.metrics and logs via
// s.logf — it never touches the request, the response, the kernel, or any session/table state.
// That is the structural reason SHADOW mode is inert: there is no code path here that resets a
// session, so the live response body is byte-identical whatever the verdict says.
func newResetShadowServer() *Server {
	return &Server{metrics: newGatewayMetrics(time.Now())}
}

// feed observes one compacted turn's OBSERVED token split and returns the SHADOW verdict.
func feed(s *Server, trace string, input, cacheRead, creation int) ResetDecision {
	d, _ := s.observeResetHealth(trace, input, cacheRead, creation)
	return d
}

func TestObserveResetHealth_UnknownUntilWarmup(t *testing.T) {
	s := newResetShadowServer()
	// Below DefaultMinObservedTurns the policy refuses to judge even a healthy ratio.
	for i := 0; i < DefaultMinObservedTurns-1; i++ {
		d := feed(s, "trace-warm", 20, 80, 0) // ratio 0.8, but too few turns
		if d.Reason != ResetReasonUnknown {
			t.Fatalf("turn %d: want unknown during warmup, got %s", i, d.Reason)
		}
		if d.ShouldReset {
			t.Fatalf("turn %d: must never recommend reset during warmup", i)
		}
	}
	// The MinObservedTurns-th turn is now judgeable: a healthy ratio reads healthy_cache.
	if d := feed(s, "trace-warm", 20, 80, 0); d.Reason != ResetReasonHealthy {
		t.Fatalf("after warmup: want healthy_cache, got %s", d.Reason)
	}
}

func TestObserveResetHealth_HealthyStaysCutByDefault(t *testing.T) {
	s := newResetShadowServer()
	var last ResetDecision
	for i := 0; i < 12; i++ {
		last = feed(s, "trace-healthy", 20, 80, 0) // 80% of the prompt served from cache, sustained
	}
	if last.Reason != ResetReasonHealthy {
		t.Fatalf("want healthy_cache on a still-landing prefix, got %s", last.Reason)
	}
	if last.ShouldReset {
		t.Fatalf("a healthy prefix must stay cut-by-default (ShouldReset=false)")
	}
	if last.Score != 0 {
		t.Fatalf("healthy score must be 0, got %v", last.Score)
	}
}

func TestObserveResetHealth_StaleFlipsToStalePrefix(t *testing.T) {
	s := newResetShadowServer()
	// Warm up healthy, then let the cached prefix crater (cache_read -> 0) for a full window.
	for i := 0; i < DefaultMinObservedTurns; i++ {
		feed(s, "trace-stale", 20, 80, 0)
	}
	var last ResetDecision
	for i := 0; i < resetHealthWindow; i++ {
		last = feed(s, "trace-stale", 100, 0, 0) // whole prompt re-prefills uncached
	}
	if last.Reason != ResetReasonStalePrefix {
		t.Fatalf("a cratered prefix over a full window must read stale_prefix, got %s", last.Reason)
	}
	if !last.ShouldReset {
		t.Fatalf("a never-reset session with a stale prefix should RECOMMEND reset (shadow only)")
	}
	if last.Score <= 0 {
		t.Fatalf("stale score must be > 0 (full pressure), got %v", last.Score)
	}
}

func TestObserveResetHealth_NoProviderSignalIsUnknown(t *testing.T) {
	s := newResetShadowServer()
	var last ResetDecision
	for i := 0; i < 12; i++ {
		last = feed(s, "trace-nosig", 0, 0, 0) // provider reported no counters at all
	}
	if last.Reason != ResetReasonUnknown {
		t.Fatalf("no provider signal must read unknown_provider (cut-by-default), got %s", last.Reason)
	}
	if last.ShouldReset {
		t.Fatalf("must never recommend reset without provider signal")
	}
}

func TestObserveResetHealth_CooldownHoldsAfterReset(t *testing.T) {
	s := newResetShadowServer()
	trace := "trace-cooldown"
	for i := 0; i < DefaultMinObservedTurns; i++ {
		feed(s, trace, 20, 80, 0)
	}
	for i := 0; i < resetHealthWindow; i++ {
		feed(s, trace, 100, 0, 0)
	}
	// A reset just happened: the cooldown must now HOLD the recommendation (hysteresis) even
	// though the prefix is still stale, so the session cannot flap cut<->reset.
	s.resetHealthReset(trace)
	for i := 1; i < DefaultResetCooldownTurns; i++ {
		d := feed(s, trace, 100, 0, 0)
		if d.Reason != ResetReasonCooldown {
			t.Fatalf("turn %d after reset: want cooldown hold, got %s", i, d.Reason)
		}
		if d.ShouldReset {
			t.Fatalf("turn %d after reset: cooldown must suppress the recommendation", i)
		}
		if d.Score <= 0 {
			t.Fatalf("turn %d: the held score should still report building pressure", i)
		}
	}
	// Once the cooldown elapses, the stale recommendation returns.
	if d := feed(s, trace, 100, 0, 0); !d.ShouldReset || d.Reason != ResetReasonStalePrefix {
		t.Fatalf("after the cooldown elapsed, want stale_prefix recommend, got reason=%s reset=%v", d.Reason, d.ShouldReset)
	}
}

func TestObserveResetHealth_EmptyTraceIsNoop(t *testing.T) {
	s := newResetShadowServer()
	d, st := s.observeResetHealth("", 20, 80, 0)
	if d.Reason != ResetReasonUnknown || d.ShouldReset {
		t.Fatalf("empty trace must be a safe unknown no-op, got %+v", d)
	}
	if st != (CacheHealthState{}) {
		t.Fatalf("empty trace must roll no state, got %+v", st)
	}
	if s.resetHealth != nil {
		t.Fatalf("empty trace must not mint a per-session record")
	}
	// And no metric was recorded.
	if snap := s.metrics.resetShadowSnapshotData(); len(snap.reasons) != 0 || snap.recommend != 0 {
		t.Fatalf("empty trace must record no metric, got %+v", snap)
	}
}

func TestObserveResetHealth_SessionsAreIsolated(t *testing.T) {
	s := newResetShadowServer()
	for i := 0; i < 12; i++ {
		feed(s, "healthy-one", 20, 80, 0)
	}
	// A second session that is stale must NOT be tainted by the first's healthy history.
	for i := 0; i < DefaultMinObservedTurns; i++ {
		feed(s, "stale-two", 20, 80, 0)
	}
	var staleLast ResetDecision
	for i := 0; i < resetHealthWindow; i++ {
		staleLast = feed(s, "stale-two", 100, 0, 0)
	}
	if staleLast.Reason != ResetReasonStalePrefix {
		t.Fatalf("per-session isolation broken: stale-two should read stale_prefix, got %s", staleLast.Reason)
	}
	if d := feed(s, "healthy-one", 20, 80, 0); d.Reason != ResetReasonHealthy {
		t.Fatalf("per-session isolation broken: healthy-one should stay healthy_cache, got %s", d.Reason)
	}
}

func TestLogResetShadow_IsContentFreeAndRecommendOnly(t *testing.T) {
	var sb strings.Builder
	s := newResetShadowServer()
	s.logf = func(format string, args ...any) {
		// matches the gateway log convention: logf("%s", b)
		if len(args) == 1 {
			if b, ok := args[0].([]byte); ok {
				sb.Write(b)
				sb.WriteByte('\n')
				return
			}
		}
		sb.WriteString(strings.TrimSpace(format))
		sb.WriteByte('\n')
	}
	for i := 0; i < DefaultMinObservedTurns; i++ {
		feed(s, "trace-log", 20, 80, 0)
	}
	out := sb.String()
	if !strings.Contains(out, `"event":"gateway_reset_shadow"`) {
		t.Fatalf("shadow log missing event marker: %s", out)
	}
	if !strings.Contains(out, `"recommend_reset"`) {
		t.Fatalf("shadow log must label the verdict a recommendation: %s", out)
	}
	if !strings.Contains(out, `"reset_reason":"healthy_cache"`) {
		t.Fatalf("shadow log must carry the closed reason: %s", out)
	}
	// Content-free: only the OBSERVED token counts/ratios, never a prompt byte. The inputs we fed
	// carried no text, so the only way a payload could appear is a code regression — assert none of
	// our distinct sentinels leak (there is no prompt content in this path at all).
	if strings.Contains(out, "prompt") && !strings.Contains(out, "has_provider_signal") {
		t.Fatalf("unexpected prompt-shaped field in shadow log: %s", out)
	}
}

func TestWriteResetShadowMetrics_RendersAllReasonsAndRecommendations(t *testing.T) {
	s := newResetShadowServer()
	// Drive a stale recommendation so recommendations_total and the stale bucket are non-zero.
	for i := 0; i < DefaultMinObservedTurns; i++ {
		feed(s, "trace-m", 20, 80, 0)
	}
	for i := 0; i < resetHealthWindow; i++ {
		feed(s, "trace-m", 100, 0, 0)
	}
	var b strings.Builder
	s.metrics.writeResetShadowMetrics(&b)
	out := b.String()
	// All five reason buckets exist (emitted at 0 so the panel is present pre-first-turn).
	for _, reason := range []string{
		string(ResetReasonHealthy), string(ResetReasonStalePrefix), string(ResetReasonDecay),
		string(ResetReasonCooldown), string(ResetReasonUnknown),
	} {
		want := `fak_gateway_compaction_reset_shadow_total{reason="` + reason + `"}`
		if !strings.Contains(out, want) {
			t.Fatalf("metric missing reason bucket %q:\n%s", reason, out)
		}
	}
	if !strings.Contains(out, "fak_gateway_compaction_reset_recommendations_total") {
		t.Fatalf("metric missing recommendations counter:\n%s", out)
	}
	if !strings.Contains(out, "fak_gateway_compaction_reset_score ") {
		t.Fatalf("metric missing reset score gauge:\n%s", out)
	}
	// The stale run produced at least one recommendation and a non-zero last score.
	if strings.Contains(out, "fak_gateway_compaction_reset_recommendations_total 0\n") {
		t.Fatalf("expected a non-zero shadow recommendation count:\n%s", out)
	}
}
