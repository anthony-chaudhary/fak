package session

import (
	"math"
	"testing"
	"time"
)

func TestParseBudgetEnvelopeCompactSyntax(t *testing.T) {
	env, err := ParseBudgetEnvelope("turns=12,tokens=5000,context=64000,wall=90m,spend=$12.34,throughput=25/s,min-tps=10tps,max-tokens=512,gap=250ms,queries=3")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	b := env.SessionBudget()
	if b.TurnsLeft != 12 || b.TokensLeft != 5000 || b.ContextTokensLeft != 64000 {
		t.Fatalf("budget = %+v, want turns=12 tokens=5000 context=64000", b)
	}
	if b.ContextTokensCap != 64000 {
		t.Fatalf("context cap = %d, want 64000", b.ContextTokensCap)
	}
	if b.ClarificationQueriesLeft != 3 || b.ClarificationQueriesCap != 3 {
		t.Fatalf("query budget = %+v, want left/cap=3", b)
	}
	if got := env.WallClockLimit(); got != 90*time.Minute {
		t.Fatalf("wall clock = %s, want 90m", got)
	}
	if env.Spend.Currency != "USD" || env.Spend.MaxCents != 1234 {
		t.Fatalf("spend = %+v, want USD 1234 cents", env.Spend)
	}
	if math.Abs(env.Throughput.ExpectedTokensPerSec-25) > 1e-9 || math.Abs(env.Throughput.MinTokensPerSec-10) > 1e-9 {
		t.Fatalf("throughput = %+v, want expected=25 min=10", env.Throughput)
	}
	if env.Pace.MaxTokensPerTurn != 512 || env.Pace.MinTurnGapMs != 250 {
		t.Fatalf("pace = %+v, want max=512 gap=250ms", env.Pace)
	}
}

func TestBudgetEnvelopeDefaultsUnboundedTurnAndOutputAxes(t *testing.T) {
	env, err := ParseBudgetEnvelope("context=off,wall=off,spend=0,throughput=0/s")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	b := env.SessionBudget()
	if b.TurnsLeft != Unbounded || b.TokensLeft != Unbounded {
		t.Fatalf("default turn/output budget = %+v, want unbounded axes", b)
	}
	if b.ContextTokensLeft != 0 || env.WallClockLimit() != 0 {
		t.Fatalf("off context/wall = context %d wall %s, want off", b.ContextTokensLeft, env.WallClockLimit())
	}
	if env.TimeBudget().Bounded() {
		t.Fatalf("wall=off should project to an unbounded TimeBudget")
	}
}

func TestBudgetEnvelopeProjectionToSessionTypes(t *testing.T) {
	env, err := ParseBudgetEnvelope("turns=inf,tokens=250,context=1000,wall=2h,max_tokens_per_turn=128,gap_ms=75,throughput=40tps")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	if got := env.SessionBudget(); got.TurnsLeft != Unbounded || got.TokensLeft != 250 || got.ContextTokensCap != 1000 {
		t.Fatalf("SessionBudget = %+v, want unbounded turns/tokens=250/context cap=1000", got)
	}
	if got := env.TimeBudget(); !got.Bounded() || got.LimitNanos != int64(2*time.Hour) {
		t.Fatalf("TimeBudget = %+v, want 2h bounded", got)
	}
	if got := env.SessionPace(); got.MaxTokensPerTurn != 128 || got.MinTurnGapMs != 75 {
		t.Fatalf("SessionPace = %+v, want max=128 gap=75", got)
	}
	if got := env.ExpectedThroughput(); got.ExpectedTokensPerSec != 40 {
		t.Fatalf("ExpectedThroughput = %+v, want expected=40", got)
	}
}

func TestParseBudgetEnvelopeRejectsMalformedSyntax(t *testing.T) {
	for _, spec := range []string{
		"",
		"turns",
		"unknown=1",
		"wall=soon",
		"spend=1.234",
		"throughput=fast",
		"max-tokens=-1",
	} {
		if _, err := ParseBudgetEnvelope(spec); err == nil {
			t.Fatalf("ParseBudgetEnvelope(%q) succeeded, want error", spec)
		}
	}
}
