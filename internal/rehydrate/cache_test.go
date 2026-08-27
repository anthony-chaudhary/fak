package rehydrate

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

func TestCacheProjectionRefusesColdPrefixAndPricesRewarm(t *testing.T) {
	projection := NewCacheProjection(resume.Input{
		IdleSeconds:           301,
		TTL:                   resume.TTL5m,
		EffectiveReuseSeconds: 300,
		ResidentTokens:        256,
		ShedBudgetTokens:      128,
		HorizonTurns:          4,
		Pricing:               resume.Pricing{InputPerMTokUSD: 1, OutputPerMTokUSD: 1},
	})
	admission := NewGate(projection.Rung()).Admit(context.Background(), dormancy.Cool)

	if admission.Admitted || admission.RefusedBy != ColdCache {
		t.Fatalf("cold cache admission = %+v, want COLD_CACHE refusal", admission)
	}
	if !strings.Contains(admission.Detail, "COLD_TTL") {
		t.Fatalf("cold cache detail = %q, want COLD_TTL break reason", admission.Detail)
	}
	if projection.Report.Posture != resume.PostureCold {
		t.Fatalf("posture = %q, want cold", projection.Report.Posture)
	}
	if projection.Report.Recommended != resume.StrategyCut {
		t.Fatalf("recommended = %q, want %q re-warm plan", projection.Report.Recommended, resume.StrategyCut)
	}
	var full, cut float64
	for _, strategy := range projection.Report.Strategies {
		switch strategy.Strategy {
		case resume.StrategyResumeFull:
			full = strategy.HorizonUSD
		case resume.StrategyCut:
			cut = strategy.HorizonUSD
		}
	}
	if full <= cut {
		t.Fatalf("cold pricing did not charge the full-prefix rewrite: resume=%v cut=%v", full, cut)
	}
}

func TestCacheProjectionKeepsUnderTTLPrefixWarm(t *testing.T) {
	projection := NewCacheProjection(resume.Input{
		IdleSeconds:           299,
		TTL:                   resume.TTL5m,
		EffectiveReuseSeconds: 300,
		ResidentTokens:        256,
		ShedBudgetTokens:      128,
		HorizonTurns:          4,
		Pricing:               resume.Pricing{InputPerMTokUSD: 1, OutputPerMTokUSD: 1},
	})
	admission := NewGate(projection.Rung()).Admit(context.Background(), dormancy.Cool)

	if !admission.Admitted || admission.RefusedBy != ReasonOK {
		t.Fatalf("warm cache admission = %+v, want admitted", admission)
	}
	if projection.Report.Posture != resume.PostureWarm {
		t.Fatalf("posture = %q, want warm", projection.Report.Posture)
	}
	if projection.Report.Recommended != resume.StrategyResumeFull {
		t.Fatalf("recommended = %q, want %q", projection.Report.Recommended, resume.StrategyResumeFull)
	}
}
