package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=300ms
func TestRequestPhaseNativeDecodeCancellationRetainsElapsed(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	planner := NewInKernelPlanner(m, nil, "synthetic-request-phase", false, nil, false)
	parent := enginestep.WithRequestObservation(context.Background(), "request-decode", "load-one")
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	_, _, _, _, _, _, err := planner.generateReusedContext(ctx, []int{1, 2}, 4, 0, 1, 1, nil, func(int) bool { cancel(); return false })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("native cancellation=%v", err)
	}
	got := enginestep.RequestPhaseObservationsFromContext(ctx)
	if got.Prefill == nil || got.Prefill.Count != 1 || got.Decode == nil || got.Decode.Count != 1 || got.Decode.DurationNS < 0 {
		t.Fatalf("native partial phases=%+v", got)
	}
}

// fak-test:runtime fast est=300ms
func TestRequestPhaseNativePreEntryCancellationUnknown(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	planner := NewInKernelPlanner(m, nil, "synthetic-prefill-phase", false, nil, false)
	parent := enginestep.WithRequestObservation(context.Background(), "request-prefill", "load-one")
	cancelled, cancel := context.WithCancel(parent)
	cancel()
	if _, _, _, _, _, _, err := planner.generateReusedContext(cancelled, []int{1, 2}, 4, 0, 1, 1, nil, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("early cancellation=%v", err)
	}
	if got := enginestep.RequestPhaseObservationsFromContext(cancelled); got.Prefill != nil || got.Decode != nil {
		t.Fatal("pre-entry cancellation measured phases")
	}
}
