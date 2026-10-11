package gateway

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

func compressedWarmupStats() agent.KVMemoryStats {
	return agent.KVMemoryStats{FitBudgetBytes: 1000000000, BytesPerToken: 13760, FixedBytesPerStream: 60893696}
}

func compressedWarmupController(streams int) *AdmissionController {
	policy := DefaultAdmissionPolicy()
	policy.MaxNumSeqs = streams
	return NewAdmissionController(policy)
}

// fak-test:runtime fast est=10ms
func TestWarmupCompressedKVAdapters(t *testing.T) {
	for _, capacityField := range []string{"fit", "free", "total"} {
		t.Run(capacityField, func(t *testing.T) {
			st := compressedWarmupStats()
			if capacityField != "fit" {
				st.FitBudgetBytes = 0
				if capacityField == "free" {
					st.CapacityFreeBytes = 1000000000
				} else {
					st.CapacityTotalBytes = 1000000000
				}
			}
			s := newTestServer(t)
			s.planner = &warmupReporterPlanner{MockPlanner: agent.NewMockPlanner("compressed"), stats: st}
			ctl := compressedWarmupController(2)
			// Bind directly here so the two fallback adapters are witnessed before
			// SetAdmissionController or RunWarmup can populate the cached capacity.
			s.admissionCtl = ctl
			want := kvbudget.WarmupCapacity{UsableBytes: 1000000000, BytesPerToken: st.BytesPerToken, FixedBytesPerStream: st.FixedBytesPerStream, MaxConcurrentStreams: 2}
			cap, ok := s.WarmupCapacity()
			if !ok || cap != want {
				t.Fatalf("fallback capacity=%+v found=%v want=%+v", cap, ok, want)
			}
			derived, ok := s.WarmupDerivedBudget()
			wantBudget := want.DeriveTokenBudget(kvbudget.DefaultReserveFraction)
			if !ok || derived != wantBudget || derived.FixedStateAmount != 2*st.FixedBytesPerStream {
				t.Fatalf("fallback derive=%+v found=%v want=%+v", derived, ok, wantBudget)
			}
			s.ArmWarmupGate()
			if _, err := s.RunWarmup(context.Background()); err != nil {
				t.Fatal(err)
			}
			if s.warmupCapacity == nil || *s.warmupCapacity != want {
				t.Fatalf("RunWarmup capacity=%+v want=%+v", s.warmupCapacity, want)
			}
			if ctl.Policy().TokenBudget != int(wantBudget.TokenBudget) || ctl.TokenBudgetProvenance() != "measured" {
				t.Fatalf("RunWarmup policy=%+v want budget=%d", ctl.Policy(), wantBudget.TokenBudget)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestWarmupCompressedKVControllerInitializationOrders(t *testing.T) {
	for _, first := range []string{"capacity", "controller", "reporter"} {
		t.Run(first, func(t *testing.T) {
			st := compressedWarmupStats()
			cap := kvbudget.WarmupCapacity{UsableBytes: st.FitBudgetBytes, BytesPerToken: st.BytesPerToken, FixedBytesPerStream: st.FixedBytesPerStream, MaxConcurrentStreams: 99}
			s := &Server{}
			ctl := compressedWarmupController(2)
			switch first {
			case "capacity":
				s.SetWarmupCapacity(cap)
				s.SetAdmissionController(ctl)
			case "controller":
				s.SetAdmissionController(ctl)
				s.SetWarmupCapacity(cap)
			case "reporter":
				s.planner = &warmupReporterPlanner{MockPlanner: agent.NewMockPlanner("compressed"), stats: st}
				s.SetAdmissionController(ctl)
			}
			cap.MaxConcurrentStreams = 2
			want := cap.DeriveTokenBudget(kvbudget.DefaultReserveFraction)
			if got := ctl.Policy().TokenBudget; got != int(want.TokenBudget) {
				t.Fatalf("budget=%d want=%d", got, want.TokenBudget)
			}
			if got, ok := s.WarmupCapacity(); !ok || got.MaxConcurrentStreams != 2 {
				t.Fatalf("cached capacity=%+v found=%v", got, ok)
			}
			if err := s.fixedWarmupAdmissionError(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestWarmupCompressedKVRefusalFailsClosed(t *testing.T) {
	for _, tc := range []struct {
		name          string
		streams       int
		usable, slope int64
		reason        kvbudget.Reason
	}{
		{"unbounded", 0, 1000000000, 13760, kvbudget.ReasonUnboundedStreamState},
		{"fixed_exhausts_room", 2, 100000000, 13760, kvbudget.ReasonBelowOneUnit},
		{"no_measurement", 2, 0, 13760, kvbudget.ReasonNoMeasuredCapacity},
		{"no_slope", 2, 1000000000, 0, kvbudget.ReasonInvalidUnitSize},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := compressedWarmupStats()
			st.FitBudgetBytes = tc.usable
			st.BytesPerToken = tc.slope
			s := newTestServer(t)
			s.planner = &warmupReporterPlanner{MockPlanner: agent.NewMockPlanner("compressed"), stats: st}
			ctl := compressedWarmupController(tc.streams)
			s.admissionCtl = ctl
			if derived, ok := s.WarmupDerivedBudget(); ok || derived.Reason != tc.reason {
				t.Fatalf("uncached fixed refusal=%+v found=%v want=%s", derived, ok, tc.reason)
			}
			s.SetAdmissionController(ctl)
			// A usable per-block fallback must not erase a known fixed-state refusal.
			s.SetWarmupBlockCapacity(kvbudget.WarmupBlockCapacity{FittedBlocks: 10000, BlockTokens: 16})
			derived, ok := s.WarmupDerivedBudget()
			if ok || derived.Reason != tc.reason || derived.TokenBudget != 0 {
				t.Fatalf("derive=%+v found=%v want=%s", derived, ok, tc.reason)
			}
			s.ArmWarmupGate()
			_, err := s.RunWarmup(context.Background())
			var refusal *fixedWarmupCapacityError
			if !errors.As(err, &refusal) || refusal.reason != tc.reason {
				t.Fatalf("warmup err=%v want=%s", err, tc.reason)
			}
			if !s.warmup.pending() {
				t.Fatal("refused fixed-state warmup released readiness")
			}
			lease, err := s.beginServedAdmission(context.Background(), servedSessionTurn{}, nil, nil, 1)
			if lease != nil || !errors.As(err, &refusal) {
				t.Fatalf("served admission escaped refusal: lease=%v err=%v", lease, err)
			}
			status, code, _, mapped := admissionErrorStatus(err)
			if !mapped || status != http.StatusServiceUnavailable || code != "warmup_capacity_unavailable" {
				t.Fatalf("served mapping=(%d,%s,%v)", status, code, mapped)
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := s.beginServedAdmission(ctx, servedSessionTurn{}, nil, nil, 1); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation lost to capacity refusal: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestWarmupCompressedKVPreservesBudgetPrecedence(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		st := compressedWarmupStats()
		s := &Server{planner: &warmupReporterPlanner{MockPlanner: agent.NewMockPlanner("compressed"), stats: st}}
		ctl := compressedWarmupController(2)
		if explicit {
			ctl.SetTokenBudgetWithProvenance(77, "explicit")
		}
		s.SetAdmissionController(ctl)
		before := ctl.Policy()
		s.SetWarmupBlockCapacity(kvbudget.WarmupBlockCapacity{FittedBlocks: 10000, BlockTokens: 16})
		if got := ctl.Policy(); got.TokenBudget != before.TokenBudget || got.TokenBudgetProvenance != before.TokenBudgetProvenance {
			t.Fatalf("block-only fallback overwrote fixed-aware policy: before=%+v after=%+v", before, got)
		}
		if explicit && (before.TokenBudget != 77 || before.TokenBudgetProvenance != "explicit") {
			t.Fatalf("explicit budget overwritten: %+v", before)
		}
		// Explicit token limits do not authorize an impossible fixed-state fit.
		s.SetWarmupCapacity(kvbudget.WarmupCapacity{UsableBytes: 1, BytesPerToken: st.BytesPerToken, FixedBytesPerStream: st.FixedBytesPerStream})
		if err := s.fixedWarmupAdmissionError(); err == nil {
			t.Fatal("fixed-state refusal escaped explicit/default budget")
		}
	}
}

// fak-test:runtime fast est=1ms
func TestWarmupCompressedKVUsesSelectedControllerBound(t *testing.T) {
	st := compressedWarmupStats()
	cap := kvbudget.WarmupCapacity{UsableBytes: st.FitBudgetBytes, BytesPerToken: st.BytesPerToken, FixedBytesPerStream: st.FixedBytesPerStream}
	bounded := compressedWarmupController(2)
	for _, ctl := range []*AdmissionController{bounded, nil, compressedWarmupController(0)} {
		err := fixedWarmupCapacityAdmissionError(&cap, ctl, kvbudget.DefaultReserveFraction)
		if ctl == bounded {
			if err != nil {
				t.Fatal(err)
			}
		} else {
			var refusal *fixedWarmupCapacityError
			if !errors.As(err, &refusal) || refusal.reason != kvbudget.ReasonUnboundedStreamState {
				t.Fatalf("selected detached/unbounded controller escaped: %v", err)
			}
		}
	}
}

// fak-test:runtime fast est=1ms
func TestWarmupCompressedKVZeroFixedExplicitUnlimitedUnchanged(t *testing.T) {
	st := agent.KVMemoryStats{FitBudgetBytes: 100000, BytesPerToken: 10}
	s := &Server{planner: &warmupReporterPlanner{MockPlanner: agent.NewMockPlanner("uniform"), stats: st}}
	ctl := compressedWarmupController(0)
	ctl.SetTokenBudgetWithProvenance(0, "explicit")
	s.SetAdmissionController(ctl)
	s.SetWarmupCapacity(kvbudget.WarmupCapacity{UsableBytes: st.FitBudgetBytes, BytesPerToken: st.BytesPerToken})
	if got := ctl.Policy(); got.TokenBudget != 0 || got.TokenBudgetProvenance != "explicit" {
		t.Fatalf("zero-fixed explicit unlimited budget changed: %+v", got)
	}
	lease, err := s.beginServedAdmission(context.Background(), servedSessionTurn{traceID: "uniform-explicit-unlimited"}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if lease == nil {
		t.Fatal("explicit unlimited admission did not acquire a lease")
	}
}

// fak-test:runtime fast est=10ms
func TestWarmupCompressedKVWaitsForBoundedController(t *testing.T) {
	s := newTestServer(t)
	s.planner = &warmupReporterPlanner{MockPlanner: agent.NewMockPlanner("compressed"), stats: compressedWarmupStats()}
	s.ArmWarmupGate()
	if _, err := s.RunWarmup(context.Background()); err == nil {
		t.Fatal("fixed warmup without stream controller succeeded")
	}
	s.SetAdmissionController(compressedWarmupController(2))
	if _, err := s.RunWarmup(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s.warmup.pending() {
		t.Fatal("bounded retry did not release readiness")
	}
	s.SetAdmissionController(nil)
	if err := s.fixedWarmupAdmissionError(); err == nil {
		t.Fatal("detached stream bound admitted fixed state")
	}
}

// fak-test:runtime fast est=1ms
func TestWarmupCompressedKVZeroFixedUnchanged(t *testing.T) {
	for _, streams := range []int{0, 1, 256} {
		st := agent.KVMemoryStats{FitBudgetBytes: 100000, BytesPerToken: 10}
		s := &Server{planner: &warmupReporterPlanner{MockPlanner: agent.NewMockPlanner("uniform"), stats: st}}
		ctl := compressedWarmupController(streams)
		s.SetAdmissionController(ctl)
		legacy := kvbudget.WarmupCapacity{UsableBytes: 100000, BytesPerToken: 10}
		got, ok := s.WarmupCapacity()
		if !ok || got != legacy {
			t.Fatalf("streams=%d capacity=%+v found=%v want=%+v", streams, got, ok, legacy)
		}
		derived, ok := s.WarmupDerivedBudget()
		if !ok || derived != legacy.DeriveTokenBudget(kvbudget.DefaultReserveFraction) || ctl.Policy().TokenBudget != 9000 {
			t.Fatalf("streams=%d derive=%+v policy=%+v", streams, derived, ctl.Policy())
		}
		if err := s.fixedWarmupAdmissionError(); err != nil {
			t.Fatal(err)
		}
	}
}
