package routing

import (
	"math"
	"reflect"
	"testing"
)

const switchGateTol = 1e-9

func switchGatePrice(v float64) *float64 { return &v }

func switchGateApprox(a, b float64) bool { return math.Abs(a-b) <= switchGateTol }

// switchGateBase is a fully priced input: 100k warm tokens, current cached
// $0.30/M, candidate uncached $3.00/M => cache loss 100000*2.70/1e6 = $0.27.
func switchGateBase(advantage float64) SwitchGateInput {
	return SwitchGateInput{
		CurrentModel:                "frontier-a",
		CandidateModel:              "local-b",
		WarmContextTokens:           100_000,
		CurrentCachedInputPrice:     switchGatePrice(0.30),
		CandidateUncachedInputPrice: switchGatePrice(3.00),
		QualityAdvantage:            switchGatePrice(advantage),
	}
}

func TestSwitchGateActiveToolLoopAlwaysStays(t *testing.T) {
	in := switchGateBase(1e9)
	in.WarmContextTokens = 0
	in.ActiveToolLoop = true
	v := SwitchGate(in)
	if v.Decision != SwitchStay || v.Reason != SwitchReasonActiveToolLoop {
		t.Fatalf("active tool loop: got %+v, want stay/%s", v, SwitchReasonActiveToolLoop)
	}
	// Even with every signal missing, the tool loop still stays.
	v = SwitchGate(SwitchGateInput{CurrentModel: "a", CandidateModel: "b", ActiveToolLoop: true})
	if v.Decision != SwitchStay {
		t.Fatalf("active tool loop with missing signals: got %+v, want stay", v)
	}
}

func TestSwitchGateCacheLossPricing(t *testing.T) {
	const wantLoss = 0.27
	t.Run("advantage_below_loss_stays", func(t *testing.T) {
		v := SwitchGate(switchGateBase(0.10))
		if v.Decision != SwitchStay || v.Reason != SwitchReasonCacheLoss {
			t.Fatalf("got %+v, want stay/%s", v, SwitchReasonCacheLoss)
		}
		if !switchGateApprox(v.CacheLossUSD, wantLoss) {
			t.Fatalf("CacheLossUSD = %v, want %v", v.CacheLossUSD, wantLoss)
		}
		if !switchGateApprox(v.AdvantageUSD, 0.10) || !switchGateApprox(v.NetUSD, 0.10-wantLoss) {
			t.Fatalf("advantage/net = %v/%v, want 0.10/%v", v.AdvantageUSD, v.NetUSD, 0.10-wantLoss)
		}
		if len(v.MissingSignals) != 0 {
			t.Fatalf("MissingSignals = %v, want none", v.MissingSignals)
		}
	})
	t.Run("advantage_above_loss_switches", func(t *testing.T) {
		v := SwitchGate(switchGateBase(0.50))
		if v.Decision != SwitchSwitch || v.Reason != SwitchReasonAdvantage {
			t.Fatalf("got %+v, want switch/%s", v, SwitchReasonAdvantage)
		}
		if !switchGateApprox(v.CacheLossUSD, wantLoss) {
			t.Fatalf("CacheLossUSD = %v, want %v", v.CacheLossUSD, wantLoss)
		}
		if !switchGateApprox(v.NetUSD, 0.23) {
			t.Fatalf("NetUSD = %v, want 0.23", v.NetUSD)
		}
	})
}

func TestSwitchGateMissingSignals(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*SwitchGateInput)
		want []string
	}{
		{"current_cached_price", func(in *SwitchGateInput) { in.CurrentCachedInputPrice = nil },
			[]string{"current_cached_input_price"}},
		{"candidate_uncached_price", func(in *SwitchGateInput) { in.CandidateUncachedInputPrice = nil },
			[]string{"candidate_uncached_input_price"}},
		{"quality_advantage", func(in *SwitchGateInput) { in.QualityAdvantage = nil },
			[]string{"quality_advantage"}},
		{"all_three", func(in *SwitchGateInput) {
			in.CurrentCachedInputPrice = nil
			in.CandidateUncachedInputPrice = nil
			in.QualityAdvantage = nil
		}, []string{"candidate_uncached_input_price", "current_cached_input_price", "quality_advantage"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := switchGateBase(1e6) // a huge advantage must not override fail-safe
			tc.mut(&in)
			v := SwitchGate(in)
			if v.Decision != SwitchStay || v.Reason != SwitchReasonMissingSignals {
				t.Fatalf("got %+v, want stay/%s", v, SwitchReasonMissingSignals)
			}
			if !reflect.DeepEqual(v.MissingSignals, tc.want) {
				t.Fatalf("MissingSignals = %v, want %v", v.MissingSignals, tc.want)
			}
		})
	}
}

func TestSwitchGateNoCandidate(t *testing.T) {
	for _, cand := range []string{"", "frontier-a"} {
		in := switchGateBase(1e6)
		in.CandidateModel = cand
		v := SwitchGate(in)
		if v.Decision != SwitchStay || v.Reason != SwitchReasonNoCandidate {
			t.Fatalf("candidate %q: got %+v, want stay/%s", cand, v, SwitchReasonNoCandidate)
		}
	}
}

func TestSwitchGateEdges(t *testing.T) {
	t.Run("negative_warm_tokens_are_zero", func(t *testing.T) {
		in := switchGateBase(0.01)
		in.WarmContextTokens = -500_000
		v := SwitchGate(in)
		if !switchGateApprox(v.CacheLossUSD, 0) || !switchGateApprox(v.NetUSD, 0.01) {
			t.Fatalf("got %+v, want loss 0 net 0.01", v)
		}
		if v.Decision != SwitchSwitch {
			t.Fatalf("got %+v, want switch", v)
		}
	})
	t.Run("cheaper_candidate_floors_loss_at_zero", func(t *testing.T) {
		in := switchGateBase(0.05)
		in.CurrentCachedInputPrice = switchGatePrice(5.00)
		in.CandidateUncachedInputPrice = switchGatePrice(1.00)
		v := SwitchGate(in)
		if !switchGateApprox(v.CacheLossUSD, 0) {
			t.Fatalf("CacheLossUSD = %v, want 0", v.CacheLossUSD)
		}
		if !switchGateApprox(v.NetUSD, 0.05) || v.Decision != SwitchSwitch {
			t.Fatalf("got %+v, want switch with net 0.05", v)
		}
	})
	t.Run("net_equal_min_advantage_stays", func(t *testing.T) {
		// Zero warm tokens keeps the arithmetic exact: net == advantage.
		in := switchGateBase(0.25)
		in.WarmContextTokens = 0
		in.MinAdvantage = 0.25
		v := SwitchGate(in)
		if v.Decision != SwitchStay || v.Reason != SwitchReasonCacheLoss {
			t.Fatalf("net == min: got %+v, want stay/%s", v, SwitchReasonCacheLoss)
		}
		in.MinAdvantage = 0.2499
		if v := SwitchGate(in); v.Decision != SwitchSwitch {
			t.Fatalf("net just above min: got %+v, want switch", v)
		}
	})
	t.Run("negative_advantage_stays", func(t *testing.T) {
		in := switchGateBase(-0.10)
		in.WarmContextTokens = 0
		if v := SwitchGate(in); v.Decision != SwitchStay {
			t.Fatalf("got %+v, want stay", v)
		}
	})
}

func TestSwitchGateDeterministic(t *testing.T) {
	loop := switchGateBase(0.5)
	loop.ActiveToolLoop = true
	missing := switchGateBase(0.5)
	missing.QualityAdvantage = nil
	missing.CurrentCachedInputPrice = nil
	inputs := map[string]SwitchGateInput{
		"stay_cache_loss": switchGateBase(0.10),
		"switch":          switchGateBase(0.50),
		"tool_loop":       loop,
		"missing":         missing,
		"no_candidate":    {CurrentModel: "a", CandidateModel: "a"},
	}
	for name, in := range inputs {
		t.Run(name, func(t *testing.T) {
			a, b := SwitchGate(in), SwitchGate(in)
			if !reflect.DeepEqual(a, b) {
				t.Fatalf("non-deterministic verdict: %+v vs %+v", a, b)
			}
		})
	}
}
