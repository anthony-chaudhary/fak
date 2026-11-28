package compute

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestGdnSplitvOccupancyPartition(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		cfg       GDNSplitVConfig
		wantSplit int
		wantChunk int
	}{
		{
			name: "small batch triggers splitV=4 to saturate idle SMs",
			cfg: GDNSplitVConfig{
				BatchSize: 1, Proposals: 8, NumKeyHeads: 4, NumValueHeads: 8,
				KeyHeadDim: 16, ValueHeadDim: 64, SMCount: 132, RMSNormEps: 1e-6,
			},
			wantSplit: 4,
			wantChunk: 16,
		},
		{
			name: "medium batch selects splitV=2",
			cfg: GDNSplitVConfig{
				BatchSize: 6, Proposals: 3, NumKeyHeads: 4, NumValueHeads: 8,
				KeyHeadDim: 16, ValueHeadDim: 64, SMCount: 132, RMSNormEps: 1e-6,
			},
			wantSplit: 2,
			wantChunk: 32,
		},
		{
			name: "large batch saturates SMs with splitV=1",
			cfg: GDNSplitVConfig{
				BatchSize: 16, Proposals: 0, NumKeyHeads: 4, NumValueHeads: 8,
				KeyHeadDim: 16, ValueHeadDim: 64, SMCount: 132, RMSNormEps: 1e-6,
			},
			wantSplit: 1,
			wantChunk: 64,
		},
		{
			name: "explicit forcedSplitV=4 with non-divisible valueHeadDim",
			cfg: GDNSplitVConfig{
				BatchSize: 2, Proposals: 2, NumKeyHeads: 2, NumValueHeads: 4,
				KeyHeadDim: 8, ValueHeadDim: 10, ForcedSplitV: 4, RMSNormEps: 1e-6,
			},
			wantSplit: 4,
			wantChunk: 3,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := PlanGDNSplitV(tc.cfg)
			if err != nil {
				t.Fatalf("PlanGDNSplitV: %v", err)
			}
			if plan.SplitV != tc.wantSplit {
				t.Fatalf("SplitV = %d, want %d", plan.SplitV, tc.wantSplit)
			}
			if plan.ChunkSize != tc.wantChunk {
				t.Fatalf("ChunkSize = %d, want %d", plan.ChunkSize, tc.wantChunk)
			}
			if plan.TokensPerRequest != 1+tc.cfg.Proposals {
				t.Fatalf("TokensPerRequest = %d, want %d", plan.TokensPerRequest, 1+tc.cfg.Proposals)
			}
			if len(plan.Partitions) != tc.wantSplit {
				t.Fatalf("len(Partitions) = %d, want %d", len(plan.Partitions), tc.wantSplit)
			}
			if plan.Partitions[0].VStart != 0 || plan.Partitions[len(plan.Partitions)-1].VEnd != tc.cfg.ValueHeadDim {
				t.Fatalf("partitions %+v do not span [0,%d)", plan.Partitions, tc.cfg.ValueHeadDim)
			}
		})
	}

	badConfigs := []struct {
		name string
		cfg  GDNSplitVConfig
	}{
		{"zero batch", GDNSplitVConfig{BatchSize: 0, NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 8, ValueHeadDim: 8, RMSNormEps: 1e-6}},
		{"proposals over K=8", GDNSplitVConfig{BatchSize: 1, Proposals: 9, NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 8, ValueHeadDim: 8, RMSNormEps: 1e-6}},
		{"negative proposals", GDNSplitVConfig{BatchSize: 1, Proposals: -1, NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 8, ValueHeadDim: 8, RMSNormEps: 1e-6}},
		{"valueHeads not divisible by keyHeads", GDNSplitVConfig{BatchSize: 1, NumKeyHeads: 3, NumValueHeads: 8, KeyHeadDim: 8, ValueHeadDim: 8, RMSNormEps: 1e-6}},
		{"invalid forcedSplitV=3", GDNSplitVConfig{BatchSize: 1, NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 8, ValueHeadDim: 8, ForcedSplitV: 3, RMSNormEps: 1e-6}},
		{"forcedSplitV exceeds valueHeadDim", GDNSplitVConfig{BatchSize: 1, NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 8, ValueHeadDim: 2, ForcedSplitV: 4, RMSNormEps: 1e-6}},
		{"zero eps", GDNSplitVConfig{BatchSize: 1, NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 8, ValueHeadDim: 8, RMSNormEps: 0}},
	}
	for _, tc := range badConfigs {
		if _, err := PlanGDNSplitV(tc.cfg); !errors.Is(err, ErrGDNSplitVInvalidGeometry) {
			t.Errorf("%s: got err=%v, want ErrGDNSplitVInvalidGeometry", tc.name, err)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGdnSplitvMatchesFullHeadOracle(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(20261106))
	randVec := func(n int, scale float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64() * scale)
		}
		return out
	}

	for _, proposals := range []int{0, 3, 8} {
		cfg := GDNSplitVConfig{
			BatchSize:     2,
			Proposals:     proposals,
			NumKeyHeads:   2,
			NumValueHeads: 4,
			KeyHeadDim:    12,
			ValueHeadDim:  24,
			RMSNormEps:    1e-5,
		}
		T := 1 + proposals
		bt := cfg.BatchSize * T
		decay := make([]float32, bt*cfg.NumValueHeads)
		beta := make([]float32, bt*cfg.NumValueHeads)
		for i := range decay {
			decay[i] = float32(0.85 + 0.1*rng.Float64())
			beta[i] = float32(0.1 + 0.8*rng.Float64())
		}
		in := GDNSplitVInputs{
			Q:              randVec(bt*cfg.NumKeyHeads*cfg.KeyHeadDim, 0.4),
			K:              randVec(bt*cfg.NumKeyHeads*cfg.KeyHeadDim, 0.4),
			V:              randVec(bt*cfg.NumValueHeads*cfg.ValueHeadDim, 0.5),
			Z:              randVec(bt*cfg.NumValueHeads*cfg.ValueHeadDim, 0.7),
			Decay:          decay,
			Beta:           beta,
			Norm:           randVec(cfg.ValueHeadDim, 1.0),
			RecurrentState: randVec(cfg.BatchSize*cfg.NumValueHeads*cfg.KeyHeadDim*cfg.ValueHeadDim, 0.25),
		}

		oracle, err := RunGDNFullHeadOracle(cfg, in)
		if err != nil {
			t.Fatalf("proposals=%d RunGDNFullHeadOracle: %v", proposals, err)
		}

		for _, splitV := range []int{1, 2, 4} {
			splitCfg := cfg
			splitCfg.ForcedSplitV = splitV
			got, err := RunGDNSplitVDecode(splitCfg, in)
			if err != nil {
				t.Fatalf("proposals=%d splitV=%d RunGDNSplitVDecode: %v", proposals, splitV, err)
			}
			for i := range oracle.RecurrentState {
				if math.Float32bits(got.RecurrentState[i]) != math.Float32bits(oracle.RecurrentState[i]) {
					t.Fatalf("proposals=%d splitV=%d state[%d]=%g != oracle=%g",
						proposals, splitV, i, got.RecurrentState[i], oracle.RecurrentState[i])
				}
			}
			for i := range oracle.Out {
				if d := math.Abs(float64(got.Out[i]) - float64(oracle.Out[i])); d > 1e-6 {
					t.Fatalf("proposals=%d splitV=%d out[%d]=%g != oracle=%g (diff=%g)",
						proposals, splitV, i, got.Out[i], oracle.Out[i], d)
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGdnSplitvDeterministicMergeAndPartialRefusals(t *testing.T) {
	t.Parallel()

	const kHd, vHd = 8, 10
	cfg := GDNSplitVConfig{
		BatchSize: 1, Proposals: 0, NumKeyHeads: 1, NumValueHeads: 1,
		KeyHeadDim: kHd, ValueHeadDim: vHd, ForcedSplitV: 4, RMSNormEps: 1e-6,
	}
	plan, err := PlanGDNSplitV(cfg)
	if err != nil {
		t.Fatalf("PlanGDNSplitV: %v", err)
	}

	rng := rand.New(rand.NewSource(77))
	initVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * 0.5)
		}
		return v
	}
	qHead := initVec(kHd)
	kHead := initVec(kHd)
	vHead := initVec(vHd)
	zHead := initVec(vHd)
	norm := initVec(vHd)
	stateAsc := initVec(kHd * vHd)

	partialsAsc := make([]GDNSplitVPartial, plan.SplitV)
	for s, slice := range plan.Partitions {
		p, err := ComputeGDNSplitVPartial(slice, qHead, kHead, vHead, stateAsc, kHd, vHd, 0.9, 0.5)
		if err != nil {
			t.Fatalf("ComputeGDNSplitVPartial(%d): %v", s, err)
		}
		partialsAsc[s] = p
	}
	wantOut, err := MergeGDNSplitVPartials(partialsAsc, zHead, norm, vHd, cfg.RMSNormEps)
	if err != nil {
		t.Fatalf("MergeGDNSplitVPartials(asc): %v", err)
	}

	// Reverse partial delivery must merge to the exact same bits as ascending split order.
	partialsRev := []GDNSplitVPartial{partialsAsc[3], partialsAsc[2], partialsAsc[1], partialsAsc[0]}
	gotRev, err := MergeGDNSplitVPartials(partialsRev, zHead, norm, vHd, cfg.RMSNormEps)
	if err != nil {
		t.Fatalf("MergeGDNSplitVPartials(rev): %v", err)
	}
	for i := range wantOut {
		if math.Float32bits(gotRev[i]) != math.Float32bits(wantOut[i]) {
			t.Fatalf("reversed delivery [%d]=%v != ascending [%d]=%v", i, gotRev[i], i, wantOut[i])
		}
	}

	// Missing split fails closed with ErrGDNSplitVMissingPartial.
	if _, err := MergeGDNSplitVPartials(partialsAsc[:3], zHead, norm, vHd, cfg.RMSNormEps); !errors.Is(err, ErrGDNSplitVMissingPartial) {
		t.Fatalf("missing split: got err=%v, want ErrGDNSplitVMissingPartial", err)
	}

	// Duplicate split fails closed with ErrGDNSplitVMissingPartial.
	dup := []GDNSplitVPartial{partialsAsc[0], partialsAsc[0], partialsAsc[2], partialsAsc[3]}
	if _, err := MergeGDNSplitVPartials(dup, zHead, norm, vHd, cfg.RMSNormEps); !errors.Is(err, ErrGDNSplitVMissingPartial) {
		t.Fatalf("duplicate split: got err=%v, want ErrGDNSplitVMissingPartial", err)
	}
}
