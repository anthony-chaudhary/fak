package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Independent scalar graph oracle. The route weight is multiplied in F32
// before BF16 down-input publication, never after down projection.
func v41RoutedExpertBF16Oracle(w1, w3, w2, input []float32, I, H int, limit, weight float32) []float32 {
	x := make([]float32, len(input))
	for i, v := range input {
		x[i] = v41OracleBF16(v)
	}
	g, u := cpuOracleMatVec(w1, x, I, H), cpuOracleMatVec(w3, x, I, H)
	h := make([]float32, I)
	for i := range h {
		a, b := v41OracleBF16(g[i]), v41OracleBF16(u[i])
		if limit > 0 {
			a = min(a, limit)
			b = max(-limit, min(b, limit))
		}
		fused := float32((a / (1 + float32(math.Exp(float64(-a))))) * b)
		h[i] = v41OracleBF16(float32(weight * fused))
	}
	y := cpuOracleMatVec(w2, h, H, I)
	for i := range y {
		y[i] = v41OracleBF16(y[i])
	}
	return y
}

// fak-test:runtime fast est=100ms lane=default
func TestV41RoutedWeightBeforeDownWitness(t *testing.T) {
	cfg := Config{HiddenSize: 1, MoEIntermediateSize: 1, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg}
	for _, sign := range []float32{1, -1} {
		for _, deviceDown := range []bool{false, true} {
			fused := []float32{sign}
			before := append([]float32(nil), fused...)
			gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				return fused, v41GateUpHandled, nil
			}
			calls := 0
			borrowed := []float32{0}
			down := func(_ int, _ string, h []float32) ([]float32, v41ExpertDownOutcome, error) {
				calls++
				if h[0] != sign*.5 {
					t.Fatalf("down input=%g want=%g", h[0], sign*.5)
				}
				if !deviceDown {
					return nil, v41DownDeclined, nil
				}
				borrowed[0] = h[0] * 1.5
				return borrowed, v41DownHandled, nil
			}
			triple := func() ([]float32, []float32, []float32, error) {
				t.Fatal("handled gate/up loaded triple")
				return nil, nil, nil, nil
			}
			got, err := m.v41FullRoutedExpert(0, "ffn.experts.0", []float32{1}, .501953125, cfg, gate, down, triple, func() ([]float32, error) { return []float32{1.5}, nil }, false)
			if err != nil || got[0] != sign*.75 || calls != 1 {
				t.Fatalf("result=%v calls=%d err=%v", got, calls, err)
			}
			borrowed[0] = 99
			if got[0] != sign*.75 || !reflect.DeepEqual(fused, before) {
				t.Fatal("borrowed callback storage changed")
			}
			old := v41OracleBF16(float32(sign * 1.5 * .501953125))
			if old == got[0] {
				t.Fatal("weight-order witness is vacuous")
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41RoutedHostSerialParallelBF16(t *testing.T) {
	cfg := Config{HiddenSize: 2, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank, ActGeluTanh: true, ActGeluErf: true, SwigluLimit: .75}
	m := &Model{Cfg: cfg}
	x := []float32{1.00390625, -.51}
	w1 := []float32{1, .2, -2, 1, .5, 1}
	w3 := []float32{2, -1, 1, 2, -.5, 3}
	w2 := []float32{1.5, -2, .7, -.2, 1, 3}
	want := v41RoutedExpertBF16Oracle(w1, w3, w2, x, 3, 2, .75, .501953125)
	for _, parallel := range []bool{false, true} {
		got, err := m.v41FullRoutedExpert(0, "ffn.experts.0", x, .501953125, cfg, nil, nil, func() ([]float32, []float32, []float32, error) { return w1, w3, w2, nil }, nil, parallel)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("parallel=%t got=%v want=%v err=%v", parallel, got, want, err)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41RoutedAscendingAccumulation(t *testing.T) {
	picks := []routePick{{expert: 2, weight: .25}, {expert: 0, weight: .5}, {expert: 1, weight: .75}}
	before := append([]routePick(nil), picks...)
	values := [][]float32{{1}, {1 << 24}, {-(1 << 24)}}
	got, err := v41FullRoutedSum(0, picks, values, 1)
	if err != nil || got[0] != 1 || !reflect.DeepEqual(picks, before) {
		t.Fatalf("ascending result=%v err=%v", got, err)
	}
	slotSum := float32(0)
	for _, row := range values {
		slotSum = float32(slotSum + row[0])
	}
	if slotSum != 0 {
		t.Fatal("order witness is vacuous")
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41RoutedGroupedWeightedOwnership(t *testing.T) {
	m := v41FullStepModel(t)
	cfg := m.Cfg
	picks := [][]routePick{{{expert: 1, weight: .501953125}, {expert: 0, weight: .25}}, {{expert: 1, weight: .75}, {expert: 1, weight: .125}}}
	inputs := [][]float32{make([]float32, cfg.HiddenSize), make([]float32, cfg.HiddenSize)}
	borrowedH, borrowedY := make([]float32, cfg.MoEIntermediateSize), make([]float32, cfg.HiddenSize)
	calls := 0
	st := &v41ForwardState{
		expertGateUp: func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
			for i := range borrowedH {
				borrowedH[i] = 1
			}
			return borrowedH, v41GateUpHandled, nil
		},
		expertDown: func(_ int, _ string, h []float32) ([]float32, v41ExpertDownOutcome, error) {
			calls++
			for i := range borrowedY {
				borrowedY[i] = float32(h[0] * 1.5)
			}
			return borrowedY, v41DownHandled, nil
		},
	}
	output := make([][]float32, len(inputs))
	scratch := &v41ProjScratch{}
	if err := m.v41ContractRoutedGrouped(0, inputs, picks, scratch, cfg, output, st); err != nil {
		t.Fatal(err)
	}
	if calls != 4 || len(scratch.exp1)+len(scratch.exp3)+len(scratch.exp2) != 0 {
		t.Fatal("grouped dispatch or materialization changed")
	}
	for i := range borrowedY {
		borrowedY[i] = 99
	}
	for token, row := range output {
		want := float32(0)
		for _, pick := range picks[token] {
			want = float32(want + v41OracleBF16(v41OracleBF16(pick.weight)*1.5))
		}
		for _, v := range row {
			if v != want {
				t.Fatalf("token=%d got=%g want=%g", token, v, want)
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41RoutedSelectedFailures(t *testing.T) {
	cfg := Config{HiddenSize: 1, MoEIntermediateSize: 1, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg}
	sentinel := errors.New("routed selected failure")
	for _, stage := range []string{"gate/up", "down"} {
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32} {
			hostCalls, downCalls := 0, 0
			gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				if stage == "gate/up" {
					return []float32{bad}, v41GateUpHandled, nil
				}
				return []float32{1}, v41GateUpHandled, nil
			}
			down := func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
				downCalls++
				return []float32{bad}, v41DownHandled, nil
			}
			got, err := m.v41FullRoutedExpert(0, "ffn.experts.0", []float32{1}, 1, cfg, gate, down, func() ([]float32, []float32, []float32, error) { hostCalls++; return nil, nil, nil, sentinel }, func() ([]float32, error) { hostCalls++; return nil, sentinel }, false)
			var operation *V41ExpertOperationError
			if got != nil || !errors.As(err, &operation) || hostCalls != 0 {
				t.Fatalf("stage=%s err=%v host=%d", stage, err, hostCalls)
			}
			if stage == "gate/up" && downCalls != 0 {
				t.Fatal("invalid intermediate reached down")
			}
		}
		for _, handled := range []bool{false, true} {
			gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				if stage == "gate/up" {
					o := v41GateUpError
					if handled {
						o = v41GateUpHandled
					}
					return nil, o, sentinel
				}
				return []float32{1}, v41GateUpHandled, nil
			}
			down := func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
				o := v41DownError
				if handled {
					o = v41DownHandled
				}
				return nil, o, sentinel
			}
			got, err := m.v41FullRoutedExpert(0, "ffn.experts.0", []float32{1}, 1, cfg, gate, down, nil, nil, false)
			if got != nil || !errors.Is(err, sentinel) {
				t.Fatalf("selected stage=%s err=%v", stage, err)
			}
		}
	}
}

// Read returns shared borrowed storage and Free poisons it. Both full device
// bindings must own their publications before another Read/Free occurs.
type v41RoutedBorrowedBackend struct {
	*v41HalSeamBackend
	scratch []float32
	reads   int
}

func (b *v41RoutedBorrowedBackend) Read(x compute.Tensor) []float32 {
	values := b.Backend.Read(x)
	b.scratch = append(b.scratch[:0], values...)
	b.reads++
	return b.scratch
}
func (b *v41RoutedBorrowedBackend) Free(x compute.Tensor) {
	for i := range b.scratch {
		b.scratch[i] = 123
	}
	b.Backend.Free(x)
}

// fak-test:runtime medium est=5s lane=default
func TestV41FullRoutedDeviceProjectionOwnership(t *testing.T) {
	m := v41IncrementalExpertFixture(t, true, false)
	b := &v41RoutedBorrowedBackend{v41HalSeamBackend: &v41HalSeamBackend{Backend: compute.Default()}}
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	cfg := m.Cfg
	x := make([]float32, cfg.HiddenSize)
	for i := range x {
		x[i] = v41OracleBF16(float32(i%3-1) * .17)
	}
	stem := "ffn.experts.0"
	w1, w3, w2, err := m.v41ExpertTripleInto(0, stem, &v41ProjScratch{}, true)
	if err != nil {
		t.Fatal(err)
	}
	want := v41RoutedExpertBF16Oracle(w1, w3, w2, x, cfg.MoEIntermediateSize, cfg.HiddenSize, float32(cfg.SwigluLimit), .501953125)
	got, err := m.v41FullRoutedExpert(0, stem, x, .501953125, cfg, s.v41ExpertGateUpFunc(), s.v41ExpertDownFunc(), func() ([]float32, []float32, []float32, error) {
		t.Fatal("selected device loaded triple")
		return nil, nil, nil, nil
	}, func() ([]float32, error) { t.Fatal("selected device loaded down"); return nil, nil }, false)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("borrowed device output mismatch err=%v", err)
	}
	if b.reads != 3 {
		t.Fatalf("device readbacks=%d want=3", b.reads)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41FullRoutedParallelDispatchIsHostOnly(t *testing.T) {
	const H, I = 256, 256
	cfg := Config{HiddenSize: H, MoEIntermediateSize: I, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg}
	x := make([]float32, H)
	x[0] = 1
	w1, w3, w2 := make([]float32, I*H), make([]float32, I*H), make([]float32, H*I)
	for i := 0; i < I; i++ {
		w1[i*H] = 1
		w3[i*H] = .5
	}
	for i := 0; i < H; i++ {
		w2[i*I] = 1.5
	}
	triple := func() ([]float32, []float32, []float32, error) { return w1, w3, w2, nil }
	old := v41SwiGLUWitnessOn
	enableV41SwiGLUWitness()
	defer func() { v41SwiGLUWitnessOn = old }()
	serial, err := m.v41FullRoutedExpert(0, "ffn.experts.0", x, .501953125, cfg, nil, nil, triple, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := m.v41FullRoutedExpert(0, "ffn.experts.0", x, .501953125, cfg, nil, nil, triple, nil, true)
	if err != nil || !reflect.DeepEqual(serial, parallel) || v41SwiGLUDispatches() != 1 {
		t.Fatalf("host parallel count=%d err=%v", v41SwiGLUDispatches(), err)
	}
	gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		return make([]float32, I), v41GateUpHandled, nil
	}
	down := func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
		return make([]float32, H), v41DownHandled, nil
	}
	if _, err := m.v41FullRoutedExpert(0, "ffn.experts.0", x, .5, cfg, gate, down, nil, nil, true); err != nil {
		t.Fatal(err)
	}
	if v41SwiGLUDispatches() != 1 {
		t.Fatal("device handled row counted as host parallel SwiGLU")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41FullRoutedDeclineAndMalformedOutcomes(t *testing.T) {
	cfg := Config{HiddenSize: 1, MoEIntermediateSize: 1, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg}
	loads := 0
	triple := func() ([]float32, []float32, []float32, error) {
		loads++
		return []float32{1}, []float32{2}, []float32{1.5}, nil
	}
	decline := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		return nil, v41GateUpDeclined, nil
	}
	forbiddenDown := func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
		t.Fatal("declined gate/up offered device down")
		return nil, v41DownError, nil
	}
	got, err := m.v41FullRoutedExpert(0, "ffn.experts.0", []float32{1}, .501953125, cfg, decline, forbiddenDown, triple, nil, false)
	want := v41RoutedExpertBF16Oracle([]float32{1}, []float32{2}, []float32{1.5}, []float32{1}, 1, 1, 0, .501953125)
	if err != nil || loads != 1 || !reflect.DeepEqual(got, want) {
		t.Fatalf("decline loads=%d err=%v", loads, err)
	}
	for _, outcome := range []v41ExpertGateUpOutcome{v41GateUpHandled, v41ExpertGateUpOutcome(255)} {
		gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) { return nil, outcome, nil }
		got, err := m.v41FullRoutedExpert(0, "ffn.experts.0", []float32{1}, .5, cfg, gate, forbiddenDown, nil, nil, false)
		var named *V41ExpertOperationError
		if got != nil || !errors.As(err, &named) || named.Stage != "gate/up" {
			t.Fatalf("malformed gate/up outcome=%v err=%v", outcome, err)
		}
	}
	for _, outcome := range []v41ExpertDownOutcome{v41DownHandled, v41ExpertDownOutcome(255)} {
		gate := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
			return []float32{1}, v41GateUpHandled, nil
		}
		down := func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) { return nil, outcome, nil }
		got, err := m.v41FullRoutedExpert(0, "ffn.experts.0", []float32{1}, .5, cfg, gate, down, nil, nil, false)
		var named *V41ExpertOperationError
		if got != nil || !errors.As(err, &named) || named.Stage != "down" {
			t.Fatalf("malformed down outcome=%v err=%v", outcome, err)
		}
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41FullRoutedGroupedHostMaterializesOnce(t *testing.T) {
	m := v41IncrementalExpertFixture(t, true, true)
	cfg := m.Cfg
	picks := [][]routePick{{{expert: 1, weight: .5}, {expert: 0, weight: .25}}, {{expert: 0, weight: .75}, {expert: 1, weight: .125}}}
	inputs := [][]float32{make([]float32, cfg.HiddenSize), make([]float32, cfg.HiddenSize)}
	inputs[0][0] = 1
	inputs[1][0] = -.5
	scratch := &v41ProjScratch{expertLayerCacheBytes: 1}
	out := make([][]float32, 2)
	before := m.expertCheckpoint.Stats().Reads
	old := v41RetentionWitnessOn
	enableV41RetentionWitness()
	defer func() { v41RetentionWitnessOn = old }()
	if err := m.v41ContractRoutedGrouped(0, inputs, picks, scratch, cfg, out, nil); err != nil {
		t.Fatal(err)
	}
	if reads := m.expertCheckpoint.Stats().Reads - before; reads != 6 {
		t.Fatalf("two distinct host triples read %d slabs, want6", reads)
	}
	if v41Retentions() != 0 || len(scratch.layerExperts) != 0 {
		t.Fatal("grouped host retained expert triples")
	}
	// Independent source-order arithmetic uses separately faulted immutable weights
	// after taking the read-count receipt above; no production activation is called.
	for token := range inputs {
		want := make([]float32, cfg.HiddenSize)
		for expert := 0; expert < 2; expert++ {
			for _, pick := range picks[token] {
				if pick.expert != expert {
					continue
				}
				w1, w3, w2, err := m.v41ExpertTripleInto(0, "ffn.experts."+itoa(expert), &v41ProjScratch{}, false)
				if err != nil {
					t.Fatal(err)
				}
				y := v41RoutedExpertBF16Oracle(w1, w3, w2, inputs[token], cfg.MoEIntermediateSize, cfg.HiddenSize, float32(cfg.SwigluLimit), pick.weight)
				for i := range want {
					want[i] = float32(want[i] + y[i])
				}
			}
		}
		if !reflect.DeepEqual(out[token], want) {
			t.Fatalf("host grouped token=%d differs from weighted oracle", token)
		}
	}
}
