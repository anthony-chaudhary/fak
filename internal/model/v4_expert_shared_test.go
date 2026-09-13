package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// sharedExpertFakeRouted is a minimal v4LiveExpertRuntime that returns a fixed
// routed vector, so the test isolates the shared-expert contribution.
type sharedExpertFakeRouted struct {
	hidden int
	out    []float32
	closed bool
}

func (f *sharedExpertFakeRouted) forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error) {
	return append([]float32(nil), f.out...), nil
}

func (f *sharedExpertFakeRouted) Close() error {
	f.closed = true
	return nil
}

func (f *sharedExpertFakeRouted) Stats() v4ExpertRuntimeStats { return v4ExpertRuntimeStats{} }

const (
	sharedTestLayer = 0
)

// xSharedProbe is length HiddenSize for the admitted Flash profile.
func sharedTestProbe(cfg Config) []float32 {
	x := make([]float32, cfg.HiddenSize)
	for i := range x {
		x[i] = float32((i % 5) - 2)
	}
	return x
}

// v4SharedExpertTestFixture builds the pinned Flash-profile model with a routed
// gate and (optionally) the shared-expert w1/w2/w3 weights in the f32 manifest.
// It uses the official admitted Flash geometry so NewFromF32Tensors admits it.
func v4SharedExpertTestFixture(t *testing.T, withShared bool) (Config, *Model) {
	t.Helper()
	_, cfg := readDeepSeekV4FlashConfig(t)
	hidden, intermediate := cfg.HiddenSize, cfg.MoEIntermediateSize
	gate := make([]float32, cfg.NumExperts*hidden)
	for i := range gate {
		gate[i] = float32((i%5)-2) / 2
	}
	w1 := sharedExpertProjection(intermediate, hidden, 0.5)
	w3 := sharedExpertProjection(intermediate, hidden, 1)
	w2 := make([]float32, hidden*intermediate)
	for i := range w2 {
		w2[i] = float32((i%3)-1) / 2
	}
	tensors := []NamedTensorF32{
		{Name: layerName(sharedTestLayer, "mlp.gate.weight"), Shape: []int{cfg.NumExperts, hidden}, Data: gate},
	}
	if withShared {
		tensors = append(tensors,
			NamedTensorF32{Name: v4SharedExpertName(sharedTestLayer, "w1"), Shape: []int{intermediate, hidden}, Data: w1},
			NamedTensorF32{Name: v4SharedExpertName(sharedTestLayer, "w2"), Shape: []int{hidden, intermediate}, Data: w2},
			NamedTensorF32{Name: v4SharedExpertName(sharedTestLayer, "w3"), Shape: []int{intermediate, hidden}, Data: w3},
		)
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	// ensureV4LiveExpert requires an indexed source directory before it honors
	// the test runtime; the fake v4Expert means no weights are ever read from it.
	m.sourceDir = t.TempDir()
	return cfg, m
}

// sharedExpertProjection fills a [rows, cols] matrix with small exact values.
func sharedExpertProjection(rows, cols int, scale float32) []float32 {
	out := make([]float32, rows*cols)
	for i := range out {
		out[i] = float32((i%4)-1) * scale
	}
	return out
}

// sharedExpertOracle is an independent host computation of the shared expert's
// contribution using the pinned asymmetric SwiGLU clamp.
func sharedExpertOracle(x, w1, w2, w3 []float32, intermediate, hidden int, limit float32) []float32 {
	out := make([]float32, hidden)
	for h := 0; h < intermediate; h++ {
		var gate, up float32
		for i := 0; i < hidden; i++ {
			gate += w1[h*hidden+i] * x[i]
			up += w3[h*hidden+i] * x[i]
		}
		if limit > 0 {
			gate = min(gate, limit)
			up = max(-limit, min(up, limit))
		}
		act := v4SiLU(gate) * up
		for o := 0; o < hidden; o++ {
			out[o] += w2[o*intermediate+h] * act
		}
	}
	return out
}

func sharedRoutedProbe(cfg Config, seed float32) []float32 {
	out := make([]float32, cfg.HiddenSize)
	for i := range out {
		out[i] = seed + float32(i%7) - 2
	}
	return out
}

func TestV4SharedExpertAddsToRoutedOutput(t *testing.T) {
	cfg, m := v4SharedExpertTestFixture(t, true)
	be := compute.Default()
	s := m.NewSession()
	s.Backend = be
	routed := sharedRoutedProbe(cfg, 0.25)
	s.v4Expert = &sharedExpertFakeRouted{hidden: cfg.HiddenSize, out: routed}

	x := sharedTestProbe(cfg)
	normalized := s.uploadHostF32([]int{cfg.HiddenSize}, x, compute.MemoryActivation, "v4-test-shared-norm")
	defer be.Free(normalized)

	gotTensor, err := s.v4ExpertForward(sharedTestLayer, 1, normalized)
	if err != nil {
		t.Fatalf("v4ExpertForward: %v", err)
	}
	defer be.Free(gotTensor)
	got := be.Read(gotTensor)

	// Independent oracle: shared expert computed inline, then the pinned helper
	// combines routed+shared. Before the fix the shared expert is never executed,
	// so the live output would equal routed alone and this differs.
	w1 := m.tensor(v4SharedExpertName(sharedTestLayer, "w1"))
	w2 := m.tensor(v4SharedExpertName(sharedTestLayer, "w2"))
	w3 := m.tensor(v4SharedExpertName(sharedTestLayer, "w3"))
	shared := sharedExpertOracle(x, w1, w2, w3, cfg.MoEIntermediateSize, cfg.HiddenSize, float32(cfg.SwigluLimit))
	want, err := v41SharedExpertAdd(routed, shared, v41RouterConfig{SharedCount: cfg.NSharedExperts})
	if err != nil {
		t.Fatalf("oracle shared add: %v", err)
	}
	if !closeSlice(got, want, 1e-4) {
		t.Fatalf("live FFN mismatch: got[0:3]=%v want[0:3]=%v shared[0:3]=%v", got[:3], want[:3], shared[:3])
	}
	// Guard against a vacuous pass: the combined output must actually differ
	// from the routed-only vector.
	if closeSlice(got, routed, 1e-3) {
		t.Fatalf("shared expert did not change the routed output: got[0:3]=%v", got[:3])
	}
	s.Close()
}

func TestV4SharedExpertFailsClosedWhenWeightsMissing(t *testing.T) {
	cfg, m := v4SharedExpertTestFixture(t, false)
	be := compute.Default()
	s := m.NewSession()
	s.Backend = be
	s.v4Expert = &sharedExpertFakeRouted{hidden: cfg.HiddenSize, out: make([]float32, cfg.HiddenSize)}

	normalized := s.uploadHostF32([]int{cfg.HiddenSize}, sharedTestProbe(cfg), compute.MemoryActivation, "v4-test-shared-missing")
	defer be.Free(normalized)

	if _, err := s.v4ExpertForward(sharedTestLayer, 1, normalized); !errors.Is(err, ErrV4SharedExpert) {
		t.Fatalf("missing shared weights err=%v, want ErrV4SharedExpert", err)
	}
	s.Close()
}

func TestV4SharedExpertSkippedWhenConfigDeclaresNone(t *testing.T) {
	cfg, m := v4SharedExpertTestFixture(t, false)
	m.Cfg.NSharedExperts = 0
	be := compute.Default()
	s := m.NewSession()
	s.Backend = be
	routed := sharedRoutedProbe(cfg, -1.5)
	s.v4Expert = &sharedExpertFakeRouted{hidden: cfg.HiddenSize, out: routed}

	normalized := s.uploadHostF32([]int{cfg.HiddenSize}, sharedTestProbe(cfg), compute.MemoryActivation, "v4-test-shared-none")
	defer be.Free(normalized)

	gotTensor, err := s.v4ExpertForward(sharedTestLayer, 1, normalized)
	if err != nil {
		t.Fatalf("shared-less config must keep routed-only: %v", err)
	}
	defer be.Free(gotTensor)
	if got := be.Read(gotTensor); !closeSlice(got, routed, 1e-6) {
		t.Fatalf("shared-less output=%v want routed-only=%v", got[:3], routed[:3])
	}
	s.Close()
}
