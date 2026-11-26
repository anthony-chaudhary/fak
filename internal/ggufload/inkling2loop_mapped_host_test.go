package ggufload

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type inkling2MappedHostTestBackend struct {
	compute.Backend
	declineAt int
	errorAt   int
	err       error
	calls     int
	frees     int
}

func inkling2MappedPackedExperts(t *testing.T, typ TensorType, experts, rows, cols int) *inkling2QuantExperts {
	t.Helper()
	rb, err := inkling2RowBytes(typ, cols)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, experts*rows*rb)
	if typ == TensorNVFP4 {
		raw = randomInkling2NVFP4(rand.New(rand.NewSource(41)), experts*rows, cols)
	}
	return &inkling2QuantExperts{typ: typ, raw: raw, experts: experts, rows: rows, cols: cols, rowLen: rb}
}

func (b *inkling2MappedHostTestBackend) UploadMappedHostWeight(t compute.Tensor, _ compute.Dtype, max int64) (compute.Tensor, bool, error) {
	b.calls++
	if b.errorAt == b.calls {
		return compute.Tensor{}, false, b.err
	}
	if b.declineAt == b.calls || max <= 0 {
		return compute.Tensor{}, false, nil
	}
	return t, true, nil
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2MappedDefaultZeroAndUnsupportedBackendPreserveHostStacks(t *testing.T) {
	m := inkling2MappedFixture(compute.Default())
	gate, down := m.Weights.Layers[0].GateUp, m.Weights.Layers[0].Down
	s := &inkling2DeviceSet{be: compute.Default()}
	if count, bytes, err := s.mapColdExperts(m, []bool{false}, 0); err != nil || count != 0 || bytes != 0 {
		t.Fatalf("default zero: count=%d bytes=%d err=%v", count, bytes, err)
	}
	if m.Weights.Layers[0].GateUp != gate || m.Weights.Layers[0].Down != down {
		t.Fatal("default-zero mapping replaced a host stack")
	}
	if _, _, err := s.mapColdExperts(m, []bool{false}, 880); err == nil {
		t.Fatal("explicit mapped-host placement on unsupported backend did not fail closed")
	}
	if m.Weights.Layers[0].GateUp != gate || m.Weights.Layers[0].Down != down {
		t.Fatal("unsupported backend replaced a host stack")
	}
}

func (b *inkling2MappedHostTestBackend) Free(t compute.Tensor) {
	b.frees++
	b.Backend.Free(t)
}

func inkling2MappedQ3Experts(be compute.Backend, seed byte) *inkling2QuantExperts {
	const experts, rows, cols = 2, 2, 256
	raw := make([]byte, experts*rows*110)
	for i := range raw {
		raw[i] = seed + byte(i)
	}
	return &inkling2QuantExperts{typ: TensorQ3_K, raw: raw, experts: experts, rows: rows, cols: cols, rowLen: 110}
}

func inkling2MappedFixture(be compute.Backend) *Inkling2LoopModel {
	return &Inkling2LoopModel{
		HP: model.Inkling2LoopHParams{Layers: 1, Passes: 1},
		Weights: &model.Inkling2LoopWeights{Layers: []model.Inkling2LoopLayerWeights{{
			GateUp: inkling2MappedQ3Experts(be, 1),
			Down:   inkling2MappedQ3Experts(be, 2),
		}}},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInkling2MappedExpertBoundDefaultAndValidation(t *testing.T) {
	lookup := func(v string) func(string) (string, bool) {
		return func(key string) (string, bool) { return v, key == inkling2MappedExpertBytesEnv && v != "" }
	}
	if got, err := inkling2MappedExpertBound(lookup("")); err != nil || got != 0 {
		t.Fatalf("unset bound = %d, %v", got, err)
	}
	if got, err := inkling2MappedExpertBound(lookup("880")); err != nil || got != 880 {
		t.Fatalf("bound = %d, %v", got, err)
	}
	if _, err := inkling2MappedExpertBound(lookup("-1")); err == nil {
		t.Fatal("negative mapped-host bound accepted")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2MappedColdExpertsPublishesOnlyCompleteSet(t *testing.T) {
	be := &inkling2MappedHostTestBackend{Backend: compute.Default()}
	m := inkling2MappedFixture(be)
	s := &inkling2DeviceSet{be: be}
	count, bytes, err := s.mapColdExperts(m, []bool{false}, 880)
	if err != nil || count != 2 || bytes != 880 || be.calls != 2 || len(s.tensors) != 2 {
		t.Fatalf("map result count=%d bytes=%d calls=%d owned=%d err=%v", count, bytes, be.calls, len(s.tensors), err)
	}
	for _, e := range []model.Inkling2LoopExperts{m.Weights.Layers[0].GateUp, m.Weights.Layers[0].Down} {
		if d, ok := e.(*inkling2DeviceExperts); !ok || len(d.w) != 2 {
			t.Fatalf("published experts = %T, want complete device stack", e)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2MappedColdExpertsRollbackPreservesHostStacks(t *testing.T) {
	be := &inkling2MappedHostTestBackend{Backend: compute.Default(), declineAt: 2}
	m := inkling2MappedFixture(be)
	gate, down := m.Weights.Layers[0].GateUp, m.Weights.Layers[0].Down
	s := &inkling2DeviceSet{be: be}
	count, bytes, err := s.mapColdExperts(m, []bool{false}, 880)
	if err == nil || count != 0 || bytes != 0 || be.calls != 2 || be.frees != 1 || len(s.tensors) != 0 {
		t.Fatalf("rollback count=%d bytes=%d calls=%d frees=%d owned=%d err=%v", count, bytes, be.calls, be.frees, len(s.tensors), err)
	}
	if m.Weights.Layers[0].GateUp != gate || m.Weights.Layers[0].Down != down {
		t.Fatal("declined transaction replaced a host expert stack")
	}

	be.calls, be.frees = 0, 0
	count, bytes, err = s.mapColdExperts(m, []bool{false}, 879)
	if err == nil || count != 0 || bytes != 0 || be.calls != 0 || be.frees != 0 {
		t.Fatalf("preflight bound count=%d bytes=%d calls=%d frees=%d err=%v", count, bytes, be.calls, be.frees, err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2MappedColdExpertsErrorAfterPartialAllocationRollsBack(t *testing.T) {
	cause := errors.New("mapped allocation failed")
	be := &inkling2MappedHostTestBackend{Backend: compute.Default(), errorAt: 2, err: cause}
	m := inkling2MappedFixture(be)
	gate, down := m.Weights.Layers[0].GateUp, m.Weights.Layers[0].Down
	s := &inkling2DeviceSet{be: be}
	count, bytes, err := s.mapColdExperts(m, []bool{false}, 880)
	if !errors.Is(err, cause) || count != 0 || bytes != 0 || be.calls != 2 || be.frees != 1 || len(s.tensors) != 0 {
		t.Fatalf("partial error count=%d bytes=%d calls=%d frees=%d owned=%d err=%v", count, bytes, be.calls, be.frees, len(s.tensors), err)
	}
	if m.Weights.Layers[0].GateUp != gate || m.Weights.Layers[0].Down != down {
		t.Fatal("partial error replaced a host expert stack")
	}
}

// TestInkling2MappedExpertSourceActualColdGeometry covers the expert counts,
// widths, padding and native packed dtypes present in staged Inkling checkpoints.
// It stays small by using one output row per expert; the expert-stride geometry
// is identical to a full 1152/2304-row projection.
//
// fak-test:runtime fast est=100ms lane=default
func TestInkling2MappedExpertSourceActualColdGeometry(t *testing.T) {
	be := compute.Default()
	cases := []struct {
		typ           TensorType
		experts, cols int
		wantPad       int
	}{
		{TensorIQ3_S, 64, 2304, 2304},
		{TensorIQ3_XXS, 96, 2304, 2304},
		{TensorIQ4_NL, 128, 1152, 1280},
		{TensorNVFP4, 128, 1152, 1152},
	}
	for _, tc := range cases {
		q := inkling2MappedPackedExperts(t, tc.typ, tc.experts, 1, tc.cols)
		source, padK, nbytes, err := inkling2MappedExpertSource(be, q)
		if err != nil {
			t.Fatalf("%s/%d: %v", tc.typ, tc.experts, err)
		}
		wantBytes := int(inkling2DeviceBytes(tc.typ, len(q.raw), tc.experts, tc.cols))
		if source.Shape[0] != tc.experts || source.Shape[1] != tc.wantPad || padK != tc.wantPad || nbytes != wantBytes {
			t.Fatalf("%s/%d: shape=%v pad=%d bytes=%d, want [%d %d] pad=%d bytes=%d", tc.typ, tc.experts, source.Shape, padK, nbytes, tc.experts, tc.wantPad, tc.wantPad, wantBytes)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInkling2MappedColdMixedNativeLayoutsPublishFor64_96_128Experts(t *testing.T) {
	for _, experts := range []int{64, 96, 128} {
		be := &inkling2MappedHostTestBackend{Backend: compute.Default()}
		gate := inkling2MappedPackedExperts(t, TensorIQ3_S, experts, 1, 2304)
		down := inkling2MappedPackedExperts(t, TensorIQ4_NL, experts, 1, 1152)
		m := &Inkling2LoopModel{HP: model.Inkling2LoopHParams{Layers: 1, Passes: 1}, Weights: &model.Inkling2LoopWeights{Layers: []model.Inkling2LoopLayerWeights{{GateUp: gate, Down: down}}}}
		s := &inkling2DeviceSet{be: be}
		bound := inkling2DeviceBytes(gate.typ, len(gate.raw), experts, gate.cols) + inkling2DeviceBytes(down.typ, len(down.raw), experts, down.cols)
		count, bytes, err := s.mapColdExperts(m, []bool{false}, bound)
		if err != nil || count != 2 || bytes != bound {
			t.Fatalf("experts=%d: count=%d bytes=%d want=%d err=%v", experts, count, bytes, bound, err)
		}
		for _, e := range []model.Inkling2LoopExperts{m.Weights.Layers[0].GateUp, m.Weights.Layers[0].Down} {
			if d, ok := e.(*inkling2DeviceExperts); !ok || len(d.w) != experts {
				t.Fatalf("experts=%d: published %T", experts, e)
			}
		}
	}
}
