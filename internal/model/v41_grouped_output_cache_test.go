package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41GroupedScalarCacheFixture(t *testing.T) (*Model, *Session, *v41GroupedRaggedRecorder) {
	t.Helper()
	m := v41GroupedFixture(t, "Q2_K", true, false)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	c := m.Cfg
	m.kqw[layerName(0, "attn.wo_b.weight")] = q3kFixtureTensor(c.HiddenSize, c.OGroups*c.OLoraRank)
	b := &v41GroupedRaggedRecorder{v41GroupedResidentRecorder: newV41GroupedResidentRecorder()}
	b.v41DeferredLedger = newV41DeferredLedger(func() compute.Backend { return b })
	return m, v41DenseTestSession(t, m, b), b
}

func v41GroupedScalarCacheCall(s *Session, input []float32) ([]float32, v41DenseProjectionOutcome, error) {
	c := s.M.Cfg
	return s.v41GroupedOutputFunc()(0, input, c.NumHeads, c.HeadDim, c.OGroups, c.OLoraRank, c.HiddenSize)
}

// fak-test:runtime medium est=5s lane=default
func TestV41GroupedOutputScalarCachedWeights(t *testing.T) {
	m, s, b := v41GroupedScalarCacheFixture(t)
	c := m.Cfg
	input := v41GroupedPanelInput(1, c.NumHeads*c.HeadDim)
	for i := range input {
		input[i] = v41RoundBF16(input[i])
	}
	want := v41GroupedPanelCacheOracle(t, m, input, 1)[0]
	readers := v41GroupedPanelCacheLazy(m)
	call := func(session *Session) {
		t.Helper()
		got, outcome, err := v41GroupedScalarCacheCall(session, input)
		if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
			t.Fatalf("scalar outcome=%d err=%v parity=%t", outcome, err, reflect.DeepEqual(got, want))
		}
	}
	call(s)
	weights := make(map[string]compute.Tensor, len(s.halW))
	for key, weight := range s.halW {
		weights[key] = weight
	}
	if len(weights) != c.OGroups+1 {
		t.Fatalf("weights=%d, want %d", len(weights), c.OGroups+1)
	}
	borrower := v41DenseTestSession(t, m, b)
	before := b.uploads
	call(s)
	borrower.cacheGeometryEpoch++ // Request geometry does not retire immutable weights.
	call(borrower)
	// Re-adoption must also work during deferred model close while this session
	// retains its lifetime pin; no new session or owner is admitted here.
	borrower.halW, borrower.borrowedHALW = map[string]compute.Tensor{}, nil
	var active *WeightSessionsActiveError
	if err := m.CloseWeights(); !errors.As(err, &active) {
		t.Fatalf("model close with live sessions: %v", err)
	}
	call(borrower)
	if b.uploads-before != 3*c.OGroups {
		t.Fatalf("warm uploads=%d, want only %d activations", b.uploads-before, 3*c.OGroups)
	}
	for i, reader := range readers {
		if reader.calls != 1 || reader.bytes != len(reader.data) {
			t.Fatalf("weight %d lazy reads=%d bytes=%d, want 1/%d", i, reader.calls, reader.bytes, len(reader.data))
		}
	}
	for key, weight := range weights {
		if borrower.halW[key].Buf() != weight.Buf() || b.freed[weight.Buf()] != 0 {
			t.Fatalf("model handle changed or freed: %s", key)
		}
		if _, ok := borrower.borrowedHALW[key]; !ok {
			t.Fatalf("missing borrowed marker: %s", key)
		}
	}
	s.Close()
	for _, weight := range weights {
		if b.freed[weight.Buf()] != 0 {
			t.Fatal("session close freed a still-pinned model weight")
		}
	}
	borrower.Close() // Last pin completes the deferred model close.
	for buf, created := range b.created {
		if created != 1 || b.freed[buf] != 1 {
			t.Fatalf("ownership: created=%d freed=%d, want once each", created, b.freed[buf])
		}
	}
	for buf := range b.freed {
		if b.created[buf] != 1 {
			t.Fatal("freed an unowned buffer")
		}
	}
}

// fak-test:runtime medium est=3s lane=default
func TestV41GroupedOutputScalarCachedMissPrevalidation(t *testing.T) {
	m, s, b := v41GroupedScalarCacheFixture(t)
	c := m.Cfg
	groupIn := c.NumHeads * c.HeadDim / c.OGroups
	name := layerName(0, "attn.wo_a.weight")
	a, ok, err := s.v41GroupedWeight(name, groupIn)
	if err != nil || !ok {
		t.Fatalf("A source: supported=%t err=%v", ok, err)
	}
	host, err := a.host(0, c.OLoraRank, groupIn)
	if err != nil {
		t.Fatal(err)
	}
	key := fmt.Sprintf("v41-grouped:%s:%v:%d:%d:%d", name, a.dtype, 0, c.OLoraRank, groupIn)
	weight := s.cachedImmutableWeight(key, key, func() compute.Tensor { return b.Upload(host, a.dtype) })
	// Only the first A group is resident. The remaining A group is a valid
	// miss; the late B miss must fail before either is uploaded or multiplied.
	q := m.kqw[layerName(0, "attn.wo_b.weight")]
	binary.LittleEndian.PutUint16(q.raw[q.kind.blockBytes()-2:], 0x7c00)
	borrower := v41DenseTestSession(t, m, b)
	input := make([]float32, c.NumHeads*c.HeadDim) // Already widened BF16.
	uploads, matmuls := b.uploads, b.matmuls
	got, outcome, err := v41GroupedScalarCacheCall(borrower, input)
	if got != nil || outcome != v41ProjectionError || !errors.Is(err, errV41ProjectionResult) || !borrower.BackendSessionClosed() {
		t.Fatalf("late invalid B: result=%v outcome=%d err=%v closed=%t", got, outcome, err, borrower.BackendSessionClosed())
	}
	if b.uploads != uploads || b.matmuls != matmuls || b.freed[weight.Buf()] != 0 {
		t.Fatal("mixed hit/miss refusal performed grouped work or freed a borrowed weight")
	}
	s.Close()
	if b.freed[weight.Buf()] != 0 {
		t.Fatal("session close retired model ownership")
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if b.freed[weight.Buf()] != 1 {
		t.Fatalf("model weight freed %d times, want once", b.freed[weight.Buf()])
	}
}

// Any backend method call panics through the nil embedded interface. Lookup
// needs only exact pointer identity and must not query capabilities or stage.
type immutableLookupBackend struct{ compute.Backend }

// fak-test:runtime fast est=1s lane=default
func TestImmutableWeightLookupAdoption(t *testing.T) {
	for _, reason := range []string{"hit", "closing", "local", "missing-owner", "missing-pool", "missing-backend", "missing-key", "other-model", "unpinned", "closed-session", "closed-owner", "closed-registry", "closed-resident", "nil-map"} {
		t.Run(reason, func(t *testing.T) {
			be := &immutableLookupBackend{}
			weight := compute.NewF32(be, []int{1}, []float32{1})
			resident := &backendHALWeightResidency{be: be, weights: map[string]compute.Tensor{"resident": weight}}
			registry := &modelHALWeightResidency{backends: []*backendHALWeightResidency{resident}}
			closer := &weightCloserState{sessions: 1, halWeights: registry}
			m := &Model{weightCloser: closer}
			s := &Session{M: m, Backend: be, modelWeightsHeld: true, halW: map[string]compute.Tensor{}}
			switch reason {
			case "closing":
				closer.closing = true
			case "local":
				s.halW["session"], s.modelWeightsHeld = weight, false
			case "missing-owner":
				m.weightCloser = nil
			case "missing-pool":
				closer.halWeights = nil
			case "missing-backend":
				s.Backend = &immutableLookupBackend{}
			case "missing-key":
				delete(resident.weights, "resident")
			case "other-model":
				s.M = &Model{}
			case "unpinned":
				s.modelWeightsHeld = false
			case "closed-session":
				s.halClosed = true
			case "closed-owner":
				closer.closed = true
			case "closed-registry":
				registry.closed = true
			case "closed-resident":
				resident.closed = true
			case "nil-map":
				s.halW = nil
			}
			owner, pool, backends := s.M.weightCloser, closer.halWeights, len(registry.backends)
			got, ok := s.lookupImmutableWeight("session", "resident")
			want := reason == "hit" || reason == "closing" || reason == "local"
			if ok != want || (ok && got.Buf() != weight.Buf()) {
				t.Fatalf("lookup hit=%t, want %t; correct handle=%t", ok, want, got.Buf() == weight.Buf())
			}
			if s.M.weightCloser != owner || closer.halWeights != pool || len(registry.backends) != backends {
				t.Fatal("lookup created owner state")
			}
			if want {
				_, borrowed := s.borrowedHALW["session"]
				if s.halW["session"].Buf() != weight.Buf() || borrowed != (reason != "local") {
					t.Fatal("lookup failed to memo protected operand or changed local ownership")
				}
			} else if len(s.halW) != 0 || len(s.borrowedHALW) != 0 {
				t.Fatal("lookup miss modified session ownership")
			}
		})
	}
}
