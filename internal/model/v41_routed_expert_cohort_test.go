package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ScalarCohortWitness struct {
	*v41RoutedLayerTestBackend
	enabled            bool
	fences             int
	matmulInputs       [][]float32
	matmulOutputs      [][]float32
	cohortRows         []int
	failMatmul         int
	failure            error
	publicationErrorAt int
	cleanupErrorAt     int
	publicationRows    map[compute.Buffer]int
}

func (b *v41ScalarCohortWitness) SupportsV41DeferredStatus(n int) bool { return b.enabled && n > 0 }
func (b *v41ScalarCohortWitness) SupportsV41BF16Publish(n int) bool    { return n > 0 }
func (b *v41ScalarCohortWitness) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	p, status, err := b.v41RoutedLayerTestBackend.V41BF16PublishDeferred(x, site)
	if b.publicationRows == nil {
		b.publicationRows = map[compute.Buffer]int{}
	}
	b.publicationRows[p.Buf()] = b.publishes
	if b.publishes == b.publicationErrorAt {
		err = b.failure
	}
	return p, status, err
}
func (b *v41ScalarCohortWitness) Free(x compute.Tensor) {
	delete(b.pending, x.Buf())
	b.Backend.Free(x)
	if row := b.publicationRows[x.Buf()]; row > 0 && row == b.cleanupErrorAt {
		panic(b.failure)
	}
}
func (b *v41ScalarCohortWitness) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	p, status, err := b.V41BF16PublishDeferred(x, site)
	if err != nil {
		return compute.Tensor{}, err
	}
	verdicts, _, err := b.V41ReadDeferred([]compute.V41DeferredStatus{status}, nil)
	if err != nil {
		b.Free(p)
		return compute.Tensor{}, err
	}
	if verdicts[0] != nil {
		b.Free(p)
		return compute.Tensor{}, verdicts[0]
	}
	return p, nil
}
func (b *v41ScalarCohortWitness) V41RoutedSwiGLU(g, u compute.Tensor, limit, weight float32) (compute.Tensor, error) {
	b.fences++
	return b.v41RoutedLayerTestBackend.V41RoutedSwiGLU(g, u, limit, weight)
}
func (b *v41ScalarCohortWitness) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmulInputs = append(b.matmulInputs, append([]float32(nil), b.Backend.Read(x)...))
	if len(b.matmulInputs) == b.failMatmul {
		panic(b.failure)
	}
	y := b.v41HalSeamBackend.MatMul(w, x)
	b.matmulOutputs = append(b.matmulOutputs, append([]float32(nil), b.Backend.Read(y)...))
	return y
}
func (b *v41ScalarCohortWitness) Read(x compute.Tensor) []float32 {
	b.fences++
	return b.v41RoutedLayerTestBackend.Read(x)
}
func (b *v41ScalarCohortWitness) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.fences++
	if len(st) > 1 {
		b.cohortRows = append(b.cohortRows, len(st)/2)
	}
	return b.v41RoutedLayerTestBackend.V41ReadDeferred(st, rows)
}

// fak-test:runtime medium est=5s lane=default
func TestV41ExpertScalarCohortProtocol(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	const count = 33 // Causally requires a 31-row cohort and a two-row tail.
	m := v41PinnedSlateFixture(t, false)
	inputs, picks := make([][]float32, count), make([][]routePick, count)
	weights := make([]float32, count)
	for r := range inputs {
		inputs[r] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[r] {
			inputs[r][i] = float32((r+i)%13-6) / 16
		}
		weights[r] = float32(r+1) / 64
		picks[r] = []routePick{{expert: 0, weight: weights[r]}}
	}
	newSession := func(enabled bool) (*Session, *v41ScalarCohortWitness) {
		b := &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: enabled}
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s, b
	}
	run := func(s *Session) [][]float32 {
		out := make([][]float32, count)
		state := s.v41State()
		state.routedSum = nil
		err := m.v41ContractRoutedGrouped(0, inputs, picks, &v41ProjScratch{}, m.Cfg, out, state)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	serial, control := newSession(false)
	batched, candidate := newSession(true)
	want, got := run(serial), run(batched)
	for r := range got {
		if !v41SameBits(got[r], want[r]) {
			t.Fatalf("row %d bits differ", r)
		}
	}
	if len(candidate.matmulInputs) != 3*count || !reflect.DeepEqual(candidate.matmulInputs, control.matmulInputs) || !reflect.DeepEqual(candidate.matmulOutputs, control.matmulOutputs) {
		t.Fatal("scalar MatMul count, operands, outputs, or execution order changed")
	}
	if !reflect.DeepEqual(candidate.cohortRows, []int{31, 2}) || candidate.fences != 2 || control.fences != 3*count {
		t.Fatalf("cohort rows=%v candidate fences=%d serial fences=%d; want [31 2],2,99", candidate.cohortRows, candidate.fences, control.fences)
	}
	if len(candidate.pending) != 0 {
		t.Fatal("cohort leaked deferred statuses")
	}
	// An earlier row's checked down verdict must remain primary even when
	// recording a later row fails before the final read. The earlier-only
	// serial arm returns the same numeric stage; no output or host replay occurs.
	t.Run("earlier_verdict_before_later_recording_failure", func(t *testing.T) {
		s, b := newSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.failMatmul = 4
		b.failure = &compute.BackendError{Backend: "test", Site: "later gate", Err: errors.New("later recording failure")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		var numeric *V41ExpertOperationError
		if outcome != v41RoutedExpertError || out != nil || !errors.As(err, &numeric) || numeric.Stage != "down" || errors.Is(err, b.failure) {
			t.Fatalf("earlier down verdict lost: outcome=%v output=%v error=%v", outcome, out, err)
		}
		if !s.BackendSessionClosed() {
			t.Fatal("later selected recording failure left session reusable")
		}
		if len(b.pending) != 0 || b.fences != 1 {
			t.Fatalf("pending=%d fences=%d", len(b.pending), b.fences)
		}
	})
	t.Run("failing_operation_record_cannot_replace_error", func(t *testing.T) {
		s, b := newSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.publicationErrorAt = 1
		b.failure = &compute.BackendError{Backend: "test", Site: "publication", Err: errors.New("selected publication error")}
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		if outcome != v41RoutedExpertError || out != nil || !errors.Is(err, b.failure) || !s.BackendSessionClosed() || len(b.pending) != 0 {
			t.Fatalf("operation error replaced or ownership retained: outcome=%v error=%v pending=%d", outcome, err, len(b.pending))
		}
	})
	t.Run("earlier_verdict_still_retires_later_cleanup_failure", func(t *testing.T) {
		s, b := newSession(true)
		b.publishFaultAt = map[int]bool{1: true}
		b.cleanupErrorAt = 2
		b.failure = errors.New("later cleanup panic")
		out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs[:2], weights[:2])
		var numeric *V41ExpertOperationError
		if outcome != v41RoutedExpertError || out != nil || !errors.As(err, &numeric) || numeric.Stage != "down" || !s.BackendSessionClosed() || len(b.pending) != 0 {
			t.Fatalf("earlier verdict or retirement lost: outcome=%v error=%v pending=%d", outcome, err, len(b.pending))
		}
	})
}
