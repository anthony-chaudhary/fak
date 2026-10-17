package model

import (
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41SumReadbackWitness struct {
	*v41SumTestBackend
	mu             sync.Mutex
	reads          int
	batches        []int
	badRow         int
	cleanupFailure compute.Buffer
}

func (b *v41SumReadbackWitness) ForwardExecutionOwner() *sync.Mutex   { return &b.mu }
func (b *v41SumReadbackWitness) SupportsV41DeferredStatus(n int) bool { return n > 0 }
func (b *v41SumReadbackWitness) V41RoutedSwiGLUDeferred(compute.Tensor, compute.Tensor, float32, float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("unused")
}
func (b *v41SumReadbackWitness) V41BF16PublishDeferred(compute.Tensor, uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("unused")
}
func (b *v41SumReadbackWitness) Read(x compute.Tensor) []float32 {
	b.reads++
	return b.v41SumTestBackend.Read(x)
}
func (b *v41SumReadbackWitness) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(statuses) != 0 {
		panic("sum supplied statuses")
	}
	b.batches = append(b.batches, len(rows))
	values := make([][]float32, len(rows))
	for row, tensor := range rows {
		if !b.live[tensor.Buf()] {
			panic("read after free")
		}
		values[row] = b.Backend.Read(tensor) // borrowed, poisoned by Free below
		if row == b.badRow {
			values[row][0] = float32(math.NaN())
		}
	}
	return nil, values, nil
}
func (b *v41SumReadbackWitness) Free(x compute.Tensor) {
	values := b.Backend.Read(x)
	b.v41SumTestBackend.Free(x)
	for i := range values {
		values[i] = float32(math.NaN())
	}
	if x.Buf() == b.cleanupFailure {
		panic(&compute.BackendError{Backend: "sum-readback-test", Site: "Free", Err: errors.New("later cleanup fault")})
	}
}

// CPU ownership/control-flow witness only: API reads, not hardware fences or timing.
// fak-test:runtime fast est=20ms lane=default
func TestV41RoutedSumReadbackCohort(t *testing.T) {
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	prior := v41PrefillSumReadback.Load()
	SetV41PrefillSumReadback(false)
	t.Cleanup(func() { SetV41PrefillSumReadback(prior) })
	const count, width = 9, 4
	picks := make([][]routePick, count)
	inputs := make([][][]float32, count)
	for row := range picks {
		picks[row] = []routePick{{expert: 3}, {expert: 1}, {expert: 2}}
		inputs[row] = [][]float32{{-1e8, -0.5, 3, float32(row)}, {1e8, 0.25, 2, 1}, {1, 0.25, -4, 2}}
	}
	newOwner := func() (*v41SumReadbackWitness, *Session, *v41RoutedOutputOwner, *backendExecutionScope) {
		b := &v41SumReadbackWitness{v41SumTestBackend: newV41SumTestBackend(), badRow: -1}
		s := &Session{M: &Model{Cfg: Config{HiddenSize: width, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank, NumLayers: 1}}, Backend: b}
		scope := s.beginBackendExecution()
		o := newV41RoutedOutputOwner(s, 0, width, picks)
		if o == nil {
			t.Fatal("owner declined")
		}
		// Mixed retained/unretained slots exercise both lifetimes.
		for row := range picks {
			tensor := b.Upload(compute.NewF32(compute.Default(), []int{width}, inputs[row][0]), compute.F32)
			if n, err := o.handoff([]v41ExpertRow{{Token: row, Slot: 0}}).adopt([]compute.Tensor{tensor}); n != 1 || err != nil {
				t.Fatalf("adopt = %d, %v", n, err)
			}
		}
		return b, s, o, scope
	}
	b, s, o, scope := newOwner()
	if o.sumReadbackEnabled() {
		t.Fatal("experiment selected by default")
	}
	want := make([][]float32, count)
	for row := range picks {
		var err error
		want[row], err = o.sum(row, picks[row], inputs[row])
		if err != nil {
			t.Fatal(err)
		}
	}
	o.release(0, true)
	s.closeBackendOwned()
	scope.finish()
	if b.reads != count || len(b.live) != 0 {
		t.Fatalf("scalar reads=%d live=%d", b.reads, len(b.live))
	}

	SetV41PrefillSumReadback(true)
	b, s, o, scope = newOwner()
	if !o.sumReadbackEnabled() {
		t.Fatal("explicit experiment declined")
	}
	got := make([][]float32, count)
	if err := o.sumReadbackGroups(picks, inputs, got); err != nil {
		t.Fatal(err)
	}
	for row := range want {
		if !v41SameBits(got[row], want[row]) {
			t.Fatalf("row %d changed bits", row)
		}
	}
	if b.reads != 0 || !reflect.DeepEqual(b.batches, []int{8, 1}) || len(b.live) != 0 {
		t.Fatalf("grouped reads=%d batches=%v live=%d", b.reads, b.batches, len(b.live))
	}
	o.release(0, true)
	s.closeBackendOwned()
	scope.finish()
	t.Log("exact scalar bits; 8 scalar reads -> 1 deferred read, tail 1; owned copies survive poisoned frees")

	b, s, o, scope = newOwner()
	b.badRow = 5    // later pending result must beat the still-later recording fault
	b.faultAdd = 22 // token 7, first addition; only 7 complete sums can be read
	b.cleanupFailure = o.rows[8].tensor.Buf()
	got = make([][]float32, count)
	err := o.sumReadbackGroups(picks, inputs, got)
	var forward *BackendForwardOperationError
	if !errors.As(err, &forward) || !errors.Is(err, errV41ExpertResult) || !s.halClosed || len(b.live) != 0 || !o.closed {
		t.Fatalf("failure precedence/retirement: err=%v closed=%v live=%d ownerClosed=%v", err, s.halClosed, len(b.live), o.closed)
	}
	if !reflect.DeepEqual(b.batches, []int{7}) {
		t.Fatalf("pending reads=%v", b.batches)
	}
	for row := range got {
		if got[row] != nil {
			t.Fatalf("published failed chunk row %d", row)
		}
	}
	scope.finish()
	t.Log("row 5 invalid result wins over row 7 add and row 8 cleanup faults; all buffers drained, session retired")
}
