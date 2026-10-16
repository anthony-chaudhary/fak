package compute

import (
	"errors"
	"sync"
	"testing"
)

type deferredReadResultRecorder struct {
	V41DeferredStatusBackend
	calls    int
	values   [][]float32
	verdicts []error
	err      error
}

func (b *deferredReadResultRecorder) V41ReadDeferred([]V41DeferredStatus, []Tensor) ([]error, [][]float32, error) {
	b.calls++
	return b.verdicts, b.values, b.err
}

type ownedDeferredReadResultRecorder struct{ *deferredReadResultRecorder }

func (b *ownedDeferredReadResultRecorder) v41ReadDeferredOwned([]V41DeferredStatus, []Tensor) ([]error, V41DeferredReadResult, error) {
	b.calls++
	return b.verdicts, V41DeferredReadResult{count: len(b.values), state: &v41DeferredReadState{rows: b.values, independent: true}}, b.err
}

// fak-test:runtime fast est=100ms lane=default
func TestV41DeferredReadResultLegacyAndSingleConsumption(t *testing.T) {
	for _, grant := range []bool{false, true} {
		numeric := errors.New("numeric verdict")
		b := &deferredReadResultRecorder{values: [][]float32{{1, 2}, {3, 4}}, verdicts: []error{nil, numeric}}
		var backend V41DeferredStatusBackend = b
		if grant {
			backend = &ownedDeferredReadResultRecorder{b}
		}
		verdicts, result, err := ReadV41DeferredResult(backend, nil, []Tensor{{}, {}})
		if result.RowCount() != 2 || err != nil || b.calls != 1 || len(verdicts) != 2 || verdicts[1] != numeric {
			t.Fatal("read/order changed")
		}
		copiedResult := result
		rows, owned := result.TakeRows()
		if owned != grant || len(rows) != 2 || &rows[0][0] != &b.values[0][0] {
			t.Fatal("row transfer changed")
		}
		rows[0][0] = 9
		if rows[1][0] != 3 {
			t.Fatal("independent rows alias")
		}
		again, owned := result.TakeRows()
		if copied, claimed := copiedResult.TakeRows(); copied != nil || claimed {
			t.Fatal("copied result consumed twice")
		}
		if result.RowCount() != 2 || copiedResult.RowCount() != 2 || again != nil || owned {
			t.Fatal("result consumed twice")
		}
	}
	var nilResult *V41DeferredReadResult
	if rows, owned := nilResult.TakeRows(); rows != nil || owned {
		t.Fatal("nil result")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41DeferredReadResultTransportDropsPartialRows(t *testing.T) {
	failure := errors.New("read transport")
	for _, grant := range []bool{false, true} {
		b := &deferredReadResultRecorder{values: [][]float32{{1}, {2}}, err: failure}
		var backend V41DeferredStatusBackend = b
		if grant {
			backend = &ownedDeferredReadResultRecorder{b}
		}
		verdicts, result, err := ReadV41DeferredResult(backend, nil, []Tensor{{}, {}})
		rows, owned := result.TakeRows()
		if result.RowCount() != 0 || err != failure || verdicts != nil || rows != nil || owned || b.calls != 1 {
			t.Fatal("partial error escaped or replayed")
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41DeferredMultirowImageStorageIndependent(t *testing.T) {
	for _, little := range []bool{false, true} {
		sizes := []int{16, 16, 8, 8}
		image := newV41HostReadbackImage(sizes, 2, little)
		if image.floats != nil {
			t.Fatal("multirow image unexpectedly retains arena")
		}
		rows := image.materializeRows(sizes, 2)
		rows[0][0] = 1
		if rows[1][0] != 0 {
			t.Fatal("peer row alias")
		}
		for i := range image.host {
			image.host[i] = 0xff
		}
		if rows[0][0] != 1 || rows[1][0] != 0 || cap(rows[0]) != len(rows[0]) || cap(rows[1]) != len(rows[1]) {
			t.Fatal("image mutation affected owned rows")
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41DeferredReadResultConcurrentCopiesConsumeOnce(t *testing.T) {
	source := [][]float32{{1, 2}, {3, 4}}
	result := V41DeferredReadResult{count: len(source), state: &v41DeferredReadState{rows: source, independent: true}}
	var wg sync.WaitGroup
	winners := make(chan [][]float32, 32)
	start := make(chan struct{})
	for i := 0; i < 32; i++ {
		copied := result
		target := &copied
		if i%2 == 0 {
			target = &result
		}
		wg.Add(1)
		go func(r *V41DeferredReadResult) {
			defer wg.Done()
			<-start
			if r.RowCount() != 2 {
				t.Error("immutable row count")
			}
			rows, owned := r.TakeRows()
			if owned {
				winners <- rows
			} else if rows != nil {
				t.Error("loser obtained rows")
			}
		}(target)
	}
	close(start)
	wg.Wait()
	close(winners)
	count := 0
	for rows := range winners {
		count++
		if len(rows) != 2 || &rows[0][0] != &source[0][0] {
			t.Fatal("ownership result changed")
		}
	}
	if count != 1 {
		t.Fatalf("ownership winners=%d", count)
	}
}
