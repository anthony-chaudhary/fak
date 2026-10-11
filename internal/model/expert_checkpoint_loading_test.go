package model

import (
	"bytes"
	"errors"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type checkpointFlightReader struct {
	data       []byte
	entered    chan int64
	release    chan struct{}
	releaseOne sync.Once
	calls      atomic.Int64
	err        error
	panicValue any
}

func (r *checkpointFlightReader) ReadAt(p []byte, off int64) (int, error) {
	r.calls.Add(1)
	half := len(p) / 2
	copy(p[:half], r.data[off:off+int64(half)])
	r.entered <- off
	<-r.release
	if r.panicValue != nil {
		panic(r.panicValue)
	}
	if r.err != nil {
		return half, r.err
	}
	copy(p[half:], r.data[off+int64(half):off+int64(len(p))])
	return len(p), nil
}

func (r *checkpointFlightReader) unblock() { r.releaseOne.Do(func() { close(r.release) }) }

type checkpointFlightResult struct {
	name   string
	weight expertWeight
	err    error
	panic  any
}

func checkpointFlightFaults(tier *ExpertCheckpointTier, names ...string) <-chan checkpointFlightResult {
	results := make(chan checkpointFlightResult, len(names))
	for _, name := range names {
		go func(name string) {
			result := checkpointFlightResult{name: name}
			defer func() {
				result.panic = recover()
				results <- result
			}()
			result.weight, result.err = tier.fault(name)
		}(name)
	}
	return results
}

func checkpointFlightFixture(t *testing.T, budget int64) (*ExpertCheckpointTier, *checkpointFlightReader, int64) {
	t.Helper()
	stride := int64(q4kBlockBytes)
	data := make([]byte, 2*stride)
	for i := range data {
		data[i] = byte(i%251 + 1)
	}
	r := &checkpointFlightReader{data: data, entered: make(chan int64, 16), release: make(chan struct{})}
	t.Cleanup(r.unblock)
	tier := NewExpertCheckpointTier(budget)
	if err := tier.AddShard(r, int64(len(data)), []FusedExpertTensor{{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
		Quant: ExpertCheckpointQ4K, Experts: 2, Rows: 1, Cols: qkK,
	}}); err != nil {
		t.Fatalf("index expert fixture: %v", err)
	}
	return tier, r, stride
}

func checkpointFlightWait(t *testing.T, ready func() bool) {
	t.Helper()
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	for !ready() {
		select {
		case <-timeout.C:
			t.Fatal("expert fault did not reach its gated synchronization point")
		default:
			runtime.Gosched()
		}
	}
}

func checkpointFlightCoalesced(tier *ExpertCheckpointTier) int64 {
	// The identical regression compiles against the pre-flight baseline, whose stats omit this field.
	field := reflect.ValueOf(tier.Stats()).FieldByName("Coalesced")
	if !field.IsValid() {
		return 0
	}
	return field.Int()
}

func checkpointFlightAttached(t *testing.T, tier *ExpertCheckpointTier, r *checkpointFlightReader, count int) {
	t.Helper()
	checkpointFlightWait(t, func() bool { return len(r.entered) > 0 })
	checkpointFlightWait(t, func() bool {
		return r.calls.Load()+checkpointFlightCoalesced(tier) == int64(count)
	})
}

func checkpointFlightCollect(t *testing.T, results <-chan checkpointFlightResult, count int) []checkpointFlightResult {
	t.Helper()
	got := make([]checkpointFlightResult, 0, count)
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	for len(got) < count {
		select {
		case result := <-results:
			got = append(got, result)
		case <-timeout.C:
			t.Fatalf("completed %d of %d expert faults", len(got), count)
		}
	}
	return got
}

func checkpointFlightComplete(t *testing.T, result checkpointFlightResult, want []byte) {
	t.Helper()
	if result.err != nil || result.panic != nil {
		t.Fatalf("expert fault failed: error=%v panic=%v", result.err, result.panic)
	}
	if result.weight.q4 == nil || !bytes.Equal(result.weight.q4.raw, want) {
		t.Fatal("expert fault returned incomplete or different checkpoint bytes")
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointConcurrentFault(t *testing.T) {
	const callers = 8
	name := expertName(0, 0, "gate_proj.weight")
	names := make([]string, callers)
	for i := range names {
		names[i] = name
	}
	for _, tc := range []struct {
		name   string
		budget int64
	}{{"stream-through", 0}, {"retained", int64(q4kBlockBytes)}} {
		t.Run(tc.name, func(t *testing.T) {
			tier, reader, stride := checkpointFlightFixture(t, tc.budget)
			results := checkpointFlightFaults(tier, names...)
			checkpointFlightAttached(t, tier, reader, callers)
			select {
			case <-results:
				t.Fatal("expert fault completed while source bytes were partial")
			default:
			}
			if st := tier.Stats(); st.ResidentCount != 0 || st.ResidentBytes != 0 || st.Reads != 0 {
				t.Fatalf("partial source bytes became visible: residents=%d bytes=%d reads=%d", st.ResidentCount, st.ResidentBytes, st.Reads)
			}
			reader.unblock()
			for _, result := range checkpointFlightCollect(t, results, callers) {
				checkpointFlightComplete(t, result, reader.data[:stride])
			}
			st := tier.Stats()
			if calls := reader.calls.Load(); calls != 1 || st.Reads != 1 || st.BytesRead != stride {
				t.Fatalf("same-key cold faults: source calls=%d reads=%d bytes=%d, want 1 read of %d bytes", calls, st.Reads, st.BytesRead, stride)
			}
			if coalesced := checkpointFlightCoalesced(tier); coalesced != callers-1 || st.Hits != 0 {
				t.Fatalf("attached waiters=%d host hits=%d, want %d waiters and 0 hits", coalesced, st.Hits, callers-1)
			}
			wantResidents := 0
			if tc.budget > 0 {
				wantResidents = 1
			}
			if st.ResidentCount != wantResidents || st.ResidentBytes != int64(wantResidents)*stride {
				t.Fatalf("completed residency: count=%d bytes=%d, want count=%d bytes=%d", st.ResidentCount, st.ResidentBytes, wantResidents, int64(wantResidents)*stride)
			}
			checkpointFlightComplete(t, checkpointFlightCollect(t, checkpointFlightFaults(tier, name), 1)[0], reader.data[:stride])
			wantCalls := int64(2 - wantResidents)
			if calls := reader.calls.Load(); calls != wantCalls {
				t.Fatalf("later fault source calls=%d, want %d", calls, wantCalls)
			}
		})
	}

	t.Run("read-failure-retry", func(t *testing.T) {
		tier, reader, stride := checkpointFlightFixture(t, int64(q4kBlockBytes))
		boom := errors.New("checkpoint read failure")
		reader.err = boom
		results := checkpointFlightFaults(tier, names...)
		checkpointFlightAttached(t, tier, reader, callers)
		reader.unblock()
		var shared error
		for _, result := range checkpointFlightCollect(t, results, callers) {
			if !errors.Is(result.err, boom) || result.panic != nil || result.weight.q4 != nil || result.weight.kq != nil {
				t.Fatalf("partial read result: error=%v panic=%v q4=%v kquant=%v", result.err, result.panic, result.weight.q4 != nil, result.weight.kq != nil)
			}
			if shared == nil {
				shared = result.err
			} else if result.err != shared {
				t.Fatal("attached waiters received separate read failures")
			}
		}
		st := tier.Stats()
		if calls := reader.calls.Load(); calls != 1 || st.Failures != 1 || st.Reads != 0 || st.BytesRead != 0 || st.ResidentCount != 0 || st.ResidentBytes != 0 {
			t.Fatalf("failed flight: calls=%d failures=%d reads=%d bytes=%d residents=%d resident_bytes=%d", calls, st.Failures, st.Reads, st.BytesRead, st.ResidentCount, st.ResidentBytes)
		}
		reader.err = nil
		checkpointFlightComplete(t, checkpointFlightCollect(t, checkpointFlightFaults(tier, name), 1)[0], reader.data[:stride])
		st = tier.Stats()
		if calls := reader.calls.Load(); calls != 2 || st.Reads != 1 || st.Failures != 1 || st.ResidentCount != 1 || st.ResidentBytes != stride {
			t.Fatalf("retry: calls=%d reads=%d failures=%d residents=%d bytes=%d", calls, st.Reads, st.Failures, st.ResidentCount, st.ResidentBytes)
		}
	})

	t.Run("independent-keys", func(t *testing.T) {
		tier, reader, stride := checkpointFlightFixture(t, 2*int64(q4kBlockBytes))
		other := expertName(0, 1, "gate_proj.weight")
		results := checkpointFlightFaults(tier, name, other)
		checkpointFlightWait(t, func() bool { return len(reader.entered) == 2 })
		first, second := <-reader.entered, <-reader.entered
		if first == second || (first != 0 && first != stride) || (second != 0 && second != stride) {
			t.Fatalf("parallel source offsets=%d,%d, want 0,%d in either order", first, second, stride)
		}
		reader.unblock()
		for _, result := range checkpointFlightCollect(t, results, 2) {
			want := reader.data[:stride]
			if result.name == other {
				want = reader.data[stride:]
			}
			checkpointFlightComplete(t, result, want)
		}
		if st := tier.Stats(); st.Reads != 2 || st.BytesRead != 2*stride || st.ResidentCount != 2 || st.ResidentBytes != 2*stride {
			t.Fatalf("parallel keys: reads=%d bytes=%d residents=%d resident_bytes=%d", st.Reads, st.BytesRead, st.ResidentCount, st.ResidentBytes)
		}
	})

	t.Run("owner-panic-retry", func(t *testing.T) {
		tier, reader, stride := checkpointFlightFixture(t, int64(q4kBlockBytes))
		panicToken := &struct{}{}
		reader.panicValue = panicToken
		results := checkpointFlightFaults(tier, names...)
		checkpointFlightAttached(t, tier, reader, callers)
		reader.unblock()
		panics, failures := 0, 0
		var shared error
		for _, result := range checkpointFlightCollect(t, results, callers) {
			if result.weight.q4 != nil || result.weight.kq != nil {
				t.Fatal("aborted read published a partial weight")
			}
			if result.panic != nil {
				if result.panic != panicToken {
					t.Fatalf("owner panic changed: %v", result.panic)
				}
				panics++
			} else if result.err != nil {
				failures++
				if shared == nil {
					shared = result.err
				} else if result.err != shared {
					t.Fatal("aborted flight waiters received separate failures")
				}
			} else {
				t.Fatal("aborted flight returned success")
			}
		}
		st := tier.Stats()
		if panics != 1 || failures != callers-1 || st.Failures != 1 || st.ResidentCount != 0 || st.ResidentBytes != 0 {
			t.Fatalf("aborted flight: owner panics=%d waiter errors=%d failures=%d residents=%d bytes=%d", panics, failures, st.Failures, st.ResidentCount, st.ResidentBytes)
		}
		reader.panicValue = nil
		checkpointFlightComplete(t, checkpointFlightCollect(t, checkpointFlightFaults(tier, name), 1)[0], reader.data[:stride])
		if calls := reader.calls.Load(); calls != 2 {
			t.Fatalf("retry after owner panic made %d source calls, want 2", calls)
		}
	})
}
