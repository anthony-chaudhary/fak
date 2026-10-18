package model

import (
	"errors"
	"reflect"
	"runtime"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type siblingOutcome struct {
	returned, resumed bool
	value             any
}

// The outer goroutine distinguishes Goexit from a recovered legacy nil panic:
// only a recovered panic resumes after the inner call.
func siblingRun(fn func()) (out siblingOutcome) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		func() {
			defer func() { out.value = recover() }()
			fn()
			out.returned = true
		}()
		out.resumed = true
	}()
	<-done
	return out
}

func siblingFault(mode string, marker error) {
	switch mode {
	case "panic":
		panic(marker)
	case "nil-panic":
		panic(nil)
	case "goexit":
		runtime.Goexit()
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestBackendSiblingCleanupCompletion(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	first, later := errors.New("first cleanup panic"), errors.New("later cleanup panic")
	for _, mode := range []string{"normal", "panic", "nil-panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			var order []int
			out := siblingRun(func() {
				runBackendSiblingCleanup(
					func() { order = append(order, 1); siblingFault(mode, first) },
					func() {
						order = append(order, 2)
						if mode != "normal" {
							panic(later)
						}
					},
					func() { order = append(order, 3) },
				)
			})
			if !reflect.DeepEqual(order, []int{1, 2, 3}) {
				t.Fatalf("sibling order/once contract: %v", order)
			}
			switch mode {
			case "normal":
				if !out.returned || !out.resumed {
					t.Fatal("normal cleanup did not return")
				}
			case "panic":
				if out.returned || !out.resumed || out.value != first {
					t.Fatalf("first panic lost: %+v", out)
				}
			case "nil-panic":
				if out.returned || !out.resumed || out.value != nil {
					t.Fatalf("legacy nil panic lost: %+v", out)
				}
			case "goexit":
				if out.returned || out.resumed || out.value != nil {
					t.Fatalf("later panic replaced Goexit: %+v", out)
				}
			}
		})
	}
}

type siblingBackend struct {
	compute.Backend
	owner              sync.Mutex
	cloneCalls, failAt int
	freeFailAt         int
	mode               string
	marker             error
	freed              []compute.Tensor
	nameHook           func()
}

func (b *siblingBackend) Name() string {
	if b.nameHook != nil {
		b.nameHook()
	}
	return b.Backend.Name()
}
func (b *siblingBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *siblingBackend) CloneTensor(x compute.Tensor) (compute.Tensor, error) {
	b.cloneCalls++
	if b.cloneCalls == b.failAt {
		siblingFault(b.mode, b.marker)
	}
	return compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...)), nil
}
func (b *siblingBackend) Free(x compute.Tensor) {
	for _, old := range b.freed {
		if old.Buf() == x.Buf() {
			panic("duplicate physical tensor release")
		}
	}
	b.freed = append(b.freed, x)
	if len(b.freed) == b.freeFailAt {
		siblingFault(b.mode, b.marker)
	}
	b.Backend.Free(x)
}

func siblingState(b *siblingBackend, layers int) *qwen35HALState {
	q := &qwen35HALState{layers: make([]qwen35HALLayerState, layers)}
	for i := range q.layers {
		q.layers[i] = newQwen35HALLayerState(b, compute.NewF32(b, []int{1}, []float32{1}), compute.NewF32(b, []int{1}, []float32{2}))
	}
	return q
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35SharedCopyInterruptionReleasesLocks(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		for _, failAt := range []int{1, 2, 4} {
			t.Run(mode+string(rune('0'+failAt)), func(t *testing.T) {
				b := &siblingBackend{Backend: compute.Default(), mode: mode, failAt: failAt, marker: errors.New("clone interruption")}
				q := siblingState(b, 2)
				defer q.free(b)
				snapshot, err := cloneQwen35HALState(q, b)
				if err != nil {
					t.Fatal(err)
				}
				defer snapshot.free(b)
				out := siblingRun(func() {
					if failAt < 3 {
						_, _, _, _ = q.mutateLayer(b, 0, func(c, r compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
							panic("mutation reached after clone fault")
						})
					} else {
						_, _ = q.mutateSequence(b, func([]compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
							panic("sequence reached after clone fault")
						})
					}
				})
				if out.returned || (mode == "goexit" && out.resumed) || (mode != "goexit" && !out.resumed) || b.cloneCalls != failAt {
					t.Fatalf("clone control not reached: %+v calls=%d", out, b.cloneCalls)
				}
				for _, state := range []*qwen35HALState{q, snapshot} {
					if !state.mu.TryLock() {
						t.Fatal("state mutex stranded")
					}
					state.mu.Unlock()
					for _, layer := range state.layers {
						if !layer.owner.mu.TryLock() {
							t.Fatal("pair mutex stranded")
						}
						layer.owner.mu.Unlock()
					}
				}
				wantPartial := 0
				if failAt == 2 || failAt == 4 {
					wantPartial = 1
				}
				if len(b.freed) != wantPartial {
					t.Fatalf("partial clone leak: freed=%d", len(b.freed))
				}
				b.failAt = 0
				q.free(b)
				snapshot.free(b)
				if len(b.freed) != 4+failAt-1 {
					t.Fatalf("physical owner release count %d", len(b.freed))
				}
			})
		}
	}
}

type siblingKV struct {
	compute.KVStore
	freed  int
	mode   string
	marker error
	child  *siblingKV
}

func (k *siblingKV) Clone() compute.KVStore { return k.child }
func (k *siblingKV) ResidentBytes() int64   { return 0 }
func (k *siblingKV) Free() {
	k.freed++
	siblingFault(k.mode, k.marker)
}

// fak-test:runtime fast est=100ms lane=default
func TestPrefixSnapshotSiblingCleanupAndAcquisition(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		t.Run("close-"+mode, func(t *testing.T) {
			b := &siblingBackend{Backend: compute.Default()}
			k := &siblingKV{mode: mode, marker: errors.New("KV close interrupted")}
			q := siblingState(b, 2)
			defer q.free(b)
			p := &PrefixSnapshot{Backend: b, halKV: k, qwen35: q}
			out := siblingRun(p.Close)
			if out.returned || (mode == "goexit" && out.resumed) || k.freed != 1 || len(b.freed) != 4 || !p.closed {
				t.Fatalf("detached sibling was skipped: %+v kv=%d tensors=%d", out, k.freed, len(b.freed))
			}
			p.Close()
			if k.freed != 1 || len(b.freed) != 4 {
				t.Fatal("repeated snapshot Close freed twice")
			}
			if !b.owner.TryLock() {
				t.Fatal("snapshot execution owner stranded")
			}
			b.owner.Unlock()
		})
		t.Run("acquire-"+mode, func(t *testing.T) {
			b := &siblingBackend{Backend: compute.Default()}
			k := &siblingKV{}
			q := siblingState(b, 2)
			defer q.free(b)
			p := &PrefixSnapshot{Backend: b, halKV: k, qwen35: q}
			var escaped *PrefixSnapshot
			out := siblingRun(func() {
				escaped, _ = acquireCompletePrefixSnapshot(b, func(scope *backendExecutionScope) (*PrefixSnapshot, error) {
					scope.afterRelease(func() { siblingFault(mode, errors.New("late scope retirement")) })
					return p, nil
				})
			})
			if out.returned || escaped != nil || !p.closed || k.freed != 1 || len(b.freed) != 4 {
				t.Fatalf("acquired candidate leaked through late failure: %+v", out)
			}
			if !b.owner.TryLock() {
				t.Fatal("acquisition owner stranded")
			}
			b.owner.Unlock()
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestQwen35RecurrentCleanupAttemptsEveryTensor(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			b := &siblingBackend{Backend: compute.Default(), freeFailAt: 1, mode: mode, marker: errors.New("first pair free")}
			q := siblingState(b, 2)
			defer q.free(b)
			out := siblingRun(func() { q.free(b) })
			if out.returned || (mode == "goexit" && out.resumed) || len(b.freed) != 4 {
				t.Fatalf("pair/layer sibling skipped: %+v frees=%d", out, len(b.freed))
			}
			q.free(b)
			if len(b.freed) != 4 {
				t.Fatal("repeated state free retried physical cleanup")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPrefixSnapshotCloneLateProfileFailure(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	prefixProfile.Lock()
	oldPath := prefixProfile.path
	prefixProfile.Unlock()
	SetPrefixProfilePath(t.TempDir() + "/never-published.jsonl")
	t.Cleanup(func() { SetPrefixProfilePath(oldPath) })
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			b := &siblingBackend{Backend: compute.Default()}
			child := &siblingKV{}
			k := &siblingKV{child: child}
			q := siblingState(b, 2)
			defer q.free(b)
			p := &PrefixSnapshot{Backend: b, Cache: NewKVCache(Config{}), halKV: k, qwen35: q}
			t.Cleanup(p.Close)
			hit := false
			b.nameHook = func() { hit = true; siblingFault(mode, errors.New("late clone profile")) }
			var escaped *PrefixSnapshot
			out := siblingRun(func() { escaped, _ = p.Clone() })
			b.nameHook = nil
			if !hit || out.returned || escaped != nil || child.freed != 1 || k.freed != 0 || len(b.freed) != 0 {
				t.Fatalf("clone escaped/leaked before deferred profile completed: %+v child=%d", out, child.freed)
			}
			if !p.mu.TryLock() {
				t.Fatal("source snapshot lock stranded")
			}
			p.mu.Unlock()
			for _, layer := range q.layers {
				if layer.owner.refs != 1 {
					t.Fatal("failed clone retained shared recurrent pair")
				}
			}
			p.Close()
			if k.freed != 1 || len(b.freed) != 4 {
				t.Fatal("source snapshot ownership changed")
			}
			if !b.owner.TryLock() {
				t.Fatal("clone execution owner stranded")
			}
			b.owner.Unlock()
		})
	}
}

type siblingSequenceBackend struct {
	Qwen35GDNPreprojectedSequenceBackend
	mode  string
	freed []Qwen35GDNAuxState
}

func (b *siblingSequenceBackend) FreeQwen35GDNAuxState(s Qwen35GDNAuxState) error {
	b.freed = append(b.freed, s)
	if len(b.freed) == 1 {
		siblingFault(b.mode, errors.New("first sequence free"))
	}
	if len(b.freed) == 2 {
		panic("later sequence cleanup panic")
	}
	return nil
}

// fak-test:runtime fast est=50ms lane=default
func TestQwen35SequenceCleanupAttemptsEveryOwner(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	for _, mode := range []string{"panic", "nil-panic", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			b := &siblingBackend{Backend: compute.Default()}
			sequence := &siblingSequenceBackend{mode: mode}
			states := []Qwen35GDNAuxState{{Convolution: 1, Recurrent: 2}, {Convolution: 3, Recurrent: 4}, {Convolution: 5, Recurrent: 6}}
			q := siblingState(b, 2)
			defer q.free(b)
			q.sequenceBackend, q.sequenceLayers = sequence, states
			out := siblingRun(func() { q.free(b) })
			if out.returned || (mode == "goexit" && out.resumed) || (mode != "goexit" && !out.resumed) || !reflect.DeepEqual(sequence.freed, states) || len(b.freed) != 4 || q.sequenceBackend != nil || q.sequenceLayers != nil {
				t.Fatalf("sequence sibling cleanup skipped/reordered: %+v %v", out, sequence.freed)
			}
			q.free(b)
			if len(sequence.freed) != 3 || len(b.freed) != 4 {
				t.Fatal("sequence owner was freed twice")
			}
		})
	}
}
