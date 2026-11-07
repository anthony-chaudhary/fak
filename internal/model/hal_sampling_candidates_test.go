package model

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Host spy, not a device producer or physical-transfer witness. It exercises the
// real Session step/output/recycle/continuation boundaries with CPU execution.
type samplingSessionProbe struct {
	compute.Backend
	owner                         sync.Mutex
	reads, recycles, calls, frees int
	refuse                        bool
	failure                       error
}

func (b *samplingSessionProbe) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *samplingSessionProbe) Read(t compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(t)
}
func (b *samplingSessionProbe) Recycle() { b.recycles++ }
func (b *samplingSessionProbe) Free(t compute.Tensor) {
	b.frees++
	b.Backend.Free(t)
}
func (b *samplingSessionProbe) SamplingCandidates(row compute.Tensor, k int, retain bool, p *compute.SamplingCandidatePacket) (compute.Tensor, bool, error) {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("candidate producer called outside execution scope")
	}
	b.calls++
	if b.failure != nil {
		return compute.Tensor{}, false, b.failure
	}
	x := b.Backend.Read(row)
	ids := make([]int, len(x))
	for i := range ids {
		ids[i] = i
	}
	sort.Slice(ids, func(i, j int) bool {
		if x[ids[i]] != x[ids[j]] {
			return x[ids[i]] > x[ids[j]]
		}
		return ids[i] < ids[j]
	})
	p.Vocab, p.Count, p.Valid = len(x), k+1, !b.refuse
	for i, id := range ids[:k+1] {
		p.IDs[i], p.Raw[i] = id, x[id]
	}
	if !retain {
		return compute.Tensor{}, true, nil
	}
	kept, err := b.Backend.(compute.TensorCloner).CloneTensor(row)
	return kept, true, err
}

// fak-test:runtime fast est=50ms lane=default
func TestStepSamplingCandidatesOwnsOutputUntilConsumed(t *testing.T) {
	m := stepGreedyTestModel()
	b := &samplingSessionProbe{Backend: compute.Pick("cpu-ref")}
	s, ref := m.NewBackendSession(b), m.NewBackendSession(compute.Pick("cpu-ref"))
	defer s.Close()
	defer ref.Close()
	s.Prefill([]int{3, 17, 5})
	ref.Prefill([]int{3, 17, 5})
	var packet compute.SamplingCandidatePacket
	for step := 0; step < 2; step++ {
		b.refuse = step == 1
		reads, recycles, calls := b.reads, b.recycles, b.calls
		want := ref.Step(11)
		full, compact := s.StepSamplingCandidates(11, .8, 2, true, &packet)
		if b.calls != calls+1 || b.recycles != recycles+1 || s.halKV.Len() != ref.halKV.Len() {
			t.Fatal("candidate step repeated forwarding or escaped its recycle boundary")
		}
		if step == 0 {
			if !compact || full != nil || b.reads != reads || packet.Count != 2 {
				t.Fatalf("accepted compact=%t full=%d reads=%d", compact, len(full), b.reads-reads)
			}
			assertSameF32(t, "retained candidate row", want, s.TakeGreedyLogits())
			if s.TakeGreedyLogits() != nil {
				t.Fatal("retained row taken twice")
			}
		} else {
			if compact || b.reads != reads+1 || s.TakeGreedyLogits() != nil {
				t.Fatal("refusal did not restore full output")
			}
			assertSameF32(t, "candidate refusal", want, full)
		}
	}
	// The compact output seam must keep token/tokenGreedy's topology refusal,
	// before it forwards a model or calls the candidate producer.
	m.Cfg.BlockTopology = PostNorm
	calls, positions := b.calls, s.halKV.Len()
	var topologyPanic any
	func() {
		defer func() { topologyPanic = recover() }()
		s.StepSamplingCandidates(11, .8, 2, true, &packet)
	}()
	m.Cfg.BlockTopology = PreNorm
	if !strings.Contains(fmt.Sprint(topologyPanic), "only PreNorm") || b.calls != calls || s.halKV.Len() != positions {
		t.Fatal("compact output bypassed the ordinary HAL topology refusal")
	}
	b.failure = errors.New("injected candidate transaction failure")
	reads, recycles := b.reads, b.recycles
	var caught any
	func() { defer func() { caught = recover() }(); s.StepSamplingCandidates(11, .8, 2, true, &packet) }()
	if caught != b.failure || b.reads != reads || b.recycles != recycles {
		t.Fatal("transaction failure was converted to fallback/recycle")
	}
}
