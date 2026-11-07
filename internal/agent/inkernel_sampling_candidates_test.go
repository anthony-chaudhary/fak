package agent

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// This producer is deliberately CPU-backed. It witnesses the actual served
// serial caller and its ID/RNG/continuation behavior, not CUDA or D2H bytes.
type samplingLaneProbe struct {
	compute.Backend
	calls           int
	refuseAlternate bool
}

func (b *samplingLaneProbe) SamplingCandidates(row compute.Tensor, k int, retain bool, p *compute.SamplingCandidatePacket) (compute.Tensor, bool, error) {
	b.calls++
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
	p.Vocab, p.Count, p.Valid = len(x), k+1, !(b.refuseAlternate && b.calls%2 == 0)
	for i, id := range ids[:k+1] {
		p.IDs[i], p.Raw[i] = id, x[id]
	}
	if !retain {
		return compute.Tensor{}, true, nil
	}
	kept, err := b.Backend.(compute.TensorCloner).CloneTensor(row)
	return kept, true, err
}

// fak-test:runtime fast est=100ms lane=default
func TestSerialSamplingCandidatesPreserveTokensRNGAndContinuation(t *testing.T) {
	t.Setenv("FAK_CUDA_SAMPLING_COMPACT", "1")
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	prompt := synthIDs(cfg.VocabSize, 7, 123)
	newLane := func(be compute.Backend) (*decodeLane, *batchLaneSink) {
		sink := &batchLaneSink{}
		s := m.NewBackendSession(be)
		t.Cleanup(s.Close)
		return &decodeLane{s: s, logits: s.Prefill(prompt), rng: rand.New(rand.NewSource(99)),
			temp: .8, topK: 4, topP: .9, minP: .05, maxNew: 6, stops: map[int]bool{},
			forwarded: []int{}, emit: sink.emit}, sink
	}
	for _, refuse := range []bool{false, true} {
		base, baseSink := newLane(compute.Pick("cpu-ref"))
		probe := &samplingLaneProbe{Backend: compute.Pick("cpu-ref"), refuseAlternate: refuse}
		compact, compactSink := newLane(probe)
		inKernelDecodeSerial(context.Background(), base)
		inKernelDecodeSerial(context.Background(), compact)
		if base.err != nil || compact.err != nil || !eqInts(baseSink.toks, compactSink.toks) ||
			!eqInts(base.forwarded, compact.forwarded) || base.rng.Float64() != compact.rng.Float64() {
			t.Fatalf("refuse=%t base=%v compact=%v errors=%v/%v", refuse, baseSink.toks, compactSink.toks, base.err, compact.err)
		}
		if probe.calls != 5 || compact.vocabSize() != cfg.VocabSize || !compact.compact.ready {
			t.Fatalf("refuse=%t calls=%d vocab=%d ready=%t", refuse, probe.calls, compact.vocabSize(), compact.compact.ready)
		}
		want, got := base.continuationLogits(), compact.continuationLogits()
		if len(want) != len(got) {
			t.Fatal("continuation lost the full vocabulary")
		}
		for i := range want {
			if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
				t.Fatalf("continuation raw mismatch at %d", i)
			}
		}
		if compact.continuationLogits() != nil {
			t.Fatal("continuation owner taken twice")
		}
	}
}
