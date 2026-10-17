package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The recorder executes actual CPU-reference math through Session. Its batch and
// graph methods record call ordering only; they do not emulate GPU execution time.
type retainedGreedyBatchBackend struct {
	compute.Backend
	cloner                                                 compute.TensorCloner
	batch, capture, captureEnabled                         bool
	argmaxInBatch, argmaxOutsideBatch, argmaxDuringCapture int
	clones, clonesInBatch, graphEnds, graphAborts, flushes int
	cloneReleases                                          map[compute.Buffer]int
	panicArgmax                                            bool
}

func (b *retainedGreedyBatchBackend) BeginBatch() { b.batch = true }
func (b *retainedGreedyBatchBackend) FlushBatch() { b.batch = false; b.flushes++ }
func (b *retainedGreedyBatchBackend) GraphBegin() bool {
	b.capture = b.captureEnabled
	return b.capture
}
func (b *retainedGreedyBatchBackend) GraphEndLaunch() { b.capture = false; b.graphEnds++ }
func (b *retainedGreedyBatchBackend) GraphAbort()     { b.capture = false; b.graphAborts++ }
func (b *retainedGreedyBatchBackend) Argmax(x compute.Tensor) int {
	if b.batch {
		b.argmaxInBatch++
	} else {
		b.argmaxOutsideBatch++
	}
	if b.capture {
		b.argmaxDuringCapture++
	}
	if b.panicArgmax {
		panic("retained greedy argmax failure")
	}
	return b.Backend.Argmax(x)
}
func (b *retainedGreedyBatchBackend) CloneTensor(x compute.Tensor) (compute.Tensor, error) {
	b.clones++
	if b.batch {
		b.clonesInBatch++
	}
	out, err := b.cloner.CloneTensor(x)
	if err == nil {
		b.cloneReleases[out.Buf()] = 0
	}
	return out, err
}
func (b *retainedGreedyBatchBackend) Free(x compute.Tensor) {
	if _, ok := b.cloneReleases[x.Buf()]; ok {
		b.cloneReleases[x.Buf()]++
	}
	b.Backend.Free(x)
}

// Exercise the optional full-logit fused branch too, with the reference math.
// This is a capability/ordering fixture, not a native fused-kernel witness.
type retainedGreedyFusedBackend struct{ *retainedGreedyBatchBackend }

func (b *retainedGreedyFusedBackend) RMSNormMatMul(w, x, norm compute.Tensor, eps float32) compute.Tensor {
	return b.Backend.MatMul(w, b.Backend.RMSNorm(x, norm, eps))
}

func retainedGreedyBatchFixture(t *testing.T, capture, fused bool) (*Session, *Session, *retainedGreedyBatchBackend) {
	t.Helper()
	ref, ok := compute.Lookup("cpu-ref")
	if !ok || ref == nil {
		t.Fatal("cpu-ref backend unavailable")
	}
	cloner, ok := ref.(compute.TensorCloner)
	if !ok {
		t.Fatal("cpu-ref tensor cloner unavailable")
	}
	m := NewSynthetic(Config{HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1})
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("close weights: %v", err)
		}
	})
	b := &retainedGreedyBatchBackend{Backend: ref, cloner: cloner, captureEnabled: capture, cloneReleases: make(map[compute.Buffer]int)}
	var backend compute.Backend = b
	if fused {
		backend = &retainedGreedyFusedBackend{b}
	}
	full, selected := m.NewBackendSession(ref), m.NewBackendSession(backend)
	t.Cleanup(full.Close)
	t.Cleanup(selected.Close)
	return full, selected, b
}

func retainedGreedySameLogits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("logit lengths %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("logit[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestStepGreedyRetainedArgmaxPrecedesBatchFlush(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		retain, full, capture, fused bool
	}{
		{name: "retained", retain: true},
		{name: "retained_fused", retain: true, fused: true},
		{name: "retained_after_capture", retain: true, capture: true},
		{name: "retained_fused_after_capture", retain: true, capture: true, fused: true},
		{name: "token_only_control"},
		{name: "full_logits_control", full: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			full, selected, b := retainedGreedyBatchFixture(t, tc.capture, tc.fused)
			prompt := []int{3, 17, 5, 23}
			id := argmaxF32(full.Prefill(prompt))
			selected.Prefill(prompt)
			selected.SetRetainGreedyLogits(tc.retain)
			beforeEnds, beforeFlushes := b.graphEnds, b.flushes
			var last []float32
			for step := 0; step < 3; step++ {
				last = full.Step(id)
				want := argmaxF32(last)
				if tc.full {
					retainedGreedySameLogits(t, selected.Step(id), last)
				} else {
					next, logits := selected.StepGreedy(id)
					if next != want || logits != nil {
						t.Fatalf("step%d next=%d logits=%d, want token%d only", step, next, len(logits), want)
					}
				}
				if selected.halKV.Len() != full.halKV.Len() {
					t.Fatal("selected forward changed KV cardinality")
				}
				id = want
			}
			wantArgmax := 3
			if tc.full {
				wantArgmax = 0
			}
			if b.argmaxInBatch != wantArgmax || b.argmaxOutsideBatch != 0 || b.argmaxDuringCapture != 0 {
				t.Fatalf("argmax in-batch=%d outside=%d during-capture=%d, want%d/0/0", b.argmaxInBatch, b.argmaxOutsideBatch, b.argmaxDuringCapture, wantArgmax)
			}
			if b.batch || b.capture || b.clonesInBatch != 0 || b.flushes-beforeFlushes != 3 {
				t.Fatalf("batch=%t capture=%t clone-in-batch=%d flushes=%d, want closed/closed/0/3", b.batch, b.capture, b.clonesInBatch, b.flushes-beforeFlushes)
			}
			wantEnds := 0
			if tc.capture {
				wantEnds = 3
			}
			if b.graphEnds-beforeEnds != wantEnds || b.graphAborts != 0 {
				t.Fatal("graph completion ordering changed")
			}
			wantClones := 0
			if tc.retain {
				wantClones = 3
				retainedGreedySameLogits(t, selected.TakeGreedyLogits(), last)
			}
			if b.clones != wantClones || selected.TakeGreedyLogits() != nil {
				t.Fatal("retained distribution lifecycle changed")
			}
			for _, frees := range b.cloneReleases {
				if frees != 1 {
					t.Fatalf("retained clone freed%d times, want1", frees)
				}
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestStepGreedyRetainedArgmaxPanicStillClosesBatch(t *testing.T) {
	_, selected, b := retainedGreedyBatchFixture(t, true, false)
	selected.Prefill([]int{3, 17, 5})
	selected.SetRetainGreedyLogits(true)
	beforeEnds, beforeFlushes := b.graphEnds, b.flushes
	b.panicArgmax = true
	var caught any
	func() { defer func() { caught = recover() }(); selected.StepGreedy(11) }()
	if caught != "retained greedy argmax failure" {
		t.Fatalf("panic=%v, want original argmax failure", caught)
	}
	if b.batch || b.capture || b.flushes-beforeFlushes != 1 || b.graphEnds-beforeEnds != 1 || b.graphAborts != 0 || b.clones != 0 {
		t.Fatalf("failed reduction cleanup batch=%t capture=%t flushes=%d ends=%d aborts=%d clones=%d", b.batch, b.capture, b.flushes-beforeFlushes, b.graphEnds-beforeEnds, b.graphAborts, b.clones)
	}
	if b.argmaxInBatch != 1 || b.argmaxOutsideBatch != 0 || b.argmaxDuringCapture != 0 {
		t.Fatal("failed reduction crossed the wrong batch/graph boundary")
	}
}

// The split path ends its batch before its host suffix. Retention must keep the
// pre-existing upload -> backend Argmax sequence, including a nonzero winner.
// fak-test:runtime fast est=50ms lane=default
func TestStepGreedyRetainedSplitKeepsBackendSelection(t *testing.T) {
	full, selected, b := retainedGreedyBatchFixture(t, true, false)
	full.DenseGPULayers, selected.DenseGPULayers = 1, 1
	prompt := []int{3, 17, 5, 23}
	retainedGreedySameLogits(t, selected.Prefill(prompt), full.Prefill(prompt))
	selected.SetRetainGreedyLogits(true)
	beforeEnds, beforeFlushes := b.graphEnds, b.flushes
	var last []float32
	for step, id := range []int{11, 29, 7} {
		last = full.Step(id)
		want := argmaxF32(last)
		if want == 0 {
			t.Fatalf("fixture control: split step%d needs a nonzero reference winner", step)
		}
		next, logits := selected.StepGreedy(id)
		if next != want || logits != nil {
			t.Fatalf("split step%d next=%d logits=%d, want token%d only", step, next, len(logits), want)
		}
		if selected.halKV.Len() != full.halKV.Len() || selected.Cache.Len() != full.Cache.Len() {
			t.Fatal("split host/backend KV cardinality changed")
		}
	}
	retainedGreedySameLogits(t, selected.TakeGreedyLogits(), last)
	if b.argmaxOutsideBatch != 3 || b.argmaxInBatch != 0 || b.argmaxDuringCapture != 0 || b.graphEnds != beforeEnds || b.graphAborts != 0 {
		t.Fatalf("split reduction route changed: outside=%d in-batch=%d capture=%d ends=%d aborts=%d", b.argmaxOutsideBatch, b.argmaxInBatch, b.argmaxDuringCapture, b.graphEnds-beforeEnds, b.graphAborts)
	}
	if b.batch || b.capture || b.clones != 3 || b.clonesInBatch != 0 || b.flushes-beforeFlushes != 6 || selected.TakeGreedyLogits() != nil {
		t.Fatal("split retention or batch lifecycle changed")
	}
	for _, frees := range b.cloneReleases {
		if frees != 1 {
			t.Fatalf("split retained clone freed%d times, want1", frees)
		}
	}
}
