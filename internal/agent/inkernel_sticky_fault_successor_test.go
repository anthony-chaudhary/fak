package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// stickySuccessorBackend models one shared execution domain, not a rebuilt
// CUDA context. Admission belongs to the backend; replacing a planner must not
// replace its latch. Only MatMul is fault-injected, before delegated math.
type stickySuccessorBackend struct {
	compute.Backend
	latch    *compute.DeviceFaultLatch
	forwards int
}

func (b *stickySuccessorBackend) DeviceFaultLatch() *compute.DeviceFaultLatch { return b.latch }

func (b *stickySuccessorBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if err := compute.AdmitDevice(b, "successor-matmul"); err != nil {
		panic(err)
	}
	b.forwards++
	if b.forwards == 1 {
		panic(b.latch.Observe(compute.DeviceFaultExecution, "injected-fatal-forward", 700))
	}
	return b.Backend.MatMul(w, x)
}

// This is a host-only Complete-boundary witness, not CUDA execution or HTTP
// parity. Unlike transient panic recovery, a fatal backend remains poisoned
// when a new planner/session uses the same execution domain.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelCompleteSuccessorPlannerPreservesStickyDeviceFault(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend unavailable")
	}
	backend := &stickySuccessorBackend{Backend: base, latch: compute.NewDeviceFaultLatch("cuda", 3)}
	tok := stickySuccessorTokenizer(t)
	newPlanner := func() *InKernelPlanner {
		p := NewInKernelPlanner(model.NewSynthetic(stickySuccessorConfig()), tok, "sticky-successor", false, backend, false)
		p.quant = false
		p.maxNew = 2
		return p
	}
	first := newPlanner()
	messages := []Message{{Role: RoleUser, Content: "hello"}}
	completion, err := first.Complete(context.Background(), messages, nil)
	var fault *compute.DeviceFaultError
	if completion != nil || !errors.As(err, &fault) || fault.Class != compute.DeviceFaultExecution || fault.Code != 700 {
		t.Fatalf("first Complete = (%#v, %v), want nil/typed injected fatal fault", completion, err)
	}
	if backend.forwards != 1 {
		t.Fatalf("first Complete admitted %d forwards, want exactly one", backend.forwards)
	}

	// Construct only after the first failure: fresh planner/session ownership
	// cannot stand in for reconstruction of the shared execution domain.
	successor := newPlanner()
	completion, err = successor.Complete(context.Background(), messages, nil)
	fault = nil
	if completion != nil || !errors.As(err, &fault) || fault.Class != compute.DeviceFaultExecution || fault.Code != 700 {
		t.Fatalf("successor Complete = (%#v, %v), want nil/sticky typed refusal", completion, err)
	}
	if backend.forwards != 1 {
		t.Fatalf("successor admitted another forward: total=%d, want one", backend.forwards)
	}
	snapshot := backend.latch.Snapshot()
	if snapshot.Health != compute.DevicePoisoned || snapshot.RefusedCalls == 0 || snapshot.Attempts != 0 || snapshot.Recoveries != 0 {
		t.Fatalf("planner replacement changed fatal backend state: %+v", snapshot)
	}
}

// Test-only fixture copied verbatim (names scoped locally) from
// internal/agent/zz_prefix_probe_test.go at 58ccf440bb467260b010157e8f1f9dd24feb4aaf.
func stickySuccessorByteVocab() string {
	// Reproduce GPT-2 byte->unicode (same table internal/tokenizer uses).
	bs := []int{}
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	for i := 0xA1; i <= 0xAC; i++ {
		bs = append(bs, i)
	}
	for i := 0xAE; i <= 0xFF; i++ {
		bs = append(bs, i)
	}
	cs := append([]int{}, bs...)
	n := 0
	inBs := func(b int) bool {
		for _, x := range bs {
			if x == b {
				return true
			}
		}
		return false
	}
	for b := 0; b < 256; b++ {
		if !inBs(b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	vocab := map[string]int{
		"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
	}
	id := 3
	for i := range bs {
		r := rune(cs[i])
		vocab[string(r)] = id
		id++
	}
	doc := map[string]any{
		"model":   map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"decoder": map[string]any{"type": "ByteLevel"},
		"added_tokens": []map[string]any{
			{"id": 0, "content": "<|endoftext|>", "special": true},
			{"id": 1, "content": "<|im_start|>", "special": true},
			{"id": 2, "content": "<|im_end|>", "special": true},
		},
	}
	b, _ := json.Marshal(doc)
	return string(b)
}

// Test-only fixture copied verbatim (names scoped locally) from
// internal/agent/zz_prefix_probe_test.go at 58ccf440bb467260b010157e8f1f9dd24feb4aaf.
func stickySuccessorTokenizer(t *testing.T) *tokenizer.Tokenizer {
	tk, err := tokenizer.ParseJSON([]byte(stickySuccessorByteVocab()))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	return tk
}

// Test-only fixture copied verbatim (names scoped locally) from
// internal/agent/inkernel_planner_concurrency_test.go at 58ccf440bb467260b010157e8f1f9dd24feb4aaf.
func stickySuccessorConfig() model.Config {
	return model.Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         320, // covers the 256-byte vocab + ChatML specials the probe tokenizer emits
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}
