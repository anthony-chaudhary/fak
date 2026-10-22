package main

import (
	"bytes"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=50ms lane=default
func TestFailureCapacityDiagnosticPreservesPanic(t *testing.T) {
	original := readVulkanMemoryCapacity
	readVulkanMemoryCapacity = func(compute.Backend) (any, error) {
		return map[string]any{"guard_refused_allocations": uint64(3)}, nil
	}
	t.Cleanup(func() { readVulkanMemoryCapacity = original })
	var stderr bytes.Buffer
	sentinel := errors.New("allocation failed")
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		func() {
			defer reportFailureCapacityOnPanic(compute.Default(), nil, &stderr)
			panic(sentinel)
		}()
	}()
	if recovered != sentinel {
		t.Fatalf("recovered=%v, want original panic", recovered)
	}
	for _, want := range [][]byte{[]byte(`"schema":"fak-dflash2-failure-capacity/1"`), []byte(`"phase":"failure"`), []byte(`"guard_refused_allocations":3`)} {
		if !bytes.Contains(stderr.Bytes(), want) {
			t.Fatalf("diagnostic %q missing %q", stderr.String(), want)
		}
	}
}

type mixedQuantBenchBackend struct{ compute.Backend }

func (b mixedQuantBenchBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = true
	caps.UploadDtype = true
	return caps
}

// fak-test:runtime fast est=500ms lane=default
func TestNewMixedQuantSessionUsesQ8WeightsWithQ4KSelected(t *testing.T) {
	const width = 256
	b := model.NewQuantBuilder(model.Config{
		HiddenSize: width, IntermediateSize: width, NumLayers: 1, NumHeads: 1,
		NumKVHeads: 1, HeadDim: width, VocabSize: width, RMSNormEps: 1e-5,
	}, false)
	addF32 := func(name string, shape []int, fill float32) {
		t.Helper()
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = fill
		}
		if err := b.AddF32Tensor(name, shape, data); err != nil {
			t.Fatalf("add %s: %v", name, err)
		}
	}
	addF32("model.embed_tokens.weight", []int{width, width}, 0.01)
	addF32("model.layers.0.input_layernorm.weight", []int{width}, 1)
	addF32("model.layers.0.self_attn.q_proj.weight", []int{width, width}, 0.01)
	addF32("model.layers.0.self_attn.k_proj.weight", []int{width, width}, 0.01)
	addF32("model.layers.0.self_attn.v_proj.weight", []int{width, width}, 0.01)
	addF32("model.layers.0.self_attn.o_proj.weight", []int{width, width}, 0.01)
	addF32("model.layers.0.post_attention_layernorm.weight", []int{width}, 1)
	addF32("model.layers.0.mlp.gate_proj.weight", []int{width, width}, 0.01)
	if err := b.AddResidentQ4K("model.layers.0.mlp.up_proj.weight", []int{width, width}, make([]byte, width*144)); err != nil {
		t.Fatal(err)
	}
	if err := b.AddResidentQ4K("model.layers.0.mlp.down_proj.weight", []int{width, width}, make([]byte, width*144)); err != nil {
		t.Fatal(err)
	}
	addF32("model.norm.weight", []int{width}, 1)
	addF32("lm_head.weight", []int{width, width}, 0.01)
	m, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	report := m.ResidentReport()
	if report.Q8Tensors == 0 || report.Q4KTensors == 0 {
		t.Fatalf("fixture is not mixed: %+v", report)
	}
	s, err := newMixedQuantSession(m, mixedQuantBenchBackend{Backend: compute.Default()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	if got := s.Prefill([]int{3, 7}); len(got) != width {
		t.Fatalf("logits=%d", len(got))
	}
}
