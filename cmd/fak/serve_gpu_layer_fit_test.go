package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serveSynthLayeredWeightSource is a 4-layer qwen2 header whose per-layer tensors carry
// "blk.<i>." names (unmapped suffixes keep the estimator on the raw payload plan) plus one
// non-layer tensor, so a --gpu-layers split has a known host share: 3 of 4 equal layers.
func serveSynthLayeredWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                   {Type: ggufload.TypeString, Value: "qwen2"},
			"qwen2.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(16)},
			"qwen2.embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(32)},
			"qwen2.block_count":                      {Type: ggufload.TypeUint64, Value: uint64(4)},
			"qwen2.feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(64)},
			"qwen2.attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(4)},
			"qwen2.attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(2)},
			"qwen2.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
			"qwen2.rope.freq_base":                   {Type: ggufload.TypeFloat32, Value: float32(10000)},
			"tokenizer.ggml.eos_token_id":            {Type: ggufload.TypeUint32, Value: uint32(2)},
		},
		Tensors: []ggufload.TensorInfo{
			{Name: "x_embd", Dims: []uint64{256 * 64}, Type: ggufload.TensorQ4_K},
			{Name: "blk.0.synth.weight", Dims: []uint64{256 * 1024}, Type: ggufload.TensorQ4_K},
			{Name: "blk.1.synth.weight", Dims: []uint64{256 * 1024}, Type: ggufload.TensorQ4_K},
			{Name: "blk.2.synth.weight", Dims: []uint64{256 * 1024}, Type: ggufload.TensorQ4_K},
			{Name: "blk.3.synth.weight", Dims: []uint64{256 * 1024}, Type: ggufload.TensorQ4_K},
		},
	}
	ws, err := ggufload.NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func TestFitServeGGUFOnDeviceHonorsGPULayers(t *testing.T) {
	ws := serveSynthLayeredWeightSource(t)
	full, err := serveGGUFMemoryPlan(ws, false, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("serveGGUFMemoryPlan: %v", err)
	}
	fullDevice := full.DeviceTotal()
	if fullDevice <= 0 {
		t.Fatalf("synthetic plan has no device weights: %+v", full)
	}
	// Device sized for about half the full weight plan (after the 15% admission headroom): the
	// full placement must refuse, --gpu-layers 1 (one of four layers on device) must admit.
	be := serveCapBackend{Backend: compute.Default(), total: fullDevice * 2 / 3, free: compute.FreeUnknown, known: true, uploadDtype: true}

	var fe *compute.FitError
	if err := fitServeGGUFOnDevice(ws, be, false, 0, 0); !errors.As(err, &fe) {
		t.Fatalf("full device placement: want *compute.FitError, got %v", err)
	}
	if err := fitServeGGUFOnDevice(ws, be, false, 0, 1); err != nil {
		t.Fatalf("--gpu-layers 1 must charge layers [1,4) to host and fit: %v", err)
	}
	// N >= NumLayers is full placement again: no re-scope, so it refuses like N=0.
	if err := fitServeGGUFOnDevice(ws, be, false, 0, 4); !errors.As(err, &fe) {
		t.Fatalf("--gpu-layers 4 of 4: want *compute.FitError, got %v", err)
	}
}

func TestApplyDenseGPULayerSplitConservesBytes(t *testing.T) {
	ws := serveSynthLayeredWeightSource(t)
	full, err := serveGGUFMemoryPlan(ws, false, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("serveGGUFMemoryPlan: %v", err)
	}
	total := func(p compute.MemoryPlan) (sum int64) {
		for _, d := range p {
			sum += d.Bytes
		}
		return sum
	}
	split := applyDenseGPULayerSplit(full, ws, 1)
	if got, want := total(split), total(full); got != want {
		t.Fatalf("split total = %d, want conserved %d", got, want)
	}
	// Three of four equal layer tensors move; the non-layer tensor stays device-charged.
	frac := serveDenseGPULayerHostFraction(ws, 1)
	if frac <= 0.5 || frac >= 0.75 {
		t.Fatalf("host fraction = %v, want in (0.5, 0.75) for 3 of 4 layers plus a device-charged embd", frac)
	}
	if split.DeviceTotal() >= full.DeviceTotal() {
		t.Fatalf("split device total %d must be below full %d", split.DeviceTotal(), full.DeviceTotal())
	}
	for _, n := range []int{-1, 0, 4, 9} {
		if got := applyDenseGPULayerSplit(full, ws, n); got.DeviceTotal() != full.DeviceTotal() {
			t.Fatalf("gpuLayers=%d must leave the plan device-charged: %d vs %d", n, got.DeviceTotal(), full.DeviceTotal())
		}
	}
}

func TestGGUFBlockLayer(t *testing.T) {
	cases := []struct {
		name string
		want int
		ok   bool
	}{
		{"blk.0.attn_q.weight", 0, true},
		{"blk.63.ffn_down.weight", 63, true},
		{"token_embd.weight", 0, false},
		{"blk.x.attn_q.weight", 0, false},
		{"blk.7", 0, false},
	}
	for _, c := range cases {
		got, ok := ggufBlockLayer(c.name)
		if got != c.want || ok != c.ok {
			t.Fatalf("ggufBlockLayer(%q) = (%d,%v), want (%d,%v)", c.name, got, ok, c.want, c.ok)
		}
	}
}
