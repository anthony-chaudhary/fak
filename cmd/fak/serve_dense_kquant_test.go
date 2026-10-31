package main

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// q2kCapBackend is a minimal backend stub whose two capability axes (native packed
// Q2_K matmul and quantized upload) are independently settable, so the serve dense
// K-quant option builder can be exercised against every combination.
type q2kCapBackend struct {
	compute.Backend
	uploadDtype bool
	q2k         bool
}

func (b q2kCapBackend) Caps() compute.Caps {
	return compute.Caps{UploadDtype: b.uploadDtype}
}

func (b q2kCapBackend) SupportsQ2K() bool { return b.q2k }

func TestServeDenseKQuantOptionsNilBackendIsEmpty(t *testing.T) {
	if got := serveDenseKQuantOptions(nil); got != nil {
		t.Fatalf("nil backend options = %v, want nil", got)
	}
}

func TestServeDenseKQuantOptionsDisableRawResidencyForDeviceBackend(t *testing.T) {
	opts := serveDenseKQuantOptions(compute.Default())
	effects := ggufload.ApplyQ4KLoadOptions(opts)
	if effects.DenseKQuantResident {
		t.Fatal("device backend must route dense k-quant through Q8 (DenseKQuantResident must be false)")
	}
	if effects.DenseQ2KResident {
		t.Fatal("backend without native Q2_K must not enable dense Q2_K residency")
	}
}

func TestServeDenseKQuantOptionsRetainQ2KOnlyWhenBackendAdvertisesBoth(t *testing.T) {
	cases := []struct {
		name        string
		q2k         bool
		uploadDtype bool
		wantDenseK  bool
		wantQ2K     bool
	}{
		// #12757: the 246 GiB UD-Q2_K_XL artifact must stay packed on a backend that
		// advertises BOTH native packed Q2_K matmul and quantized upload. Expanding it
		// to Q8/F32 at load is exactly what makes a 128 GB Halo refuse the model.
		{name: "q2k+native+upload", q2k: true, uploadDtype: true, wantDenseK: false, wantQ2K: true},
		// Either capability alone is NOT enough: without native Q2_K the packed tensor
		// is unreachable at decode; without UploadDtype there is no quantized device
		// seam to hold it.
		{name: "q2k-only", q2k: true, uploadDtype: false, wantDenseK: false, wantQ2K: false},
		{name: "upload-only", q2k: false, uploadDtype: true, wantDenseK: false, wantQ2K: false},
		{name: "neither", q2k: false, uploadDtype: false, wantDenseK: false, wantQ2K: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			backend := q2kCapBackend{Backend: compute.Default(), uploadDtype: tc.uploadDtype, q2k: tc.q2k}
			// Drive the options through the loader's own option-application path so the
			// assertion is on the EFFECT of the option list, not its slice length.
			effects := ggufload.ApplyQ4KLoadOptions(serveDenseKQuantOptions(backend))
			if effects.DenseKQuantResident != tc.wantDenseK {
				t.Fatalf("DenseKQuantResident = %v, want %v", effects.DenseKQuantResident, tc.wantDenseK)
			}
			if effects.DenseQ2KResident != tc.wantQ2K {
				t.Fatalf("DenseQ2KResident = %v, want %v", effects.DenseQ2KResident, tc.wantQ2K)
			}
		})
	}
}

type q6kCapBackend struct {
	compute.Backend
	deviceMemory, uploadDtype, q6k bool
}

func (b q6kCapBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = b.deviceMemory, b.uploadDtype
	return c
}
func (b q6kCapBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.Q6_K && b.q6k
}

type optionalQ6KCapBackend struct {
	q6kCapBackend
	native bool
}

func (b optionalQ6KCapBackend) SupportsQ6KMatMul() bool { return b.native }

func TestServeDenseKQuantOptionsEnableQ6KOnlyWhenBackendAdvertisesBoth(t *testing.T) {
	backends := []struct {
		name    string
		backend compute.Backend
	}{
		{name: "common-dtype-contract", backend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}},
		{name: "optional-native-contract", backend: optionalQ6KCapBackend{q6kCapBackend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}, native: true}},
	}
	for _, tc := range backends {
		t.Run(tc.name, func(t *testing.T) {
			effects := ggufload.ApplyQ4KLoadOptions(serveDenseKQuantOptions(tc.backend))
			if !effects.DenseQ6KResident {
				t.Fatal("DenseQ6KResident = false, want true")
			}
			if effects.DenseKQuantResident || effects.DenseQ2KResident {
				t.Fatalf("Q6_K admission changed other residency: %+v", effects)
			}
		})
	}
}

func TestServeDenseKQuantOptionsDeclinesQ6KPartialCapability(t *testing.T) {
	backends := []struct {
		name    string
		backend compute.Backend
	}{
		{name: "nil", backend: nil},
		{name: "cpu", backend: compute.Default()},
		{name: "upload-only", backend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true}},
		{name: "execution-only", backend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, q6k: true}},
		{name: "host-memory", backend: q6kCapBackend{Backend: compute.Default(), uploadDtype: true, q6k: true}},
		{name: "optional-bundle-missing", backend: optionalQ6KCapBackend{q6kCapBackend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}}},
	}
	for _, tc := range backends {
		t.Run(tc.name, func(t *testing.T) {
			effects := ggufload.ApplyQ4KLoadOptions(serveDenseKQuantOptions(tc.backend))
			if effects.DenseQ6KResident {
				t.Fatal("DenseQ6KResident = true, want false")
			}
			if effects.DenseKQuantResident || effects.DenseQ2KResident {
				t.Fatalf("declined Q6_K changed other residency: %+v", effects)
			}
		})
	}
}

// serveGDNTestBackend exposes the real CUDA source dtype declarations without
// opening a device. The non-CUDA name is a control for the same loader inputs,
// not a claim that either device has executed this fixture.
type serveGDNTestBackend struct {
	compute.Backend
	name string
}

func (b serveGDNTestBackend) Name() string { return b.name }
func (b serveGDNTestBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b serveGDNTestBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	for _, op := range compute.CUDADeviceOpTable() {
		if op.Operation == "matmul" && op.Format == dt.String() {
			return true
		}
	}
	return false
}

// fak-test:runtime fast est=100ms lane=default
func TestServeCUDAGDNLoaderKeepsSupportedWeights(t *testing.T) {
	for _, key := range []string{"FAK_QWEN35_NATIVE_QK_ROWS", "FAK_QWEN35_NATIVE_GDN_INPROJ_ROWS", "FAK_QWEN35_NATIVE_SSM_OUT"} {
		t.Setenv(key, "1")
	}
	t.Setenv("FAK_W3_MLP", "0")
	// Only the relevant projection payloads are needed to witness real loader
	// storage. The 64-layer hybrid topology is real, but this is not a complete
	// inference model or an arithmetic/device parity fixture.
	meta := map[string]ggufload.Value{
		"general.architecture":                    {Type: ggufload.TypeString, Value: "qwen35"},
		"qwen35.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
	}
	for key, value := range map[string]uint64{
		"context_length": 16, "embedding_length": 256, "block_count": 64,
		"feed_forward_length": 256, "attention.head_count": 2,
		"attention.head_count_kv": 1, "attention.key_length": 128,
		"attention.value_length": 128, "full_attention_interval": 4,
		"nextn_predict_layers": 0, "ssm.conv_kernel": 4, "ssm.state_size": 64,
		"ssm.group_count": 2, "ssm.time_step_rank": 4, "ssm.inner_size": 256,
	} {
		meta["qwen35."+key] = ggufload.Value{Type: ggufload.TypeUint64, Value: value}
	}
	projections := []struct {
		gguf, canonical  string
		kind             ggufload.TensorType
		rows, blockBytes int
	}{
		{"blk.0.attn_qkv.weight", "model.layers.0.linear_attn.in_proj_qkv.weight", ggufload.TensorQ4_K, 512, 144},
		{"blk.1.attn_qkv.weight", "model.layers.1.linear_attn.in_proj_qkv.weight", ggufload.TensorQ6_K, 512, 210},
		{"blk.0.attn_gate.weight", "model.layers.0.linear_attn.in_proj_z.weight", ggufload.TensorQ4_K, 256, 144},
		{"blk.0.ssm_out.weight", "model.layers.0.linear_attn.out_proj.weight", ggufload.TensorQ4_K, 256, 144},
		{"blk.3.attn_q.weight", "model.layers.3.self_attn.q_proj.weight", ggufload.TensorQ4_K, 512, 144},
		{"blk.3.attn_k.weight", "model.layers.3.self_attn.k_proj.weight", ggufload.TensorQ4_K, 128, 144},
	}
	var payload bytes.Buffer
	file := &ggufload.File{Metadata: meta}
	for _, p := range projections {
		file.Tensors = append(file.Tensors, ggufload.TensorInfo{
			Name: p.gguf, Dims: []uint64{256, uint64(p.rows)}, Type: p.kind, FileOffset: int64(payload.Len()),
		})
		payload.Write(make([]byte, p.rows*p.blockBytes))
	}
	// An unrelated dense IQ projection must remain packed on both paths.
	file.Tensors = append(file.Tensors, ggufload.TensorInfo{
		Name: "blk.0.ffn_gate.weight", Dims: []uint64{256, 256}, Type: ggufload.TensorIQ2_XXS, FileOffset: int64(payload.Len()),
	})
	payload.Write(make([]byte, 256*66))
	cfg, err := file.Config()
	if err != nil || !fakmodel.Qwen38ExactMetalQ8Runtime(cfg) {
		t.Fatalf("fixture must reach exact hybrid loader policy: cfg=%+v err=%v", cfg, err)
	}
	for _, name := range []string{"cuda", "vulkan"} {
		t.Run(name, func(t *testing.T) {
			backend := serveGDNTestBackend{Backend: compute.Default(), name: name}
			ws, err := ggufload.NewWeightSource(file, bytes.NewReader(payload.Bytes()), int64(payload.Len()))
			if err != nil {
				t.Fatal(err)
			}
			defer ws.Close()
			m, err := ws.QuantModelQ4KProfileOptions(nil, serveDenseKQuantOptions(backend)...)
			if err != nil {
				t.Fatal(err)
			}
			defer m.CloseWeights()
			for _, p := range projections {
				q8, q4, kq := m.HasQ8(p.canonical), m.HasQ4K(p.canonical), m.HasKQuant(p.canonical)
				t.Logf("%s: Q8=%v Q4_K=%v KQuant=%v", p.canonical, q8, q4, kq)
				if name == "cuda" {
					if !q8 || q4 || kq {
						t.Errorf("CUDA whole-GDN compatibility requires prior Q8 storage: %s Q8=%v Q4_K=%v KQuant=%v", p.canonical, q8, q4, kq)
					}
				} else if q8 || q4 != (p.kind == ggufload.TensorQ4_K) || kq != (p.kind == ggufload.TensorQ6_K) {
					t.Errorf("non-CUDA raw projection changed: %s Q8=%v Q4_K=%v KQuant=%v", p.canonical, q8, q4, kq)
				}
			}
			if got, want := m.Qwen35OutProjTiled("model.layers.0.linear_attn.out_proj.weight"), name != "cuda"; got != want {
				t.Errorf("tiled GDN out = %v, want %v", got, want)
			}
			const iq = "model.layers.0.mlp.gate_proj.weight"
			if !m.HasKQuant(iq) || m.HasQ8(iq) || !backend.SupportsDeviceWeightDtype(compute.IQ2_XXS) {
				t.Error("independent IQ projection/capability must remain packed and enabled")
			}
		})
	}
}
