package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

type v41PackedBackend struct {
	compute.Backend
	device, upload, q2, q3, q6 bool
	dtypes                     map[compute.Dtype]bool
}

func (b v41PackedBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = b.device, b.upload
	return c
}
func (b v41PackedBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool { return b.dtypes[dt] }
func (b v41PackedBackend) SupportsQ2K() bool                               { return b.q2 }
func (b v41PackedBackend) SupportsQ3KMatMul() bool                         { return b.q3 }
func (b v41PackedBackend) SupportsQ6KMatMul() bool                         { return b.q6 }

// The absent probe is a separate method set: false and absent are distinct contracts.
type v41PackedWithoutQ3Probe struct {
	compute.Backend
	caps   compute.Caps
	dtypes map[compute.Dtype]bool
}

func (b v41PackedWithoutQ3Probe) Caps() compute.Caps { return b.caps }
func (b v41PackedWithoutQ3Probe) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return b.dtypes[dt]
}
func (b v41PackedWithoutQ3Probe) SupportsQ2K() bool       { return true }
func (b v41PackedWithoutQ3Probe) SupportsQ6KMatMul() bool { return true }

func v41PackedNativeBackend() v41PackedBackend {
	return v41PackedBackend{Backend: compute.Default(), device: true, upload: true, q2: true, q3: true, q6: true,
		dtypes: map[compute.Dtype]bool{compute.Q2_K: true, compute.Q3_K: true, compute.Q6_K: true}}
}

func v41PackedTestEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"FAK_STREAM_Q4K", "FAK_METAL_STREAM_Q4K", "FAK_W3_MLP", "FAK_F32", "FAK_EXPERT_STREAM"} {
		t.Setenv(key, "")
	}
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_GGUF_MMAP", "0")
}

// fak-test:runtime fast est=1s
func TestServeV41PackedFamilyCapabilityContract(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_PACKED_ISOLATED_TEST") != t.Name() {
		t.Parallel()
		v41PackedIsolatedTest(t)
		return
	}
	v41PackedTestEnv(t)
	artifact := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_v.weight", Type: ggufload.TensorQ2_K},
		{Name: "blk.0.ffn_down_shexp.weight", Type: ggufload.TensorQ3_K},
		{Name: "output.weight", Type: ggufload.TensorQ6_K},
		{Name: "output_norm.weight", Type: ggufload.TensorF32},
	})
	if artifact.Name != "mixed(Q2_K+Q3_K+Q6_K)" || artifact.Recipe != "" || artifact.Q4KResident {
		t.Fatalf("encoding/recipe contract: name=%q recipe=%q q4=%t", artifact.Name, artifact.Recipe, artifact.Q4KResident)
	}
	t.Run("native", func(t *testing.T) {
		if !serveArtifactResidentQ4K(v41PackedNativeBackend(), artifact) {
			t.Fatal("native Q2/Q3/Q6 family must select packed load")
		}
	})
	for _, axis := range []string{"device", "upload", "q2-kernel", "q3-kernel", "q6-kernel", "q2-dtype", "q3-dtype", "q6-dtype"} {
		t.Run(axis, func(t *testing.T) {
			b := v41PackedNativeBackend()
			switch axis {
			case "device":
				b.device = false
			case "upload":
				b.upload = false
			case "q2-kernel":
				b.q2 = false
			case "q3-kernel":
				b.q3 = false
			case "q6-kernel":
				b.q6 = false
			case "q2-dtype":
				b.dtypes[compute.Q2_K] = false
			case "q3-dtype":
				b.dtypes[compute.Q3_K] = false
			case "q6-dtype":
				b.dtypes[compute.Q6_K] = false
			}
			if serveArtifactResidentQ4K(b, artifact) {
				t.Fatalf("incomplete %s must decline packed family", axis)
			}
		})
	}
	t.Run("q3-probe-absent", func(t *testing.T) {
		b := v41PackedNativeBackend()
		absent := v41PackedWithoutQ3Probe{Backend: b.Backend, caps: b.Caps(), dtypes: b.dtypes}
		if serveArtifactResidentQ4K(absent, artifact) {
			t.Fatal("dtype presence cannot replace a missing Q3 execution probe")
		}
	})
	t.Run("rollback", func(t *testing.T) {
		t.Setenv("FAK_Q4K", "0")
		if serveArtifactResidentQ4K(v41PackedNativeBackend(), artifact) {
			t.Fatal("explicit rollback must select legacy load")
		}
	})
	for _, name := range []string{"unknown", "Q8_0", "mixed(Q2_K+type999)", "Q2_K+Q3_K+Q6_K"} {
		t.Run(name, func(t *testing.T) {
			if serveArtifactResidentQ4K(v41PackedNativeBackend(), ggufload.ArtifactQuant{Name: name}) {
				t.Fatal("unqualified family must decline packed load")
			}
		})
	}
	for _, art := range []ggufload.ArtifactQuant{{Name: "Q4_K", Q4KResident: true}, {Name: "mixed(Q2_K+Q4_K+Q6_K)", Recipe: "UD-Q2_K_XL"}} {
		if !serveArtifactResidentQ4K(v41PackedNativeBackend(), art) {
			t.Fatalf("legacy family changed: %q", art.Name)
		}
	}
	if !serveArtifactResidentQ4K(nil, ggufload.ArtifactQuant{Name: "PQ2_0"}) || serveArtifactResidentQ4K(v41PackedNativeBackend(), ggufload.ArtifactQuant{Name: "PQ2_0"}) {
		t.Fatal("legacy CPU PQ2 arm changed")
	}
}

type v41PackedFixtureTensor struct {
	name string
	typ  ggufload.TensorType
	data []byte
}

func v41PackedQ3Block() []byte {
	raw := make([]byte, 110)
	// Zero scale codes mean -32; zero low/high codes mean -4; d=1 yields 128.
	binary.LittleEndian.PutUint16(raw[108:], 0x3c00)
	return raw
}

func writeV41PackedFixture(t *testing.T, path string, tensors []v41PackedFixtureTensor, shard, count, total int) {
	t.Helper()
	var b bytes.Buffer
	kvs := uint64(9)
	if count > 1 {
		kvs += 3
	}
	writeMinimalHeaderForTest(&b, uint64(len(tensors)), kvs)
	writeKVStringForTest(&b, "general.architecture", "llama")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "llama.embedding_length", 256)
	writeKVUint32ForTest(&b, "llama.block_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.key_length", 256)
	writeKVUint32ForTest(&b, "llama.feed_forward_length", 256)
	writeKVUint32ForTest(&b, "llama.context_length", 16)
	writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	if count > 1 {
		writeKVUint32ForTest(&b, "split.no", uint32(shard))
		writeKVUint32ForTest(&b, "split.count", uint32(count))
		writeKVUint32ForTest(&b, "split.tensors.count", uint32(total))
	}
	offset := uint64(0)
	for _, tensor := range tensors {
		dims := []uint64{256, 1}
		if tensor.typ == ggufload.TensorF32 || tensor.typ == ggufload.TensorBF16 {
			dims = []uint64{1}
		}
		writeTensorInfoForTest(&b, tensor.name, dims, uint32(tensor.typ), offset)
		offset = (offset + uint64(len(tensor.data)) + 31) &^ 31
	}
	padToAlignmentForTest(&b, 32)
	for _, tensor := range tensors {
		b.Write(tensor.data)
		padToAlignmentForTest(&b, 32)
	}
	if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=1s
func TestServeV41DenseQ3PayloadRetentionAndEstimate(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_PACKED_ISOLATED_TEST") != t.Name() {
		t.Parallel()
		v41PackedIsolatedTest(t)
		return
	}
	v41PackedTestEnv(t)
	path := filepath.Join(t.TempDir(), "dense-q3.gguf")
	payload := v41PackedQ3Block()
	writeV41PackedFixture(t, path, []v41PackedFixtureTensor{{name: "blk.0.ffn_down.weight", typ: ggufload.TensorQ3_K, data: payload}}, 0, 1, 1)
	artifact, err := ggufload.ClassifyArtifactQuant(path)
	if err != nil {
		t.Fatal(err)
	}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	for _, native := range []bool{true, false} {
		name := "native"
		if !native {
			name = "shader-absent"
		}
		t.Run(name, func(t *testing.T) {
			backend := v41PackedNativeBackend()
			backend.q3 = native
			opts := serveResidentQ4KLoadOptions(backend, path, native, artifact)
			m, err := ws.QuantModelQ4KProfileOptions(nil, opts...)
			if err != nil {
				t.Fatal(err)
			}
			defer m.CloseWeights()
			raw, packed := m.KQuantRaw("model.layers.0.mlp.down_proj.weight")
			if packed != native {
				t.Fatalf("actual Q3 retention=%t want=%t", packed, native)
			}
			plan, err := ws.EstimateQ4KLoadMemoryPlan(opts...)
			if err != nil {
				t.Fatal(err)
			}
			report := m.ResidentReport()
			if plan.Total() != report.TotalResidentBytes {
				t.Fatalf("estimate=%d actual=%d", plan.Total(), report.TotalResidentBytes)
			}
			if native {
				if !bytes.Equal(raw, payload) || report.KQuantBytes != 110 || report.Q8Bytes != 0 || plan.Total() != 110 {
					t.Fatalf("packed storage: raw=%d K=%d Q8=%d plan=%d", len(raw), report.KQuantBytes, report.Q8Bytes, plan.Total())
				}
				decoded := make([]float32, 256)
				fakmodel.DequantQ3K(decoded, raw)
				for i, value := range decoded {
					if value != 128 {
						t.Fatalf("decoded[%d]=%g want=128", i, value)
					}
				}
			} else if report.KQuantBytes != 0 || report.Q8Bytes <= 110 {
				t.Fatalf("fallback storage K=%d Q8=%d", report.KQuantBytes, report.Q8Bytes)
			}
		})
	}
}
