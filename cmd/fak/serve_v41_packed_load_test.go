package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
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
	for _, axis := range []string{"device", "upload", "q2-kernel", "q6-kernel", "q2-dtype", "q6-dtype"} {
		t.Run(axis, func(t *testing.T) {
			b := v41PackedNativeBackend()
			switch axis {
			case "device":
				b.device = false
			case "upload":
				b.upload = false
			case "q2-kernel":
				b.q2 = false
			case "q6-kernel":
				b.q6 = false
			case "q2-dtype":
				b.dtypes[compute.Q2_K] = false
			case "q6-dtype":
				b.dtypes[compute.Q6_K] = false
			}
			if serveArtifactResidentQ4K(b, artifact) {
				t.Fatalf("incomplete %s must decline packed family", axis)
			}
		})
	}
	// The missing-Q3 axes are no longer a full refusal: the NATIVE packed gate still declines
	// (the negative control is preserved), but the explicit mixed host-fallback arm admits the
	// artifact with honest storage — Q2_K packed device-native, dense Q3_K converted to Q8 with
	// charged bytes, routed experts retaining raw source bytes (fak#12640).
	for _, axis := range []string{"q3-kernel", "q3-dtype"} {
		t.Run(axis+"-fallback", func(t *testing.T) {
			b := v41PackedNativeBackend()
			if axis == "q3-kernel" {
				b.q3 = false
			} else {
				b.dtypes[compute.Q3_K] = false
			}
			if serveArtifactNativeKQuant(b, artifact) {
				t.Fatal("missing Q3 route must not relabel into native packed residency")
			}
			if !serveArtifactMixedQuantHostFallback(b, artifact) || !serveArtifactResidentQ4K(b, artifact) {
				t.Fatalf("CUDA-like backend without a Q3 probe must take the mixed host-fallback arm (%s)", axis)
			}
		})
	}
	t.Run("q3-probe-absent", func(t *testing.T) {
		b := v41PackedNativeBackend()
		absent := v41PackedWithoutQ3Probe{Backend: b.Backend, caps: b.Caps(), dtypes: b.dtypes}
		if serveArtifactNativeKQuant(absent, artifact) {
			t.Fatal("dtype presence cannot replace a missing Q3 execution probe")
		}
		if !serveArtifactMixedQuantHostFallback(absent, artifact) || !serveArtifactResidentQ4K(absent, artifact) {
			t.Fatal("an absent Q3 probe is the CUDA first-light class: the mixed host-fallback arm must admit it")
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

// v41MixedExpertPatternByte is the distinct, recoverable per-expert byte at offset i: keyed on
// BOTH the expert index and the offset so expert 1's region can never be confused with expert 0's.
func v41MixedExpertPatternByte(expert, i int) byte {
	return byte((expert*131 + i*7 + 17) & 0xff)
}

func v41MixedF32Bytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// v41MixedMoEExpertGGUF writes a tiny qwen3moe checkpoint shaped like the DS4.1 Flash intake
// inventory (fak#12640): packed Q2_K gate/up routed-expert slabs beside a packed Q3_K down slab
// plus an F32 router gate. Each expert's region carries a distinct byte pattern so a retained
// resident store can be matched back to the exact source expert.
func v41MixedMoEExpertGGUF(t *testing.T, E, H, I int) []byte {
	t.Helper()
	const q2kBlock, q3kBlock = 84, 110 // 256-weight super-blocks: Q2_K 84 B, Q3_K 110 B
	perQ2 := H * I / 256 * q2kBlock
	perQ3 := I * H / 256 * q3kBlock
	slab := func(per int) []byte {
		out := make([]byte, E*per)
		for x := 0; x < E; x++ {
			for i := 0; i < per; i++ {
				out[x*per+i] = v41MixedExpertPatternByte(x, i)
			}
		}
		return out
	}
	router := make([]float32, E*H)
	for i := range router {
		router[i] = float32(i%97) * 0.001
	}
	tensors := []struct {
		name    string
		dims    []uint64
		typ     ggufload.TensorType
		payload []byte
	}{
		{name: "blk.0.ffn_down_exps.weight", dims: []uint64{uint64(I), uint64(H), uint64(E)}, typ: ggufload.TensorQ3_K, payload: slab(perQ3)},
		{name: "blk.0.ffn_gate_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: ggufload.TensorQ2_K, payload: slab(perQ2)},
		{name: "blk.0.ffn_gate_inp.weight", dims: []uint64{uint64(H), uint64(E)}, typ: ggufload.TensorF32, payload: v41MixedF32Bytes(router)},
		{name: "blk.0.ffn_up_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: ggufload.TensorQ2_K, payload: slab(perQ2)},
	}
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, uint64(len(tensors)), 10)
	writeKVStringForTest(&b, "general.architecture", "qwen3moe")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen3moe.embedding_length", uint32(H))
	writeKVUint32ForTest(&b, "qwen3moe.block_count", 1)
	writeKVUint32ForTest(&b, "qwen3moe.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen3moe.feed_forward_length", uint32(I*2))
	writeKVFloat32ForTest(&b, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32ForTest(&b, "qwen3moe.expert_count", uint32(E))
	writeKVUint32ForTest(&b, "qwen3moe.expert_used_count", 1)
	writeKVUint32ForTest(&b, "qwen3moe.expert_feed_forward_length", uint32(I))
	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		off += len(tt.payload)
		off = (off + 31) / 32 * 32
	}
	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, uint32(tt.typ), offsets[i])
	}
	padToAlignmentForTest(&b, 32)
	for _, tt := range tensors {
		b.Write(tt.payload)
		padToAlignmentForTest(&b, 32)
	}
	return b.Bytes()
}

// fak-test:runtime fast est=1s
func TestServeV41MixedQuantHostFallbackLoadArm(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_PACKED_ISOLATED_TEST") != t.Name() {
		t.Parallel()
		v41PackedIsolatedTest(t)
		return
	}
	v41PackedTestEnv(t)
	const E, H, I = 2, 256, 256
	path := filepath.Join(t.TempDir(), "mixed-moe.gguf")
	if err := os.WriteFile(path, v41MixedMoEExpertGGUF(t, E, H, I), 0o600); err != nil {
		t.Fatal(err)
	}
	artifact, err := ggufload.ClassifyArtifactQuant(path)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Name != "mixed(Q2_K+Q3_K)" {
		t.Fatalf("intake-shape inventory classified as %q", artifact.Name)
	}
	backend := v41PackedBackend{Backend: compute.Default(), device: true, upload: true, q2: true, q3: false,
		dtypes: map[compute.Dtype]bool{compute.Q2_K: true}}
	if serveArtifactNativeKQuant(backend, artifact) {
		t.Fatal("no Q3 probe/dtype: the native packed gate must stay declined")
	}
	if !serveArtifactMixedQuantHostFallback(backend, artifact) || !serveArtifactResidentQ4K(backend, artifact) {
		t.Fatal("CUDA-like backend must enter the mixed host-fallback load arm")
	}
	fit := serveFitBudget{Base: 64 << 20, Headroom: serveGGUFDeviceHeadroom}
	m, inKernelQ4K, profile, _ := loadServeInKernelModelPlaced(path, backend, false, false, 16, nil, 1, &fit)
	if m == nil || !inKernelQ4K {
		t.Fatalf("load arm did not select resident packed storage (q4k=%t)", inKernelQ4K)
	}
	defer m.CloseWeights()
	report := m.ResidentReport()
	wantGateUp := int64(2 * E * (H * I / 256 * 84)) // both Q2_K slabs, raw source bytes
	wantDown := int64(E * (I * H / 256 * 110))      // Q3_K slab, raw source bytes
	wantRouterF32 := int64(E * H * 4)               // F32 router gate, on disk
	if report.KQuantTensors != 3*E || report.KQuantBytes != wantGateUp+wantDown {
		t.Fatalf("retained packed expert bytes: tensors=%d bytes=%d want tensors=%d bytes=%d",
			report.KQuantTensors, report.KQuantBytes, 3*E, wantGateUp+wantDown)
	}
	// The F32 router gate is a matmul weight: the Q4_K-profile loader stores it as its native-Q8
	// copy (the established loader contract), never as a whole-slab FP32 decode.
	wantRouterQ8 := (int64(E*H)*36 + 31) / 32
	if report.Q8Bytes != wantRouterQ8 || report.F32Bytes != 0 {
		t.Fatalf("router storage: Q8=%d want=%d, f32=%d want=0 (no whole-slab FP32 decode)",
			report.Q8Bytes, wantRouterQ8, report.F32Bytes)
	}
	region := func(x, per int) []byte {
		out := make([]byte, per)
		for i := 0; i < per; i++ {
			out[i] = v41MixedExpertPatternByte(x, i)
		}
		return out
	}
	for e := 0; e < E; e++ {
		for proj, per := range map[string]int{"gate_proj": H * I / 256 * 84, "down_proj": I * H / 256 * 110} {
			name := fmt.Sprintf("model.layers.0.mlp.experts.%d.%s.weight", e, proj)
			raw, ok := m.KQuantRaw(name)
			if !ok || !bytes.Equal(raw, region(e, per)) {
				t.Fatalf("expert %d %s resident bytes != source pattern (present=%t len=%d want=%d)", e, proj, ok, len(raw), per)
			}
		}
	}
	replicated, expert, ok := m.MoEResidentWeightBytes()
	if !ok || expert != wantGateUp+wantDown || replicated != wantRouterQ8 {
		t.Fatalf("selected consumers: replicated=%d expert=%d ok=%t want replicated=%d expert=%d",
			replicated, expert, ok, wantRouterQ8, wantGateUp+wantDown)
	}
	// Estimate/live honesty: the MoE arm admits on the prior conservative RAW-PAYLOAD plan, which
	// may OVER-charge (the router lands smaller as native Q8) but must never UNDER-charge the
	// bytes the loader retains.
	weights := int64(0)
	for _, d := range profile.MemoryPlan {
		if d.Class == string(compute.MemoryWeights) {
			weights += d.Bytes
		}
	}
	wantRawPayload := wantGateUp + wantDown + wantRouterF32
	if weights != wantRawPayload {
		t.Fatalf("MoE admission plan charged %d bytes, want the raw-payload total %d", weights, wantRawPayload)
	}
	if weights < report.TotalResidentBytes {
		t.Fatalf("under-charge: plan weights=%d < live resident=%d", weights, report.TotalResidentBytes)
	}
	loadMode := ""
	for _, msg := range profile.Messages {
		if msg.Kind == "load-mode" && msg.Level == "info" {
			loadMode = msg.Text
		}
	}
	if !containsAll(loadMode, "mixed(Q2_K+Q3_K)", "packed Q2_K device-native", "dense Q3_K converted to Q8") {
		t.Fatalf("load-mode message must name the fallback arm honestly: %q", loadMode)
	}
}

// fak-test:runtime fast est=1s
func TestServeV41MixedQuantHostFallbackFitRefusal(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_PACKED_ISOLATED_TEST") != t.Name() {
		t.Parallel()
		v41PackedIsolatedTest(t)
		return
	}
	v41PackedTestEnv(t)
	const E, H, I = 2, 256, 256
	path := filepath.Join(t.TempDir(), "mixed-moe-small-device.gguf")
	if err := os.WriteFile(path, v41MixedMoEExpertGGUF(t, E, H, I), 0o600); err != nil {
		t.Fatal(err)
	}
	backend := v41PackedBackend{Backend: compute.Default(), device: true, upload: true, q2: true, q3: false,
		dtypes: map[compute.Dtype]bool{compute.Q2_K: true}}
	// 64 KiB of device budget against ~141 KiB of weights PLUS the KV-cache row the plan adds:
	// the headroom-adjusted ceiling (~54 KiB) must reject before any byte is materialized.
	fit := serveFitBudget{Base: 64 << 10, Headroom: serveGGUFDeviceHeadroom}
	_, err := fitAndPlanServeGGUFPathOnDevice(path, backend, false, 16, &fit)
	var fe *compute.FitError
	if !errors.As(err, &fe) || fe.Verdict != compute.FitTooBig {
		t.Fatalf("small device budget must refuse typed before materialization: %v", err)
	}
}

// fak-test:runtime fast est=1s
func TestServeV41MixedQuantHostFallbackDenseQ3Charge(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_PACKED_ISOLATED_TEST") != t.Name() {
		t.Parallel()
		v41PackedIsolatedTest(t)
		return
	}
	v41PackedTestEnv(t)
	// Dense llama side of the same family: Q2_K stays packed (charged at its 84 B/256 payload),
	// Q3_K converts to native Q8 and is charged at the EXPANDED logical bytes, never the packed
	// payload — the honesty the packet requires of any dense Q3 conversion.
	const dim = 256
	q2Payload := make([]byte, dim*dim/256*84) // [dim,dim] tensor: dim blocks of 84 B each
	q3Payload := make([]byte, dim*dim/256*110)
	binary.LittleEndian.PutUint16(q3Payload[len(q3Payload)-2:], 0x3c00) // d=1
	path := filepath.Join(t.TempDir(), "mixed-dense.gguf")
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 2, 9)
	writeKVStringForTest(&b, "general.architecture", "llama")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "llama.embedding_length", dim)
	writeKVUint32ForTest(&b, "llama.block_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.key_length", dim)
	writeKVUint32ForTest(&b, "llama.feed_forward_length", dim)
	writeKVUint32ForTest(&b, "llama.context_length", 16)
	writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	q2Padded := len(q2Payload) + ((32 - len(q2Payload)%32) % 32) // next tensor offset must be 32-aligned
	writeTensorInfoForTest(&b, "blk.0.ffn_up.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ2_K), 0)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ3_K), uint64(q2Padded))
	padToAlignmentForTest(&b, 32)
	b.Write(q2Payload)
	padToAlignmentForTest(&b, 32)
	b.Write(q3Payload)
	padToAlignmentForTest(&b, 32)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	artifact, err := ggufload.ClassifyArtifactQuant(path)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Name != "mixed(Q2_K+Q3_K)" {
		t.Fatalf("dense fixture classified as %q", artifact.Name)
	}
	backend := v41PackedBackend{Backend: compute.Default(), device: true, upload: true, q2: true, q3: false,
		dtypes: map[compute.Dtype]bool{compute.Q2_K: true}}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	opts := serveResidentQ4KLoadOptions(backend, path, false, artifact)
	plan, err := ws.EstimateQ4KLoadMemoryPlan(opts...)
	if err != nil {
		t.Fatalf("estimate must qualify the mixed arm, not fall back to raw-payload admission: %v", err)
	}
	m, err := ws.QuantModelQ4KProfileOptions(nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	report := m.ResidentReport()
	wantPackedQ2 := int64(dim * dim / 256 * 84)
	wantQ8Logical := int64((dim*dim*36 + 31) / 32) // native Q8: f32 scale + i8 codes, 36 B/32
	if report.KQuantBytes != wantPackedQ2 || report.Q8Bytes != wantQ8Logical {
		t.Fatalf("live charge: Q2 packed=%d want=%d, Q3->Q8=%d want=%d (expanded, not packed)",
			report.KQuantBytes, wantPackedQ2, report.Q8Bytes, wantQ8Logical)
	}
	if plan.Total() != report.TotalResidentBytes {
		t.Fatalf("estimate/live disagreement: plan=%d live=%d", plan.Total(), report.TotalResidentBytes)
	}
	q8Row := int64(0)
	for _, d := range plan {
		if d.DType == "native-q8-f32-scales" {
			q8Row += d.Bytes
		}
	}
	if q8Row != wantQ8Logical {
		t.Fatalf("estimate charged the Q3 conversion at %d bytes, want the expanded %d", q8Row, wantQ8Logical)
	}
}

type v41CUDAFallbackBackend struct {
	v41PackedBackend
	free int64
}

func (b *v41CUDAFallbackBackend) Name() string                       { return "cuda" }
func (b *v41CUDAFallbackBackend) Tier() string                       { return "discrete:cpu-contract" }
func (b *v41CUDAFallbackBackend) DeviceMemory() (int64, int64, bool) { return b.free, b.free, true }
func (b *v41CUDAFallbackBackend) Caps() compute.Caps {
	c := b.v41PackedBackend.Caps()
	c.CapacityProbe = true
	return c
}

func newV41CUDAFallbackBackend(free int64) *v41CUDAFallbackBackend {
	b := v41PackedNativeBackend()
	b.q3 = false
	b.dtypes[compute.Q3_K] = false
	return &v41CUDAFallbackBackend{v41PackedBackend: b, free: free}
}

// A real bounded checkpoint exercises the production loader, not a 246 GiB
// allocation. The expert slab ratio still distinguishes full residency from
// the two-expert ring; this fixture is not a complete runnable V4.1 model.
func writeV41CUDAFallbackFixture(t *testing.T, experts int) string {
	t.Helper()
	type tensor struct {
		name string
		dims []uint64
		typ  ggufload.TensorType
		raw  []byte
	}
	tensors := []tensor{
		{"token_embd.weight", []uint64{256, 1}, ggufload.TensorQ2_K, make([]byte, 84)},
		{"output.weight", []uint64{256, 1}, ggufload.TensorQ6_K, make([]byte, 210)},
		{"output_norm.weight", []uint64{256}, ggufload.TensorF32, make([]byte, 1024)},
		{"blk.0.ffn_down_shexp.weight", []uint64{256, 256}, ggufload.TensorQ3_K, bytes.Repeat(v41PackedQ3Block(), 256)},
		{"blk.0.ffn_gate_exps.weight", []uint64{256, 256, uint64(experts)}, ggufload.TensorQ2_K, make([]byte, 256*experts*84)},
		{"blk.0.ffn_up_exps.weight", []uint64{256, 256, uint64(experts)}, ggufload.TensorQ2_K, make([]byte, 256*experts*84)},
		{"blk.0.ffn_down_exps.weight", []uint64{256, 256, uint64(experts)}, ggufload.TensorQ3_K, bytes.Repeat(v41PackedQ3Block(), 256*experts)},
	}
	keys := []struct {
		key   string
		value uint64
	}{
		{"embedding_length", 256}, {"block_count", 1}, {"attention.head_count", 1},
		{"attention.head_count_kv", 1}, {"attention.key_length", 32}, {"feed_forward_length", 256},
		{"context_length", 2048}, {"expert_count", uint64(experts)}, {"expert_used_count", 2},
		{"expert_feed_forward_length", 256}, {"attention.q_lora_rank", 32}, {"attention.kv_lora_rank", 32},
		{"attention.qk_nope_head_dim", 16}, {"attention.qk_rope_head_dim", 16}, {"attention.value_length_mla", 16},
	}
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, uint64(len(tensors)), uint64(3+len(keys)))
	writeKVStringForTest(&b, "general.architecture", "deepseek41")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVFloat32ForTest(&b, "deepseek41.attention.layer_norm_rms_epsilon", 1e-6)
	for _, key := range keys {
		writeKVUint64ForTest(&b, "deepseek41."+key.key, key.value)
	}
	var offset uint64
	for _, tensor := range tensors {
		writeTensorInfoForTest(&b, tensor.name, tensor.dims, uint32(tensor.typ), offset)
		offset = (offset + uint64(len(tensor.raw)) + 31) &^ 31
	}
	padToAlignmentForTest(&b, 32)
	for _, tensor := range tensors {
		b.Write(tensor.raw)
		padToAlignmentForTest(&b, 32)
	}
	path := filepath.Join(t.TempDir(), "v41-mixed-cpu-contract.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

// fak-test:runtime fast est=1s
func TestServeV41MixedQuantCUDAHostFallbackLoadArm(t *testing.T) {
	v41PackedTestEnv(t)
	path := writeV41CUDAFallbackFixture(t, 4)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	be := newV41CUDAFallbackBackend(1 << 30)
	artifact := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if serveArtifactNativeKQuant(be, artifact) || compute.BackendSupportsDeviceWeightDtype(be, compute.Q3_K) {
		t.Fatal("host storage admission fabricated Q3 CUDA execution capability")
	}
	if resolveDeviceServeLoadArm(ws, be, false, false) != serveLoadArmResidentQ4K {
		t.Fatal("ordinary sizing did not select the supported packed-storage arm")
	}
	fit := serveFitBudget{Base: 1 << 30}
	m, packed, profile, _, _, err := loadServeInKernelModelPlacedErr(path, be, false, false, 16, nil, 1, &fit)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	if !packed || profile == nil {
		t.Fatal("ordinary production loader did not retain packed storage")
	}
	raw, ok := m.KQuantRaw("model.layers.0.ffn.shared_experts.w2.weight")
	if !ok || !bytes.Equal(raw, bytes.Repeat(v41PackedQ3Block(), 256)) {
		t.Fatal("Q3 shared projection was expanded or changed")
	}
	decoded := make([]float32, 256*256)
	fakmodel.DequantQ3K(decoded, raw)
	for _, value := range decoded {
		if value != 128 {
			t.Fatalf("packed host Q3 consumer value=%g want=128", value)
		}
	}
	opts := serveResidentQ4KLoadOptions(be, path, true, artifact)
	plan, err := serveGGUFWeightMemoryPlanForArm(ws, serveLoadArmResidentQ4K, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Total() != m.ResidentReport().TotalResidentBytes || m.ResidentReport().Q8Bytes != 0 {
		t.Fatalf("packed estimate/live storage mismatch: plan=%d actual=%+v", plan.Total(), m.ResidentReport())
	}
	// Native-only admission stays strict; explicit rollback and missing native
	// Q2 support must still decline this new storage-only route.
	be.q2 = false
	if resolveDeviceServeLoadArm(ws, be, false, false) == serveLoadArmResidentQ4K {
		t.Fatal("missing Q2 native capability admitted")
	}
	be.q2 = true
	t.Setenv("FAK_Q4K", "0")
	if resolveDeviceServeLoadArm(ws, be, false, false) == serveLoadArmResidentQ4K {
		t.Fatal("explicit legacy rollback ignored")
	}
}
