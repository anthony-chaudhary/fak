package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func TestServeArtifactResidentQ4KUsesArtifactNotArmLabel(t *testing.T) {
	t.Setenv("FAK_Q4K", "1")
	backend := serveCapBackend{Backend: compute.Default(), uploadDtype: true}
	q8 := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ8_0},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})
	if serveArtifactResidentQ4K(backend, q8) {
		t.Fatal("all-Q8_0 artifact must not select resident Q4_K even when FAK_Q4K=1")
	}
	q4k := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ4_K},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})
	if !serveArtifactResidentQ4K(backend, q4k) {
		t.Fatal("Q4_K artifact should preserve resident Q4_K when backend and environment enable it")
	}
	t.Setenv("FAK_Q4K", "0")
	if serveArtifactResidentQ4K(backend, q4k) {
		t.Fatal("FAK_Q4K=0 must retain the Q8 staging rollback")
	}
	t.Setenv("FAK_Q4K", "1")
	udq2 := ggufload.ArtifactQuant{Name: "UD-Q2_K_XL", Recipe: "UD-Q2_K_XL", Q4KResident: true}
	if !serveArtifactResidentQ4K(backend, udq2) {
		t.Fatal("UD-Q2_K_XL artifact must select resident loading when backend and environment enable it")
	}
	t.Setenv("FAK_Q4K", "0")
	if serveArtifactResidentQ4K(backend, udq2) {
		t.Fatal("FAK_Q4K=0 must retain the Q8 staging rollback for UD-Q2_K_XL")
	}
}

func TestServeQuantProvenanceDistinguishesArtifactResidentAndSession(t *testing.T) {
	q8 := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ8_0},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})
	got := serveQuantProvenance(q8, false)
	for _, want := range []string{"artifact_quant=Q8_0", "artifact_inventory=mixed(F32+Q8_0)", "resident_quant=Q8_0", "session_quant=Q8_0"} {
		if !strings.Contains(got.Text, want) {
			t.Fatalf("Q8 provenance %q missing %q", got.Text, want)
		}
	}
	if strings.Contains(got.Text, "resident_quant=Q4_K") || strings.Contains(got.Text, "session_quant=Q4_K") {
		t.Fatalf("Q8 artifact emitted Q4_K provenance: %q", got.Text)
	}
}

func TestServeArtifactResidentQ4KMetalAutoSelect(t *testing.T) {
	orig := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = orig })

	q4k := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ4_K},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})
	q8 := ggufload.ClassifyTensorQuant([]ggufload.TensorInfo{
		{Name: "blk.0.attn_q.weight", Type: ggufload.TensorQ8_0},
		{Name: "blk.0.attn_norm.weight", Type: ggufload.TensorF32},
	})

	t.Run("metal available defaults to resident Q4_K when backend is nil", func(t *testing.T) {
		serveMetalAvailable = func() bool { return true }
		t.Setenv("FAK_Q4K", "")
		if !serveDeviceResidentQ4K(nil) {
			t.Fatal("serveDeviceResidentQ4K(nil) must auto-select true when Metal is available")
		}
		if !serveArtifactResidentQ4K(nil, q4k) {
			t.Fatal("serveArtifactResidentQ4K(nil, q4k) must auto-select true for Q4_K artifact on Metal")
		}
		if serveArtifactResidentQ4K(nil, q8) {
			t.Fatal("serveArtifactResidentQ4K(nil, q8) must remain false for non-Q4_K artifact even when Metal is available")
		}
	})

	t.Run("metal available with explicit FAK_Q4K=1", func(t *testing.T) {
		serveMetalAvailable = func() bool { return true }
		t.Setenv("FAK_Q4K", "1")
		if !serveDeviceResidentQ4K(nil) {
			t.Fatal("serveDeviceResidentQ4K(nil) must return true when FAK_Q4K=1")
		}
		if !serveArtifactResidentQ4K(nil, q4k) {
			t.Fatal("serveArtifactResidentQ4K(nil, q4k) must return true when FAK_Q4K=1")
		}
	})

	t.Run("metal available with FAK_Q4K=0 rolls back to Q8 staging", func(t *testing.T) {
		serveMetalAvailable = func() bool { return true }
		t.Setenv("FAK_Q4K", "0")
		if serveDeviceResidentQ4K(nil) {
			t.Fatal("FAK_Q4K=0 must roll back serveDeviceResidentQ4K(nil) to false")
		}
		if serveArtifactResidentQ4K(nil, q4k) {
			t.Fatal("FAK_Q4K=0 must roll back serveArtifactResidentQ4K(nil, q4k) to false")
		}
	})

	t.Run("metal unavailable returns false when backend is nil", func(t *testing.T) {
		serveMetalAvailable = func() bool { return false }
		t.Setenv("FAK_Q4K", "")
		if serveDeviceResidentQ4K(nil) {
			t.Fatal("serveDeviceResidentQ4K(nil) must return false when Metal is unavailable")
		}
		if serveArtifactResidentQ4K(nil, q4k) {
			t.Fatal("serveArtifactResidentQ4K(nil, q4k) must return false when Metal is unavailable")
		}
	})

	t.Run("device backend preserves non-metal behavior", func(t *testing.T) {
		serveMetalAvailable = func() bool { return false }
		backendWithQuant := serveCapBackend{Backend: compute.Default(), uploadDtype: true}
		backendNoQuant := serveCapBackend{Backend: compute.Default(), uploadDtype: false}

		t.Setenv("FAK_Q4K", "")
		if !serveDeviceResidentQ4K(backendWithQuant) {
			t.Fatal("backend with UploadDtype should enable resident Q4_K")
		}
		if serveDeviceResidentQ4K(backendNoQuant) {
			t.Fatal("backend without UploadDtype must not enable resident Q4_K")
		}
	})

	if metalgemm.Available() {
		t.Run("hardware witness on Apple Silicon Metal", func(t *testing.T) {
			serveMetalAvailable = metalgemm.Available
			t.Setenv("FAK_Q4K", "")
			if !serveDeviceResidentQ4K(nil) {
				t.Fatal("live Apple Silicon Metal must auto-select resident Q4_K when backend is nil")
			}
			if !serveArtifactResidentQ4K(nil, q4k) {
				t.Fatal("live Apple Silicon Metal must auto-select resident Q4_K for Q4_K artifact")
			}
		})
	}
}

func TestServeLoadModelMetalResidentDefault(t *testing.T) {
	orig := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = orig })

	q4kPath := createTestQ4KGGUF(t)
	q8Path := createTestQ8GGUF(t)

	t.Run("metal resident default loads Q4K model in resident mode", func(t *testing.T) {
		serveMetalAvailable = func() bool { return true }
		t.Setenv("FAK_Q4K", "")

		model, q4k, profile, phase := loadServeInKernelModel(q4kPath, nil, false, 0, nil, 1, nil)
		if model == nil {
			t.Fatal("expected non-nil model")
		}
		if !q4k {
			t.Fatal("expected inKernelQ4K to be true by default on Apple Silicon Metal")
		}
		if profile == nil {
			t.Fatal("expected non-nil load profile")
		}
		if profile.Mode != "gguf-resident-q4k" {
			t.Fatalf("profile mode = %q, want gguf-resident-q4k", profile.Mode)
		}
		if phase.Name != "model-load" {
			t.Fatalf("phase name = %q, want model-load", phase.Name)
		}
		foundMessage := false
		for _, msg := range profile.Messages {
			if strings.Contains(msg.Text, "GGUF Apple-Silicon Metal load -> resident quantized weights") {
				foundMessage = true
				break
			}
		}
		if !foundMessage {
			t.Fatalf("expected Metal resident startup message in profile.Messages, got: %+v", profile.Messages)
		}
	})

	t.Run("FAK_Q4K=0 rolls back to lean Q8 staging", func(t *testing.T) {
		serveMetalAvailable = func() bool { return true }
		t.Setenv("FAK_Q4K", "0")

		model, q4k, profile, _ := loadServeInKernelModel(q4kPath, nil, false, 0, nil, 1, nil)
		if model == nil {
			t.Fatal("expected non-nil model")
		}
		if q4k {
			t.Fatal("expected inKernelQ4K to be false when FAK_Q4K=0")
		}
		if profile == nil {
			t.Fatal("expected non-nil load profile")
		}
		if profile.Mode != "gguf-lean-q8" {
			t.Fatalf("profile mode = %q, want gguf-lean-q8", profile.Mode)
		}
	})

	t.Run("all Q8 artifact stays lean Q8 even when Metal is available", func(t *testing.T) {
		serveMetalAvailable = func() bool { return true }
		t.Setenv("FAK_Q4K", "")

		model, q4k, profile, _ := loadServeInKernelModel(q8Path, nil, false, 0, nil, 1, nil)
		if model == nil {
			t.Fatal("expected non-nil model")
		}
		if q4k {
			t.Fatal("expected inKernelQ4K to be false for all-Q8 artifact")
		}
		if profile == nil {
			t.Fatal("expected non-nil load profile")
		}
		if profile.Mode != "gguf-lean-q8" {
			t.Fatalf("profile mode = %q, want gguf-lean-q8", profile.Mode)
		}
	})
}

func createTestQ4KGGUF(t *testing.T) string {
	t.Helper()
	const dim = 256
	const blkBytes = 36864

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 4, 9)
	writeKVStringForTest(&b, "general.architecture", "llama")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "llama.embedding_length", dim)
	writeKVUint32ForTest(&b, "llama.block_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.key_length", dim)
	writeKVUint32ForTest(&b, "llama.feed_forward_length", dim)
	writeKVUint32ForTest(&b, "llama.context_length", 16)
	writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)

	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), 0)
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), blkBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), 2*blkBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), 3*blkBytes)
	padToAlignmentForTest(&b, 32)
	for i := 0; i < 4; i++ {
		b.Write(bytes.Repeat([]byte{0}, blkBytes))
	}

	path := filepath.Join(t.TempDir(), "test_q4k.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func createTestQ8GGUF(t *testing.T) string {
	t.Helper()
	const dim = 256
	const blkBytes = 69632 // 256 rows * (256/32 * 34 bytes) = 69632

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

	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ8_0), 0)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ8_0), blkBytes)
	padToAlignmentForTest(&b, 32)
	for i := 0; i < 2; i++ {
		b.Write(bytes.Repeat([]byte{0}, blkBytes))
	}

	path := filepath.Join(t.TempDir(), "test_q8.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeMinimalHeaderForTest(b *bytes.Buffer, tensors, kvs uint64) {
	b.WriteString("GGUF")
	_ = binary.Write(b, binary.LittleEndian, uint32(3))
	_ = binary.Write(b, binary.LittleEndian, tensors)
	_ = binary.Write(b, binary.LittleEndian, kvs)
}

func writeKVStringForTest(b *bytes.Buffer, key, value string) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeString))
	writeStringForTest(b, value)
}

func writeKVUint32ForTest(b *bytes.Buffer, key string, value uint32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeUint32))
	_ = binary.Write(b, binary.LittleEndian, value)
}

func writeKVFloat32ForTest(b *bytes.Buffer, key string, value float32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeFloat32))
	_ = binary.Write(b, binary.LittleEndian, math.Float32bits(value))
}

func writeStringForTest(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func writeTensorInfoForTest(b *bytes.Buffer, name string, shape []uint64, typ uint32, offset uint64) {
	writeStringForTest(b, name)
	_ = binary.Write(b, binary.LittleEndian, uint32(len(shape)))
	for _, dim := range shape {
		_ = binary.Write(b, binary.LittleEndian, dim)
	}
	_ = binary.Write(b, binary.LittleEndian, typ)
	_ = binary.Write(b, binary.LittleEndian, offset)
}

func padToAlignmentForTest(b *bytes.Buffer, alignment int) {
	rem := b.Len() % alignment
	if rem != 0 {
		b.Write(bytes.Repeat([]byte{0}, alignment-rem))
	}
}

// createMixedQuantGGUF writes the #12983 fixture shape: a Q4_K dense side (attn_v/attn_q/output)
// with one ffn_down tensor of the named expert type. The resident loader holds an eligible
// non-Q4_K ffn_down raw in the k-quant store exactly like the mixed-quant checkpoints that
// reached READY and then wedged in unbounded CPU dequant on the first turn.
func createMixedQuantGGUF(t *testing.T, expertType ggufload.TensorType) string {
	t.Helper()
	const dim = 256
	q4kBytes := uint64(36864) // 256 super-blocks x 144 B
	var expertBytes uint64
	switch expertType {
	case ggufload.TensorQ4_K:
		expertBytes = 36864 // 256 x 144
	case ggufload.TensorIQ3_XXS:
		expertBytes = 256 * 98 // 256 x blockIQ3XXSBytes
	case ggufload.TensorQ6_K:
		expertBytes = 256 * 210 // 256 x blockQ6KBytes
	default:
		t.Fatalf("unsupported fixture expert type %s", expertType)
	}

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 4, 9)
	writeKVStringForTest(&b, "general.architecture", "llama")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "llama.embedding_length", dim)
	writeKVUint32ForTest(&b, "llama.block_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.key_length", dim)
	writeKVUint32ForTest(&b, "llama.feed_forward_length", dim)
	writeKVUint32ForTest(&b, "llama.context_length", 16)
	writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)

	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), 0)
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), q4kBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, uint32(expertType), 2*q4kBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), 2*q4kBytes+expertBytes)
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, 3*q4kBytes+expertBytes))

	path := filepath.Join(t.TempDir(), "mixed-quant.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// fak-test:runtime fast est=1ms
func TestServeQuantBackendNameMapsMetalNativeArm(t *testing.T) {
	if got := serveQuantBackendName(nil); got != "metal" {
		t.Fatalf("serveQuantBackendName(nil) = %q, want metal (the Metal-native load arm)", got)
	}
	if be := compute.Default(); be == nil {
		t.Skip("no reference backend registered; device mapping not exercisable")
	}
	if got := serveQuantBackendName(compute.Default()); got != "cpu" {
		t.Fatalf("serveQuantBackendName(device) = %q, want cpu (bounded host routes)", got)
	}
}

// fak-test:runtime fast est=10ms
func TestServeRefuseUnsupportedQuantsNilOnBoundedCPU(t *testing.T) {
	// A nil model is a no-op: arms that never load retain the historical contract.
	if err := serveRefuseUnsupportedQuants(nil, nil); err != nil {
		t.Fatalf("nil model produced %v, want nil", err)
	}
	// A pure-Q4_K artifact retains no k-quant bulk: every arm's census must pass, and the
	// helper must return a literal nil error (not a wrapped concrete refusal pointer).
	path := createMixedQuantGGUF(t, ggufload.TensorQ4_K)
	mm, err := ggufload.LoadModelQ4KProfileOptions(path, ggufload.NewLoadProfiler())
	if err != nil {
		t.Fatal(err)
	}
	for _, be := range []compute.Backend{nil, compute.Default()} {
		if err := serveRefuseUnsupportedQuants(mm, be); err != nil {
			t.Fatalf("bounded Q4_K load refused on backend %v: %v", be, err)
		}
	}
}

// TestServeQuantCapabilityRefusalGatesEveryLoadArm is the #12983 fails-before/passes-after
// witness: a mixed-quant checkpoint whose retained expert bulk has no Metal kernel and no
// bounded host route must be refused at LOAD TIME (before READY) by the typed post-load census,
// naming the offending kind, instead of reaching READY and hanging in unbounded CPU dequant on
// the first turn. Before the gate this call returned a loaded model (err == nil).
//
// fak-test:runtime fast est=50ms
func TestServeQuantCapabilityRefusalGatesEveryLoadArm(t *testing.T) {
	orig := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = orig })
	serveMetalAvailable = func() bool { return true }
	t.Setenv("FAK_Q4K", "")

	path := createMixedQuantGGUF(t, ggufload.TensorIQ3_XXS)
	_, _, _, _, _, err := loadServeInKernelModelPlacedErr(path, nil, false, false, 2048, nil, 1, nil)
	if err == nil {
		t.Fatal("mixed IQ3_XXS checkpoint reached READY on the Metal-native arm, want typed quant-capability refusal before readiness")
	}
	var refusal *fakmodel.LoadQuantCapabilityRefusal
	if !errors.As(err, &refusal) {
		t.Fatalf("load error type = %T (%v), want *model.LoadQuantCapabilityRefusal", err, err)
	}
	if !strings.Contains(refusal.Error(), "IQ3_XXS") {
		t.Errorf("refusal %q does not name the unsupported IQ3_XXS band", refusal.Error())
	}
}

// TestServeQuantCapabilityBoundedTurnOnSupportedBands pins the other side of the gate: a mixed
// checkpoint whose retained expert bulk dispatches to a resident Metal device kernel (Q6_K) or a
// bounded host route must LOAD through the census unchanged — the GLM-5.2-style Q5_K/Q6_K expert
// checkpoint the gate exists to keep serving.
//
// fak-test:runtime fast est=50ms
func TestServeQuantCapabilityBoundedTurnOnSupportedBands(t *testing.T) {
	orig := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = orig })
	serveMetalAvailable = func() bool { return true }
	t.Setenv("FAK_Q4K", "")

	path := createMixedQuantGGUF(t, ggufload.TensorQ6_K)
	mm, q4k, _, _, _, err := loadServeInKernelModelPlacedErr(path, nil, false, false, 2048, nil, 1, nil)
	if err != nil {
		t.Fatalf("supported-bands mixed checkpoint refused at load: %v", err)
	}
	if mm == nil || !q4k {
		t.Fatalf("load returned model=%v q4k=%v, want the resident Q4_K arm", mm, q4k)
	}
	if ref := mm.RefuseUnsupportedQuants("metal"); ref != nil {
		t.Fatalf("retained Q6_K expert bulk refused on metal: %v", ref)
	}
}
