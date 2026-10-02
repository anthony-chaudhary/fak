package main

// serve_pq2_reconcile_seam_test.go probes the seam where the Prism Bonsai-2 PQ2_0
// (GGUF type 142) CPU-resident change meets the pre-existing Metal/Q4_K resident
// machinery in cmd/fak. The contract:
//
//   - a cmd/fak serve of a PQ2_0-only checkpoint MUST select the packed
//     CPU-resident load arm on a host where Metal is unavailable;
//   - an explicit device/Metal request MUST refuse rather than silently
//     relabel a CPU path.
//
// The tests here are adversarial and reuse the REAL predicates/estimators
// (serveArtifactResidentQ4K, resolveHostServeLoadArm, EstimateQ4KLoadMemoryPlan,
// LoadModelQ4KProfileOptions); nothing is mocked except the empirically
// device-dependent serveMetalAvailable probe, which is forced to model a
// non-Metal host.

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// TestServePQ2ReconcilePredicateAgreesWithHostArm pins the invariant that the
// runtime load-path gate (serveArtifactResidentQ4K) and the host arm resolver
// (resolveHostServeLoadArm) cannot disagree about a PQ2_0-only checkpoint on a
// host where Metal is NOT available.
//
// serveMetalAvailable is build-tag/device dependent (metal Available() is
// true only on darwin/arm64 with cgo AND a usable device; the stub returns
// false), so this test forces it false to construct the non-Metal case the
// contract names.
func TestServePQ2ReconcilePredicateAgreesWithHostArm(t *testing.T) {
	oldMetal := serveMetalAvailable
	serveMetalAvailable = func() bool { return false } // model a host with no Metal device
	t.Cleanup(func() { serveMetalAvailable = oldMetal })
	t.Setenv("FAK_Q4K", "")

	path := writeServePQ2OnlyFixture(t)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	art := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if !servePQ2Artifact(art) {
		t.Fatalf("fixture quant = %q, want PQ2_0", art.Name)
	}

	gateResident := serveArtifactResidentQ4K(nil, art) // the runtime load-path gate
	hostArm := resolveHostServeLoadArm(ws, false, false)
	armResident := hostArm == serveLoadArmResidentQ4K

	if gateResident != armResident {
		t.Fatalf("PREDICATE/ARM DIVERGENCE on a non-Metal host:\n"+
			"  serveArtifactResidentQ4K(nil, pq2) = %t\n"+
			"  resolveHostServeLoadArm(ws,false,false) = %q (resident=%t)\n"+
			"Contract: a PQ2_0-only checkpoint MUST select the packed CPU-resident arm when Metal is unavailable.\n"+
			"resolveHostServeLoadArm is correct (it returns the packed resident arm).\n"+
			"serveArtifactResidentQ4K is the divergent predicate: its PQ2_0 branch ANDs serveMetalAvailable(),\n"+
			"but PQ2_0 is a CPU-resident format and a *device* has no PQ2 kernel, so the absence of Metal\n"+
			"cannot disqualify it. Fix: PQ2 branch should return backend == nil && os.Getenv(\"FAK_Q4K\") != \"0\"\n"+
			"(drop the serveMetalAvailable() conjunct).",
			gateResident, hostArm, armResident)
	}
	// Also assert the agreement is on the RESIDENT side, not the Q8 side: a
	// future change could make both false and silently expand the packed band.
	if !gateResident {
		t.Fatalf("both predicates agree FALSE on a non-Metal host; contract requires the packed "+
			"CPU-resident arm, not the Q8 expansion arm (hostArm=%q)", hostArm)
	}
}

// TestServePQ2ReconcileEstimateChargesPackedBytes probes the estimate seam: the
// merged PQ2 change must charge a PQ2_0 matmul AND a PQ2_0 embedding at the
// PACKED 34-bytes-per-128-weights size, not a native-Q8 expanded size. A merge
// regression that drops PQ2 out of the packed residency branches would show up
// as a "native-q8-f32-scales" demand (about 1.125 B/weight) and FAIL here.
func TestServePQ2ReconcileEstimateChargesPackedBytes(t *testing.T) {
	path := writeServePQ2PrismFixture(t)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })

	// Expected packed bytes per tensor: rows * (cols/128) * 34. GGUF dims are
	// [cols, rows] and modelShapeFromGGUFDims reverses them.
	const (
		embedRows, embedCols = 4, 256
		matRows, matCols     = 256, 256
	)
	embedBytes := int64(embedRows * (embedCols / 128) * 34)
	matBytes := int64(matRows * (matCols / 128) * 34)
	// The fixture carries token_embd + output + one matmul, all PQ2_0.
	wantTotal := 2*embedBytes + matBytes

	plan, err := ws.EstimateQ4KLoadMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateQ4KLoadMemoryPlan: %v", err)
	}
	var pq2Bytes, q8Bytes int64
	for _, demand := range plan {
		switch demand.DType {
		case "pq2_0":
			pq2Bytes += demand.Bytes
		case "native-q8-f32-scales":
			q8Bytes += demand.Bytes
		}
	}
	if pq2Bytes == 0 {
		t.Fatalf("no pq2_0 demand row: PQ2 tensors fell out of the packed residency branches; plan=%+v", plan)
	}
	if q8Bytes != 0 {
		t.Fatalf("plan charged %d bytes as native-q8-f32-scales; PQ2_0 must stay packed (plan=%+v)", q8Bytes, plan)
	}
	if pq2Bytes != wantTotal {
		t.Fatalf("pq2_0 demand total = %d, want packed %d (rows*(cols/128)*34); plan=%+v",
			pq2Bytes, wantTotal, plan)
	}
	if plan.Total() != wantTotal {
		t.Fatalf("plan total = %d, want %d packed bytes; plan=%+v", plan.Total(), wantTotal, plan)
	}

	// The real loader must agree with the header-only estimate: the packed
	// embedding and matmul tensors are stored raw, with zero Q8 expansion.
	prof := ggufload.NewLoadProfiler()
	m, err := ggufload.LoadModelQ4KProfileOptions(path, prof)
	if err != nil {
		t.Fatalf("LoadModelQ4KProfileOptions: %v", err)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	rep := m.ResidentReport()
	if rep.Q2Bytes != matBytes+embedBytes {
		t.Fatalf("loader Q2Bytes = %d, want packed %d (matmul %d + lm_head %d)",
			rep.Q2Bytes, matBytes+embedBytes, matBytes, embedBytes)
	}
	if rep.PQ2EmbedBytes != embedBytes {
		t.Fatalf("loader PQ2EmbedBytes = %d, want packed %d", rep.PQ2EmbedBytes, embedBytes)
	}
	if rep.Q8Tensors != 0 {
		t.Fatalf("loader expanded %d tensor(s) to Q8; PQ2_0 must stay packed (report=%+v)",
			rep.Q8Tensors, rep)
	}
}

// TestServePQ2ReconcileDeviceBackendNeverResident pins the predicate-level half
// of the explicit-device/Metal refusal: a non-nil backend (a compute HAL device)
// has no PQ2 execution kernel, so serveArtifactResidentQ4K must be false for a
// PQ2_0 artifact regardless of that backend's capabilities. This is distinct
// from TestServePQ2ExplicitMetalRefuses, which exercises the whole serve CLI.
func TestServePQ2ReconcileDeviceBackendNeverResident(t *testing.T) {
	oldMetal := serveMetalAvailable
	serveMetalAvailable = func() bool { return true } // even with Metal "available"
	t.Cleanup(func() { serveMetalAvailable = oldMetal })
	t.Setenv("FAK_Q4K", "1")

	path := writeServePQ2OnlyFixture(t)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	art := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if !servePQ2Artifact(art) {
		t.Fatalf("fixture quant = %q, want PQ2_0", art.Name)
	}

	for _, be := range []compute.Backend{
		compute.Default(),
		serveCapBackend{Backend: compute.Default(), uploadDtype: true},
		serveCapBackend{Backend: compute.Default(), uploadDtype: false},
	} {
		if be == nil {
			t.Fatal("test backend is nil; cannot probe device refusal")
		}
		if serveArtifactResidentQ4K(be, art) {
			t.Fatalf("serveArtifactResidentQ4K(%T, pq2) = true; a device backend has no PQ2 kernel "+
				"and must not be admitted to the packed CPU-resident arm", be)
		}
	}
}

// -- fixture ------------------------------------------------------------------

// writeServePQ2PrismFixture writes the smallest Qwen3.5-family hybrid GGUF that
// carries a PQ2_0 embedding, a PQ2_0 lm_head, and a PQ2_0 matmul tensor, plus
// the prism.hadamard metadata the estimate/loader use to enable PQ2 residency.
// It is deliberately NOT a redefinition of writeServePQ2OnlyFixture: that
// helper has no embedding and no prism metadata, so it cannot exercise the
// embedding residency path.
func writeServePQ2PrismFixture(t *testing.T) string {
	t.Helper()
	const dim = 256
	const vocab = 4
	align32 := func(n uint64) uint64 { return (n + 31) &^ 31 }

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 3, 19)
	writeKVStringForTest(&b, "general.architecture", "qwen35")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen35.embedding_length", dim)
	writeKVUint32ForTest(&b, "qwen35.block_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count_kv", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.key_length", dim)
	writeKVUint32ForTest(&b, "qwen35.full_attention_interval", 4)
	writeKVUint32ForTest(&b, "qwen35.feed_forward_length", dim)
	writeKVFloat32ForTest(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32ForTest(&b, "prism.hadamard.version", 1)
	writeKVUint32ForTest(&b, "prism.hadamard.block_size", dim)
	writeKVStringForTest(&b, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
	writeKVStringForTest(&b, "prism.hadamard.axis", "input-last-dimension")
	writeKVStringForTest(&b, "prism.hadamard.sign_mode", "explicit")
	writeServeKVIntArrayForTest(&b, "prism.hadamard.sign_widths", []int32{dim})
	signs := make([]int32, dim)
	for i := range signs {
		signs[i] = 1
	}
	writeServeKVIntArrayForTest(&b, "prism.hadamard.sign_values", signs)
	writeServeKVStringArrayForTest(&b, "prism.hadamard.weight_names", []string{"blk.0.ffn_up.weight"})
	writeServeKVBoolForTest(&b, "prism.hadamard.gdn_v_grouped", true)

	embedBytes := uint64(vocab * (dim / 128) * 34)
	matBytes := uint64(dim * (dim / 128) * 34)
	off := uint64(0)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{dim, vocab}, uint32(ggufload.TensorPQ2_0), off)
	off = align32(off + embedBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, vocab}, uint32(ggufload.TensorPQ2_0), off)
	off = align32(off + embedBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_up.weight", []uint64{dim, dim}, uint32(ggufload.TensorPQ2_0), off)
	off = align32(off + matBytes)
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(off)))

	path := filepath.Join(t.TempDir(), "bonsai2-pq2-prism.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeServeKVIntArrayForTest(b *bytes.Buffer, key string, values []int32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeInt32))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		_ = binary.Write(b, binary.LittleEndian, v)
	}
}

func writeServeKVStringArrayForTest(b *bytes.Buffer, key string, values []string) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeString))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		writeStringForTest(b, v)
	}
}

func writeServeKVBoolForTest(b *bytes.Buffer, key string, val bool) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeBool))
	if val {
		b.WriteByte(1)
	} else {
		b.WriteByte(0)
	}
}
