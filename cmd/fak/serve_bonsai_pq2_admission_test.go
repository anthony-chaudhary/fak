package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// A Bonsai-2 artifact can contain PQ2_0 matrices without any Q4_K tensor.
// The default CPU serve arm and fit estimate must follow the packed loader for
// that header; routing it through Q8 silently expands the dominant tensor band.
func TestServePQ2OnlyArtifactSelectsPackedResidentOnCPU(t *testing.T) {
	t.Setenv("FAK_Q4K", "")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	oldMetal := serveMetalAvailable
	serveMetalAvailable = func() bool { return false }
	t.Cleanup(func() { serveMetalAvailable = oldMetal })
	path := writeServePQ2OnlyFixture(t)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	artifact := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if artifact.Name != "PQ2_0" {
		t.Fatalf("artifact quant = %q, want PQ2_0", artifact.Name)
	}

	const packedBytes int64 = 128 * 34 // 128 rows, one group-128 block per row
	if arm := resolveHostServeLoadArm(ws, false, false); arm != serveLoadArmResidentQ4K {
		t.Fatalf("default CPU fit arm = %q, want packed resident", arm)
	}
	plan, err := serveGGUFWeightMemoryPlanForArm(ws, serveLoadArmResidentQ4K,
		serveQ4KFitOptions(path, ws, nil, serveLoadArmResidentQ4K, serveFitBudget{})...)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Total() != packedBytes {
		t.Fatalf("CPU fit weight bytes = %d, want packed %d", plan.Total(), packedBytes)
	}

	// Exercise the real load switch as well as its fit arm with no opt-in flag.
	m, resident, _, _ := loadServeInKernelModel(path, nil, false, 0, nil, 1, nil)
	q2, q8 := 0, 0
	if m != nil {
		q2, q8 = m.Q2Count(), m.ResidentReport().Q8Tensors
	}
	if m == nil || !resident || q2 != 1 || q8 != 0 {
		t.Fatalf("CPU load: model=%v resident=%t q2=%d report=%+v, want one packed PQ2 tensor and no Q8",
			m != nil, resident, q2, q8)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
}

// --plan-json runs before the live compute resolver. On a Metal-capable Mac it
// must still describe the PQ2 CPU path that the live resolver will select.
func TestServePQ2PlanJSONReportsPackedCPUPlacement(t *testing.T) {
	t.Setenv("FAK_Q4K", "")
	oldMetal := serveMetalAvailable
	serveMetalAvailable = func() bool { return true }
	t.Cleanup(func() { serveMetalAvailable = oldMetal })
	path := writeServePQ2OnlyFixture(t)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	art, err := buildServeSizingArtifact(ws, nil, false, 16, path, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	if art.Arm != "cpu-resident-pq2_0" {
		t.Fatalf("PQ2 plan arm = %q, want cpu-resident-pq2_0", art.Arm)
	}
	if art.Tiers.VRAMBytes != 0 || len(art.Pools) != 1 || art.Pools[0].Pool != "host" {
		t.Fatalf("PQ2 plan placement: tiers=%+v pools=%+v, want host-only with zero VRAM", art.Tiers, art.Pools)
	}
	const packedBytes int64 = 128 * 34
	var pq2WeightBytes int64
	for _, demand := range art.Demands {
		if demand.DType == "pq2_0" { // plan dtype labels are canonical lowercase
			pq2WeightBytes += demand.Bytes
		}
	}
	if pq2WeightBytes != packedBytes {
		t.Fatalf("PQ2 plan weights = %d, want %d packed bytes (no Q8 expansion)", pq2WeightBytes, packedBytes)
	}
}

// Explicit Metal must fail before a PQ2 checkpoint is loaded. The packed
// ternary model path is currently CPU only, even on a Mac with Metal present.
func TestServePQ2ExplicitMetalRefuses(t *testing.T) {
	const child = "FAK_TEST_PQ2_METAL_CHILD"
	if os.Getenv(child) == "1" {
		path := os.Getenv("FAK_TEST_PQ2_METAL_PATH")
		backendName, kvPrecision, nativePrefixProfile := "", "", ""
		metal, gdn, slab, vulkanProfile, vulkanStage := true, false, false, false, false
		prefillChunk := 0
		sf := &serveFlags{
			backendName: &backendName, ggufPath: &path, metal: &metal,
			kvPrecision: &kvPrecision, nativeQwenQ4KPrefillChunk: &prefillChunk,
			nativeQwen35MetalGDNSequence: &gdn, nativeQ4KGateUpOutputSlab: &slab,
			nativePrefixProfile: &nativePrefixProfile,
			vulkanQ4KProfile:    &vulkanProfile, vulkanStageQ4K: &vulkanStage,
		}
		(&serveRuntime{}).resolveCompute(sf)
		t.Fatal("explicit Metal unexpectedly admitted PQ2_0 checkpoint")
	}
	if !serveMetalAvailable() {
		t.Skip("requires an available Metal device to reach the PQ2-specific refusal")
	}
	path := writeServePQ2OnlyFixture(t)
	cmd := exec.Command(os.Args[0], "-test.run=^TestServePQ2ExplicitMetalRefuses$")
	cmd.Env = append(os.Environ(), child+"=1", "FAK_TEST_PQ2_METAL_PATH="+path, "FAK_BACKEND=", "FAK_METAL=")
	out, err := cmd.CombinedOutput()
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() != 2 {
		t.Fatalf("explicit Metal exit = %v, want code 2; output: %s", err, out)
	}
	if !strings.Contains(string(out), "PQ2_0 checkpoint has no model-wired Metal kernel") {
		t.Fatalf("explicit Metal refusal did not name the unsupported kernel: %s", out)
	}
}

func writeServePQ2OnlyFixture(t *testing.T) string {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 1, 10)
	writeKVStringForTest(&b, "general.architecture", "qwen35")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen35.embedding_length", 128)
	writeKVUint32ForTest(&b, "qwen35.block_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count_kv", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.key_length", 128)
	writeKVUint32ForTest(&b, "qwen35.full_attention_interval", 4)
	writeKVUint32ForTest(&b, "qwen35.feed_forward_length", 128)
	writeKVFloat32ForTest(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)
	writeTensorInfoForTest(&b, "blk.0.ffn_up.weight", []uint64{128, 128}, uint32(ggufload.TensorPQ2_0), 0)
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, 128*34))
	path := filepath.Join(t.TempDir(), "bonsai2-pq2-only.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
