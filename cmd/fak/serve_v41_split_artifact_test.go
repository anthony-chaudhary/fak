package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func v41PackedIsolatedTest(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^"+regexp.QuoteMeta(t.Name())+"$", "-test.v")
	cmd.Env = append(os.Environ(), "FAK_TEST_V41_PACKED_ISOLATED_TEST="+t.Name())
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("isolated packed-load contract failed: %v\n%s", err, out)
	}
	t.Logf("isolated packed-load contract:\n%s", out)
}

func v41PackedDefaultFlag(t *testing.T) {
	t.Helper()
	t.Setenv("FAK_Q4K", "")
	if err := os.Unsetenv("FAK_Q4K"); err != nil {
		t.Fatal(err)
	}
	if _, present := os.LookupEnv("FAK_Q4K"); present {
		t.Fatal("default packed-load witness requires FAK_Q4K absent")
	}
}

// fak-test:runtime fast est=1s
func TestServeV41PackedMeasuredSevenShardInventory(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_PACKED_ISOLATED_TEST") != t.Name() {
		t.Parallel()
		v41PackedIsolatedTest(t)
		return
	}
	v41PackedTestEnv(t)
	v41PackedDefaultFlag(t)
	// Header census at 85e0ca215404b055b8675460edbd6a7d07208eb8. Payload geometry is
	// deliberately miniature; these counts do not claim a runnable V4.1 model.
	counts := [7][5]int{
		{34, 4, 1, 30, 2}, {107, 18, 0, 102, 9}, {35, 6, 0, 32, 3},
		{29, 6, 0, 25, 2}, {105, 18, 0, 101, 9}, {103, 18, 0, 99, 9}, {63, 10, 0, 60, 6},
	}
	types := []ggufload.TensorType{ggufload.TensorQ2_K, ggufload.TensorQ3_K, ggufload.TensorQ6_K, ggufload.TensorF32, ggufload.TensorBF16}
	sizes := []int{84, 110, 210, 4, 2}
	var total int
	for _, row := range counts {
		for _, n := range row {
			total += n
		}
	}
	dir := t.TempDir()
	var first string
	wantTensors := make(map[string]ggufload.TensorType, total)
	for shard, row := range counts {
		var tensors []v41PackedFixtureTensor
		for typ, n := range row {
			for i := 0; i < n; i++ {
				name := fmt.Sprintf("census.%d.%d.%d.weight", shard, typ, i)
				tensors = append(tensors, v41PackedFixtureTensor{name: name, typ: types[typ], data: make([]byte, sizes[typ])})
				wantTensors[name] = types[typ]
			}
		}
		path := filepath.Join(dir, fmt.Sprintf("census-%05d-of-%05d.gguf", shard+1, len(counts)))
		if shard == 0 {
			first = path
		}
		writeV41PackedFixture(t, path, tensors, shard, len(counts), total)
		gg, err := ggufload.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		got := make(map[ggufload.TensorType]int)
		for _, tensor := range gg.Tensors {
			got[tensor.Type]++
		}
		for typ, n := range row {
			if got[types[typ]] != n {
				t.Fatalf("shard=%d type=%s count=%d want=%d", shard+1, types[typ], got[types[typ]], n)
			}
		}
	}
	ws, err := ggufload.OpenWeights(first)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	if len(ws.File.Tensors) != len(wantTensors) {
		t.Fatalf("merged tensors=%d want fixture inventory=%d", len(ws.File.Tensors), len(wantTensors))
	}
	for _, tensor := range ws.File.Tensors {
		want, ok := wantTensors[tensor.Name]
		if !ok || tensor.Type != want {
			t.Fatalf("merged tensor %q type=%s missing, duplicated, or changed from fixture type=%s", tensor.Name, tensor.Type, want)
		}
		delete(wantTensors, tensor.Name)
	}
	if len(wantTensors) != 0 {
		t.Fatalf("merged inventory omitted fixture tensors: %v", wantTensors)
	}
	merged := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if merged.Name != "mixed(Q2_K+Q3_K+Q6_K)" || merged.Inventory != "mixed(BF16+F32+Q2_K+Q3_K+Q6_K)" || merged.Recipe != "" || merged.Q4KResident {
		t.Fatalf("merged classification name=%q inventory=%q recipe=%q q4=%t", merged.Name, merged.Inventory, merged.Recipe, merged.Q4KResident)
	}
	if !serveArtifactResidentQ4K(v41PackedNativeBackend(), merged) {
		t.Fatal("measured merged family must admit native packed load")
	}
}

func writeV41PackedSevenShardLoadFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	var first string
	norms := []string{"output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"}
	for shard := 0; shard < 7; shard++ {
		var tensors []v41PackedFixtureTensor
		if shard == 0 {
			tensors = []v41PackedFixtureTensor{
				{name: "blk.0.attn_v.weight", typ: ggufload.TensorQ2_K, data: make([]byte, 84)},
				{name: "output.weight", typ: ggufload.TensorQ6_K, data: make([]byte, 210)},
			}
		} else if shard == 6 {
			tensors = []v41PackedFixtureTensor{{name: "blk.0.ffn_down.weight", typ: ggufload.TensorQ3_K, data: v41PackedQ3Block()}}
		} else {
			tensors = []v41PackedFixtureTensor{{name: norms[shard-1], typ: ggufload.TensorF32, data: make([]byte, 256*4)}}
		}
		path := filepath.Join(dir, fmt.Sprintf("late-q3-000%02d-of-00007.gguf", shard+1))
		if shard == 0 {
			first = path
		}
		writeV41PackedLoadShard(t, path, tensors, shard)
	}
	return first
}

func writeV41PackedLoadShard(t *testing.T, path string, tensors []v41PackedFixtureTensor, shard int) {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, uint64(len(tensors)), 12)
	writeKVStringForTest(&b, "general.architecture", "llama")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "llama.embedding_length", 256)
	writeKVUint32ForTest(&b, "llama.block_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.key_length", 256)
	writeKVUint32ForTest(&b, "llama.feed_forward_length", 256)
	writeKVUint32ForTest(&b, "llama.context_length", 16)
	writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32ForTest(&b, "split.no", uint32(shard))
	writeKVUint32ForTest(&b, "split.count", 7)
	writeKVUint32ForTest(&b, "split.tensors.count", 8)
	var offset uint64
	for _, tensor := range tensors {
		dims := []uint64{256, 1}
		if tensor.typ == ggufload.TensorF32 {
			dims = []uint64{256}
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
func TestServeV41PackedSevenShardProductionArmUsesMergedInventory(t *testing.T) {
	// CLI must(err) exits the process; isolate the production entrypoint so a
	// refusal becomes a test failure without suppressing the other contracts.
	if os.Getenv("FAK_TEST_V41_PACKED_LOAD_CHILD") != "1" {
		t.Parallel()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeV41PackedSevenShardProductionArmUsesMergedInventory$", "-test.v")
		cmd.Env = append(os.Environ(), "FAK_TEST_V41_PACKED_LOAD_CHILD=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("bounded production load subprocess failed: %v\n%s", err, out)
		}
		t.Logf("bounded production load subprocess:\n%s", out)
		return
	}
	v41PackedTestEnv(t)
	v41PackedDefaultFlag(t)
	path := writeV41PackedSevenShardLoadFixture(t)
	first, err := ggufload.ClassifyArtifactQuant(path)
	if err != nil {
		t.Fatal(err)
	}
	if first.Name != "mixed(Q2_K+Q6_K)" {
		t.Fatalf("first-shard discriminator=%q", first.Name)
	}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	merged := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if merged.Name != "mixed(Q2_K+Q3_K+Q6_K)" {
		t.Fatalf("merged discriminator=%q", merged.Name)
	}
	for _, tc := range []struct {
		name, flag string
		q3, packed bool
	}{
		{name: "default-all-native", q3: true, packed: true},
		{name: "default-late-q3-shader-absent"},
		{name: "explicit-opt-in", flag: "1", q3: true, packed: true},
		{name: "rollback", flag: "0", q3: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.flag == "" {
				v41PackedDefaultFlag(t)
			} else {
				t.Setenv("FAK_Q4K", tc.flag)
			}
			backend := v41PackedNativeBackend()
			backend.q3 = tc.q3
			arm := resolveDeviceServeLoadArm(ws, backend, false, false)
			wantArm := serveLoadArmQuantProfileQ8
			if tc.packed {
				wantArm = serveLoadArmResidentQ4K
			}
			if arm != wantArm {
				t.Errorf("sizing arm=%v want=%v", arm, wantArm)
			}
			fit := serveFitBudget{Base: 1 << 30, Headroom: 0}
			m, packed, _, _ := loadServeInKernelModel(path, backend, false, 1, nil, 1, &fit)
			if m == nil {
				t.Fatal("bounded production load returned no model")
			}
			defer m.CloseWeights()
			if packed != tc.packed {
				t.Fatalf("production packed arm=%t want=%t", packed, tc.packed)
			}
			if got := m.HasKQuant("model.layers.0.mlp.down_proj.weight"); got != tc.packed {
				t.Fatalf("late-shard actual Q3 retention=%t want=%t", got, tc.packed)
			}
			if tc.packed {
				for _, weight := range []struct {
					name    string
					payload []byte
				}{
					{name: "model.layers.0.self_attn.v_proj.weight", payload: make([]byte, 84)},
					{name: "lm_head.weight", payload: make([]byte, 210)},
					{name: "model.layers.0.mlp.down_proj.weight", payload: v41PackedQ3Block()},
				} {
					raw, ok := m.KQuantRaw(weight.name)
					if !ok || !bytes.Equal(raw, weight.payload) {
						t.Fatalf("native weight %s packed=%t bytes=%d want=%d", weight.name, ok, len(raw), len(weight.payload))
					}
				}
				report := m.ResidentReport()
				if report.KQuantBytes != 404 || report.Q8Bytes != 0 || report.F32Bytes != 5120 || report.TotalResidentBytes != 5524 {
					t.Fatalf("native storage K=%d Q8=%d F32=%d total=%d want 404/0/5120/5524", report.KQuantBytes, report.Q8Bytes, report.F32Bytes, report.TotalResidentBytes)
				}
				opts := serveResidentQ4KLoadOptions(backend, path, true, merged)
				plan, err := serveGGUFWeightMemoryPlanForArm(ws, arm, opts...)
				if err != nil {
					t.Fatal(err)
				}
				if got := m.ResidentReport().TotalResidentBytes; plan.Total() != got {
					t.Fatalf("merged selected-arm estimate=%d actual=%d", plan.Total(), got)
				}
			}
		})
	}
	// All dtypes stay present in the late-shader negative: the missing execution
	// probe alone must defeat first-shard admission.
	if !compute.BackendSupportsDeviceWeightDtype(v41PackedNativeBackend(), compute.Q3_K) {
		t.Fatal("fixture lost Q3 dtype capability")
	}
}
