package sotamatrix

import (
	"path"
	"strings"
	"testing"
)

// anyGlobMatches mirrors the PRIOR_ART gate's path.Match coverage semantics closely
// enough to assert that a row covers a specific kernel file: both operate on
// forward-slash paths and no glob in the matrix needs to cross a separator.
func anyGlobMatches(globs []string, p string) bool {
	for _, g := range globs {
		if ok, _ := path.Match(g, p); ok {
			return true
		}
	}
	return false
}

// TestCollectiveCommRowCoversNCCLProcessGroup pins the fix for a SOTA-matrix blind spot:
// internal/compute/cuda_nccl_pg.cu (the multi-PROCESS NCCL bootstrap) was covered by no
// prior-art row, so an agent optimizing the collective could re-derive a ring/tree
// all-reduce with no reference or oracle in hand. This asserts the collective-comm row
// exists and that its FileGlobs actually cover the previously-orphaned kernel file. RED
// before the row was added (BySlug fails), GREEN after.
func TestCollectiveCommRowCoversNCCLProcessGroup(t *testing.T) {
	op, ok := BySlug("collective-comm")
	if !ok {
		t.Fatal("collective-comm row missing from the SOTA matrix")
	}
	const orphan = "internal/compute/cuda_nccl_pg.cu"
	if !anyGlobMatches(op.FileGlobs, orphan) {
		t.Fatalf("collective-comm FileGlobs %v do not cover %s", op.FileGlobs, orphan)
	}
	// The device file also binds -lnccl and calls NCCL directly; the honest route is bind,
	// not a from-scratch all-reduce.
	if op.Route != RouteBind {
		t.Errorf("collective-comm route = %q, want %q", op.Route, RouteBind)
	}
}

func TestCUDAQwenGDNPriorArt(t *testing.T) {
	op, ok := BySlug("cuda-qwen-gdn")
	if !ok {
		t.Fatal("cuda-qwen-gdn row missing from the SOTA matrix")
	}
	for _, p := range []string{
		"internal/compute/cuda_qwen35_gdn.go",
		"internal/compute/cuda_qwen35_gdn_sequence.go",
		"internal/compute/cuda_kernels.go",
		"internal/compute/cuda_kernels.cu",
	} {
		if !anyGlobMatches(op.FileGlobs, p) {
			t.Errorf("cuda-qwen-gdn FileGlobs %v do not cover %s", op.FileGlobs, p)
		}
	}
	if op.Route != RouteBorrow {
		t.Errorf("cuda-qwen-gdn route = %q, want %q", op.Route, RouteBorrow)
	}
	for _, pin := range []string{
		"FLA@bccaf2d3cf4d9badc8be050a2c71616220b246d7",
		"llama.cpp@ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0",
		"MLX-LM@cc8521569694a3240b52c98acffd100d59b4c755",
	} {
		if !strings.Contains(op.SOTA, pin) {
			t.Errorf("cuda-qwen-gdn SOTA %q does not name pinned source %q", op.SOTA, pin)
		}
	}
	for _, obligation := range []string{
		"per-step output",
		"convolution-state",
		"recurrent-state",
		"exact deterministic greedy-token equality",
		"fak-native CUDA",
		"zero fallback",
	} {
		if !strings.Contains(op.Oracle, obligation) {
			t.Errorf("cuda-qwen-gdn oracle %q does not name %q", op.Oracle, obligation)
		}
	}
	if !strings.Contains(op.Note, "no runtime or backend fallback") {
		t.Errorf("cuda-qwen-gdn note %q does not reject runtime/backend fallback", op.Note)
	}
}

func TestMetalQwenGDNPriorArt(t *testing.T) {
	op, ok := BySlug("metal-qwen-gdn")
	if !ok {
		t.Fatal("metal-qwen-gdn row missing from the SOTA matrix")
	}
	for _, p := range []string{
		"internal/model/qwen35.go",
		"internal/model/qwen35_gdn.go",
		"internal/metalgemm/qwen35_decode.m",
	} {
		if !anyGlobMatches(op.FileGlobs, p) {
			t.Errorf("metal-qwen-gdn FileGlobs %v do not cover %s", op.FileGlobs, p)
		}
	}
	if op.Route != RouteBorrow {
		t.Errorf("metal-qwen-gdn route = %q, want %q", op.Route, RouteBorrow)
	}
	for _, pin := range []string{
		"llama.cpp@ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0",
		"MLX@43d2f06cb87e76895bf9a152bade4fee83408643",
		"MLX-LM@cc8521569694a3240b52c98acffd100d59b4c755",
	} {
		if !strings.Contains(op.SOTA, pin) {
			t.Errorf("metal-qwen-gdn SOTA %q does not name pinned source %q", op.SOTA, pin)
		}
	}
	for _, obligation := range []string{
		"CPU reference",
		"per-step output parity",
		"convolution-state parity",
		"recurrent-state parity",
		"exact greedy-token equality",
	} {
		if !strings.Contains(op.Oracle, obligation) {
			t.Errorf("metal-qwen-gdn oracle %q does not name %q", op.Oracle, obligation)
		}
	}
	if !strings.Contains(op.Note, "no runtime or backend fallback") {
		t.Errorf("metal-qwen-gdn note %q does not reject runtime/backend fallback", op.Note)
	}
}

func TestAMDVulkanPriorArtPaths(t *testing.T) {
	tests := []struct {
		path, slug, license string
		route               Route
		pins, obligations   []string
	}{
		{
			path: "internal/compute/vulkan.go", slug: "amd-vulkan-quant-gemm", route: RouteBorrow, license: "MIT",
			pins:        []string{"Nathanw1014/strix-halo-llamacpp@45bec945dd4c7944fba45ebd02e6b713b342d0ba"},
			obligations: []string{"CPU dequant/GEMM", "argmax exact", "cosine"},
		},
		{
			path: "internal/compute/shaders/q2k_matmul.comp", slug: "amd-vulkan-quant-gemm", route: RouteBorrow, license: "MIT",
			pins:        []string{"Nathanw1014/strix-halo-llamacpp@45bec945dd4c7944fba45ebd02e6b713b342d0ba"},
			obligations: []string{"CPU dequant/GEMM", "argmax exact", "cosine"},
		},
		{
			path: "internal/compute/vulkan_qwen35_gdn.go", slug: "amd-vulkan-qwen-gdn", route: RouteStayMinimal, license: "Apache-2.0",
			pins: []string{
				"julianmb/q38rocm@6c142530031c923ece1adf4d8c9c824b7369fef8",
				"FLA@bccaf2d3cf4d9badc8be050a2c71616220b246d7",
			},
			obligations: []string{"production CPU", "multi-token", "convolution-state", "recurrent-state", "logit", "exact greedy-token"},
		},
		{
			path: "internal/compute/vulkan_qwen35_sequence.go", slug: "amd-vulkan-sequence-attention", route: RouteBorrow, license: "MIT",
			pins:        []string{"Nathanw1014/strix-halo-llamacpp@45bec945dd4c7944fba45ebd02e6b713b342d0ba"},
			obligations: []string{"production CPU", "full sequence", "exact token identity", "finite logits", "cosine"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			op, ok := BySlug(tt.slug)
			if !ok {
				t.Fatalf("%s row missing from the SOTA matrix", tt.slug)
			}
			if !anyGlobMatches(op.FileGlobs, tt.path) {
				t.Errorf("%s FileGlobs %v do not cover %s", tt.slug, op.FileGlobs, tt.path)
			}
			if op.Route != tt.route {
				t.Errorf("%s route = %q, want %q", tt.slug, op.Route, tt.route)
			}
			for _, pin := range tt.pins {
				if !strings.Contains(op.SOTA, pin) {
					t.Errorf("%s SOTA %q does not name pinned source %q", tt.slug, op.SOTA, pin)
				}
			}
			if !strings.Contains(op.PrimaryLink, "45bec945dd4c7944fba45ebd02e6b713b342d0ba") &&
				!strings.Contains(op.PrimaryLink, "6c142530031c923ece1adf4d8c9c824b7369fef8") {
				t.Errorf("%s PrimaryLink %q is not revision-pinned", tt.slug, op.PrimaryLink)
			}
			if !strings.Contains(op.SOTA, tt.license) {
				t.Errorf("%s SOTA %q does not state license %q", tt.slug, op.SOTA, tt.license)
			}
			for _, obligation := range tt.obligations {
				if !strings.Contains(op.Oracle, obligation) {
					t.Errorf("%s oracle %q does not name %q", tt.slug, op.Oracle, obligation)
				}
			}
			if !strings.Contains(op.Note, "no runtime or backend fallback") {
				t.Errorf("%s note %q does not reject runtime/backend fallback", tt.slug, op.Note)
			}
		})
	}
}

func TestKVCacheTransformCompressionPriorArt(t *testing.T) {
	op, ok := BySlug("kv-cache-transform-compression")
	if !ok {
		t.Fatal("kv-cache-transform-compression row missing from the SOTA matrix")
	}
	for _, p := range []string{
		"internal/model/kvquant.go",
		"internal/model/coldkv.go",
		"internal/engine/kv_quantization.go",
		"internal/compute/kvprecision.go",
	} {
		if !anyGlobMatches(op.FileGlobs, p) {
			t.Errorf("kv-cache-transform-compression FileGlobs %v do not cover %s", op.FileGlobs, p)
		}
	}
	if op.Route != RouteBorrow {
		t.Errorf("kv-cache-transform-compression route = %q, want %q", op.Route, RouteBorrow)
	}
	if !strings.Contains(op.SOTA, "SPECTRA@272032275106dc5944fbfa7091a1ceb403fa7e28") {
		t.Errorf("SOTA %q does not pin the studied SPECTRA revision", op.SOTA)
	}
	for _, obligation := range []string{
		"fak-native Qwen3.8",
		"resident bytes",
		"peak transient bytes",
		"end-to-end latency and throughput",
		"exact pre-RoPE eviction/reuse compatibility",
		"zero fallback",
	} {
		if !strings.Contains(op.Oracle, obligation) {
			t.Errorf("kv-cache-transform-compression oracle %q does not name %q", op.Oracle, obligation)
		}
	}
}

// TestEveryRowResolvesAndIsComplete mirrors the coverage scorecard's per-row HARD KPIs
// in-binary: every row resolves by its own slug and carries a primary http(s) link, an
// oracle, and a fak-path. A row that fails this is a matrix that silently stopped being
// load-bearing — the exact rot the datum exists to prevent.
func TestEveryRowResolvesAndIsComplete(t *testing.T) {
	seen := map[string]bool{}
	for _, op := range Operations() {
		if seen[op.Slug] {
			t.Errorf("duplicate slug %q", op.Slug)
		}
		seen[op.Slug] = true
		if got, ok := BySlug(op.Slug); !ok || got.Slug != op.Slug {
			t.Errorf("row %q does not resolve via BySlug", op.Slug)
		}
		if !strings.HasPrefix(op.PrimaryLink, "http") {
			t.Errorf("row %q PrimaryLink = %q, want an http(s) SOTA link", op.Slug, op.PrimaryLink)
		}
		if strings.TrimSpace(op.Oracle) == "" {
			t.Errorf("row %q has no oracle (a kernel with no oracle is not done)", op.Slug)
		}
		if strings.TrimSpace(op.FakPath) == "" {
			t.Errorf("row %q has no FakPath", op.Slug)
		}
	}
}
