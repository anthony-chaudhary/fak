package perfscout

import (
	"encoding/json"
	"fmt"
	"strings"
)

// CrossInnovation defines an architectural or kernel innovation discovered on one platform
// that is portable and applicable to other inference architectures.
type CrossInnovation struct {
	ID                     string   `json:"id"`
	Title                  string   `json:"title"`
	SourcePlatform         string   `json:"source_platform"`
	SourceRepo             string   `json:"source_repo"`
	SourceAnchor           string   `json:"source_anchor"`
	TargetPlatforms        []string `json:"target_platforms"`
	ProblemSolvedOnSource  string   `json:"problem_solved_on_source"`
	TargetBottleneckSolved string   `json:"target_bottleneck_solved"`
	MathematicalBasis      string   `json:"mathematical_basis,omitempty"`
	TranslationRecipe      string   `json:"translation_recipe"`
	ExpectedGain           string   `json:"expected_gain"`
	Status                 string   `json:"status"` // "STUDIED", "ADAPTED", "PROVEN"
}

// DefaultCrossInnovations contains the 10 canonical cross-architecture translation vectors.
var DefaultCrossInnovations = []CrossInnovation{
	{
		ID:                     "XINNOV-01",
		Title:                  "Multi-Tensor Preallocated Slot-Streaming for Large MoE",
		SourcePlatform:         "Apple Silicon (MLX / Metal)",
		SourceRepo:             "carloslfu/slotstream",
		SourceAnchor:           "App/Model/SlotStreamEngine.swift:42-180@53028cf3",
		TargetPlatforms:        []string{"NVIDIA CUDA", "AMD ROCm"},
		ProblemSolvedOnSource:  "Metal single-buffer length cap (28.1 GB) prevented loading 125B MoE weights (105 GB) on 48 GB Macs.",
		TargetBottleneckSolved: "Enables single-GPU serving of 125B-180B MoE models on consumer 24GB/32GB cards (RTX 4090, RX 7900 XTX) without 4x GPU clusters.",
		MathematicalBasis:      "Dense trunk (3.8 GB) resident; 48 routed layers x 2.76 MB expert records streamed asynchronously via QD32 pread (17.3 GB/s line rate).",
		TranslationRecipe:      "Decompose VRAM slot buffer into 9 distinct layer tensors (W1/W2/W3 x weight/scale/bias). Use Linux io_uring with O_DIRECT or DirectStorage 1.3 to stream expert records directly to GPU BAR.",
		ExpectedGain:           "Unlocks 125B MoE inference on a single $1,600 RTX 4090 at 18-25 tok/s.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-02",
		Title:                  "Compact DSA / MLA Latent Cache with 16-Token Rolling Reversion Journal",
		SourcePlatform:         "Apple Silicon (MLX)",
		SourceRepo:             "kiojuvr/glm53-flash-mlx",
		SourceAnchor:           "models/glm53_flash/compact_cache.py:45-189@04c4e9e",
		TargetPlatforms:        []string{"NVIDIA CUDA", "AMD ROCm"},
		ProblemSolvedOnSource:  "DSA/MLA NoPE latent cache state scaled to multiple gigabytes at 256K context depths.",
		TargetBottleneckSolved: "Eliminates 8-10 GB transient allocator spikes during chunked prefill on Grace-Blackwell GB10 and dual RTX 5090s.",
		MathematicalBasis:      "Contiguous pool buffer with <=3 token active tail and rolling 16-token rollback journal: (16 + index_kpool - 1 = 19 tokens max history state).",
		TranslationRecipe:      "Port the circular pool and 16-token rolling history journal into CUDA/HIP attention metadata allocators, replacing unbounded per-token latent storage.",
		ExpectedGain:           "85.95% reduction in latent cache state memory; saves ~1.28 GB VRAM per sequence.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-03",
		Title:                  "Stream-Async GPU-Doorbell Direct RDMA / RoCE Collective All-Reduce",
		SourcePlatform:         "AMD ROCm (USB4 / Strix Halo RoCE)",
		SourceRepo:             "davidcanar/vllm-strix-halo",
		SourceAnchor:           "container/native/tbv_ar2.hip:212-409@22ecd28",
		TargetPlatforms:        []string{"NVIDIA CUDA", "Apple Metal"},
		ProblemSolvedOnSource:  "RCCL CPU dispatch overhead (~100 µs/layer) and TCP latency collapsed token decode across a $30 USB4 RoCE cable.",
		TargetBottleneckSolved: "1) Cuts collective latency on non-NVLink dual RTX 5090s; 2) Creates the first multi-Mac Thunderbolt clustering transport for Metal.",
		MathematicalBasis:      "8 µs NHI interrupt moderation + direct UMA GTT buffers + GPU acquire-spin kernel (__builtin_amdgcn_s_sleep / __nanosleep) over RoCE v2.",
		TranslationRecipe:      "Allocate send/recv slots via cudaHostAlloc or MTLResourceStorageModeShared. GPU thread 0 acquire-spins on peer arrival flag, followed by vectorized add reduction with zero CPU intervention.",
		ExpectedGain:           "~105 µs all-reduce over USB4/Thunderbolt RoCE; boosts dual-GPU decode by 15-20%.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-04",
		Title:                  "HashK Dual-Subtable PLE Embedding Compression with Identity Ridge Bypass",
		SourcePlatform:         "NVIDIA CUDA (Grace-Blackwell GB10)",
		SourceRepo:             "airawatraj/dgx-spark-qwen38-flash-agent",
		SourceAnchor:           "tools/build_hashk_ple.py:45-226@659fa229",
		TargetPlatforms:        []string{"Apple Silicon", "AMD Strix Halo"},
		ProblemSolvedOnSource:  "51.2 GB FP8 PLE n-gram table consumed 40% of unified memory on a 128 GB workstation.",
		TargetBottleneckSolved: "Eliminates disk I/O dependency on unified memory Apple Silicon Macs and AMD Strix Halo APUs.",
		MathematicalBasis:      "Dual SplitMix64 polynomial hash (S_h = ceil(V_h / 4)); ridge projection Wh asymptotically converges to Identity (I_160) and is bypassed, saving 409,600 MACs/token.",
		TranslationRecipe:      "Implement dual SplitMix64 hashing directly in Metal and Vulkan compute shaders; gather 80-dim sub-table slices and bypass matrix Wh directly into 1D depthwise conv.",
		ExpectedGain:           "Compresses 51.2 GB table to 12.8 GB VRAM (4x reduction); reclaims 38 GB unified RAM on Mac/Strix Halo.",
		Status:                 "PROVEN",
	},
	{
		ID:                     "XINNOV-05",
		Title:                  "Decoupled Speculative Draft Micro-Batching",
		SourcePlatform:         "AMD Vulkan (Strix Halo)",
		SourceRepo:             "Gr33n93/llama.cpp-qwen3.8-flash-next-mtp-vulkan",
		SourceAnchor:           "patches/0009-spec-draft-ubatch.patch:12-85@f0a073bd",
		TargetPlatforms:        []string{"NVIDIA CUDA", "Apple Metal"},
		ProblemSolvedOnSource:  "Deep context prefills (237K tokens) tripped AMD compute-ring driver watchdog timeouts.",
		TargetBottleneckSolved: "Prevents CUDA graph cache misses and execution stalls when draft length varies across sequence depths.",
		MathematicalBasis:      "Decouple draft micro-batch size (--spec-draft-ubatch-size 512) from primary prompt micro-batch (1024).",
		TranslationRecipe:      "In CUDA/Metal graph planners, allocate a dedicated fixed micro-batch capture dimension for speculative verification independent of primary prompt chunking.",
		ExpectedGain:           "Unlocks 237K-262K context without driver timeouts; preserves 125 tok/s prefill.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-06",
		Title:                  "Direct-I/O Loopback Ext4 KV Cache Tier with Idle-Gated Eviction",
		SourcePlatform:         "NVIDIA CUDA (RTX 5090)",
		SourceRepo:             "adrienbrault/qwen3.8-27b-rtx5090",
		SourceAnchor:           "scripts/setup-native-l2.sh:8-19@00461a30",
		TargetPlatforms:        []string{"Apple Metal", "AMD ROCm"},
		ProblemSolvedOnSource:  "Host page-cache double buffering consumed 37 GB RAM, and background block eviction caused mid-turn engine crashes.",
		TargetBottleneckSolved: "Gives Apple Silicon and AMD ROCm a persistent local disk KV tier (where LMCache is absent or unmaintained).",
		MathematicalBasis:      "losetup --sector-size 4096 --direct-io=on with page-aligned DMA; gate deletion on server idleness (running == 0 && waiting == 0).",
		TranslationRecipe:      "Create a fixed-size loopback ext4 volume formatted with direct-I/O on NVMe. Implement page-aligned block reads/writes and gate background pruning on agent pause boundaries.",
		ExpectedGain:           "0.45s warm revisit of 32K context (16.6x speedup vs cold prefill); zero host RAM double-buffering.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-07",
		Title:                  "WYF Chunkwise Parallel Recurrence for Linear Attention (GDN)",
		SourcePlatform:         "Rust / CUDA (sm100a)",
		SourceRepo:             "MindLab-Research/ferrite",
		SourceAnchor:           "crates/ferrite-kernel/src/wyf.rs:38-147@d771576a",
		TargetPlatforms:        []string{"Apple Metal", "AMD ROCm"},
		ProblemSolvedOnSource:  "Sequential token recurrence loops stalled GPU SIMD lanes during long-prompt prefill.",
		TargetBottleneckSolved: "Accelerates Gated DeltaNet prefill on Apple Metal and AMD RDNA3/RDNA4 by 4x-8x.",
		MathematicalBasis:      "32-token triangular block solve: w_t = beta_t * (v_t - b_t - sum_{s<t} c[t,s] * w_s) with parallel prefix scan for decay L[t,i].",
		TranslationRecipe:      "Port the 32-token triangular block solve into Metal compute shaders and ROCm Wave32/Wave64 HIP kernels, unrolling forward substitution across SIMD registers.",
		ExpectedGain:           "32x reduction in kernel launches; 4x-8x faster linear attention prefill.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-08",
		Title:                  "Mamba / DeltaNet Recurrent State Rollback Checkpointing",
		SourcePlatform:         "NVIDIA CUDA (DGX Spark GB10)",
		SourceRepo:             "airawatraj/dgx-spark-qwen38-flash-agent",
		SourceAnchor:           "patches/qsa_kv_pool.py:98-102@659fa229",
		TargetPlatforms:        []string{"Apple MLX", "llama.cpp"},
		ProblemSolvedOnSource:  "Rejected speculative draft tokens permanently poisoned recurrent linear attention states, causing NaN / token-0 collapse.",
		TargetBottleneckSolved: "Prevents numerical corruption and repetition loops during multi-token speculative decoding across all backends.",
		MathematicalBasis:      "Maintain auxiliary state snapshots at tracking interval 64 (--mamba-track-interval 64); rewind recurrent state S_t to last verified step upon draft rejection.",
		TranslationRecipe:      "Add state snapshot buffers in MLX and llama.cpp execution graphs; on draft verification failure, restore S_t before resuming autoregressive generation.",
		ExpectedGain:           "100% numerical stability across 260K context under aggressive multi-token speculative decoding.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-09",
		Title:                  "Expert-Union MoE Batching & SIMD Table Lookups",
		SourcePlatform:         "Host CPU (Native C)",
		SourceRepo:             "shyringo/qwen3.8-flash-next-in-c",
		SourceAnchor:           "qwen4.c:840-995@ca31f82",
		TargetPlatforms:        []string{"GPU MoE Kernels (CUDA / HIP / Metal)"},
		ProblemSolvedOnSource:  "Nested OpenMP fork-joins during batched token MoE evaluation caused severe cache thrashing on CPU.",
		TargetBottleneckSolved: "Eliminates GPU warp divergence during prefill when tokens in the same warp route to disjoint experts.",
		MathematicalBasis:      "Pre-compute the union of routed experts (<= 40 out of 512 for batch 4); evaluate each active expert once across all tokens; AVX2 vpshufb table lookups.",
		TranslationRecipe:      "Implement expert-union bucketing in CUDA/HIP MoE dispatch kernels, grouping tokens by active expert index prior to GEMM launch.",
		ExpectedGain:           "Linearizes GPU memory access; avoids thread divergence on fine-grained MoE architectures.",
		Status:                 "STUDIED",
	},
	{
		ID:                     "XINNOV-10",
		Title:                  "Dialect-Conforming SSE Keepalive & Cancellation Reverse Proxy",
		SourcePlatform:         "Multi-Node SGLang",
		SourceRepo:             "hasso5703/dgx-spark-qwen38",
		SourceAnchor:           "keepalive-proxy.py:403-541@a08dea9c",
		TargetPlatforms:        []string{"Universal FAK Gateway (All Platforms)"},
		ProblemSolvedOnSource:  "SGLang buffered tool-call arguments for 120s+, causing Claude Code and opencode CLI agent timeouts.",
		TargetBottleneckSolved: "Protects coding agents from disconnects during long prefill or reasoning across ALL local inference engines.",
		MathematicalBasis:      "Emit official SSE comment frames (: ping\\n\\n on /v1/messages, empty choices on /v1/chat) every 10s strictly at event boundaries; monitor client disconnect to abort GPU compute.",
		TranslationRecipe:      "Incorporate the keepalive injection filter into fak's reverse proxy layer. If upstream latency exceeds 10s without an emitted chunk, inject dialect-conforming ping frames.",
		ExpectedGain:           "Zero client timeouts during 5-minute deep prefill or reasoning passes; eliminates zombie GPU execution.",
		Status:                 "ADAPTED",
	},
}

// FilterInnovations filters the canonical innovations by source, target, or keyword.
func FilterInnovations(source, target, keyword string) []CrossInnovation {
	source = strings.ToLower(strings.TrimSpace(source))
	target = strings.ToLower(strings.TrimSpace(target))
	keyword = strings.ToLower(strings.TrimSpace(keyword))

	var out []CrossInnovation
	for _, in := range DefaultCrossInnovations {
		if source != "" && !strings.Contains(strings.ToLower(in.SourcePlatform), source) && !strings.Contains(strings.ToLower(in.SourceRepo), source) {
			continue
		}
		if target != "" {
			matched := false
			for _, t := range in.TargetPlatforms {
				if strings.Contains(strings.ToLower(t), target) {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		if keyword != "" {
			combined := strings.ToLower(in.ID + " " + in.Title + " " + in.SourcePlatform + " " + in.SourceRepo + " " + in.ProblemSolvedOnSource + " " + in.TargetBottleneckSolved + " " + in.TranslationRecipe + " " + in.MathematicalBasis + " " + in.ExpectedGain)
			if !strings.Contains(combined, keyword) {
				continue
			}
		}
		out = append(out, in)
	}
	return out
}

// RenderCrossInnovationsMarkdown renders the innovations into a formatted Markdown report.
func RenderCrossInnovationsMarkdown(innovations []CrossInnovation) string {
	var b strings.Builder
	b.WriteString("# Cross-Architecture Innovation Translation Matrix\n\n")
	b.WriteString(fmt.Sprintf("- **Total Portable Innovations**: %d\n\n", len(innovations)))
	b.WriteString("| ID | Innovation | Source Platform & Repo | Target Platforms | Target Bottleneck Solved | Expected Gain |\n")
	b.WriteString("|---|---|---|---|---|---|\n")

	for _, in := range innovations {
		targets := strings.Join(in.TargetPlatforms, ", ")
		b.WriteString(fmt.Sprintf("| **%s** | **%s** | %s<br/>`%s` | %s | %s | %s |\n",
			in.ID, in.Title, in.SourcePlatform, in.SourceRepo, targets, in.TargetBottleneckSolved, in.ExpectedGain))
	}

	b.WriteString("\n## Detailed Translation Recipes\n\n")
	for _, in := range innovations {
		b.WriteString(fmt.Sprintf("### %s: %s\n\n", in.ID, in.Title))
		b.WriteString(fmt.Sprintf("- **Originating Seam**: `%s` (%s)\n", in.SourceAnchor, in.SourcePlatform))
		b.WriteString(fmt.Sprintf("- **Target Platforms**: %s\n", strings.Join(in.TargetPlatforms, ", ")))
		b.WriteString(fmt.Sprintf("- **Problem on Source**: %s\n", in.ProblemSolvedOnSource))
		b.WriteString(fmt.Sprintf("- **Bottleneck Solved on Target**: %s\n", in.TargetBottleneckSolved))
		if in.MathematicalBasis != "" {
			b.WriteString(fmt.Sprintf("- **Mathematical Formulation**: %s\n", in.MathematicalBasis))
		}
		b.WriteString(fmt.Sprintf("- **Translation Recipe**: %s\n", in.TranslationRecipe))
		b.WriteString(fmt.Sprintf("- **Expected Impact**: %s\n\n", in.ExpectedGain))
	}

	return b.String()
}

// RenderCrossInnovationsJSON formats the innovations as indented JSON.
func RenderCrossInnovationsJSON(innovations []CrossInnovation) ([]byte, error) {
	return json.MarshalIndent(innovations, "", "  ")
}
