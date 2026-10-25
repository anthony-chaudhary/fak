//go:build cuda && cgo

package compute

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	cudaQwen38AttnBenchEnv       = "FAK_CUDA_QWEN38_ATTN_BENCH"
	cudaQwen38AttnBenchPromptEnv = "FAK_CUDA_QWEN38_ATTN_BENCH_P"
	cudaQwen38AttnBenchItersEnv  = "FAK_CUDA_QWEN38_ATTN_BENCH_ITERS"
	// Qwen3.8-27B interleaves one full-attention layer after every three GDN
	// layers across 64 main layers.
	qwen38FullAttentionLayers = 16
)

type cudaQwen38AttnBenchRow struct {
	Schema         string    `json:"schema"`
	Kernel         string    `json:"kernel"`
	GPU            string    `json:"gpu"`
	Prompt         int       `json:"prompt_tokens"`
	Prefix         int       `json:"prefix"`
	Iterations     int       `json:"iterations"`
	LayerMSMedian  float64   `json:"layer_ms_median"`
	LayerMSSamples []float64 `json:"layer_ms_samples"`
	ModelMS        float64   `json:"full_attention_layers_ms"`
	FullAttnLayers int       `json:"full_attention_layers"`
	EvidenceClass  string    `json:"evidence_class"`
}

func cudaQwen38AttnBenchPrompts(t *testing.T) []int {
	t.Helper()
	raw := strings.TrimSpace(os.Getenv(cudaQwen38AttnBenchPromptEnv))
	if raw == "" {
		return []int{512, 4096}
	}
	var prompts []int
	for _, field := range strings.Split(raw, ",") {
		value, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil || value <= 0 {
			t.Fatalf("%s=%q: want comma-separated positive prompt lengths", cudaQwen38AttnBenchPromptEnv, raw)
		}
		prompts = append(prompts, value)
	}
	return prompts
}

func cudaQwen38AttnBenchGPU() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0])
}

// TestCUDAQwen38PrefillAttentionBench times the Qwen3.8 hd256 prompt-attention
// panel at prefix 0 for each prompt length and logs one JSON receipt row per
// length. The step-2 profile gate of P-05 multiplies the per-layer time by the
// 16 full-attention layers and divides by end-to-end prefill time on the same
// GPU UUID.
//
// fak-test:runtime slow est=60s
func TestCUDAQwen38PrefillAttentionBench(t *testing.T) {
	if os.Getenv(cudaQwen38AttnBenchEnv) != "1" {
		t.Skipf("set %s=1 on an sm_80+ CUDA host to time Qwen3.8 prompt attention", cudaQwen38AttnBenchEnv)
	}
	iterations := 5
	if raw := os.Getenv(cudaQwen38AttnBenchItersEnv); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value <= 0 {
			t.Fatalf("%s=%q: want a positive iteration count", cudaQwen38AttnBenchItersEnv, raw)
		}
		iterations = value
	}
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	const (
		heads   = qwen38CausalAttentionPanelHeads
		kvHeads = qwen38CausalAttentionPanelKVHeads
		headDim = qwen38CausalAttentionPanelHeadDim
	)
	scale := float32(1 / math.Sqrt(headDim))
	gpu := cudaQwen38AttnBenchGPU()
	for _, prompt := range cudaQwen38AttnBenchPrompts(t) {
		rng := qwen35PanelLCG(0x13984)
		upload := func(shape []int, data []float32, site string) Tensor {
			resident := be.UploadClass(NewF32(Default(), shape, data), F32, MemoryScratchpad, site)
			t.Cleanup(func() { be.Free(resident) })
			return resident
		}
		q := upload([]int{prompt, heads * headDim}, rng.values(prompt*heads*headDim, 0.5), "qwen38-attn-bench-query")
		k := upload([]int{prompt, kvHeads * headDim}, rng.values(prompt*kvHeads*headDim, 0.5), "qwen38-attn-bench-key")
		v := upload([]int{prompt, kvHeads * headDim}, rng.values(prompt*kvHeads*headDim, 0.5), "qwen38-attn-bench-value")
		out := upload([]int{prompt, heads * headDim}, make([]float32, prompt*heads*headDim), "qwen38-attn-bench-output")
		run := func() {
			if err := be.qwen35CausalAttentionPanelIntoForTest(q, k, v, out, prompt, 0, heads, kvHeads, headDim, scale); err != nil {
				t.Fatalf("prompt attention P=%d: %v", prompt, err)
			}
		}
		run() // warm-up: module load and first-touch costs stay out of the samples
		samples := make([]float64, 0, iterations)
		for range iterations {
			start := time.Now()
			run()
			samples = append(samples, float64(time.Since(start).Microseconds())/1000)
		}
		sorted := append([]float64(nil), samples...)
		sort.Float64s(sorted)
		median := sorted[len(sorted)/2]
		kernel := os.Getenv("FAK_CUDA_QWEN38_PREFILL_ATTN")
		if kernel == "" {
			kernel = "default"
		}
		row, err := json.Marshal(cudaQwen38AttnBenchRow{
			Schema: "fak.qwen38-prefill-attention-bench/1", Kernel: kernel, GPU: gpu,
			Prompt: prompt, Iterations: iterations, LayerMSMedian: median, LayerMSSamples: samples,
			ModelMS: median * qwen38FullAttentionLayers, FullAttnLayers: qwen38FullAttentionLayers,
			EvidenceClass: "HW-WITNESSED",
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s", row)
	}
}
