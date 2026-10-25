//go:build cuda && cgo

// fak-test:runtime slow est=600s

package ggufload

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// cudaGraphParityKVPositions exceeds the 1024-position default so the rung also proves
// the fixed graph KV is sized by SetCUDAGraphKVCapacity rather than refusing the prompt.
const cudaGraphParityKVPositions = 4096

func greedyCUDAQwen35Decode(t *testing.T, m *model.Model, be compute.Backend, prompt []int, steps int) []int {
	t.Helper()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	defer s.Close()
	next := argmaxF32(s.Prefill(prompt))
	toks := make([]int, 0, steps)
	for i := 0; i < steps; i++ {
		toks = append(toks, next)
		next = argmaxF32(s.Step(next))
	}
	return toks
}

// TestCUDAGraphQwen35HybridGreedyParity is the P-06 (fak#13985) hardware rung: CUDA-graph
// replay must greedy-decode the same tokens as eager decode for a Qwen3.8 hybrid on the same
// device, for 32 tokens on 3 prompts. It needs an NVIDIA GPU and FAK_QWEN38_Q4KM_GGUF, and
// FAK_CUDA_GRAPH unset so the eager half really runs uncaptured.
func TestCUDAGraphQwen35HybridGreedyParity(t *testing.T) {
	path := os.Getenv("FAK_QWEN38_Q4KM_GGUF")
	if path == "" {
		t.Skip("set FAK_QWEN38_Q4KM_GGUF to a Qwen3.8 Q4_K_M GGUF to run the CUDA graph parity rung")
	}
	if os.Getenv("FAK_CUDA_GRAPH") == "1" {
		t.Skip("unset FAK_CUDA_GRAPH: the eager half must run with graph replay off")
	}
	be := compute.Pick("cuda")
	if be == nil || be.Name() != "cuda" {
		t.Skip("no CUDA device")
	}
	m, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.IsQwen35Hybrid() {
		t.Skipf("want a Qwen3.5-family hybrid, got model_type=%q", m.Cfg.ModelType)
	}
	_, free, _ := compute.DeviceMemoryInfo(be)
	sizing := model.CUDAGraphKVSizingFor(m.Cfg, be, true, cudaGraphParityKVPositions, free)
	if !sizing.Eligible {
		t.Fatalf("config is not capture-eligible (%s); record the reason and use an unscaled config", sizing.Reason)
	}
	t.Logf("graph KV sizing: rows=%d positions=%d prealloc=%d bytes reason=%s", sizing.KVRows, cudaGraphParityKVPositions, sizing.Prealloc.Bytes, sizing.Prealloc.Reason)

	prompts := [][]int{
		{9707, 11, 1124, 374, 264, 1273, 315, 279},
		{785, 3974, 13876, 38835, 34208, 13, 576},
		{40, 1079, 264, 3460, 1618, 315, 7377, 15226, 13},
	}
	const steps = 32
	eager := make([][]int, len(prompts))
	for i, prompt := range prompts {
		eager[i] = greedyCUDAQwen35Decode(t, m, be, prompt, steps)
	}

	compute.SetCUDAGraphKVCapacity(cudaGraphParityKVPositions)
	compute.EnableCUDAGraph()
	before := compute.CUDAGraphReplays()
	for i, prompt := range prompts {
		got := greedyCUDAQwen35Decode(t, m, be, prompt, steps)
		for j := range got {
			if got[j] != eager[i][j] {
				t.Fatalf("prompt %d token %d diverged: graph=%d eager=%d\ngraph=%v\neager=%v", i, j, got[j], eager[i][j], got, eager[i])
			}
		}
	}
	if replays := compute.CUDAGraphReplays() - before; replays == 0 {
		t.Fatal("graph-enabled decode made no CUDA graph replay; the parity compared eager with eager")
	}
}
