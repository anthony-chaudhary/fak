package compute

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
)

// PromptPanelKQuantMode is how the CUDA prompt panel runs Q4_K/Q6_K projections.
type PromptPanelKQuantMode uint32

const (
	// PromptPanelKQuantBatched reads each weight once per panel chunk through the prefill GEMM
	// arms (Q4_K mmq or tile panel, Q6_K tile panel). Rows are not bitwise the decode GEMV, so
	// the panel is approximate against the token loop.
	PromptPanelKQuantBatched PromptPanelKQuantMode = iota
	// PromptPanelKQuantRowExact runs the decode GEMV once per row, bitwise the token loop.
	PromptPanelKQuantRowExact
	// PromptPanelKQuantOff keeps K-quant models on the token loop.
	PromptPanelKQuantOff
)

func (m PromptPanelKQuantMode) String() string {
	switch m {
	case PromptPanelKQuantBatched:
		return "batched"
	case PromptPanelKQuantRowExact:
		return "rowexact"
	case PromptPanelKQuantOff:
		return "off"
	}
	return fmt.Sprintf("PromptPanelKQuantMode(%d)", uint32(m))
}

// CUDAPromptPanelKQuantEnv selects PromptPanelKQuantMode: batched (default), rowexact or off.
const CUDAPromptPanelKQuantEnv = "FAK_CUDA_PROMPT_PANEL_KQUANT"

// CUDAKQuantPromptPanelEnv=1 is the legacy spelling of rowexact; an explicit
// CUDAPromptPanelKQuantEnv value wins over it.
const CUDAKQuantPromptPanelEnv = "FAK_CUDA_KQUANT_PROMPT_PANEL"

// ErrPromptPanelKQuantModeUnknown refuses a CUDAPromptPanelKQuantEnv value no mode answers to.
var ErrPromptPanelKQuantModeUnknown = errors.New("unknown CUDA prompt-panel K-quant mode")

// ResolvePromptPanelKQuantMode reads the mode env value and the legacy opt-in value.
func ResolvePromptPanelKQuantMode(mode, legacy string) (PromptPanelKQuantMode, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "":
		if strings.TrimSpace(legacy) == "1" {
			return PromptPanelKQuantRowExact, nil
		}
		return PromptPanelKQuantBatched, nil
	case "batched", "1", "on":
		return PromptPanelKQuantBatched, nil
	case "rowexact", "row-exact", "exact":
		return PromptPanelKQuantRowExact, nil
	case "off", "0":
		return PromptPanelKQuantOff, nil
	}
	return PromptPanelKQuantOff, fmt.Errorf("%s=%q: %w", CUDAPromptPanelKQuantEnv, mode, ErrPromptPanelKQuantModeUnknown)
}

var promptPanelKQuantMode atomic.Uint32

func init() {
	mode, err := ResolvePromptPanelKQuantMode(os.Getenv(CUDAPromptPanelKQuantEnv), os.Getenv(CUDAKQuantPromptPanelEnv))
	if err != nil {
		// A typo must not silently enable an approximate path: fall back to the token loop.
		fmt.Fprintf(os.Stderr, "fak: %v; K-quant prompt panel off\n", err)
	}
	promptPanelKQuantMode.Store(uint32(mode))
}

// CurrentPromptPanelKQuantMode is the mode prompt-panel admission reads per prompt.
func CurrentPromptPanelKQuantMode() PromptPanelKQuantMode {
	return PromptPanelKQuantMode(promptPanelKQuantMode.Load())
}

// SetPromptPanelKQuantMode replaces the mode and returns the previous one. Change it only
// between prompts: admission and the panel GEMM each read it.
func SetPromptPanelKQuantMode(m PromptPanelKQuantMode) PromptPanelKQuantMode {
	return PromptPanelKQuantMode(promptPanelKQuantMode.Swap(uint32(m)))
}

// CUDAPromptPanelCounters are the panel engagement witness: scopes opened, panel GEMMs issued
// (BatchedMatMuls is the subset that read the weight once for all rows), and panel buffers
// still held when the last scope closed.
type CUDAPromptPanelCounters struct {
	Begins, MatMuls, BatchedMatMuls uint64
	LastReleaseLive                 int64
}

// promptPanelRowExactFor is the dtype set whose panel GEMM is bitwise the decode GEMV per row.
func promptPanelRowExactFor(mode PromptPanelKQuantMode, dt Dtype) bool {
	return dt == Q8_0 || mode == PromptPanelKQuantRowExact && (dt == Q4_K || dt == Q6_K)
}

// promptPanelBatchedFor is the dtype set whose panel GEMM reads the weight once for many rows
// instead of once per row. Disjoint from promptPanelRowExactFor in every mode.
func promptPanelBatchedFor(mode PromptPanelKQuantMode, dt Dtype) bool {
	return mode == PromptPanelKQuantBatched && (dt == Q4_K || dt == Q6_K)
}

// promptPanelBatchRows is the most rows one batched K-quant GEMM launch may take. A Q4_K
// panel of DequantGEMMScratchRows or more rows on any arm but mmq would allocate a
// weight-sized dequant or FP8 image the chunk planner does not price, so those panels run as
// sub-batches that stay on the scratch-free tile kernel. mmq and the Q6_K arms only allocate
// activation-sized scratch.
func promptPanelBatchRows(dt Dtype, tokens int, q4kMMQ bool) int {
	if dt == Q4_K && !q4kMMQ && tokens >= DequantGEMMScratchRows {
		return DequantGEMMScratchRows - 1
	}
	return tokens
}
