package main

import (
	"fmt"
	"os"
	"strconv"
)

const (
	serveNativeQwenQ4KPrefillChunkFlag = "native-qwen-q4k-prefill-chunk-tokens"
	nativeQwenQ4KPrefillChunkEnv       = "FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS"
	defaultNativeQwenQ4KPrefillChunk   = 512
	minNativeQwenQ4KPrefillChunk       = 128
	maxNativeQwenQ4KPrefillChunk       = 4096
)

// applyServeNativeQwenQ4KPrefillChunk validates the serve-facing bound before
// model loading and explicitly carries it over the strict agent environment
// contract consumed once by NewInKernelPlanner. The native planner remains the
// only execution path; this control never selects or falls back to llama.cpp.
func applyServeNativeQwenQ4KPrefillChunk(tokens int) error {
	if tokens < minNativeQwenQ4KPrefillChunk || tokens > maxNativeQwenQ4KPrefillChunk {
		return fmt.Errorf("--%s=%d: want %d..%d", serveNativeQwenQ4KPrefillChunkFlag, tokens, minNativeQwenQ4KPrefillChunk, maxNativeQwenQ4KPrefillChunk)
	}
	if err := os.Setenv(nativeQwenQ4KPrefillChunkEnv, strconv.Itoa(tokens)); err != nil {
		return fmt.Errorf("--%s: propagate to %s: %w", serveNativeQwenQ4KPrefillChunkFlag, nativeQwenQ4KPrefillChunkEnv, err)
	}
	return nil
}
