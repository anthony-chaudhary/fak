package main

import (
	"fmt"
)

const (
	nativeQwenQ4KPrefillChunkFlag    = "native-qwen-q4k-prefill-chunk-tokens"
	defaultNativeQwenQ4KPrefillChunk = 512
	minNativeQwenQ4KPrefillChunk     = 128
	maxNativeQwenQ4KPrefillChunk     = 8192
)

// validateNativeQwenQ4KPrefillChunk validates the shared serve/run/guard bound before
// model loading. The parsed value is carried through typed planner configuration;
// this function intentionally performs no ambient process mutation.
func validateNativeQwenQ4KPrefillChunk(tokens int) error {
	if tokens < minNativeQwenQ4KPrefillChunk || tokens > maxNativeQwenQ4KPrefillChunk {
		return fmt.Errorf("--%s=%d: want %d..%d", nativeQwenQ4KPrefillChunkFlag, tokens, minNativeQwenQ4KPrefillChunk, maxNativeQwenQ4KPrefillChunk)
	}
	return nil
}
