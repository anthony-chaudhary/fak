package engine

import (
	"os"
	"strconv"
	"strings"
)

// Serve knobs for the decode-starvation bound (fak-private#3630). The registered
// batcher engine is built from DefaultBatchingEngineConfig, so these env values
// reach ContinuousBatcherConfig on the `fak serve` dispatch path. Unset, malformed,
// or non-positive values keep the zero config (prefill-first, today's behavior).
const (
	envBatcherWaitingServedRatio    = "FAK_BATCHER_WAITING_SERVED_RATIO"
	envBatcherMaxWaitingDecodeSteps = "FAK_BATCHER_MAX_WAITING_DECODE_STEPS"
)

func applyBatcherStarvationEnv(c *ContinuousBatcherConfig) {
	if c == nil {
		return
	}
	if r, err := strconv.ParseFloat(strings.TrimSpace(os.Getenv(envBatcherWaitingServedRatio)), 64); err == nil && r > 0 && r <= 1e6 {
		c.WaitingServedRatio = r
	}
	if n, err := strconv.Atoi(strings.TrimSpace(os.Getenv(envBatcherMaxWaitingDecodeSteps))); err == nil && n > 0 {
		c.MaxWaitingDecodeSteps = n
	}
}
