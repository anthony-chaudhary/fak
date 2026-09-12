package model

import (
	"errors"
	"fmt"
)

// ErrV41DSparkUnsupported is the typed fail-closed refusal for the V4.1 Flash
// DSpark decoder. The official checkpoint declares dspark_* metadata, so a
// config that requests the speculative decoder must be DETECTED and rejected
// here rather than silently running the plain autoregressive text forward.
// This is admission and routing only: no draft model, verify loop, or
// speculative rollback is implemented, and none is claimed. DSpark is
// DISABLED until the text baseline is qualified.
var ErrV41DSparkUnsupported = errors.New("model: DeepSeek V4.1 Flash DSpark decoder is not implemented")

// DeepSeekV41DSparkConfig retains the official checkpoint's dspark_* markers
// without admitting an execution path. It is metadata only; its presence makes
// the config a DSpark request that RefuseDeepSeekV41DSpark rejects.
type DeepSeekV41DSparkConfig struct {
	// NextNPredictLayers is the checkpoint's num_nextn_predict_layers: the
	// multi-token-prediction head count. It is a speculative marker on its own,
	// so its presence triggers the refusal even when the dspark_* fields are
	// absent.
	NextNPredictLayers int
	BlockSize          int
	NoiseTokenID       int
	TargetLayerIDs     []int
	MarkovRank         int
	NumRoutedExperts   int
	NumExpertsPerTok   int
}

// HasDeepSeekV41DSpark reports whether the config requests the V4.1 DSpark
// speculative decoder. The retained dspark metadata being present is the
// fail-closed marker.
func (c Config) HasDeepSeekV41DSpark() bool {
	return c.DeepSeekV41 != nil && c.DeepSeekV41.DSpark != nil
}

// RefuseDeepSeekV41DSpark is the admission hook the V4.1 dispatch seam calls
// before any speculative work. It is nil (no-op) for every non-V4.1 family and
// for a V4.1 config that requests no DSpark, so the plain text path is
// byte-for-byte unchanged. A V4.1 DSpark request is refused with
// ErrV41DSparkUnsupported.
func (c Config) RefuseDeepSeekV41DSpark() error {
	if !c.IsDeepSeekV41() || !c.HasDeepSeekV41DSpark() {
		return nil
	}
	d := c.DeepSeekV41.DSpark
	return fmt.Errorf("%w: %s@%s declares DSpark(block=%d targets=%v markov=%d); speculation stays disabled until the text baseline is qualified", ErrV41DSparkUnsupported, DeepSeekV41FlashModelID, DeepSeekV41FlashRevision, d.BlockSize, d.TargetLayerIDs, d.MarkovRank)
}
