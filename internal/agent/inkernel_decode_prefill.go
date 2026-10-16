package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const inKernelQwenQ4KPrefillChunkTokens = 512

type inKernelPrefillSession interface {
	PrefillNoLogits([]int)
	Prefill([]int) []float32
}

// prefillDivergentSuffix bounds the temporary prompt panels used by the resident
// Qwen hybrid Q4_K path and makes each completed panel a cancellation boundary.
// #9066 proves the append equivalence for the resident CPU route; only the last
// chunk needs the distribution consumed by decode. A device route is admitted
// only when its backend implements the complete native sequence operation and
// advertises both the sequence and bounded embedding-row capability identities.
// This is a source-level capability gate, not hardware parity evidence. Other
// forward paths keep the historical single Prefill call because they do not share
// the required append contract.
func (p *InKernelPlanner) prefillDivergentSuffix(ctx context.Context, s inKernelPrefillSession, ids []int, measurementOpt ...*nativeInferenceMeasurement) ([]float32, error) {
	var measurement *nativeInferenceMeasurement
	if len(measurementOpt) > 0 {
		measurement = measurementOpt[0]
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if p.qwenQ4KPrefillChunkExplicit && p.qwenQ4KPrefillChunkTarget() {
		if _, err := p.deviceBoundedQwenQ4KPrefillChunkTokens(); err != nil {
			return nil, err
		}
	}
	chunkTokens := p.effectiveQwenQ4KPrefillChunkTokens()
	if !p.qwenQ4KPrefillChunkTarget() || len(ids) <= chunkTokens {
		chunkStart := time.Now()
		logits := s.Prefill(ids)
		enginestep.Default.ObservePrefillChunkContext(ctx, len(ids), time.Since(chunkStart))
		recordQwen35SequencePrefillRoute(measurement, s, len(ids))
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return logits, nil
	}
	for len(ids) > chunkTokens {
		chunk := ids[:chunkTokens]
		chunkStart := time.Now()
		s.PrefillNoLogits(chunk)
		enginestep.Default.ObservePrefillChunkContext(ctx, len(chunk), time.Since(chunkStart))
		recordQwen35SequencePrefillRoute(measurement, s, len(chunk))
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ids = ids[chunkTokens:]
	}
	chunkStart := time.Now()
	logits := s.Prefill(ids)
	enginestep.Default.ObservePrefillChunkContext(ctx, len(ids), time.Since(chunkStart))
	recordQwen35SequencePrefillRoute(measurement, s, len(ids))
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return logits, nil
}

func (p *InKernelPlanner) qwenQ4KPrefillChunkTarget() bool {
	if p == nil || p.m == nil || !p.q4k || !p.m.Cfg.IsQwen35Hybrid() {
		return false
	}
	if p.backend == nil {
		return true
	}
	sequence, ok := p.backend.(model.Qwen35SequencePrefillBackend)
	if !ok || sequence.Qwen35SequencePrefillPath() != compute.Qwen35SequencePrefillPath {
		return false
	}
	rows, ok := p.backend.(compute.Qwen35SequenceEmbeddingRowsBackend)
	return ok && rows.Qwen35SequenceEmbeddingRowsPath() == compute.Qwen35SequenceEmbeddingRowsPath
}

// qwenPrefillPanelBufferCap is implemented by backends that publish the
// single-resource buffer ceiling used to bound an explicitly configured prompt
// panel. It mirrors the model-side weightBufferCapBackend seam.
type qwenPrefillPanelBufferCap interface {
	MaxWeightBufferBytes() int64
}

type qwenQ4KPrefillPanelCapacityError struct {
	BufferBytes int64
	RowBytes    int64
	Reason      string
}

func (e *qwenQ4KPrefillPanelCapacityError) Error() string {
	if e.Reason != "" {
		return fmt.Sprintf("native Qwen Q4_K prefill panel: device single-buffer cap %d bytes cannot be applied: %s", e.BufferBytes, e.Reason)
	}
	return fmt.Sprintf("native Qwen Q4_K prefill panel: device single-buffer cap %d bytes cannot fit one %d-byte token row", e.BufferBytes, e.RowBytes)
}

// qwenPrefillMaxTokenPanelWidth returns the widest float32 row allocated per
// token by an individual Qwen 3.5 sequence panel. This is a single-allocation
// bound; persistent KV and request-wide capacity remain backend admission work.
func qwenPrefillMaxTokenPanelWidth(cfg model.Config) (int64, bool) {
	const maxInt64 = int64(^uint64(0) >> 1)
	mul := func(a, b int) (int64, bool) {
		if a <= 0 || b <= 0 || int64(a) > maxInt64/int64(b) {
			return 0, false
		}
		return int64(a) * int64(b), true
	}
	qWidth, ok := mul(cfg.NumHeads, cfg.HeadDim)
	if !ok {
		return 0, false
	}
	kvWidth, ok := mul(cfg.NumKVHeads, cfg.HeadDim)
	if !ok {
		return 0, false
	}
	keyWidth, ok := mul(cfg.LinearNumKeyHeads, cfg.LinearKeyHeadDim)
	if !ok {
		return 0, false
	}
	valueWidth, ok := mul(cfg.LinearNumValueHeads, cfg.LinearValueHeadDim)
	if !ok || qWidth > maxInt64/2 || keyWidth > (maxInt64-valueWidth)/2 {
		return 0, false
	}
	widths := [...]int64{
		int64(cfg.HiddenSize), int64(cfg.IntermediateSize),
		2 * qWidth, qWidth, kvWidth, 2*keyWidth + valueWidth, valueWidth,
	}
	var widest int64
	for _, width := range widths {
		if width <= 0 {
			return 0, false
		}
		if width > widest {
			widest = width
		}
	}
	return widest, true
}

// deviceBoundedQwenQ4KPrefillChunkTokens derives the largest explicitly
// configured prompt panel that fits the backend's single-resource ceiling.
// Unknown caps preserve the requested experiment setting. A known cap that
// cannot fit one token is refused instead of silently increasing the chunk.
func (p *InKernelPlanner) deviceBoundedQwenQ4KPrefillChunkTokens() (int, *qwenQ4KPrefillPanelCapacityError) {
	if p == nil || p.backend == nil || p.m == nil || !p.qwenQ4KPrefillChunkTarget() {
		return 0, nil
	}
	capper, ok := p.backend.(qwenPrefillPanelBufferCap)
	if !ok || capper.MaxWeightBufferBytes() <= 0 {
		return 0, nil
	}
	widest, ok := qwenPrefillMaxTokenPanelWidth(p.m.Cfg)
	const maxInt64 = int64(^uint64(0) >> 1)
	bufferBytes := capper.MaxWeightBufferBytes()
	if !ok {
		return 0, &qwenQ4KPrefillPanelCapacityError{
			BufferBytes: bufferBytes,
			Reason:      "model panel dimensions are invalid or overflow int64",
		}
	}
	if widest > maxInt64/4 {
		return 0, &qwenQ4KPrefillPanelCapacityError{
			BufferBytes: bufferBytes,
			Reason:      "widest float32 token row overflows int64 bytes",
		}
	}
	rowBytes := widest * 4
	maxTokens := bufferBytes / rowBytes
	if maxTokens < 1 {
		return 0, &qwenQ4KPrefillPanelCapacityError{BufferBytes: bufferBytes, RowBytes: rowBytes}
	}
	if maxTokens > 8192 {
		maxTokens = 8192
	}
	return int(maxTokens), nil
}

func (p *InKernelPlanner) effectiveQwenQ4KPrefillChunkTokens() int {
	configured := inKernelQwenQ4KPrefillChunkTokens
	if p != nil && p.qwenQ4KPrefillChunkTokens > 0 {
		configured = p.qwenQ4KPrefillChunkTokens
	}
	if p == nil || !p.qwenQ4KPrefillChunkExplicit {
		return configured
	}
	bounded, err := p.deviceBoundedQwenQ4KPrefillChunkTokens()
	if err != nil {
		return 0
	}
	if bounded > 0 && configured > bounded {
		return bounded
	}
	return configured
}

func (p *InKernelPlanner) nativeInferencePrefillChunkTokens() int {
	if !p.qwenQ4KPrefillChunkTarget() {
		return 0
	}
	return p.effectiveQwenQ4KPrefillChunkTokens()
}
