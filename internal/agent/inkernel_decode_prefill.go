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

// Context-aware Sessions preserve the selected operation's boundary; this does
// not rechunk generic forwards or cancel another participant's shared runCtx.
type inKernelContextPrefillSession interface {
	PrefillContext(context.Context, []int) ([]float32, error)
	PrefillNoLogitsContext(context.Context, []int) error
}

func prefillSessionContext(ctx context.Context, s inKernelPrefillSession, ids []int, wantLogits bool) ([]float32, error) {
	if contextual, ok := s.(inKernelContextPrefillSession); ok {
		if wantLogits {
			return contextual.PrefillContext(ctx, ids)
		}
		return nil, contextual.PrefillNoLogitsContext(ctx, ids)
	}
	if wantLogits {
		return s.Prefill(ids), nil
	}
	s.PrefillNoLogits(ids)
	return nil, nil
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
		logits, err := prefillSessionContext(ctx, s, ids, true)
		if err != nil {
			return nil, err
		}
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
		if _, err := prefillSessionContext(ctx, s, chunk, false); err != nil {
			return nil, err
		}
		enginestep.Default.ObservePrefillChunkContext(ctx, len(chunk), time.Since(chunkStart))
		recordQwen35SequencePrefillRoute(measurement, s, len(chunk))
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ids = ids[chunkTokens:]
	}
	chunkStart := time.Now()
	logits, err := prefillSessionContext(ctx, s, ids, true)
	if err != nil {
		return nil, err
	}
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
	if p == nil {
		return configured
	}
	if p.qwenQ4KPrefillChunkExplicit {
		bounded, err := p.deviceBoundedQwenQ4KPrefillChunkTokens()
		if err != nil {
			return 0
		}
		if bounded > 0 && configured > bounded {
			configured = bounded
		}
	}
	return p.headroomBoundedQwenQ4KPrefillChunkTokens(configured)
}

// Rows bounds for the headroom-derived Qwen Q4_K prompt panel (fak#14097). Below
// qwenQ4KSplitScratchRows the Q4_K GEMM takes the scratch-free panel kernel, so a
// derived chunk may drop under it to escape the P-independent fp16 weight split.
const (
	qwenQ4KPrefillHeadroomMinRows = 8
	qwenQ4KSplitScratchRows       = 128
)

// headroomBoundedQwenQ4KPrefillChunkTokens lowers configured to the largest row
// count in [qwenQ4KPrefillHeadroomMinRows, configured] whose priced sequence
// prefill transient fits the live device headroom less the request admission
// margin. The headroom is the backend's DeviceMemoryInfo reading, the same seam
// request admission sizes against. Unknown headroom keeps configured.
func (p *InKernelPlanner) headroomBoundedQwenQ4KPrefillChunkTokens(configured int) int {
	if p == nil || p.backend == nil || p.m == nil || configured <= qwenQ4KPrefillHeadroomMinRows || !p.qwenQ4KPrefillChunkTarget() {
		return configured
	}
	_, free, known := compute.DeviceMemoryInfo(p.backend)
	if !known || free < 0 {
		return configured
	}
	arm, err := compute.CUDAQ4KPrefillArmInUse()
	if err != nil {
		arm = compute.CUDAQ4KPrefillCompensated
	}
	return qwenSequencePrefillChunkForBudget(p.m.Cfg, configured, ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom), arm)
}

// qwenSequencePrefillChunkForBudget returns the largest rows in
// [qwenQ4KPrefillHeadroomMinRows, configured] with peak(rows) <= budget. The peak
// is not monotonic (the split scratch appears at qwenQ4KSplitScratchRows), so the
// search walks down from configured. When nothing fits, the smallest panel is the
// least transient the planner can choose; admission still owns the refusal.
func qwenSequencePrefillChunkForBudget(cfg model.Config, configured int, budget int64, arm compute.CUDAQ4KPrefillArm) int {
	if configured <= qwenQ4KPrefillHeadroomMinRows {
		return configured
	}
	for rows := configured; rows >= qwenQ4KPrefillHeadroomMinRows; rows-- {
		peak, ok := qwenSequencePrefillPeakBytes(cfg, rows, arm)
		if !ok {
			return configured
		}
		if peak <= budget {
			return rows
		}
	}
	return qwenQ4KPrefillHeadroomMinRows
}

// qwenSequencePrefillPeakBytes prices the per-layer transient device peak of one
// Qwen 3.5 hybrid sequence-prefill panel of rows tokens: the widest live f32
// activation set of any layer stage plus the largest Q4_K GEMM scratch for arm
// (cuda_qwen35_sequence.go, quant_matmul.cuh fcuda_q4k_matmul_f32, q4k_mmq.cuh).
// Every projection is priced as Q4_K, the widest-scratch resident dtype on this
// path. Rows in [qwenQ4KPrefillHeadroomMinRows, qwenQ4KSplitScratchRows) are priced
// with MMQ activation scratch so the bound stays safe if small panels route to MMQ.
// ok is false for invalid or overflowing dimensions.
func qwenSequencePrefillPeakBytes(cfg model.Config, rows int, arm compute.CUDAQ4KPrefillArm) (int64, bool) {
	if rows <= 0 || cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 {
		return 0, false
	}
	if _, ok := qwenPrefillMaxTokenPanelWidth(cfg); !ok {
		return 0, false
	}
	hidden, inter := int64(cfg.HiddenSize), int64(cfg.IntermediateSize)
	qWidth := int64(cfg.NumHeads) * int64(cfg.HeadDim)
	kvWidth := int64(cfg.NumKVHeads) * int64(cfg.HeadDim)
	keyDim := int64(cfg.LinearNumKeyHeads) * int64(cfg.LinearKeyHeadDim)
	valueDim := int64(cfg.LinearNumValueHeads) * int64(cfg.LinearValueHeadDim)
	convDim := 2*keyDim + valueDim

	// Live f32 values per token at each stage's widest point.
	ffn := 3*hidden + 3*inter                                                              // x, post-norm, delta + gate, up, swiglu
	attention := 3*hidden + 4*qWidth + 2*kvWidth                                           // x, normed, projected + qg, q, gate, rope'd q + k, v
	gdn := 3*hidden + 2*convDim + 2*valueDim + 2*keyDim + 2*int64(cfg.LinearNumValueHeads) // x, normed, output + gdn scratch set
	width := max(ffn, attention, gdn)
	r := int64(rows)
	activation := hostDemandMul(hostDemandMul(r, width), 4)

	type shape struct{ out, in int64 }
	shapes := []shape{{inter, hidden}, {hidden, inter}, {2 * qWidth, hidden}, {kvWidth, hidden}, {hidden, qWidth}}
	if convDim > 0 && valueDim > 0 {
		shapes = append(shapes, shape{convDim, hidden}, shape{valueDim, hidden}, shape{hidden, valueDim})
	}
	var scratch int64
	for _, s := range shapes {
		if s.out <= 0 || s.in <= 0 {
			continue
		}
		scratch = max(scratch, qwenQ4KGEMMScratchBytes(s.out, s.in, r, arm))
	}
	return hostDemandAdd(activation, scratch), true
}

// qwenQ4KGEMMScratchBytes mirrors compute.cudaQ4KScratch for one Q4_K GEMM call.
func qwenQ4KGEMMScratchBytes(out, in, rows int64, arm compute.CUDAQ4KPrefillArm) int64 {
	const q81Block, q81Values = 144, 128 // block_q8_1_mmq
	mmq := func() int64 {
		padded := (in + 511) / 512 * 512
		return hostDemandAdd(hostDemandMul(hostDemandMul(rows, padded)/q81Values, q81Block), 128*q81Block)
	}
	if rows < qwenQ4KSplitScratchRows {
		if rows < qwenQ4KPrefillHeadroomMinRows {
			return 0
		}
		return mmq()
	}
	switch arm {
	case compute.CUDAQ4KPrefillMMQ:
		return mmq()
	case compute.CUDAQ4KPrefillFP8LT:
		padded := (rows + 15) / 16 * 16
		bytes := hostDemandAdd(hostDemandAdd(hostDemandMul(out, in), hostDemandMul(padded, in)), 16)
		if padded != rows {
			bytes = hostDemandAdd(bytes, hostDemandMul(hostDemandMul(padded, out), 4))
		}
		return bytes
	default:
		// Compensated: fp16 hi/lo halves of the dequantized weight and of the activations.
		return hostDemandAdd(hostDemandMul(hostDemandMul(out, in), 4), hostDemandMul(hostDemandMul(rows, in), 4))
	}
}

func (p *InKernelPlanner) nativeInferencePrefillChunkTokens() int {
	if !p.qwenQ4KPrefillChunkTarget() {
		return 0
	}
	return p.effectiveQwenQ4KPrefillChunkTokens()
}
