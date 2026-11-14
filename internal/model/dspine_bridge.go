package model

// dspine_bridge.go — DSpine adjacent-causal-injection drafter lineage
// (anthony-chaudhary/fak#14441), the third drafter lineage for the spec-decode
// path alongside chain-style and MTP drafters and the DFlash 2 reference shape.
//
// Concept-level port of the DSpine mechanisms (Zhang et al., "Draft in Parallel,
// Condition Through Depth: Adjacent Causal Injection for Speculative Decoding",
// arXiv:2609.36173; Apache-2.0) re-expressed against fak's own drafter shapes.
// No upstream bytes are vendored; attribution/NOTICE preserved.
//
// Reference math (fp32; every reduction is serial and index-ordered so the
// drafter and its witness oracle stay bit-identical):
//
// Block: B rows, row 0 = anchor token A, rows 1..B-1 = placeholder M.
// E[t] in R^H is embedding row t; P in R^{R x H} projects a layer output into
// the fixed transfer space made of whitened target-head rows; G in R^{R x H}
// produces the receiver gate; U in R^{H x R} lifts the gated message back to
// the hidden width; D_l in R^{H x H} is layer l's dense transform; Head in
// R^{V x H} is the shared target head.
//
//   rms(v) = v / sqrt(mean(v^2) + eps)
//
// forward(A, B):
//   x[i] = E[A] (i == 0); E[M] (i > 0)
//   for l in 0..L-1:
//     n[i] = rms(x[i])
//     if l > 0:                          // bridge write replaces the input norm
//       i == 0:  (the anchor row is never written)
//       i >= 1:  pred(i) = P . E[A]          (i == 1: the anchor sends its exact
//                                            token code)
//                   P . h[l-1][i-1]        (i >= 2: the predecessor's predicted
//                                            feature, the layer output projected
//                                            into the transfer space)
//                gamma(i) = sigmoid(G . h[l-1][i])
//                n[i] += U . (gamma(i) * pred(i))   (fused next-input-norm write)
//     mix(l, i) = sum of n[j] over j allowed by layer l's direction:
//                  causal j <= i; bidirectional all j; sliding max(0, i-W+1) <= j <= i
//     h[l][i] = x[i] + D_l . n[i] + mix(l, i)
//   f[i] = rms(h[L-1][i]); logits[i] = Head . f[i]
//
// Five mechanisms land together so the invariant is testable as a whole:
//  1. Per-layer bridge write — gated predecessor message plus fused
//     next-input-norm; the anchor row is never written.
//  2. Cap-tracking draft width — a block-causal head drafts anchor + cap rows;
//     the truncated block's rows equal the full block's (exact-truncation
//     invariant), and a non-block-causal head refuses truncation.
//  3. Last-write refinement — the final bridge write redone with each candidate
//     sender's exact code, scored against the target-head rows, into the shared
//     [slots, K predecessor, K candidate] lattice the existing greedy/sampled
//     walk consumes unchanged.
//  4. Fail-closed checkpoint contract — int64 version-stamp tensors must equal
//     1; unsupported variants refuse with structured reasons instead of loading
//     or mis-scheduling (the RefuseDeepSeekV41SpecDecodeUnqualified shape).
//  5. Per-layer attention-direction semantics — causal vs bidirectional vs
//     sliding-window mask selection per layer; shipped heads declare both
//     directions across their layers.
//
// Quarantine fallback: the whole lineage sits behind DSpineBridgeConfig.Enabled
// defaulting to false; an absent bridge is today's plain DFlash behavior,
// byte-identical. The bridge projections stay unquantized (semantics-sensitive;
// kept unquantized upstream too). No Metal kernels, no GPU-side graph
// construction, no serving/gateway wiring in this leaf: the receipt is a model
// unit test on synthetic fp32 weights.

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// DSpineAttentionDirection names the attention direction a drafter layer applies
// across its block rows.
type DSpineAttentionDirection int

const (
	// DSpineCausal mixes row i with rows j <= i only. Truncation-safe: row i's
	// result depends on no row beyond i.
	DSpineCausal DSpineAttentionDirection = iota
	// DSpineBidirectional mixes row i with every block row. Not truncation-safe:
	// dropping tail rows changes every row's mix.
	DSpineBidirectional
	// DSpineSlidingWindow mixes row i with the trailing Window rows ending at i.
	// Truncation-safe like causal.
	DSpineSlidingWindow
)

// String renders the direction for receipts.
func (d DSpineAttentionDirection) String() string {
	switch d {
	case DSpineCausal:
		return "causal"
	case DSpineBidirectional:
		return "bidirectional"
	case DSpineSlidingWindow:
		return "sliding-window"
	default:
		return "unknown"
	}
}

// truncationSafe reports whether the direction keeps row i's result independent
// of any row beyond i, which is what makes cap-truncated blocks exact.
func (d DSpineAttentionDirection) truncationSafe() bool {
	return d == DSpineCausal || d == DSpineSlidingWindow
}

// DSpineLayerSpec declares one drafter layer's attention direction. Shipped
// heads declare both directions across their layers; the declared set decides
// whether cap-truncated drafting is admissible at all.
type DSpineLayerSpec struct {
	Direction DSpineAttentionDirection
	// Window is the sliding-window width in block rows; required in
	// [2, BlockSize] when Direction is DSpineSlidingWindow and zero otherwise.
	Window int
}

var (
	// ErrDSpineDisabled is the inert refusal for a zero-value or disabled config:
	// the bridge never engages, so plain DFlash behavior applies byte-identically.
	ErrDSpineDisabled = errors.New("model: dspine bridge is disabled; plain DFlash behavior applies")

	// ErrDSpineConfigInvalid is the structured refusal for a malformed runtime
	// config (layer-set, block-size, top-k, rank, window bounds) or weight shape.
	ErrDSpineConfigInvalid = errors.New("model: dspine bridge config is invalid")

	// Contract refusals: a closed vocabulary, one sentinel per unsupported
	// checkpoint variant.
	ErrDSpineContractVersion            = errors.New("model: dspine checkpoint version stamp must equal 1")
	ErrDSpineContractTerminalEnabled    = errors.New("model: dspine terminal-enabled bridge is not supported")
	ErrDSpineContractWritesDependent    = errors.New("model: dspine per-layer writes are not independent")
	ErrDSpineContractDecodeNonSerial    = errors.New("model: dspine decode mode is not serial")
	ErrDSpineContractSelectorCoLocated  = errors.New("model: dspine selector and convolution are co-located")
	ErrDSpineContractFinalWrite         = errors.New("model: dspine final-layer write is missing or out of range")
	ErrDSpineContractRefinementWidths   = errors.New("model: dspine refinement widths disagree")
	ErrDSpineContractTopK               = errors.New("model: dspine top-k is outside [2,64]")
	ErrDSpineContractTerminalProjection = errors.New("model: dspine terminal projection is non-zero despite a disabled terminal")

	// ErrDSpineTruncationRefused is the fail-closed refusal to draft a truncated
	// block from a head whose layers are not all block-causal.
	ErrDSpineTruncationRefused = errors.New("model: dspine block truncation is refused for a non-block-causal head")

	// ErrDSpineWalkDeadEnd is the fail-closed refusal when the lattice walk loses
	// the decided sender inside a slot's predecessor set.
	ErrDSpineWalkDeadEnd = errors.New("model: dspine lattice walk reached a dead end")
)

// DSpineCheckpointContract is the version-stamped, fail-closed checkpoint
// contract the DSpine load path enforces before any drafter arithmetic. It
// mirrors the RefuseDeepSeekV41SpecDecodeUnqualified shape: an unsupported
// checkpoint variant refuses with a structured reason instead of loading or
// mis-scheduling.
type DSpineCheckpointContract struct {
	// Version is the int64 version-stamp tensor value; it must equal 1.
	Version int64
	// TerminalEnabled declares the terminal projection enabled; the supported
	// bridge is terminal-disabled, so any terminal-enabled checkpoint refuses.
	TerminalEnabled bool
	// WritesIndependent declares each layer's bridge write self-contained
	// (last-write-wins); dependent writes refuse.
	WritesIndependent bool
	// SerialDecode declares the serial decode mode; parallel decode refuses.
	SerialDecode bool
	// SelectorSeparated declares the selector kept apart from the convolution;
	// co-location refuses.
	SelectorSeparated bool
	// FinalLayerWrite is the layer index carrying the final write; it must name
	// the last layer exactly. Missing (< 0) or out-of-range values refuse.
	FinalLayerWrite int
	// RefinementWidths is the per-slot refinement width declared by the
	// checkpoint; it must be non-empty and agree across slots.
	RefinementWidths []int
	// TopK is the checkpoint-declared refinement top-k; it must lie in [2, 64].
	TopK int
	// TerminalProjection holds the terminal projection weights; they must be
	// all-zero whenever TerminalEnabled is false.
	TerminalProjection []float32
}

// Validate applies the fail-closed checkpoint contract gate against numLayers
// drafter layers. An enabled bridge must pass every check or refuse with the
// first failing structured reason.
func (c DSpineCheckpointContract) Validate(numLayers int) error {
	if c.Version != 1 {
		return fmt.Errorf("%w: got %d", ErrDSpineContractVersion, c.Version)
	}
	if c.TerminalEnabled {
		return ErrDSpineContractTerminalEnabled
	}
	if !c.WritesIndependent {
		return ErrDSpineContractWritesDependent
	}
	if !c.SerialDecode {
		return ErrDSpineContractDecodeNonSerial
	}
	if !c.SelectorSeparated {
		return ErrDSpineContractSelectorCoLocated
	}
	if c.FinalLayerWrite < 0 || c.FinalLayerWrite >= numLayers {
		return fmt.Errorf("%w: final write at layer %d of %d", ErrDSpineContractFinalWrite, c.FinalLayerWrite, numLayers)
	}
	if len(c.RefinementWidths) == 0 {
		return fmt.Errorf("%w: no refinement widths declared", ErrDSpineContractRefinementWidths)
	}
	for _, w := range c.RefinementWidths[1:] {
		if w != c.RefinementWidths[0] {
			return fmt.Errorf("%w: slot widths %v disagree", ErrDSpineContractRefinementWidths, c.RefinementWidths)
		}
	}
	if c.TopK < 2 || c.TopK > 64 {
		return fmt.Errorf("%w: top-k=%d", ErrDSpineContractTopK, c.TopK)
	}
	if !c.TerminalEnabled {
		for _, v := range c.TerminalProjection {
			if v != 0 {
				return ErrDSpineContractTerminalProjection
			}
		}
	}
	return nil
}

// DSpineBridgeConfig is the config-gated admission for the DSpine drafter
// lineage. The zero value (Enabled=false) never engages: absent bridge =
// today's plain DFlash behavior, byte-identical. Only an Enabled config whose
// checkpoint contract passes and whose runtime bounds hold may construct a
// drafter.
type DSpineBridgeConfig struct {
	// Enabled requests the DSpine bridge; false keeps plain DFlash behavior.
	Enabled bool
	// Checkpoint is the version-stamped checkpoint contract gate.
	Checkpoint DSpineCheckpointContract
	// Layers declares the per-layer attention direction; at least two layers,
	// because the bridge write lands between layers.
	Layers []DSpineLayerSpec
	// BlockSize is the trained block width in rows (16 upstream): the anchor row
	// plus BlockSize-1 placeholder rows.
	BlockSize int
	// TopK is the runtime refinement top-k; it must match the checkpoint's
	// agreed refinement width and lie in [2, 64].
	TopK int
	// Rank is the transfer-space width.
	Rank int
	// MaskToken is the placeholder token embedding for non-anchor block rows.
	MaskToken int
	// Eps is the RMS normalization epsilon.
	Eps float64
}

// Validate applies the runtime config gate. A disabled config reports
// ErrDSpineDisabled (inert by design); an enabled one must hold every bound.
func (c DSpineBridgeConfig) Validate() error {
	if !c.Enabled {
		return ErrDSpineDisabled
	}
	if len(c.Layers) < 2 {
		return fmt.Errorf("%w: %d layers declared, need at least 2", ErrDSpineConfigInvalid, len(c.Layers))
	}
	if c.BlockSize < 2 {
		return fmt.Errorf("%w: block size %d", ErrDSpineConfigInvalid, c.BlockSize)
	}
	if c.Rank < 1 {
		return fmt.Errorf("%w: rank %d", ErrDSpineConfigInvalid, c.Rank)
	}
	if c.TopK < 2 || c.TopK > 64 {
		return fmt.Errorf("%w: top-k %d outside [2,64]", ErrDSpineConfigInvalid, c.TopK)
	}
	if c.Eps <= 0 || math.IsNaN(c.Eps) || math.IsInf(c.Eps, 0) {
		return fmt.Errorf("%w: eps %v", ErrDSpineConfigInvalid, c.Eps)
	}
	for i, l := range c.Layers {
		switch l.Direction {
		case DSpineCausal, DSpineBidirectional:
			if l.Window != 0 {
				return fmt.Errorf("%w: layer %d window %d on a %s layer", ErrDSpineConfigInvalid, i, l.Window, l.Direction)
			}
		case DSpineSlidingWindow:
			if l.Window < 2 || l.Window > c.BlockSize {
				return fmt.Errorf("%w: layer %d window %d outside [2,%d]", ErrDSpineConfigInvalid, i, l.Window, c.BlockSize)
			}
		default:
			return fmt.Errorf("%w: layer %d direction %v", ErrDSpineConfigInvalid, i, l.Direction)
		}
	}
	return nil
}

// DSpineWeights holds the fp32 bridge parameters. Tests use synthetic weights
// only — no checkpoint download — and the bridge projections stay unquantized.
type DSpineWeights struct {
	Hidden int
	Rank   int
	Vocab  int
	// Dense is the per-layer dense transform, [layers][Hidden*Hidden] row-major.
	Dense [][]float32
	// Transfer projects a layer output into the fixed transfer space made of
	// whitened target-head rows: P, [Rank*Hidden] row-major.
	Transfer []float32
	// Gate produces the receiver gate over the transfer dimension: G,
	// [Rank*Hidden] row-major.
	Gate []float32
	// Lift maps the gated transfer message back to the hidden width: U,
	// [Hidden*Rank] row-major.
	Lift []float32
	// Embed is the shared embedding matrix, [Vocab*Hidden] row-major.
	Embed []float32
	// Head is the target head matrix, [Vocab*Hidden] row-major; its whitened
	// rows span the transfer space.
	Head []float32
}

// DSpineBridgeDrafter is the adjacent-causal-injection drafter: a multi-layer
// draft head whose per-layer conditioning passes a gated message along the
// block (the bridge write), whose draft width tracks the verify cap, and whose
// final write is refined into the shared [slots, K predecessor, K candidate]
// lattice the existing greedy/sampled walk consumes unchanged. DraftProposer
// consumers select this lineage additively; the verify/accept contract
// (DraftProposer/DraftVerifier) stays intact.
type DSpineBridgeDrafter struct {
	cfg DSpineBridgeConfig
	w   *DSpineWeights
}

// NewDSpineBridge admits a drafter through the config gate and the fail-closed
// checkpoint contract, then verifies the weight shapes. Every unsupported
// variant refuses here; nothing loads or mis-schedules past this seam.
func NewDSpineBridge(cfg DSpineBridgeConfig, w *DSpineWeights) (*DSpineBridgeDrafter, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if err := cfg.Checkpoint.Validate(len(cfg.Layers)); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, fmt.Errorf("%w: nil weights", ErrDSpineConfigInvalid)
	}
	if w.Hidden < 1 || w.Rank != cfg.Rank || w.Vocab < 1 {
		return nil, fmt.Errorf("%w: weight geometry hidden=%d rank=%d vocab=%d", ErrDSpineConfigInvalid, w.Hidden, w.Rank, w.Vocab)
	}
	if len(w.Dense) != len(cfg.Layers) {
		return nil, fmt.Errorf("%w: %d dense matrices for %d layers", ErrDSpineConfigInvalid, len(w.Dense), len(cfg.Layers))
	}
	for _, dense := range w.Dense {
		if len(dense) != w.Hidden*w.Hidden {
			return nil, fmt.Errorf("%w: dense width %d for hidden %d", ErrDSpineConfigInvalid, len(dense), w.Hidden)
		}
	}
	if len(w.Transfer) != cfg.Rank*w.Hidden || len(w.Gate) != cfg.Rank*w.Hidden || len(w.Lift) != w.Hidden*cfg.Rank || len(w.Embed) != w.Vocab*w.Hidden || len(w.Head) != w.Vocab*w.Hidden {
		return nil, fmt.Errorf("%w: bridge/embedding/head width", ErrDSpineConfigInvalid)
	}
	if cfg.MaskToken < 0 || cfg.MaskToken >= w.Vocab {
		return nil, fmt.Errorf("%w: mask token %d outside vocab %d", ErrDSpineConfigInvalid, cfg.MaskToken, w.Vocab)
	}
	if cfg.Checkpoint.RefinementWidths[0] != cfg.TopK {
		return nil, fmt.Errorf("%w: checkpoint width %d vs runtime top-k %d", ErrDSpineContractRefinementWidths, cfg.Checkpoint.RefinementWidths[0], cfg.TopK)
	}
	return &DSpineBridgeDrafter{cfg: cfg, w: w}, nil
}

// DSPineLayerTrace captures one layer's input norms and outputs so the
// reference-equivalence witness can compare them against an independent
// recomputation.
type DSPineLayerTrace struct {
	// Norm is the input norm after the bridge write (the replaced norm),
	// [rows*Hidden] row-major.
	Norm []float32
	// Out is the layer output features, [rows*Hidden] row-major.
	Out []float32
}

// DSPineForwardTrace is the full forward-pass decomposition.
type DSPineForwardTrace struct {
	Anchor int
	Rows   int
	// Layers is the per-layer trace, index order.
	Layers []DSPineLayerTrace
	// Features is the final-normed feature per row, [rows*Hidden] row-major.
	Features []float32
	// Logits is the target-head logit row per row, [rows*Vocab].
	Logits [][]float32
}

// Forward runs the drafter backbone over rows block rows (row 0 = anchor) and
// returns the per-layer trace plus final features and target-head logits. rows
// must equal BlockSize unless every layer is truncation-safe, in which case
// rows may be any width in [2, BlockSize] (cap-tracking width). A truncated
// request for a non-block-causal head refuses with ErrDSpineTruncationRefused.
func (d *DSpineBridgeDrafter) Forward(ctx context.Context, anchor, rows int) (DSPineForwardTrace, error) {
	if err := ctx.Err(); err != nil {
		return DSPineForwardTrace{}, err
	}
	if d == nil || d.w == nil {
		return DSPineForwardTrace{}, fmt.Errorf("%w: drafter is nil", ErrDSpineConfigInvalid)
	}
	h, r, b := d.w.Hidden, d.w.Rank, d.cfg.BlockSize
	if anchor < 0 || anchor >= d.w.Vocab {
		return DSPineForwardTrace{}, fmt.Errorf("%w: anchor %d outside vocab %d", ErrDSpineConfigInvalid, anchor, d.w.Vocab)
	}
	if rows < 2 || rows > b {
		return DSPineForwardTrace{}, fmt.Errorf("%w: rows %d outside [2,%d]", ErrDSpineConfigInvalid, rows, b)
	}
	if rows < b && !d.truncationSafe() {
		return DSPineForwardTrace{}, fmt.Errorf("%w: %d rows < trained block %d", ErrDSpineTruncationRefused, rows, b)
	}
	trace := DSPineForwardTrace{Anchor: anchor, Rows: rows}
	x := make([]float32, rows*h)
	anchorEmb := d.w.Embed[anchor*h : (anchor+1)*h]
	maskEmb := d.w.Embed[d.cfg.MaskToken*h : (d.cfg.MaskToken+1)*h]
	copy(x[:h], anchorEmb)
	for i := 1; i < rows; i++ {
		copy(x[i*h:], maskEmb)
	}
	hPrev := []float32(nil)
	for l := range d.cfg.Layers {
		var msgs [][]float32
		if l > 0 {
			msgs = d.dspineBridgeWrite(hPrev, anchorEmb, rows, h, r)
		}
		n := make([]float32, rows*h)
		for i := 0; i < rows; i++ {
			norm := dspineRMS(x[i*h:(i+1)*h], float32(d.cfg.Eps))
			if i >= 1 && l > 0 {
				for k := 0; k < h; k++ {
					norm[k] += msgs[i][k]
				}
			}
			copy(n[i*h:], norm)
		}
		out := make([]float32, rows*h)
		dense := d.w.Dense[l]
		for i := 0; i < rows; i++ {
			proj := dspineRow(dense, n[i*h:(i+1)*h], h, h)
			mix := d.dspineMix(l, i, n, h)
			for k := 0; k < h; k++ {
				out[i*h+k] = x[i*h+k] + proj[k] + mix[k]
			}
		}
		trace.Layers = append(trace.Layers, DSPineLayerTrace{Norm: n, Out: out})
		hPrev = out
	}
	feat := make([]float32, rows*h)
	for i := 0; i < rows; i++ {
		copy(feat[i*h:], dspineRMS(hPrev[i*h:(i+1)*h], float32(d.cfg.Eps)))
	}
	logits := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		logits[i] = dspineRow(d.w.Head, feat[i*h:(i+1)*h], d.w.Vocab, h)
	}
	trace.Features = feat
	trace.Logits = logits
	return trace, nil
}

// truncationSafe reports whether every layer's mask keeps each row independent
// of the rows beyond it, which is what makes a cap-truncated block exact.
func (d *DSpineBridgeDrafter) truncationSafe() bool {
	for _, l := range d.cfg.Layers {
		if !l.Direction.truncationSafe() {
			return false
		}
	}
	return true
}

// dspineBridgeWrite computes the layer-to-layer bridge write for one block.
// For row i >= 1 the predecessor transfer feature is the anchor's exact token
// code P·E[anchor] when i == 1 and the predecessor row's predicted layer output
// P·hPrev[i-1] when i >= 2; the receiver gate is sigmoid(G·hPrev[i]). The
// returned slice holds the fused next-input-norm contribution
// msg_i = U·(γ_i ⊙ pred(i)); msgs[0] stays nil because the anchor row is never
// written.
func (d *DSpineBridgeDrafter) dspineBridgeWrite(hPrev []float32, anchorEmb []float32, rows, h, r int) [][]float32 {
	msgs := make([][]float32, rows)
	for i := 1; i < rows; i++ {
		var src []float32
		if i == 1 {
			src = anchorEmb
		} else {
			src = hPrev[(i-1)*h : i*h]
		}
		pred := dspineRow(d.w.Transfer, src, r, h)
		gamma := dspineRow(d.w.Gate, hPrev[i*h:(i+1)*h], r, h)
		gated := make([]float32, r)
		for k := 0; k < r; k++ {
			gated[k] = dspineSigmoid(gamma[k]) * pred[k]
		}
		msgs[i] = dspineRow(d.w.Lift, gated, h, r)
	}
	return msgs
}

// dspineMix accumulates the allowed predecessor rows for row i under layer l's
// direction: causal j <= i; bidirectional all j; sliding max(0, i-W+1) <= j <= i.
func (d *DSpineBridgeDrafter) dspineMix(l, i int, n []float32, h int) []float32 {
	spec := d.cfg.Layers[l]
	start := 0
	if spec.Direction == DSpineSlidingWindow {
		start = i - spec.Window + 1
		if start < 0 {
			start = 0
		}
	}
	out := make([]float32, h)
	for j := start; j <= i; j++ {
		for k := 0; k < h; k++ {
			out[k] += n[j*h+k]
		}
	}
	return out
}

// dspineRMS normalizes one row: v / sqrt(mean(v^2) + eps). The serial
// index-ordered reduction is part of the reference contract: the drafter and
// its witness oracle share one deterministic accumulation order.
func dspineRMS(v []float32, eps float32) []float32 {
	var ss float32
	for _, x := range v {
		ss += x * x
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(v))+eps)))
	y := make([]float32, len(v))
	for i, x := range v {
		y[i] = x * inv
	}
	return y
}

// dspineRow is one row-major matrix-vector product: y[o] = Σ_i W[o*in+i]·x[i],
// serial index-ordered accumulation (HF Linear: y = x @ W^T).
func dspineRow(w, x []float32, out, in int) []float32 {
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		var s float32
		base := o * in
		for i := 0; i < in; i++ {
			s += w[base+i] * x[i]
		}
		y[o] = s
	}
	return y
}

// dspineSigmoid mirrors the dflash2 gate form: g / (1 + exp(-g)), exp in
// float64, division in float32.
func dspineSigmoid(g float32) float32 {
	return g / (1 + float32(math.Exp(float64(-g))))
}

// dspineTopK selects the k highest-scoring tokens, descending score, ties toward
// the lower token ID (the dflash2 selector convention). Nonfinite scores are
// skipped rather than poison the ranking.
func dspineTopK(logits []float32, k int) []int {
	if k < 1 {
		return nil
	}
	cands := make([]int, 0, k)
	for token, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			continue
		}
		at := len(cands)
		for at > 0 && (value > logits[cands[at-1]] || (value == logits[cands[at-1]] && token < cands[at-1])) {
			at--
		}
		if at < k {
			if len(cands) < k {
				cands = append(cands, 0)
			}
			copy(cands[at+1:k], cands[at:k-1])
			cands[at] = token
		}
	}
	return cands
}

// DSPineRefinementTable is the shared [slots, K predecessor, K candidate]
// transition lattice. Slot s (block row s+1) scores each candidate c against
// each sender σ: the final bridge write redone with σ's exact token code,
// scored against the target-head rows. Sentinel sender -1 scores -Inf. The
// existing greedy/sampled lattice walk consumes this table unchanged.
type DSPineRefinementTable struct {
	K int
	// Senders is [slot][K] sender token IDs (-1 = sentinel).
	Senders [][]int
	// Candidates is [slot][K] candidate token IDs.
	Candidates [][]int
	// Cells is [slot][K sender][K candidate] refined scores.
	Cells [][][]float32
}

// Refine redoes the final bridge write — the write landing in the last layer's
// input norm, built from the second-to-last layer's outputs — with each
// candidate sender's exact code, and scores the result against the target-head
// rows, producing the shared [slots, K predecessor, K candidate] lattice.
func (d *DSpineBridgeDrafter) Refine(t DSPineForwardTrace) (DSPineRefinementTable, error) {
	if d == nil || d.w == nil {
		return DSPineRefinementTable{}, fmt.Errorf("%w: drafter is nil", ErrDSpineConfigInvalid)
	}
	if t.Rows < 2 || len(t.Layers) != len(d.cfg.Layers) {
		return DSPineRefinementTable{}, fmt.Errorf("%w: trace holds %d rows / %d layers for a %d-layer head", ErrDSpineConfigInvalid, t.Rows, len(t.Layers), len(d.cfg.Layers))
	}
	k := d.cfg.TopK
	h, r := d.w.Hidden, d.w.Rank
	slots := t.Rows - 1
	lastL := len(d.cfg.Layers) - 1
	nLast := t.Layers[lastL].Norm
	hGate := t.Layers[lastL-1].Out
	dense := d.w.Dense[lastL]
	maskEmb := d.w.Embed[d.cfg.MaskToken*h : (d.cfg.MaskToken+1)*h]
	tab := DSPineRefinementTable{K: k, Senders: make([][]int, slots), Candidates: make([][]int, slots), Cells: make([][][]float32, slots)}
	for s := 1; s < t.Rows; s++ {
		cand := dspineTopK(t.Logits[s], k)
		send := make([]int, k)
		if s == 1 {
			send[0] = t.Anchor
			for j := 1; j < k; j++ {
				send[j] = -1
			}
		} else {
			copy(send, dspineTopK(t.Logits[s-1], k))
		}
		tab.Senders[s-1] = send
		tab.Candidates[s-1] = cand
		tab.Cells[s-1] = make([][]float32, k)
		mix := d.dspineMix(lastL, s, nLast, h)
		ns := nLast[s*h : (s+1)*h]
		for si, sigma := range send {
			row := make([]float32, k)
			tab.Cells[s-1][si] = row
			if sigma < 0 {
				for ci := range row {
					row[ci] = float32(math.Inf(-1))
				}
				continue
			}
			sigmaEmb := d.w.Embed[sigma*h : (sigma+1)*h]
			pred := dspineRow(d.w.Transfer, sigmaEmb, r, h)
			gamma := dspineRow(d.w.Gate, hGate[s*h:(s+1)*h], r, h)
			gated := make([]float32, r)
			for kk := 0; kk < r; kk++ {
				gated[kk] = dspineSigmoid(gamma[kk]) * pred[kk]
			}
			nPrime := dspineRMS(maskEmb, float32(d.cfg.Eps))
			for kk, m := range dspineRow(d.w.Lift, gated, h, r) {
				nPrime[kk] += m
			}
			proj := dspineRow(dense, nPrime, h, h)
			hPrime := make([]float32, h)
			for kk := 0; kk < h; kk++ {
				hPrime[kk] = maskEmb[kk] + proj[kk] + mix[kk] - ns[kk] + nPrime[kk]
			}
			fPrime := dspineRMS(hPrime, float32(d.cfg.Eps))
			logitsPrime := dspineRow(d.w.Head, fPrime, d.w.Vocab, h)
			for ci, c := range cand {
				row[ci] = logitsPrime[c]
			}
		}
	}
	return tab, nil
}

// Walk consumes the refinement lattice the way the existing greedy tree walk
// does: from the decided anchor, descend slot by slot taking the highest-scoring
// candidate under the decided sender, ties toward the lower token ID. It
// returns the drafted token path, one token per slot.
func (tab DSPineRefinementTable) Walk(anchor int) ([]int, error) {
	path := make([]int, 0, len(tab.Cells))
	decided := anchor
	for s, cells := range tab.Cells {
		senderIdx := -1
		for i, tok := range tab.Senders[s] {
			if tok == decided {
				senderIdx = i
				break
			}
		}
		if senderIdx < 0 {
			return nil, fmt.Errorf("%w: slot %d lost sender %d", ErrDSpineWalkDeadEnd, s+1, decided)
		}
		row := cells[senderIdx]
		best, bestScore := -1, float32(math.Inf(-1))
		for ci, score := range row {
			if math.IsInf(float64(score), -1) {
				continue
			}
			if best < 0 || score > bestScore || (score == bestScore && tab.Candidates[s][ci] < tab.Candidates[s][best]) {
				best, bestScore = ci, score
			}
		}
		if best < 0 {
			return nil, fmt.Errorf("%w: slot %d has no finite candidate", ErrDSpineWalkDeadEnd, s+1)
		}
		path = append(path, tab.Candidates[s][best])
		decided = tab.Candidates[s][best]
	}
	return path, nil
}

// Draft runs one cap-tracking round: it drafts cap+1 rows (the anchor plus cap
// placeholder rows), refines the final write into the shared lattice, and walks
// it greedily. A truncated round is refused for non-block-causal heads. The
// returned IDs are drafts only: they must never be emitted before exact native
// target verification.
func (d *DSpineBridgeDrafter) Draft(ctx context.Context, anchor, cap int) ([]int, error) {
	if d == nil {
		return nil, fmt.Errorf("%w: drafter is nil", ErrDSpineConfigInvalid)
	}
	if cap < 1 || cap > d.cfg.BlockSize-1 {
		return nil, fmt.Errorf("%w: cap %d outside [1,%d]", ErrDSpineConfigInvalid, cap, d.cfg.BlockSize-1)
	}
	trace, err := d.Forward(ctx, anchor, cap+1)
	if err != nil {
		return nil, err
	}
	tab, err := d.Refine(trace)
	if err != nil {
		return nil, err
	}
	return tab.Walk(anchor)
}
