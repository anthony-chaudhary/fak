package model

// kimik31_attn_res.go — the Kimi-K3.1 attention-residual bank (ticket
// 1611-kimi-k31, port leaf sglang-kimi-k31-2; goal Wave 1.6/1.7; device-port
// plan Stage 3). K3.1 replaces the plain residual stream: every attention-residual
// block (depth 12) is a block-write layer that snapshots its pre-attention prefix
// into a per-forward bank, and each aggregation point mixes the banked rows
// with the current prefix by a learned per-token score, softmax over the rows —
// so a port keeping add-and-norm will NOT match the checkpoint.
//
// SOURCE (direct port, pinned): SGLang
// python/sglang/srt/layers/attn_residual.py at
// a7d9f81ee24c5cceeceeb4ccb9072f68d56cb10d (Apache-2.0). The authoritative
// semantics are aggregate_stream_torch (the file's own eager reference):
//
//	rows   = cat([bank[:, 0:nvb, :], prefix.unsqueeze(1)], dim=1)   # [T, nvb+1, H]
//	normed = score_norm(rows)                                        # RMSNorm(scoreNormW)
//	scores = score_proj(normed)                                      # [T, R], proj is [1,H]
//	probs  = softmax(scores.float(), dim=-1)
//	mixed  = sum(probs * rows, dim=1).to(prefix.dtype)
//
// and the normed aggregation point additionally applies out_norm:
//
//	aggregate(prefix) = out_norm(mix(prefix))
//
// with nvb == 0 (the layer-0 attention side, nothing banked yet) short-circuiting
// to out_norm(prefix). write() snapshots the pre-attention prefix into the NEXT
// bank row (bank[:, nvb, :]) and advances num_valid_blocks. The score projection
// folds the score-norm weight into one vector cw = scoreNormW ⊙ scoreProjW
// (get_cw), so score = dot(v, cw) * rms(v) — the host form below computes it
// directly for clarity.
//
// WHAT THIS OWNS: the bank + the two aggregation forms (normed and pre-norm),
// pinned against the reference semantics. WHAT THIS DOES NOT OWN: the per-layer
// REPLACE_AT_EXIT delayed-residual wiring (the composite's job) and the CUDA/
// Triton kernels (the reference dispatches by capability; the host form is the
// eager reference every device path must match).

import "math"

// KimiK31AttnResBank is the per-forward snapshot bank: T rows (tokens) x up to
// BlockNum snapshots x Hidden, plus a write cursor. It mirrors SGLang's
// AttnResidual (block_residual [T, NB, H] + num_valid_blocks).
type KimiK31AttnResBank struct {
	Hidden int
	Tokens int
	// Residual is [Tokens * BlockNum * Hidden], row-major: token-major, then
	// snapshot index, then hidden.
	Residual []float32
	// NumValidBlocks is the write cursor (nvb): snapshots [0, NumValidBlocks).
	NumValidBlocks int
}

// NewKimiK31AttnResBank allocates a bank for tokens x hidden with room for
// blockNum snapshots (the model's attention-residual block depth).
func NewKimiK31AttnResBank(tokens, hidden, blockNum int) *KimiK31AttnResBank {
	if tokens < 0 {
		tokens = 0
	}
	if hidden < 0 {
		hidden = 0
	}
	if blockNum < 0 {
		blockNum = 0
	}
	return &KimiK31AttnResBank{
		Hidden:   hidden,
		Tokens:   tokens,
		Residual: make([]float32, tokens*blockNum*hidden),
	}
}

// blockNum is the bank capacity (snapshots along the middle axis).
func (b *KimiK31AttnResBank) blockNum() int {
	if b.Tokens == 0 || b.Hidden == 0 {
		return 0
	}
	return len(b.Residual) / (b.Tokens * b.Hidden)
}

// row returns the [Hidden] view of snapshot j for token t.
func (b *KimiK31AttnResBank) row(t, j int) []float32 {
	off := (t*b.blockNum() + j) * b.Hidden
	return b.Residual[off : off+b.Hidden]
}

// Write snapshots the pre-attention prefix ([Tokens*Hidden]) into bank row
// NumValidBlocks and advances the cursor. It panics when the bank is full or
// the shape is wrong — an unrepresentable state, never a silent drop.
func (b *KimiK31AttnResBank) Write(prefix []float32) {
	nb := b.blockNum()
	if b.NumValidBlocks >= nb {
		panic("model: kimi-k31 attn-res: bank is full")
	}
	if len(prefix) != b.Tokens*b.Hidden {
		panic("model: kimi-k31 attn-res: prefix length != Tokens*Hidden")
	}
	for t := 0; t < b.Tokens; t++ {
		copy(b.row(t, b.NumValidBlocks), prefix[t*b.Hidden:(t+1)*b.Hidden])
	}
	b.NumValidBlocks++
}

// NumValidBlocksFor reports the bank's current write cursor. Exported helper so
// a composite can branch on "nothing banked yet" (the layer-0 short circuit).
func (b *KimiK31AttnResBank) NumValidBlocksFor() int {
	if b == nil {
		return 0
	}
	return b.NumValidBlocks
}

// KimiK31AttnResScore computes one aggregation point's softmax mixture of the
// banked rows plus the current prefix, returning the mixed [Tokens*Hidden] (the
// pre-norm value; the normed aggregation point applies outNormW after).
//
// It is the host transcription of aggregate_stream_torch: per token, rows =
// bank[0..nvb-1] ++ prefix; score_r = dot(rmsnorm(rows_r, scoreNormW, eps),
// scoreProjW); probs = softmax(scores); mixed = sum_r probs_r * rows_r, summed
// in float32 and stored in prefix's dtype (float32 here).
//
// With nvb == 0 the mixture is just the prefix (nothing to mix), matching the
// reference's early return.
func KimiK31AttnResScore(prefix []float32, b *KimiK31AttnResBank, scoreProjW, scoreNormW []float32, eps float32) []float32 {
	if b == nil || len(prefix) != b.Tokens*b.Hidden {
		panic("model: kimi-k31 attn-res: prefix shape does not match the bank")
	}
	if len(scoreNormW) != b.Hidden || len(scoreProjW) != b.Hidden {
		panic("model: kimi-k31 attn-res: score weights must be [Hidden]")
	}
	nvb := b.NumValidBlocks
	out := make([]float32, len(prefix))
	if nvb == 0 {
		copy(out, prefix)
		return out
	}
	R := nvb + 1
	normed := make([]float32, b.Hidden)
	scores := make([]float32, R)
	for t := 0; t < b.Tokens; t++ {
		for r := 0; r < R; r++ {
			var v []float32
			if r < nvb {
				v = b.row(t, r)
			} else {
				v = prefix[t*b.Hidden : (t+1)*b.Hidden]
			}
			rmsnormInto(normed, v, scoreNormW, eps)
			scores[r] = dot(normed, scoreProjW)
		}
		probs := softmaxCopyF32(scores)
		acc := out[t*b.Hidden : (t+1)*b.Hidden]
		for r := 0; r < R; r++ {
			p := probs[r]
			if p == 0 {
				continue
			}
			var v []float32
			if r < nvb {
				v = b.row(t, r)
			} else {
				v = prefix[t*b.Hidden : (t+1)*b.Hidden]
			}
			for h := 0; h < b.Hidden; h++ {
				acc[h] += p * v[h]
			}
		}
	}
	return out
}

// KimiK31AttnResAggregate is the normed aggregation point: the score mixture
// (KimiK31AttnResScore) followed by out_norm (RMSNorm(outNormW)). With nvb == 0
// it short-circuits to out_norm(prefix), matching the reference's layer-0
// attention side.
func KimiK31AttnResAggregate(prefix []float32, b *KimiK31AttnResBank, scoreProjW, scoreNormW, outNormW []float32, eps float32) []float32 {
	if b == nil || len(outNormW) != b.Hidden {
		panic("model: kimi-k31 attn-res: out-norm weight must be [Hidden]")
	}
	mixed := KimiK31AttnResScore(prefix, b, scoreProjW, scoreNormW, eps)
	normed := make([]float32, len(mixed))
	rmsnormInto(normed, mixed, outNormW, eps)
	return normed
}

// softmaxCopyF32 returns a max-subtracted float32 softmax of s (a copy; s is
// not mutated). It mirrors the reference's torch.softmax(scores.float(), -1).
func softmaxCopyF32(s []float32) []float32 {
	out := append([]float32(nil), s...)
	if len(out) == 0 {
		return out
	}
	m := out[0]
	for _, v := range out[1:] {
		if v > m {
			m = v
		}
	}
	var sum float64
	for i, v := range out {
		e := math.Exp(float64(v - m))
		out[i] = float32(e)
		sum += e
	}
	inv := 1.0 / sum
	for i := range out {
		out[i] *= float32(inv)
	}
	return out
}
