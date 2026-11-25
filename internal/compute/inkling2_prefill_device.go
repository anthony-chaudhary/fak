package compute

import (
	"os"
	"strings"
)

// ink2PrefillFast reports whether the prefill kernel family name runs its fast
// variant; FAK_INKLING2_PREFILL_OLD lists families (comma separated, or "all") that
// keep the original kernel, for A/B attribution.
func ink2PrefillFast(name string) bool {
	old := os.Getenv("FAK_INKLING2_PREFILL_OLD")
	if old == "" {
		return true
	}
	for _, f := range strings.Split(old, ",") {
		if f = strings.TrimSpace(f); f == name || f == "all" {
			return false
		}
	}
	return true
}

// Inkling2PrefillDevice batches the Inkling2Device decode ops along the token axis
// for a prompt chunk: token t of a T-token chunk sits at position pos0+t and every
// activation is [T][width] rows. Ops that are already row-batched (Ink2RMS rows,
// Ink2SigmoidMul, Ink2Situ and Ink2ExpertMatVec over T*k slots) are reused as is.
type Inkling2PrefillDevice interface {
	// Ink2PMatMul is Y[T,out] = X[T,in] times W transposed for one resident dense weight.
	Ink2PMatMul(w Tensor, x, y Ink2Ref, T int) error
	// Ink2PMix writes query-major [nq][T][n].
	Ink2PMix(states []Ink2Ref, queries Ink2Ref, nq int, out Ink2Ref, n, T int) error
	Ink2PCtrlDown(cur, msg, w, g Ink2Ref, n, rank, T int) error
	Ink2PCtrlApply(cur, msg, g, gainW, forgetW, out Ink2Ref, n, rank, T int) error
	// Ink2PConv walks the chunk in token order and leaves the final history in hist.
	Ink2PConv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error
	Ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error
	Ink2PAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error
	Ink2PRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error
	Ink2PCombine(down, wts, out Ink2Ref, k, n, T int) error
	// Ink2PRMS and Ink2PSitu are the chunk forms of Ink2RMS and Ink2Situ; they may
	// reduce in float32 where the decode ops keep float64.
	Ink2PRMS(x, w, y Ink2Ref, rows, n int, eps float32) error
	Ink2PSitu(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error
	// Ink2PExpertGrouped multiplies npairs routed (token, slot) pairs through one
	// stacked expert tensor grouped by expert, so each expert block is decoded once
	// per tile of pairs; pair i reads x row (i/xDiv)*xStride and writes y row i.
	// scratch holds Ink2PGroupScratch(npairs, nexp) ints; bucketed reuses the bucket an
	// earlier call left in scratch for the same ids. ok is false (nothing
	// launched) for a dtype without a grouped kernel (NVFP4 runs a BF16 tensor-core MMA,
	// the IQ and Q3_K stacks an FP32 tile kernel).
	Ink2PExpertGrouped(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch Ink2Ref, bucketed bool) (ok bool, err error)
	// Ink2PExpertSorted runs the decode expert GEMV over the npairs pairs in expert
	// order (pair inputs gathered into sortX, outputs through sortY), so consecutive
	// slots reuse one expert's rows from cache. sortX holds npairs*xStride floats,
	// sortY npairs*expertRows, sortIds npairs ints. ok is false for a dtype without
	// an indexed kernel.
	Ink2PExpertSorted(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch, sortIds, sortX, sortY Ink2Ref) (ok bool, err error)
}

// Ink2PGroupScratch is the int count of the grouped expert GEMM bucket scratch.
func Ink2PGroupScratch(npairs, nexp int) int {
	tiles := npairs/8 + nexp
	return nexp + (nexp + 1) + npairs + 2*tiles + 1
}

// Inkling2VerifyDevice adds what a speculative verify chunk needs beyond prefill:
// a conv that also logs its entry history and every chunk input, and the commit that
// rewinds a history to the state after the first keep tokens of that chunk.
type Inkling2VerifyDevice interface {
	// Ink2PConvLog is Ink2PConv that also writes log as [n][kn-1+T]: the entry
	// history followed by the T inputs of each channel.
	Ink2PConvLog(x, w, dyn, hist, log, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error
	// Ink2PCombineConvLog is Ink2PConvLog (no dynamic taps) whose input row t is
	// Ink2PCombine(down, wts, k) row t, so the MoE combine needs no launch of its own.
	Ink2PCombineConvLog(down, wts Ink2Ref, k int, w, hist, log, out, resid Ink2Ref, n, kn int, norm float32, T int) error
	// Ink2PConvCommit sets hist[c][j] = log[c][keep+j] for channels*(past) entries,
	// where every log row is L floats.
	Ink2PConvCommit(hist, log Ink2Ref, channels, past, L, keep int) error
	// Ink2VRows is Y[t] = W.X[t] for a Q5_K or Q6_K W and up to 9 rows in one weight
	// pass, each row equal to the single-row GEMV; ok is false (nothing launched) for
	// a dtype without one.
	Ink2VRows(w Tensor, x, y Ink2Ref, rows int) (ok bool, err error)
	// Ink2VQ8Rows is Y[t] = W.X[t] for a Q8_0 W and up to 9 rows, each row equal to the
	// decode GEMV; q holds rows*(in/4 + in/32) floats of codes and scales, which
	// quantize=false reuses from an earlier call on the same x. A set gate (needs
	// quantize) multiplies x*sigmoid(gate) instead, the gate applied while x is
	// quantized and x left unscaled. ok is false (nothing launched) outside the
	// kernel's contract.
	Ink2VQ8Rows(w Tensor, x, gate, q, y Ink2Ref, rows int, quantize bool) (ok bool, err error)
	// Ink2VMix is Ink2PMix ([nq][T][n] out) on the cluster mix kernel, with each query
	// RMS-normalized by norms[q] when norms is set; ok is false (nothing launched)
	// without cluster launch.
	Ink2VMix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n, T int) (ok bool, err error)
	// Ink2PControlMulti runs J <= 4 controllers of one visit over T rows (Ink2PCtrlDown
	// then Ink2PCtrlApply for each) in one launch per stage; slices index by controller.
	Ink2PControlMulti(cur, msg, down, gain, forget, g, out []Ink2Ref, n, rank, T int) error
	// Ink2PConvPairLog runs two Ink2PConvLog calls of the same geometry (no residual)
	// in one launch; every slice holds the two convs' arguments.
	Ink2PConvPairLog(x, w, dyn, hist, log, out []Ink2Ref, norm []float32, n, heads, kn, T int) error
	// Ink2VRepeatRows writes out[t*k+j] = x[t] for rows x of n floats.
	Ink2VRepeatRows(x, out Ink2Ref, n, rows, k int) error
	// Ink2PConvCommitDev is Ink2PConvCommit with keep read from the int at keepDev.
	Ink2PConvCommitDev(hist, log, keepDev Ink2Ref, channels, past, L int) error
	// Ink2PQKVPostDev and Ink2PAttendDev read the chunk's first position from the int
	// at posDev, so a captured verify replays at any position.
	Ink2PQKVPostDev(q, r, k, v, qclip, rclip, kcache, vcache, posDev Ink2Ref, p Ink2AttnParams, cap, T int) error
	Ink2PAttendDev(q, r, sinks, relBias, kcache, vcache, out, posDev Ink2Ref, p Ink2AttnParams, extent, window, cap, T int) error
	// Ink2GraphEndSlot ends a capture opened by Ink2GraphBegin into slot (1..3), beside
	// the decode graph; Ink2GraphLaunchSlot replays it.
	Ink2GraphEndSlot(slot int) error
	Ink2GraphLaunchSlot(slot int) error
}

// Inkling2DraftDevice runs the DSpark draft's dense block forward. Positions are
// read from device ints so the draft and its context commit replay in graphs.
type Inkling2DraftDevice interface {
	// Ink2DFuse writes rows of RMS(sum_t alpha[t]*taps[t]) times w.
	Ink2DFuse(taps []Ink2Ref, alpha []float32, w, out Ink2Ref, rows, n int, eps float32) error
	// Ink2DQK RMS-normalizes each head of x by w and rotates it (neox RoPE, theta) at
	// position *posDev+row into dst: row order, or the pos%cap slot of a ring.
	Ink2DQK(x, w, dst, posDev Ink2Ref, rows, heads, dim, cap int, ring bool, eps float32, theta float64) error
	// Ink2DRingStore copies src row r into ring slot (*posDev+r)%cap.
	Ink2DRingStore(src, ring, posDev Ink2Ref, rows, width, cap int) error
	// Ink2DAttend attends each block query row to the context ring before *posDev
	// (the last window positions when window > 0) and to every block key.
	Ink2DAttend(q, kb, vb, kc, vc, out, posDev Ink2Ref, rows, heads, kvHeads, dim, window, cap int) error
	// Ink2DSiluMul is out = silu(g) * u.
	Ink2DSiluMul(g, u, out Ink2Ref, n int) error
	// Ink2DTopM writes the M largest of the first vocab logits of each row into
	// out[row][M]; candV/candI are 64*M scratch per row.
	Ink2DTopM(logits Ink2Ref, rows, rowStride, vocab, M int, candV, candI, out Ink2Ref) error
	// Ink2DMarkovIn assembles step's Markov input [state | w1[prev] | h | fRNN], w1 a
	// BF16 [vocab][rank] table, prev the anchor at step 0 else idx[4*(step-1)], fRNN
	// the ring row of position *posDev-1.
	Ink2DMarkovIn(state, w1, anchor, idx Ink2Ref, step int, h, frnnRing, posDev Ink2Ref, cap, rank, hidden int, in Ink2Ref) error
	// Ink2DMarkovGate updates state and writes latent from raw+bias.
	Ink2DMarkovGate(raw, bias, state, latent Ink2Ref, rank, step int) error
	// Ink2DMarkovPick writes idx[4*step] = argmax over the M shortlisted tokens cand
	// of base[c] + head[c].(z+zb), head the resident Q5_K LM head.
	Ink2DMarkovPick(head Tensor, base, cand Ink2Ref, M int, z, zb, idx Ink2Ref, step int) error
}
