//go:build darwin && arm64 && cgo

package compute

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// The DFlash2 draft block ops of the Metal backend. Weights are BF16 [rows,
// cols] tensors: a packed GGUF view of GGML type BF16, or a BF16 device buffer
// holding the checkpoint's bytes as uploaded.

// ink2MetalBF16 resolves a BF16 weight tensor.
func ink2MetalBF16(w Tensor) (metalgemm.Ink2Weight, error) {
	if b, ok := w.buf.(*metalBuf); ok && w.Dtype == BF16 && len(w.Shape) == 2 && b.ptr != nil {
		if b.n < 2*w.Shape[0]*w.Shape[1] {
			return metalgemm.Ink2Weight{}, fmt.Errorf("BF16 weight %v in a %d-byte buffer", w.Shape, b.n)
		}
		return metalgemm.Ink2Weight{Buf: metalgemm.Ink2AdoptBuffer(b.ptr), Type: 30, Rows: w.Shape[0], Cols: w.Shape[1], Experts: 1}, nil
	}
	v, _, err := ink2MetalWeight(w, false)
	if err == nil && v.Type != 30 {
		err = fmt.Errorf("GGML type %d weight, want BF16", v.Type)
	}
	return v, err
}

// Ink2DFLinear is y = x W^T for tokens <= 8 rows of the BF16 weight w; x holds
// 1 to 4 column segments of [tokens][cols/len(x)] each, so the fc fuse reads
// the four target taps in place. y is [tokens][rows of w].
func (c *metalBackend) Ink2DFLinear(x []Ink2Ref, w Tensor, y Ink2Ref, tokens int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(y)
	if err != nil {
		return ink2MetalErr("draft linear", err)
	}
	xs, err := ink2MetalRefs(x)
	if err != nil {
		return ink2MetalErr("draft linear input", err)
	}
	wv, err := ink2MetalBF16(w)
	if err != nil {
		return ink2MetalErr("draft linear", err)
	}
	return ink2MetalErr("draft linear", d.DraftLinear(xs, wv, r[0], tokens))
}

// Ink2DFConv is one side of the DFlash2 dynamic short conv over a block:
// out = resid + sum_t x[row-t] * (base[side][t] + dyn[row][side][t][c/group]),
// dyn the kernel projection output and base the BF16 [2*kn][hidden] kernel.
func (c *metalBackend) Ink2DFConv(x, dyn Ink2Ref, base Tensor, resid, out Ink2Ref, rows, hidden, group, kn, side int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, dyn, resid, out)
	if err != nil {
		return ink2MetalErr("draft conv", err)
	}
	bv, err := ink2MetalBF16(base)
	if err != nil {
		return ink2MetalErr("draft conv", err)
	}
	return ink2MetalErr("draft conv", d.DraftConv(r[0], r[1], bv, r[2], r[3], rows, hidden, group, kn, side))
}

// Ink2DFAttend is Ink2DAttend for the DFlash2 draft: non-causal inside the
// block, and the window applied per row, each row seeing the context
// positions within window-1 of its own.
func (c *metalBackend) Ink2DFAttend(q, kb, vb, kc, vc, out, posDev Ink2Ref, rows, heads, kvHeads, dim, window, cap int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(q, kb, vb, kc, vc, out, posDev)
	if err != nil {
		return ink2MetalErr("draft attend", err)
	}
	return ink2MetalErr("draft attend", d.DraftAttend(r[0], r[1], r[2], r[3], r[4], r[5], r[6], rows, heads, kvHeads, dim, window, cap))
}

// Ink2DFSelectScratch is the floats of scratch Ink2DFSelect needs.
func Ink2DFSelectScratch(rows, k int) int { return metalgemm.DraftSelectScratch(rows, k) }

// Ink2DFSelect runs the DFlash2 codebook candidate selector's greedy chain:
// each row's top k logits, scored with the predecessor/hidden/successor
// codebook edge and softcapped after that correction, the best becoming
// path[row] and the next row's predecessor (row 0's the int at anchorDev).
// cand, when set, receives the [rows][k] candidate ids.
func (c *metalBackend) Ink2DFSelect(logits Ink2Ref, rows, rowStride, vocab, k int, proj Ink2Ref, pred, succ Tensor,
	anchorDev Ink2Ref, softcap float32, scratch, path, cand Ink2Ref) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(logits, proj, anchorDev, scratch, path, cand)
	if err != nil {
		return ink2MetalErr("draft select", err)
	}
	pv, err := ink2MetalBF16(pred)
	if err != nil {
		return ink2MetalErr("draft select predecessor codebook", err)
	}
	sv, err := ink2MetalBF16(succ)
	if err != nil {
		return ink2MetalErr("draft select successor codebook", err)
	}
	return ink2MetalErr("draft select", d.DraftSelect(r[0], rows, rowStride, vocab, k, r[1], pv, sv, r[2], softcap, r[3], r[4], r[5]))
}

var _ Inkling2DFlash2FusedDevice = (*metalBackend)(nil)

func (c *metalBackend) Ink2DFLinearMulti(x []Ink2Ref, w []Tensor, y []Ink2Ref, tokens int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	if len(w) == 0 || len(y) != len(w) {
		return ink2MetalErr("draft linear multi", fmt.Errorf("%d weights into %d outputs", len(w), len(y)))
	}
	d, ys, err := ink2MetalBind(y...)
	if err != nil {
		return ink2MetalErr("draft linear multi", err)
	}
	xs, err := ink2MetalRefs(x)
	if err != nil {
		return ink2MetalErr("draft linear multi input", err)
	}
	ws := make([]metalgemm.Ink2Weight, len(w))
	for i := range w {
		if ws[i], err = ink2MetalBF16(w[i]); err != nil {
			return ink2MetalErr("draft linear multi", err)
		}
	}
	return ink2MetalErr("draft linear multi", d.DraftLinearMulti(xs, ws, ys, tokens))
}

func (c *metalBackend) Ink2DFLinearSiluMul(x Ink2Ref, gate, up Tensor, y, scratch Ink2Ref, tokens int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, y, scratch)
	if err != nil {
		return ink2MetalErr("draft gated linear", err)
	}
	gv, err := ink2MetalBF16(gate)
	if err != nil {
		return ink2MetalErr("draft gated linear gate", err)
	}
	uv, err := ink2MetalBF16(up)
	if err != nil {
		return ink2MetalErr("draft gated linear up", err)
	}
	return ink2MetalErr("draft gated linear", d.DraftLinearSiluMul(r[0], gv, uv, r[1], r[2], tokens))
}
