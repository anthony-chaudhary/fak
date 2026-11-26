//go:build !(darwin && arm64 && cgo)

package metalgemm

func (*Ink2Device) DraftLinear(x []Ink2Ref, w Ink2Weight, y Ink2Ref, tokens int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) DraftConv(x, dyn Ink2Ref, base Ink2Weight, resid, out Ink2Ref, rows, hidden, group, kn, side int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) DraftAttend(q, kb, vb, kc, vc, out, posDev Ink2Ref, rows, heads, kvHeads, dim, window, cap int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) DraftSelect(logits Ink2Ref, rows, rowStride, vocab, k int, proj Ink2Ref, pred, succ Ink2Weight,
	anchorDev Ink2Ref, softcap float32, scratch, path, cand Ink2Ref) error {
	return ErrMetalUnavailable
}
