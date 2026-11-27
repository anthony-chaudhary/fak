//go:build !(darwin && arm64 && cgo)

package metalgemm

func PAttendSpan(window, cap, pos0, T int) int { return 0 }

func (*Ink2Device) RowsMatMul(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PMix(states []Ink2Ref, queries Ink2Ref, nq int, out Ink2Ref, n, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PCtrlDown(cur, msg, down, g []Ink2Ref, n, rank, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PCtrlApply(cur, msg, g, gain, forget, out []Ink2Ref, n, rank, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PConv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PQKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PAttend(q, r, sinks, relBias, kcache, vcache, scores Ink2Ref, scoreFloats int, out Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PAttendHalf(q, r, sinks, relBias, kcache, vcache, scores Ink2Ref, scoreFloats int, out Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PCombine(down, wts, out Ink2Ref, k, n, T int) error { return ErrMetalUnavailable }
func (*Ink2Device) PBucket(ids, counts, offsets, order Ink2Ref, npairs, nexp int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PExpertGrouped(w Ink2Weight, ids, order, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale Ink2Ref) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) RowsGEMM(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) (bool, error) {
	return false, ErrMetalUnavailable
}
func (*Ink2Device) PExpertGEMM(w Ink2Weight, ids, order, offsets, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale Ink2Ref) (bool, error) {
	return false, ErrMetalUnavailable
}
func CtrlGEMMTiles(n, rank int) bool { return false }
func (*Ink2Device) PCtrlDownGEMM(cur, msg, down, g, xs Ink2Ref, n, rank, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PCtrlApplyGEMM(cur, msg, g, gain, forget, out, gs Ink2Ref, n, rank, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PFlashAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) (bool, error) {
	return false, ErrMetalUnavailable
}
func (*Ink2Device) PBucketSort(ids, counts, offsets, order Ink2Ref, npairs, nexp int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) PConvRows(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	return ErrMetalUnavailable
}
func CtrlTiles(n, rank int) bool              { return false }
func CtrlDownSplitScratch(n, rank, T int) int { return 0 }
func (*Ink2Device) PCtrlDownSplit(cur, msg, down, g, part Ink2Ref, n, rank, T int) (bool, error) {
	return false, ErrMetalUnavailable
}
func (*Ink2Device) PCtrlApplyTile(cur, msg, g, gain, forget, out Ink2Ref, n, rank, T int) (bool, error) {
	return false, ErrMetalUnavailable
}
