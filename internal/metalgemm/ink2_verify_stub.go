//go:build !(darwin && arm64 && cgo)

package metalgemm

func (*Ink2Device) VerifyRows(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) VerifyMix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) VerifyControl(ctl Ink2Controls, g, out []Ink2Ref, n, rank, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) VerifyConv(x, w, dyn, hist, log, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) VerifyConvPair(x, w, dyn, hist, log, out []Ink2Ref, norm []float32, n, heads, kn, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) VerifyConvCommit(hist, log, keepDev Ink2Ref, channels, past, L, keep int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) VerifyQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, posDev Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func VerifyAttendSpan(devicePos bool, window, cap, pos0, T int) int { return 0 }
func (*Ink2Device) VerifyAttend(q, r, sinks, relBias, kcache, vcache, scores Ink2Ref, scoreFloats int, out, posDev Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) VerifyRepeatRows(x, out Ink2Ref, n, rows, k int) error { return ErrMetalUnavailable }
