//go:build !(darwin && arm64 && cgo)

package metalgemm

func (*Ink2Device) TreeVerifyQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, posDev, parent Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) TreeVerifyAttend(q, r, sinks, relBias, kcache, vcache, scores Ink2Ref, scoreFloats int, out, posDev, parent Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) TreeVerifyConv(x, w, dyn, hist, log, out, resid, parent Ink2Ref, n, heads, kn int, norm float32, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) TreeVerifyConvCommit(hist, log, colsDev Ink2Ref, channels, past, L int, cols []int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) TreeVerifyKVCompact(kcache, vcache, pathDev, posDev Ink2Ref, width, cap, pos0 int, path []int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) TreeVerifyGatherRows(in, out, pathDev Ink2Ref, width, inStride, rows int, path []int) error {
	return ErrMetalUnavailable
}
