//go:build !(darwin && arm64 && cgo)

package metalgemm

// Ink2Device cannot be opened without the Metal backend; its methods exist so
// portable callers compile.
type Ink2Device struct{}

func NewInk2Device() (*Ink2Device, error) { return nil, ErrMetalUnavailable }

func (*Ink2Device) Close()                         {}
func (*Ink2Device) Alloc(n int) (Ink2Buf, error)   { return Ink2Buf{}, ErrMetalUnavailable }
func (*Ink2Device) Free(b Ink2Buf)                 {}
func (*Ink2Device) Flush() error                   { return ErrMetalUnavailable }
func (*Ink2Device) Stats() Ink2Stats               { return Ink2Stats{} }
func (*Ink2Device) ProfileMode() Ink2ProfileMode   { return Ink2ProfileOff }
func (*Ink2Device) SetProfileScope(string)         {}
func (*Ink2Device) Profile() Ink2Profile           { return Ink2Profile{} }
func (*Ink2Device) ResetProfile() error            { return nil }
func (*Ink2Device) Zero(dst Ink2Ref, n int) error  { return ErrMetalUnavailable }
func (*Ink2Device) Write(Ink2Ref, []float32) error { return ErrMetalUnavailable }
func (*Ink2Device) Read(Ink2Ref, int) ([]float32, error) {
	return nil, ErrMetalUnavailable
}
func (*Ink2Device) ReadInts(Ink2Ref, int) ([]int32, error) {
	return nil, ErrMetalUnavailable
}
func (*Ink2Device) MatVec(w Ink2Weight, x, y Ink2Ref) error { return ErrMetalUnavailable }
func (*Ink2Device) MatVecQ8(w Ink2Weight, x, gate, q Ink2Ref, quantize bool, y Ink2Ref) (bool, error) {
	return false, ErrMetalUnavailable
}
func (*Ink2Device) ExpertMatVec(w Ink2Weight, ids, x Ink2Ref, xStride int, y Ink2Ref, k int, scale Ink2Ref, situ *Ink2Situ) (bool, error) {
	return false, ErrMetalUnavailable
}
func (*Ink2Device) ExpertScale(y, ids, scale Ink2Ref, k, rows int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) RMS(x, w, y Ink2Ref, rows, n int, eps float32) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) CtrlPair(cur, msg, pair Ink2Ref, n int) error { return ErrMetalUnavailable }
func (*Ink2Device) Mix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) Control(ctl Ink2Controls, g, out Ink2Ref, n, rank int, norms []Ink2Ref, eps float32) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) Conv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) CombineConv(down, wts Ink2Ref, k int, w, hist, out, resid Ink2Ref, n, kn int, norm float32) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) QKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) QKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) Attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) AttendHalf(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) SigmoidMul(x, gate Ink2Ref, n int) error { return ErrMetalUnavailable }
func (*Ink2Device) Route(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) Situ(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) Combine(down, wts, out Ink2Ref, k, n int) error        { return ErrMetalUnavailable }
func (*Ink2Device) CombineRows(down, wts, out Ink2Ref, k, n, T int) error { return ErrMetalUnavailable }
func (*Ink2Device) RouteRows(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) Add(a, b, out Ink2Ref, n int) error { return ErrMetalUnavailable }
func (*Ink2Device) SoftcapArgmax(logits Ink2Ref, n int, softcap float32, outIdx Ink2Ref) error {
	return ErrMetalUnavailable
}

func (*Ink2Device) MakeResident([]Ink2Buf) error { return ErrMetalUnavailable }
