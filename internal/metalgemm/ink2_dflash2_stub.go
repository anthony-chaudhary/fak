//go:build !(darwin && arm64 && cgo)

package metalgemm

func (*Ink2Device) DraftQKRope(x, w, dst, posDev Ink2Ref, rows, heads, dim, cap, base int, ring bool, eps float32, theta float64) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) DraftRingStore(src, ring, posDev Ink2Ref, rows, width, cap, base int) error {
	return ErrMetalUnavailable
}
func (*Ink2Device) DraftSiluMul(g, u, out Ink2Ref, n int) error { return ErrMetalUnavailable }
func (*Ink2Device) DraftHead(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, tokens int) (bool, error) {
	return false, ErrMetalUnavailable
}
