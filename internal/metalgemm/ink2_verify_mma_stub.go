//go:build !(darwin && arm64 && cgo)

package metalgemm

func (*Ink2Device) VerifyRowsMMA(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) (bool, error) {
	return false, ErrMetalUnavailable
}
func (*Ink2Device) VerifyExpertsMMA(w Ink2Weight, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, tab Ink2Ref) (bool, error) {
	return false, ErrMetalUnavailable
}
