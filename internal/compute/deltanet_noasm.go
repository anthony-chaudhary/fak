//go:build !amd64

package compute

func hasDeltaNetSIMD() bool { return false }

func tryDeltaNetSIMD(st, qn, kn, vh []float32, bt, g float32, od, kvmem, delta []float32) bool {
	return false
}
