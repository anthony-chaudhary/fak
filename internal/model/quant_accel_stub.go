//go:build !(fakaccel && darwin && arm64 && cgo)

package model

func qgemmAccelDefault() bool { return false }

func q8PrepareAccelWeight(qt *q8Tensor) {}

func q8RememberAccelPanel(qp *q8Panel, X []float32) {
	qp.f32 = nil
}

func qGemm8AccelInto(qt *q8Tensor, qp *q8Panel, Y []float32) bool {
	return false
}
