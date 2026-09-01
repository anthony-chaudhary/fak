//go:build !(darwin && arm64 && cgo)

package metalgemm

type NVFP4Weight struct{ Out, In int }
type NVFP4Execution int

const (
	NVFP4NotExecuted NVFP4Execution = iota
	NVFP4ExecutedM5GEMV
)

func UploadNVFP4(packed, scales []byte, out, in int) *NVFP4Weight { return nil }
func (w *NVFP4Weight) GEMV(x, y []float32) NVFP4Execution         { return NVFP4NotExecuted }
func ResetNVFP4()                                                 {}
