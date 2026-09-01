//go:build darwin && arm64 && cgo

package metalgemm

/*
int mg_nvfp4_upload(const unsigned char*, const unsigned char*, int, int);
int mg_nvfp4_gemv(int, const float*, float*);
void mg_nvfp4_reset(void);
*/
import "C"

import (
	"math"
	"strings"
	"unsafe"
)

// NVFP4Weight is a resident W4A16 matrix for the Apple M5 M=1 GEMV candidate.
type NVFP4Weight struct {
	id      C.int
	Out, In int
}

// NVFP4Execution binds candidate selection to actual Metal completion.
type NVFP4Execution int

const (
	NVFP4NotExecuted NVFP4Execution = iota
	NVFP4ExecutedM5GEMV
)

// UploadNVFP4 selects the candidate only on an Apple M5 Metal device. The
// existing caller path remains the fallback when this returns nil.
func UploadNVFP4(packed, scales []byte, out, in int) *NVFP4Weight {
	if !strings.Contains(DeviceName(), "M5") {
		return nil
	}
	return uploadNVFP4(packed, scales, out, in)
}

func uploadNVFP4(packed, scales []byte, out, in int) *NVFP4Weight {
	if !Available() || !nvfp4ValidPayload(packed, scales, out, in) {
		return nil
	}
	for _, raw := range scales {
		if math.IsNaN(float64(nvfp4E4M3FN(raw))) {
			return nil
		}
	}
	id := C.mg_nvfp4_upload((*C.uchar)(unsafe.Pointer(&packed[0])), (*C.uchar)(unsafe.Pointer(&scales[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	return &NVFP4Weight{id: id, Out: out, In: in}
}

// GEMV executes exactly one activation row. Invalid slices fail closed without
// touching output; successful return identifies the kernel that completed.
func (w *NVFP4Weight) GEMV(x, y []float32) NVFP4Execution {
	if w == nil || w.id < 0 || len(x) != w.In || len(y) != w.Out {
		return NVFP4NotExecuted
	}
	if C.mg_nvfp4_gemv(w.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0]))) != 1 {
		return NVFP4NotExecuted
	}
	return NVFP4ExecutedM5GEMV
}

// ResetNVFP4 releases all resident candidate weights and scratch buffers.
func ResetNVFP4() { C.mg_nvfp4_reset() }
