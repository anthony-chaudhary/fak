//go:build cuda && cgo && cudaarchspecific && fak_cuda_deepgemm_wkv

package compute

/*
#cgo LDFLAGS: -lcuda
#include "cuda_deepgemm_wkv.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"
)

type cudaWKVWeights struct {
	owner         *cudaBackend
	codes, scales *cudaBuf
}

func (c *cudaBackend) wkvStatus(site string, code C.int) error {
	if code == 0 {
		return nil
	}
	var cause error
	if code > 0 {
		cause = c.faultLatch.Observe(DeviceFaultExecution, "deepgemm-wkv", int(code))
	}
	return &CUDAOpError{Op: "V41FP8WKV", Site: site, Code: int(code), Err: cause}
}
func (c *cudaBackend) V41FP8WKVSupported() bool {
	if c == nil {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var f C.fak_deepgemm_wkv_facts
	return C.fcuda_deepgemm_wkv_facts(&f) == 0
}

// V41FP8WKVQualified stays false until exact physical numerical evidence is returned.
func (c *cudaBackend) V41FP8WKVQualified() bool { return false }

func (c *cudaBackend) wkvRelease(b *cudaBuf) error {
	if b == nil || b.ptr == nil {
		return nil
	}
	if err := c.wkvStatus("release", C.fcuda_deepgemm_wkv_release(b.ptr)); err != nil {
		return err
	}
	b.ptr = nil
	return nil
}
func (w *cudaWKVWeights) Release() error {
	if w == nil {
		return nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if w.owner == nil {
		return ErrV41FP8WKVUnavailable
	}
	if err := w.owner.wkvRelease(w.codes); err != nil {
		return err
	}
	return w.owner.wkvRelease(w.scales)
}
func (c *cudaBackend) UploadV41FP8WKV(codes, tileScales []byte) (V41FP8WKVWeights, error) {
	if c == nil {
		return nil, ErrV41FP8WKVUnavailable
	}
	if len(codes) != 512*5120 {
		return nil, fmt.Errorf("WKV codes must be512x5120")
	}
	for _, v := range codes {
		if v&127 == 127 {
			return nil, fmt.Errorf("WKV E4M3 NaN")
		}
	}
	packed, err := packV41WKVScales(tileScales)
	if err != nil {
		return nil, err
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if err = c.faultLatch.Admit("deepgemm-wkv-upload"); err != nil {
		return nil, err
	}
	if err = c.wkvStatus("state", C.fcuda_deepgemm_wkv_state()); err != nil {
		return nil, err
	}
	w := &cudaWKVWeights{owner: c}
	w.codes, err = c.dallocDeviceOnlyClass(len(codes), MemoryWeights, "deepgemm-wkv-codes")
	if err != nil {
		return nil, err
	}
	w.scales, err = c.dallocDeviceOnlyClass(len(packed), MemoryWeights, "deepgemm-wkv-scales")
	if err != nil {
		return nil, errors.Join(err, c.wkvRelease(w.codes))
	}
	err = c.wkvStatus("codes-upload", C.fcuda_deepgemm_wkv_upload(w.codes.ptr, unsafe.Pointer(&codes[0]), C.size_t(len(codes))))
	if err == nil {
		err = c.wkvStatus("scales-upload", C.fcuda_deepgemm_wkv_upload(w.scales.ptr, unsafe.Pointer(&packed[0]), C.size_t(len(packed))))
	}
	runtime.KeepAlive(codes)
	runtime.KeepAlive(packed)
	if err != nil {
		return nil, errors.Join(err, c.wkvRelease(w.codes), c.wkvRelease(w.scales))
	}
	// Blocking H2D does not certify unrelated producer-stream buffers Ready.
	return w, nil
}
func (c *cudaBackend) V41FP8WKV(weights V41FP8WKVWeights, input Tensor, rows int) (Tensor, error) {
	var zero Tensor
	if c == nil || rows <= 0 || rows%128 != 0 || rows > int(^uint32(0)>>1)/5120 {
		return zero, ErrV41FP8WKVUnavailable
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if err := c.faultLatch.Admit("deepgemm-wkv"); err != nil {
		return zero, err
	}
	if err := c.wkvStatus("state", C.fcuda_deepgemm_wkv_state()); err != nil {
		return zero, err
	}
	w, ok := weights.(*cudaWKVWeights)
	if !ok || w == nil || w.owner != c || w.codes == nil || w.scales == nil || w.codes.ptr == nil || w.scales.ptr == nil {
		return zero, fmt.Errorf("WKV owner/live weights mismatch")
	}
	x, ok := input.buf.(*cudaBuf)
	if input.Backend() != c || input.Dtype != F32 || input.Layout != RowMajor || len(input.Shape) != 2 || input.Shape[0] != rows || input.Shape[1] != 5120 || !ok || x == nil || x.ptr == nil || x.managed || x.device != 0 || x.n < rows*5120*4 || x.invalidForSubmit() {
		return zero, fmt.Errorf("WKV requires owned device F32[M,5120]")
	}
	var bufs [4]*cudaBuf
	sizes := [4]int{rows * 5120, rows * 160, 4, rows * 512 * 2}
	cleanup := func() error {
		var out error
		for _, b := range bufs {
			out = errors.Join(out, c.wkvRelease(b))
		}
		return out
	}
	for i, n := range sizes {
		b, err := c.dallocDeviceOnlyClass(n, MemoryScratchpad, "deepgemm-wkv-operation")
		if err != nil {
			return zero, errors.Join(err, cleanup())
		}
		bufs[i] = b
	}
	rc := C.fcuda_deepgemm_wkv_run((*C.float)(x.ptr), (*C.uint8_t)(w.codes.ptr), (*C.uint32_t)(w.scales.ptr), (*C.uint8_t)(bufs[0].ptr), (*C.uint32_t)(bufs[1].ptr), (*C.uint32_t)(bufs[2].ptr), bufs[3].ptr, C.int(rows))
	if err := c.wkvStatus("launch", rc); err != nil {
		return zero, errors.Join(err, cleanup())
	}
	// Native success positively drained the actual producer stream.
	atomic.AddUint64(&c.fenceGen, 1)
	result := bufs[3]
	bufs[3] = nil
	if err := cleanup(); err != nil {
		return zero, errors.Join(err, c.wkvRelease(result))
	}
	return makeTensor(c, BF16, RowMajor, []int{rows, 512}, nil, result), nil
}
