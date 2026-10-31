//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
extern int fcuda_blockscaled_wkv_support(int rows, int *runtime_version, size_t *blas_version);
extern int fcuda_blockscaled_wkv_upload(void *dst, const void *src, size_t bytes);
extern int fcuda_blockscaled_wkv_release(void *ptr, size_t bytes);
extern int fcuda_blockscaled_wkv_run(const float *x, const uint8_t *w, const uint8_t *ws,
    uint8_t *a, uint32_t *recipe, uint8_t *tiled, uint32_t *status, float *y, int rows);
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// ErrBlockScaledGEMMUnsupported is a capability refusal, not a request to silently
// fall back after execution. No serving caller selects this experimental route.
var ErrBlockScaledGEMMUnsupported = errors.New("experimental SM120 blockscaled WKV GEMM unsupported")

// BlockScaledWKVCapability describes an admitted vendor heuristic, not numerical
// qualification. Qualified stays false until exact physical evidence is returned.
type BlockScaledWKVCapability struct {
	Rows, ComputeMajor, ComputeMinor int
	CUDARuntimeVersion               int
	CUBLASVersion                    uint64
	Qualified                        bool
}

// BlockScaledWKVWeights owns an immutable device copy; Release is explicit.
type BlockScaledWKVWeights interface{ Release() error }

// BlockScaledWKVBackend is the explicit, default-off cuBLASLt baseline for the
// V4.1 dense wkv_b shape, separate from the SM100 DeepGEMM/BF16 bridge. Enable
// FAK_CUDA_BLOCKSCALED_WKV=1 before probing/uploading/running. Only SM120,
// CUDA >=12.9 and cuBLAS >=12.9.1 with a matching heuristic are admitted.
//
// Weights are row-major E4M3[512,5120] and E8M0[16,160] (one scale per 32x32
// weight tile). Input is owned device F32[P,5120], P in {128,512,4096}; the
// existing recipe32 SIMT cast rounds it to E4M3 with per-32 E8M0 scales. Result
// is owned device F32[P,512], freed via Backend.Free. Two 4-byte status reads
// and producer-stream fences are deliberate baseline costs, not zero-transfer
// or graph-capture claims. No operand matrix is staged through the host.
type BlockScaledWKVBackend interface {
	Backend
	BlockScaledWKVSupported(rows int) (BlockScaledWKVCapability, error)
	UploadBlockScaledWKV(codes, tileScales []byte) (BlockScaledWKVWeights, error)
	BlockScaledWKV(weights BlockScaledWKVWeights, input Tensor, rows int) (Tensor, error)
}

var _ BlockScaledWKVBackend = (*cudaBackend)(nil)

type cudaBlockScaledWKVWeights struct {
	owner         *cudaBackend
	codes, scales *cudaBuf
	latch         *DeviceFaultLatch
	faults        int
}

func blockScaledWKVOptIn(rows int) error {
	if os.Getenv("FAK_CUDA_BLOCKSCALED_WKV") != "1" || (rows != 128 && rows != 512 && rows != 4096) {
		return fmt.Errorf("%w: require FAK_CUDA_BLOCKSCALED_WKV=1 and P=128,512,4096", ErrBlockScaledGEMMUnsupported)
	}
	return nil
}

// Status values match cuda/blockscaled_wkv.cuh; positive values are execution
// or CUDA/cuBLAS API faults, never a capability-based fallback after mutation.
func (c *cudaBackend) blockScaledWKVStatus(site string, rc C.int) error {
	if rc == 0 {
		return nil
	}
	var cause error
	msg := "native boundary refused the operation"
	switch rc {
	case -72001, -72002, -72004:
		cause = ErrBlockScaledGEMMUnsupported
		msg = "require opted-in SM120, CUDA 12.9/cuBLAS 12.9.1+, exact shape, uncaptured stream and vendor heuristic"
	case -72003:
		msg = "require disjoint owned device-0 allocations with sufficient extent and alignment"
	case -72005:
		msg = "nonfinite input or result; output withheld"
	case -72006:
		msg = "shared CUDA stream is terminally uncertain; storage retained"
	default:
		if rc > 0 {
			cause = c.faultLatch.Observe(DeviceFaultExecution, "blockscaled-wkv-"+site, int(rc))
		}
	}
	return &CUDAOpError{Op: "BlockScaledWKV", Site: site, Code: int(rc), Msg: msg, Err: cause}
}

// Caller holds cudaMu. Admission is repeated before every operation, before its
// device reservations or operand mutation; no stale capability result is cached.
func (c *cudaBackend) blockScaledWKVSupport(rows int) (BlockScaledWKVCapability, error) {
	var zero BlockScaledWKVCapability
	if c == nil {
		return zero, ErrBlockScaledGEMMUnsupported
	}
	if err := blockScaledWKVOptIn(rows); err != nil {
		return zero, err
	}
	if c.capturing {
		return zero, fmt.Errorf("%w: capture is not supported", ErrBlockScaledGEMMUnsupported)
	}
	if err := c.faultLatch.Admit("blockscaled-wkv"); err != nil {
		return zero, err
	}
	if cudaQ4KAsyncReleaseBlockedLocked() {
		return zero, c.blockScaledWKVStatus("peer-owner", C.int(-72006))
	}
	var rt C.int
	var lt C.size_t
	if err := c.blockScaledWKVStatus("support", C.fcuda_blockscaled_wkv_support(C.int(rows), &rt, &lt)); err != nil {
		return zero, err
	}
	return BlockScaledWKVCapability{Rows: rows, ComputeMajor: 12, ComputeMinor: 0,
		CUDARuntimeVersion: int(rt), CUBLASVersion: uint64(lt), Qualified: false}, nil
}

func (c *cudaBackend) BlockScaledWKVSupported(rows int) (BlockScaledWKVCapability, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.blockScaledWKVSupport(rows)
}

func (c *cudaBackend) blockScaledWKVRelease(b *cudaBuf) error {
	if b == nil || b.ptr == nil {
		return nil
	}
	if cudaQ4KAsyncReleaseBlockedLocked() {
		return c.blockScaledWKVStatus("release-peer-owner", C.int(-72006))
	}
	if err := c.blockScaledWKVStatus("release", C.fcuda_blockscaled_wkv_release(b.ptr, C.size_t(b.n))); err != nil {
		return err // Native live record remains owned, never recycled on uncertainty.
	}
	b.ptr = nil
	return nil
}

func (w *cudaBlockScaledWKVWeights) Release() error {
	if w == nil {
		return nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if (w.codes == nil || w.codes.ptr == nil) && (w.scales == nil || w.scales.ptr == nil) {
		return nil
	}
	if w.owner == nil || w.owner.faultLatch != w.latch || w.latch.faultCount() != w.faults {
		return fmt.Errorf("blockscaled WKV weights are stale; storage retained")
	}
	// Disabling the opt-in does not disable release of a still-valid owner.
	return errors.Join(w.owner.blockScaledWKVRelease(w.codes), w.owner.blockScaledWKVRelease(w.scales))
}

func (c *cudaBackend) UploadBlockScaledWKV(codes, tileScales []byte) (BlockScaledWKVWeights, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if _, err := c.blockScaledWKVSupport(128); err != nil {
		return nil, err
	}
	if len(codes) != 512*5120 || len(tileScales) != 16*160 {
		return nil, fmt.Errorf("blockscaled WKV requires E4M3[512,5120] and E8M0[16,160]")
	}
	for _, v := range codes {
		if v&127 == 127 {
			return nil, fmt.Errorf("blockscaled WKV E4M3 NaN weight")
		}
	}
	for _, v := range tileScales {
		if v == 255 {
			return nil, fmt.Errorf("blockscaled WKV E8M0 NaN scale")
		}
	}
	// cuBLAS 12.9.1 section 3.1.4.3.2 uses the same byte permutation as the
	// existing NVFP4 layout helper, but K/32 (160), NOT K/16 or recipe32 order.
	// Replicate a weight tile's exponent across its 32 rows; never requantize.
	packed := make([]byte, 512*160)
	for row := 0; row < 512; row++ {
		for kb := 0; kb < 160; kb++ {
			packed[nvfp4ScaleLayoutOffset(row, kb, 40)] = tileScales[(row/32)*160+kb]
		}
	}
	w := &cudaBlockScaledWKVWeights{owner: c, latch: c.faultLatch, faults: c.faultLatch.faultCount()}
	var err error
	w.codes, err = c.dallocDeviceOnlyClass(len(codes), MemoryWeights, "blockscaled-wkv-codes")
	if err != nil {
		return nil, err
	}
	w.scales, err = c.dallocDeviceOnlyClass(len(packed), MemoryWeights, "blockscaled-wkv-scales")
	if err != nil {
		return nil, errors.Join(err, c.blockScaledWKVRelease(w.codes))
	}
	err = c.blockScaledWKVStatus("codes-upload", C.fcuda_blockscaled_wkv_upload(w.codes.ptr, unsafe.Pointer(&codes[0]), C.size_t(len(codes))))
	if err == nil {
		err = c.blockScaledWKVStatus("scales-upload", C.fcuda_blockscaled_wkv_upload(w.scales.ptr, unsafe.Pointer(&packed[0]), C.size_t(len(packed))))
	}
	runtime.KeepAlive(codes)
	runtime.KeepAlive(packed)
	if err != nil {
		return nil, errors.Join(err, c.blockScaledWKVRelease(w.codes), c.blockScaledWKVRelease(w.scales))
	}
	return w, nil
}

func (c *cudaBackend) BlockScaledWKV(weights BlockScaledWKVWeights, input Tensor, rows int) (Tensor, error) {
	var zero Tensor
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if _, err := c.blockScaledWKVSupport(rows); err != nil {
		return zero, err
	}
	w, ok := weights.(*cudaBlockScaledWKVWeights)
	if !ok || w == nil || w.owner != c || w.codes == nil || w.scales == nil || w.codes.ptr == nil || w.scales.ptr == nil || w.latch != c.faultLatch || w.faults != c.faultLatch.faultCount() {
		return zero, fmt.Errorf("blockscaled WKV owner/live weight epoch mismatch")
	}
	x, ok := input.buf.(*cudaBuf)
	if input.Backend() != c || input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil || len(input.Shape) != 2 || input.Shape[0] != rows || input.Shape[1] != 5120 || !ok || x == nil || x.ptr == nil || x.managed || x.device != 0 || x.scales != nil || x.n < rows*5120*4 || x.invalidForSubmit() {
		return zero, fmt.Errorf("blockscaled WKV requires owned device F32[P,5120]")
	}
	if _, pending := cudaAsyncPending[x.ptr]; pending {
		return zero, fmt.Errorf("blockscaled WKV input upload fence must be satisfied before submission")
	}
	if cudaV41CompressorPendingIndex(x) >= 0 {
		return zero, fmt.Errorf("blockscaled WKV input is retained by a pending compressor owner")
	}
	var bufs [5]*cudaBuf
	sizes := [5]int{rows * 5120, rows * 160, rows * 160, 4, rows * 512 * 4}
	cleanup := func() error {
		var err error
		for _, b := range bufs {
			err = errors.Join(err, c.blockScaledWKVRelease(b))
		}
		return err
	}
	for i, n := range sizes {
		b, err := c.dallocDeviceOnlyClass(n, MemoryScratchpad, "blockscaled-wkv-operation")
		if err != nil {
			return zero, errors.Join(err, cleanup())
		}
		bufs[i] = b
	}
	rc := C.fcuda_blockscaled_wkv_run((*C.float)(x.ptr), (*C.uint8_t)(w.codes.ptr), (*C.uint8_t)(w.scales.ptr),
		(*C.uint8_t)(bufs[0].ptr), (*C.uint32_t)(bufs[1].ptr), (*C.uint8_t)(bufs[2].ptr), (*C.uint32_t)(bufs[3].ptr), (*C.float)(bufs[4].ptr), C.int(rows))
	runtime.KeepAlive(input)
	runtime.KeepAlive(weights)
	if err := c.blockScaledWKVStatus("run", rc); err != nil {
		return zero, errors.Join(err, cleanup())
	}
	// Native success positively drained the actual producer stream, not stream 0.
	atomic.AddUint64(&c.fenceGen, 1)
	result := bufs[4]
	bufs[4] = nil
	if err := cleanup(); err != nil {
		return zero, errors.Join(err, c.blockScaledWKVRelease(result))
	}
	return makeTensor(c, F32, RowMajor, []int{rows, 512}, nil, result), nil
}
