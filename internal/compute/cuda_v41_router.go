//go:build cuda && cgo && fak_cuda_v41_router

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
// Candidate-local ABI from cuda_v41_router.cu (always compiled; this tag is the opt-in).
int fak_v41_router_state(void);
int fak_v41_router_topk_f32(const float *, const float *, const int *, int *, unsigned *,
    unsigned *, float *, unsigned *, int, int, int, int, int, int, float);
uint64_t fak_v41_router_completed(void);
int fak_qwen35_embedding_gather_f32(const float *, const int *, int *, float *, int, int);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var (
	_ V41RouterTopKBackend      = (*cudaBackend)(nil)
	_ V41RouterHashBackend      = (*cudaBackend)(nil)
	_ V41RouterGateTopKBackend  = (*cudaBackend)(nil)
	_ V41EmbeddingGatherBackend = (*cudaBackend)(nil)
)

// The CUDA router (ds4 router_select_warp_topk_kernel adapted to fak's
// contract, see cuda_v41_router.cu) and the F32 embedding gather. Compiling the
// tag is the opt-in; the default cuda build lacks the method set, so the model
// keeps the host route and gather.
func (c *cudaBackend) V41RouterTopKAvailable() bool      { return c.cudaV41RouterAvailable() }
func (c *cudaBackend) V41RouterHashAvailable() bool      { return c.cudaV41RouterAvailable() }
func (c *cudaBackend) V41RouterGateTopKAvailable() bool  { return c.cudaV41RouterAvailable() }
func (c *cudaBackend) V41EmbeddingGatherAvailable() bool { return c.cudaV41RouterAvailable() }

func (c *cudaBackend) cudaV41RouterAvailable() bool {
	if c == nil {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.cudaV41RouterState("V41RouterAvailable") == nil
}

func cudaV41RouterError(site, message string, status int, cause error) error {
	return &BackendError{Backend: "cuda", Site: site, Class: "cuda-v41-router-failed",
		Message: fmt.Sprintf("%s (status %d)", message, status), Err: cause, Recovered: cause}
}

// Caller holds cudaMu. Faults take precedence over native admission.
func (c *cudaBackend) cudaV41RouterState(site string) error {
	if c == nil || c.capturing {
		return cudaV41RouterError(site, "nil backend or active capture", -3, nil)
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return cudaV41RouterError(site, "sticky CUDA fault", -3, err)
	}
	if status := int(C.fak_v41_router_state()); status != 0 {
		return cudaV41RouterError(site, "native admission refused", status, nil)
	}
	return nil
}

// cudaV41RouterOperand admits only live device-0, non-managed, unscaled
// storage owned by c of at least bytes. Caller holds cudaMu.
func (c *cudaBackend) cudaV41RouterOperand(site, name string, t Tensor, bytes int) (*cudaBuf, error) {
	b, ok := t.buf.(*cudaBuf)
	if t.be != c || !ok || b == nil || b.ptr == nil || b.n < bytes || b.managed || b.device != 0 ||
		b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
		return nil, cudaV41RouterError(site, name+" is not owned device-only F32 storage", -2, nil)
	}
	if invalid := b.invalidStateError(site + " " + name); invalid != nil {
		return nil, cudaV41RouterError(site, "invalid storage lifetime", -2, invalid)
	}
	return b, nil
}

func (c *cudaBackend) V41RouterTopK(logits, bias Tensor, rows, experts, topK int, routeScale float32, full, hasBias bool) (V41RouterPicks, error) {
	const site = "V41RouterTopK"
	logitBytes, biasBytes, pickBytes, statusBytes, err := validateV41RouterTopK(logits, bias, rows, experts, topK, routeScale, hasBias)
	if err != nil {
		return V41RouterPicks{}, cudaV41RouterError(site, "invalid geometry", 2, err)
	}
	return c.cudaV41Route(site, logits, bias, nil, logitBytes, biasBytes, pickBytes, statusBytes, rows, experts, topK, routeScale, full, hasBias)
}

func (c *cudaBackend) V41RouterHash(logits Tensor, expertIDs []int32, rows, experts, topK int, routeScale float32, full bool) (V41RouterPicks, error) {
	const site = "V41RouterHash"
	logitBytes, pickBytes, statusBytes, err := validateV41RouterHash(logits, expertIDs, rows, experts, topK, routeScale)
	if err != nil {
		return V41RouterPicks{}, cudaV41RouterError(site, "invalid geometry", 2, err)
	}
	return c.cudaV41Route(site, logits, Tensor{}, expertIDs, logitBytes, 0, pickBytes, statusBytes, rows, experts, topK, routeScale, full, false)
}

// cudaV41Route runs one checked route. hash != nil selects the table mode, in
// which bias is absent and the logits buffer stands in for the unused operand.
func (c *cudaBackend) cudaV41Route(site string, logits, bias Tensor, hash []int32, logitBytes, biasBytes, pickBytes, statusBytes, rows, experts, topK int, routeScale float32, full, hasBias bool) (picks V41RouterPicks, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var scratch, hashBuf *cudaBuf
	leak := false
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "router")
			err = cudaV41RouterError(site, "candidate panicked", -1, cause)
		}
		// An uncertain drain may leave the kernel writing scratch: never pool it.
		for _, b := range []*cudaBuf{scratch, hashBuf} {
			if b != nil && b.ptr != nil && !leak {
				C.fcuda_free(b.ptr)
				b.ptr, b.n = nil, 0
			}
		}
		if err != nil {
			picks = V41RouterPicks{}
		}
	}()
	if err = c.cudaV41RouterState(site); err != nil {
		return V41RouterPicks{}, err
	}
	logitBuf, err := c.cudaV41RouterOperand(site, "logits", logits, logitBytes)
	if err != nil {
		return V41RouterPicks{}, err
	}
	biasBuf := logitBuf
	if hash == nil {
		if biasBuf, err = c.cudaV41RouterOperand(site, "bias", bias, biasBytes); err != nil {
			return V41RouterPicks{}, err
		}
	}
	scratch, err = c.dallocDeviceOnlyClass(2*pickBytes+statusBytes, MemoryActivation, "v41-router-picks")
	if err != nil {
		return V41RouterPicks{}, cudaV41RouterError(site, "device-only scratch allocation failed", -2, err)
	}
	var hashHost *C.int
	var hashDev *C.int
	hashMode := C.int(0)
	if hash != nil {
		if hashBuf, err = c.dallocDeviceOnlyClass(pickBytes, MemoryScratchpad, "v41-router-hash-ids"); err != nil {
			return V41RouterPicks{}, cudaV41RouterError(site, "device-only id allocation failed", -2, err)
		}
		hashHost, hashDev, hashMode = (*C.int)(unsafe.Pointer(&hash[0])), (*C.int)(hashBuf.ptr), 1
	}
	status := make([]uint32, statusBytes/4)
	experts32 := make([]uint32, pickBytes/4)
	weights := make([]float32, pickBytes/4)
	var fullFlag, biasFlag C.int
	if full {
		fullFlag = 1
	}
	if hasBias {
		biasFlag = 1
	}
	rc := int(C.fak_v41_router_topk_f32((*C.float)(logitBuf.ptr), (*C.float)(biasBuf.ptr), hashHost, hashDev,
		(*C.unsigned)(scratch.ptr), (*C.unsigned)(unsafe.Pointer(&experts32[0])), (*C.float)(unsafe.Pointer(&weights[0])),
		(*C.unsigned)(unsafe.Pointer(&status[0])), C.int(rows), C.int(experts), C.int(topK), fullFlag, biasFlag, hashMode, C.float(routeScale)))
	runtime.KeepAlive(hash)
	if rc != 0 {
		var cause error
		if rc >= 30000 {
			leak = rc >= 40000
			cause = c.faultLatch.Observe(DeviceFaultExecution, site, rc)
		}
		return V41RouterPicks{}, cudaV41RouterError(site, "checked CUDA router refused", rc, cause)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	picks, decodeErr := decodeV41RouterTopK(status, experts32, weights, rows, experts, topK)
	var fault *V41RouterTopKFault
	if errors.As(decodeErr, &fault) {
		return V41RouterPicks{}, fault // a numeric verdict on the inputs, not a device failure
	}
	if decodeErr != nil {
		return V41RouterPicks{}, cudaV41RouterError(site, decodeErr.Error(), 1, decodeErr)
	}
	return picks, nil
}

// V41RouterGateTopK projects the gate with this backend's own one-token F32
// MatMul, so the logits equal that path by construction, then routes them on
// the device. Only the picks and status words leave the device.
func (c *cudaBackend) V41RouterGateTopK(weight, x, bias Tensor, experts, hidden, topK int, routeScale float32, full, hasBias bool) (V41RouterPicks, error) {
	const site = "V41RouterGateTopK"
	weightBytes, xBytes, _, _, _, err := validateV41RouterGateTopK(weight, x, bias, experts, hidden, topK, routeScale, hasBias)
	if err != nil {
		return V41RouterPicks{}, cudaV41RouterError(site, "invalid geometry", 2, err)
	}
	cudaMu.Lock()
	for _, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{{"weight", weight, weightBytes}, {"x", x, xBytes}} {
		if _, opErr := c.cudaV41RouterOperand(site, operand.name, operand.t, operand.bytes); opErr != nil {
			cudaMu.Unlock()
			return V41RouterPicks{}, opErr
		}
	}
	cudaMu.Unlock()
	logits := c.MatMul(weight, x)
	defer c.Free(logits)
	row := makeTensor(c, F32, RowMajor, []int{1, experts}, nil, logits.buf)
	return c.V41RouterTopK(row, bias, 1, experts, topK, routeScale, full, hasBias)
}

// V41EmbeddingGather reuses the resident-sequence gather kernel: each output
// row is a byte copy of its table row, in id order with repetitions.
func (c *cudaBackend) V41EmbeddingGather(table Tensor, ids []int, hidden int) (out Tensor, err error) {
	const site = "V41EmbeddingGather"
	rowBytes, err := validateV41EmbeddingGather(table, ids, hidden)
	if err != nil {
		return Tensor{}, cudaV41RouterError(site, "invalid geometry", 2, err)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output, idBuf *cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "embedding")
			err = cudaV41RouterError(site, "candidate panicked", -1, cause)
		}
		if idBuf != nil && idBuf.ptr != nil {
			C.fcuda_free(idBuf.ptr)
		}
		if err != nil {
			if output != nil && output.ptr != nil {
				C.fcuda_free(output.ptr)
			}
			out = Tensor{}
		}
	}()
	if err = c.cudaV41RouterState(site); err != nil {
		return Tensor{}, err
	}
	tableBuf, err := c.cudaV41RouterOperand(site, "table", table, table.Shape[0]*rowBytes)
	if err != nil {
		return Tensor{}, err
	}
	if output, err = c.dallocDeviceOnlyClass(len(ids)*rowBytes, MemoryActivation, "v41-embedding-gather-output"); err != nil {
		return Tensor{}, cudaV41RouterError(site, "device-only output allocation failed", -2, err)
	}
	if idBuf, err = c.dallocDeviceOnlyClass(len(ids)*4, MemoryScratchpad, "v41-embedding-gather-ids"); err != nil {
		return Tensor{}, cudaV41RouterError(site, "device-only id allocation failed", -2, err)
	}
	hostIDs := make([]C.int, len(ids))
	for i, id := range ids {
		hostIDs[i] = C.int(id)
	}
	rc := int(C.fak_qwen35_embedding_gather_f32((*C.float)(tableBuf.ptr), &hostIDs[0], (*C.int)(idBuf.ptr),
		(*C.float)(output.ptr), C.int(len(ids)), C.int(hidden)))
	if rc != 0 {
		var cause error
		if rc >= 20000 {
			cause = c.faultLatch.Observe(DeviceFaultExecution, site, rc)
		}
		return Tensor{}, cudaV41RouterError(site, "checked CUDA gather refused", rc, cause)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return makeTensor(c, F32, RowMajor, []int{len(ids), hidden}, nil, output), nil
}

func cudaV41RouterCompleted() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fak_v41_router_completed())
}
