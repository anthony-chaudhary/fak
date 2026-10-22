//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"unsafe"
)

var _ V41RouterGateTopKBackend = (*vulkanBackend)(nil)

// Device verdicts held by the shim for the life of the device.
const (
	vulkanV41RouterGateUnknown  = 0
	vulkanV41RouterGateEqual    = 1
	vulkanV41RouterGateDiffered = 2
)

// V41RouterGateTopKAvailable reports the fused module only once its device
// self-check found it bit-identical to MatMul + V41RouterTopK. The check runs
// at most once per device, outside any foreign batch; a device fault during it
// records no verdict and reports false.
func (v *vulkanBackend) V41RouterGateTopKAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if C.fvk_have_v41_router_gate_topk() == 0 || C.fvk_have_v41_router_topk() == 0 {
		return false
	}
	switch int(C.fvk_v41_router_gate_verdict()) {
	case vulkanV41RouterGateEqual:
		return true
	case vulkanV41RouterGateDiffered:
		return false
	}
	if C.fvk_submission_status() != 0 || bool(C.fvk_batch_foreign_active()) {
		return false
	}
	equal, err := v.v41RouterGateSelfCheckLocked()
	if err != nil {
		return false
	}
	verdict := vulkanV41RouterGateDiffered
	if equal {
		verdict = vulkanV41RouterGateEqual
	}
	C.fvk_v41_router_gate_set_verdict(C.int(verdict))
	return equal
}

// V41RouterGateTopK projects, routes and reads back only the picks, weights
// and status words behind one fence. Every output buffer is freed here.
func (v *vulkanBackend) V41RouterGateTopK(weight, x, bias Tensor, experts, hidden, topK int, routeScale float32, full, hasBias bool) (V41RouterPicks, error) {
	picks, _, err := v.v41RouterGateTopK(weight, x, bias, experts, hidden, topK, routeScale, full, hasBias, false)
	return picks, err
}

// v41RouterGateTopK also returns the device logits when keepLogits is set,
// for witnesses only; that adds one region to the same readback.
func (v *vulkanBackend) v41RouterGateTopK(weight, x, bias Tensor, experts, hidden, topK int, routeScale float32, full, hasBias, keepLogits bool) (picks V41RouterPicks, logits []float32, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var logitBuf, expertBuf, weightBuf, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RouterGateTopK")
		}
		v.freeV41RouterGateLocked(logitBuf, expertBuf, weightBuf, statusBuf)
		if err != nil {
			picks, logits = V41RouterPicks{}, nil
		}
	}()
	if v == nil || C.fvk_have_v41_router_gate_topk() == 0 || int(C.fvk_v41_router_gate_verdict()) != vulkanV41RouterGateEqual {
		return V41RouterPicks{}, nil, vulkanV41RouterGateStatusError(3)
	}
	weightBytes, xBytes, biasBytes, logitBytes, pickBytes, validationErr := validateV41RouterGateTopK(weight, x, bias, experts, hidden, topK, routeScale, hasBias)
	if validationErr != nil {
		return V41RouterPicks{}, nil, &BackendError{Backend: "vulkan", Site: "V41RouterGateTopK", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error(), Recovered: validationErr}
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{
		{"weight", weight, weightBytes}, {"x", x, xBytes}, {"bias", bias, biasBytes},
	} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n != operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return V41RouterPicks{}, nil, &BackendError{Backend: "vulkan", Site: "V41RouterGateTopK", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: operand.name + " must be a live, exact-sized F32 device buffer owned by this backend"}
		}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return V41RouterPicks{}, nil, vulkanV41RouterGateStatusError(status)
	}
	logitBuf = v.dallocForClass(logitBytes, MemoryActivation, "V4.1 router gate logits")
	expertBuf = v.dallocForClass(pickBytes, MemoryActivation, "V4.1 router gate experts")
	weightBuf = v.dallocForClass(pickBytes, MemoryActivation, "V4.1 router gate weights")
	statusBuf = v.dallocForClass(16, MemoryActivation, "V4.1 router gate status")
	status := v.v41RouterGateDispatchLocked(weight.buf.(*vulkanBuf), x.buf.(*vulkanBuf), bias.buf.(*vulkanBuf), logitBuf, expertBuf, weightBuf, statusBuf, experts, hidden, topK, routeScale, full, hasBias)
	if status != 0 {
		return V41RouterPicks{}, nil, vulkanV41RouterGateStatusError(status)
	}
	srcs := []*vulkanBuf{statusBuf, expertBuf, weightBuf}
	sizes := []int{16, pickBytes, pickBytes}
	if keepLogits {
		srcs, sizes = append(srcs, logitBuf), append(sizes, logitBytes)
	}
	image, readErr := vulkanV41ReadMany("V41RouterGateTopK", srcs, sizes)
	if readErr != nil {
		return V41RouterPicks{}, nil, readErr
	}
	picks, decodeErr := decodeV41RouterGateImage(image, experts, topK)
	var fault *V41RouterTopKFault
	if errors.As(decodeErr, &fault) {
		return V41RouterPicks{}, nil, fault
	}
	if decodeErr != nil {
		return V41RouterPicks{}, nil, &BackendError{Backend: "vulkan", Site: "V41RouterGateTopK", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: decodeErr.Error(), Recovered: decodeErr}
	}
	if keepLogits {
		logits = vulkanLEFloat32s(image[3])
	}
	return picks, logits, nil
}

// v41RouterGateDispatchLocked seeds the status words and records the fused
// dispatch. Caller holds vulkanMu and owns every buffer.
func (v *vulkanBackend) v41RouterGateDispatchLocked(weight, x, bias, logits, experts, weights, status *vulkanBuf, numExperts, hidden, topK int, routeScale float32, full, hasBias bool) int {
	seed := v41RouterGateStatusSeed
	C.fvk_h2d(status.ptr, unsafe.Pointer(&seed[0]), C.size_t(len(seed)*4))
	if code := int(C.fvk_submission_status()); code != 0 {
		return code
	}
	var fullFlag, biasFlag C.int
	if full {
		fullFlag = 1
	}
	if hasBias {
		biasFlag = 1
	}
	return int(C.fvk_v41_router_gate_topk_f32(weight.ptr, x.ptr, bias.ptr, logits.ptr, experts.ptr, weights.ptr, status.ptr,
		C.int(numExperts), C.int(hidden), C.int(topK), fullFlag, biasFlag, C.float(routeScale)))
}

// vulkanV41RouterGateModule, vulkanV41RouterGateVerdict and
// vulkanSubmissionStatusForTest expose native state to the device witnesses.
func vulkanV41RouterGateModule() int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_have_v41_router_gate_topk())
}

func vulkanV41RouterGateVerdict() int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_v41_router_gate_verdict())
}

func (v *vulkanBackend) v41RouterGateSelfCheckImage() ([][]byte, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.v41RouterGateSelfCheckImageLocked()
}

func vulkanSubmissionStatusForTest() int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_submission_status())
}

func (v *vulkanBackend) freeV41RouterGateLocked(bufs ...*vulkanBuf) {
	for _, b := range bufs {
		if b != nil && b.ptr != nil {
			C.fvk_free(b.ptr)
			b.ptr = nil
			b.n = 0
		}
	}
}

func decodeV41RouterGateImage(image [][]byte, experts, topK int) (V41RouterPicks, error) {
	words := make([]uint32, 4)
	expertWords := make([]uint32, topK)
	weights := make([]float32, topK)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(image[0][4*i:])
	}
	for i := range expertWords {
		expertWords[i] = binary.LittleEndian.Uint32(image[1][4*i:])
		weights[i] = math.Float32frombits(binary.LittleEndian.Uint32(image[2][4*i:]))
	}
	return decodeV41RouterTopK(words, expertWords, weights, 1, experts, topK)
}

func vulkanLEFloat32s(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func vulkanV41RouterGateStatusError(status int) *BackendError {
	err := vulkanV41RouterStatusError(status)
	err.Site = "V41RouterGateTopK"
	err.Message = fmt.Sprintf("fused router gate top-k failed closed (code %d)", status)
	return err
}

// v41RouterGateSelfCheckLocked runs the fused module and MatMul followed by
// the router top-k on one synthetic row and compares every logit, pick,
// weight and status bit. The row has three tied leading experts, a partial
// last chunk and more experts than lanes. Caller holds vulkanMu.
func (v *vulkanBackend) v41RouterGateSelfCheckLocked() (bool, error) {
	image, err := v.v41RouterGateSelfCheckImageLocked()
	if err != nil {
		return false, err
	}
	for i := 0; i < 4; i++ {
		if !bytes.Equal(image[i], image[4+i]) {
			return false, nil
		}
	}
	// A faulted reference exercised no route, so it cannot admit the module.
	return bytes.Equal(image[3], make([]byte, 16)), nil
}

// v41RouterGateSelfCheckImageLocked returns the reference logits, experts,
// weights and status, then the fused ones, from one readback.
func (v *vulkanBackend) v41RouterGateSelfCheckImageLocked() (image [][]byte, err error) {
	const experts, hidden, topK = 72, 3*1024 + 40, 6
	const routeScale = float32(1.5)
	w, x, bias := v41RouterGateProbeFixture(experts, hidden)
	var owned []*vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			image, err = nil, ClassifyVulkanPanic(recovered, "V41RouterGateTopK self-check")
		}
		v.freeV41RouterGateLocked(owned...)
	}()
	upload := func(values []float32, what string) *vulkanBuf {
		b := v.dallocForClass(len(values)*4, MemoryActivation, what)
		owned = append(owned, b)
		C.fvk_h2d(b.ptr, unsafe.Pointer(&values[0]), C.size_t(len(values)*4))
		return b
	}
	alloc := func(n int, what string) *vulkanBuf {
		b := v.dallocForClass(n, MemoryActivation, what)
		owned = append(owned, b)
		return b
	}
	wb := upload(w, "V4.1 router gate self-check weight")
	xb := upload(x, "V4.1 router gate self-check input")
	bb := upload(bias, "V4.1 router gate self-check bias")
	refLogits := alloc(experts*4, "V4.1 router gate self-check reference logits")
	refExperts, refWeights, refStatus := alloc(topK*4, "V4.1 router gate self-check reference experts"), alloc(topK*4, "V4.1 router gate self-check reference weights"), alloc(16, "V4.1 router gate self-check reference status")
	logits, gotExperts, gotWeights, gotStatus := alloc(experts*4, "V4.1 router gate self-check logits"), alloc(topK*4, "V4.1 router gate self-check experts"), alloc(topK*4, "V4.1 router gate self-check weights"), alloc(16, "V4.1 router gate self-check status")
	C.fvk_matmul_f32(wb.ptr, xb.ptr, refLogits.ptr, C.int(experts), C.int(hidden), 1)
	if code := int(C.fvk_submission_status()); code != 0 {
		return nil, vulkanV41RouterGateStatusError(code)
	}
	if code := int(C.fvk_v41_router_topk_f32(refLogits.ptr, bb.ptr, refExperts.ptr, refWeights.ptr, refStatus.ptr, 1, C.int(experts), C.int(topK), 1, 1, C.float(routeScale))); code != 0 {
		return nil, vulkanV41RouterGateStatusError(code)
	}
	if code := v.v41RouterGateDispatchLocked(wb, xb, bb, logits, gotExperts, gotWeights, gotStatus, experts, hidden, topK, routeScale, true, true); code != 0 {
		return nil, vulkanV41RouterGateStatusError(code)
	}
	sizes := []int{experts * 4, topK * 4, topK * 4, 16}
	return vulkanV41ReadMany("V41RouterGateTopK self-check", []*vulkanBuf{refLogits, refExperts, refWeights, refStatus, logits, gotExperts, gotWeights, gotStatus}, append(sizes, sizes...))
}
