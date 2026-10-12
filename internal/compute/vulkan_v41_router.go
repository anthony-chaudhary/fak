//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

var _ V41RouterTopKBackend = (*vulkanBackend)(nil)

// V41RouterTopKAvailable reports the optional router module. It is absent from
// the trusted shader registry, so a registry-built bundle never enables it.
func (v *vulkanBackend) V41RouterTopKAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_router_topk() != 0
}

// V41RouterTopK fences the route and reads back only the per-row picks,
// weights and status words. Logits stay resident; every output buffer is
// freed here. No public Upload, Read or Free method runs under vulkanMu.
func (v *vulkanBackend) V41RouterTopK(logits, bias Tensor, rows, experts, topK int, routeScale float32, full, hasBias bool) (picks V41RouterPicks, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var expertBuf, weightBuf, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RouterTopK")
		}
		for _, b := range []*vulkanBuf{expertBuf, weightBuf, statusBuf} {
			if b != nil && b.ptr != nil {
				C.fvk_free(b.ptr)
				b.ptr = nil
				b.n = 0
			}
		}
		if err != nil {
			picks = V41RouterPicks{}
		}
	}()
	if v == nil || C.fvk_have_v41_router_topk() == 0 {
		return V41RouterPicks{}, vulkanV41RouterStatusError(3)
	}
	logitBytes, biasBytes, pickBytes, statusBytes, validationErr := validateV41RouterTopK(logits, bias, rows, experts, topK, routeScale, hasBias)
	if validationErr != nil {
		return V41RouterPicks{}, &BackendError{Backend: "vulkan", Site: "V41RouterTopK", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error(), Recovered: validationErr}
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{
		{"logits", logits, logitBytes}, {"bias", bias, biasBytes},
	} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n != operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return V41RouterPicks{}, &BackendError{Backend: "vulkan", Site: "V41RouterTopK", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: operand.name + " must be a live, exact-sized F32 device buffer owned by this backend"}
		}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return V41RouterPicks{}, vulkanV41RouterStatusError(status)
	}
	expertBuf = v.dallocForClass(pickBytes, MemoryActivation, "V4.1 router experts")
	weightBuf = v.dallocForClass(pickBytes, MemoryActivation, "V4.1 router weights")
	statusBuf = v.dallocForClass(statusBytes, MemoryActivation, "V4.1 router status")
	var fullFlag, biasFlag C.int
	if full {
		fullFlag = 1
	}
	if hasBias {
		biasFlag = 1
	}
	status := int(C.fvk_v41_router_topk_f32(logits.buf.(*vulkanBuf).ptr, bias.buf.(*vulkanBuf).ptr, expertBuf.ptr, weightBuf.ptr, statusBuf.ptr,
		C.int(rows), C.int(experts), C.int(topK), fullFlag, biasFlag, C.float(routeScale)))
	if status != 0 {
		return V41RouterPicks{}, vulkanV41RouterStatusError(status)
	}
	// Status, experts and weights share one host fence (#13820).
	image, readErr := vulkanV41ReadMany("V41RouterTopK", []*vulkanBuf{statusBuf, expertBuf, weightBuf}, []int{statusBytes, pickBytes, pickBytes})
	if readErr != nil {
		return V41RouterPicks{}, readErr
	}
	words := make([]uint32, statusBytes/4)
	expertWords := make([]uint32, pickBytes/4)
	weights := make([]float32, pickBytes/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(image[0][4*i:])
	}
	for i := range expertWords {
		expertWords[i] = binary.LittleEndian.Uint32(image[1][4*i:])
		weights[i] = math.Float32frombits(binary.LittleEndian.Uint32(image[2][4*i:]))
	}
	picks, decodeErr := decodeV41RouterTopK(words, expertWords, weights, rows, experts, topK)
	var fault *V41RouterTopKFault
	if errors.As(decodeErr, &fault) {
		// A numeric verdict on the inputs, not a device failure.
		return V41RouterPicks{}, fault
	}
	if decodeErr != nil {
		return V41RouterPicks{}, &BackendError{Backend: "vulkan", Site: "V41RouterTopK", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: decodeErr.Error(), Recovered: decodeErr}
	}
	return picks, nil
}

func vulkanV41RouterStatusError(status int) *BackendError {
	class, cause := VulkanClassExecutionFailed, ErrVulkanExecutionFailed
	switch {
	case status == -4: // VK_ERROR_DEVICE_LOST
		class, cause = VulkanClassDeviceLost, ErrVulkanDeviceLost
	case status == -1 || status == -2:
		class, cause = VulkanClassAllocationFailed, ErrVulkanAllocationFailed
	case status < 0:
		class, cause = VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
	case status == 2:
		class, cause = VulkanClassInvalidGeometry, ErrVulkanInvalidGeometry
	}
	return &BackendError{Backend: "vulkan", Site: "V41RouterTopK", Class: class, Err: cause, Message: fmt.Sprintf("dedicated router top-k failed closed (code %d)", status)}
}
