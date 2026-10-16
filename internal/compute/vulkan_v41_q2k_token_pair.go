//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41Q2KTokenPairBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41Q2KTokenPair(gate, up Tensor, rows int) bool {
	if v == nil || gate.Dtype != Q2_K || up.Dtype != Q2_K || gate.Layout != RowMajor || up.Layout != RowMajor || len(gate.Shape) != 2 || len(up.Shape) != 2 || gate.Shape[0] != up.Shape[0] || gate.Shape[1] != up.Shape[1] {
		return false
	}
	p, err := PlanV41Q2KTokenPair(gate.Shape[0], gate.Shape[1], rows)
	if err != nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	g, gok := v.ownedVulkanBuf(gate)
	u, uok := v.ownedVulkanBuf(up)
	return gok && uok && g.n >= p.WeightBytes && u.n >= p.WeightBytes && !bool(C.fvk_batch_foreign_active()) && C.fvk_have_v41_q2k_token_pair() != 0 && C.fvk_v41_q2k_token_pair_shape(C.int(p.Out), C.int(p.In), C.int(p.Rows)) != 0
}

// The pair is a single selected transport operation. Both outputs are allocated
// before dispatch; a second allocation failure may precede work that the old
// sequential gate call would already have recorded. It never retries either arm.
func (v *vulkanBackend) V41Q2KTokenPair(gate, up, input Tensor, rows int) (gateOut, upOut Tensor, err error) {
	const site = "V41Q2KTokenPair"
	p, invalid := validateV41Q2KTokenPair(gate, up, input, rows)
	if invalid != nil {
		return Tensor{}, Tensor{}, vulkanV41RoutedIDError(site, 2, invalid.Error())
	}
	if v == nil {
		return Tensor{}, Tensor{}, vulkanV41RoutedIDError(site, 3, "nil backend")
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var gOut, uOut *vulkanBuf
	completed := false
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, site)
			}
		}
		if !completed {
			if err == nil {
				err = ClassifyVulkanPanic("incomplete paired contraction", site)
			}
			gateOut, upOut = Tensor{}, Tensor{}
			release := func(b *vulkanBuf) { defer func() { _ = recover() }(); v.freeDeferredLocked(b) }
			defer release(uOut)
			defer release(gOut)
		}
	}()
	if C.fvk_have_v41_q2k_token_pair() == 0 {
		return Tensor{}, Tensor{}, vulkanV41RoutedIDError(site, 3, "optional token-pair module unavailable")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, Tensor{}, vulkanV41RoutedIDError(site, status, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() || C.fvk_v41_q2k_token_pair_shape(C.int(p.Out), C.int(p.In), C.int(p.Rows)) == 0 {
		return Tensor{}, Tensor{}, vulkanV41RoutedIDError(site, 2, "foreign recording window or unsupported paired launch geometry")
	}
	g, gok := v.ownedVulkanBuf(gate)
	u, uok := v.ownedVulkanBuf(up)
	x, xok := v.ownedVulkanBuf(input)
	if !gok || !uok || !xok || g.n < p.WeightBytes || u.n < p.WeightBytes || x.n < p.InputBytes {
		return Tensor{}, Tensor{}, vulkanV41RoutedIDError(site, 2, "weights and input must have sufficient live storage owned by this backend")
	}
	gOut = v.dallocForClass(p.OutputBytes, MemoryActivation, "V4.1 Q2_K paired gate panel")
	uOut = v.dallocForClass(p.OutputBytes, MemoryActivation, "V4.1 Q2_K paired up panel")
	gateOut = makeTensor(v, F32, RowMajor, []int{p.OutputBytes / 4}, nil, gOut)
	upOut = makeTensor(v, F32, RowMajor, []int{p.OutputBytes / 4}, nil, uOut)
	v.recordBatchDispatch(
		VulkanBufferAccess{BufferID: v.bufferIdentity(gate), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(up), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(input), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(gateOut), Size: 0, Mode: VulkanAccessWrite},
		VulkanBufferAccess{BufferID: v.bufferIdentity(upOut), Size: 0, Mode: VulkanAccessWrite},
	)
	if status := int(C.fvk_v41_q2k_token_pair_f32(g.ptr, u.ptr, x.ptr, gOut.ptr, uOut.ptr, C.int(p.Out), C.int(p.In), C.int(p.Rows))); status != 0 {
		return Tensor{}, Tensor{}, vulkanV41RoutedIDError(site, status, "selected paired gate/up dispatch failed")
	}
	gOut.v41CheckedRead = true
	uOut.v41CheckedRead = true
	completed = true
	return gateOut, upOut, nil
}
