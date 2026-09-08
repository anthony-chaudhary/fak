//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakvulkan
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

// VulkanGLMKDAWave32Available reports whether the selected physical device can
// enforce a 32-lane compute subgroup through VK_EXT_subgroup_size_control.
func (v *vulkanBackend) VulkanGLMKDAWave32Available() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_glm_kda_wave32() != 0
}

func glmKDAShapeEqual(got []int, want ...int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func (v *vulkanBackend) validateGLMKDATensor(name string, t Tensor, want ...int) (*vulkanBuf, error) {
	if t.Backend() != v {
		return nil, &GLMKDAContractError{Operand: name, Reason: "tensor is not owned by this Vulkan backend"}
	}
	if t.Dtype != F32 || t.Layout != RowMajor {
		return nil, &GLMKDAContractError{Operand: name, Reason: "requires row-major F32 storage"}
	}
	if !glmKDAShapeEqual(t.Shape, want...) {
		return nil, &GLMKDAContractError{Operand: name, Reason: fmt.Sprintf("shape %v, want %v", t.Shape, want)}
	}
	b, ok := t.buf.(*vulkanBuf)
	if !ok || b == nil || b.ptr == nil || b.n < t.Numel()*F32.Bytes() {
		return nil, &GLMKDAContractError{Operand: name, Reason: "missing live full-size Vulkan allocation"}
	}
	return b, nil
}

// VulkanGLMKDAStep executes exactly one fixed-D GLM KDA recurrence. The seven
// buffers must be pairwise disjoint: each invocation owns one state column and
// one output element, while all other operands are immutable during dispatch.
// The method has no CPU fallback and performs no host/device transfer.
func (v *vulkanBackend) VulkanGLMKDAStep(
	state, q, k, value, alpha, beta, output Tensor,
	variant VulkanGLMKDAVariant,
) error {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()

	if C.fvk_have_glm_kda_wave32() == 0 {
		return &GLMKDAContractError{Operand: "backend", Reason: "32-lane compute subgroup control is unavailable"}
	}
	if variant != VulkanGLMKDAReread && variant != VulkanGLMKDAWave32Retain {
		return &GLMKDAContractError{Operand: "variant", Reason: fmt.Sprintf("unsupported value %d", variant)}
	}
	if len(state.Shape) != 3 || state.Shape[0] <= 0 || state.Shape[1] != GLMKDAHeadDim || state.Shape[2] != GLMKDAHeadDim {
		return &GLMKDAContractError{Operand: "state", Reason: fmt.Sprintf("shape %v, want [heads>0,%d,%d]", state.Shape, GLMKDAHeadDim, GLMKDAHeadDim)}
	}
	heads := state.Shape[0]
	operands := []struct {
		name string
		t    Tensor
		want []int
	}{
		{"state", state, []int{heads, GLMKDAHeadDim, GLMKDAHeadDim}},
		{"q", q, []int{heads, GLMKDAHeadDim}},
		{"k", k, []int{heads, GLMKDAHeadDim}},
		{"value", value, []int{heads, GLMKDAHeadDim}},
		{"alpha", alpha, []int{heads}},
		{"beta", beta, []int{heads}},
		{"output", output, []int{heads, GLMKDAHeadDim}},
	}
	bufs := make([]*vulkanBuf, len(operands))
	for i, operand := range operands {
		b, err := v.validateGLMKDATensor(operand.name, operand.t, operand.want...)
		if err != nil {
			return err
		}
		bufs[i] = b
	}
	for i := range bufs {
		for j := i + 1; j < len(bufs); j++ {
			if bufs[i].ptr == bufs[j].ptr {
				return &GLMKDAContractError{Operand: operands[j].name, Reason: "aliases " + operands[i].name}
			}
		}
	}
	status := C.fvk_glm_kda_step_f32(
		v.vp(state), v.vp(q), v.vp(k), v.vp(value), v.vp(alpha), v.vp(beta), v.vp(output),
		C.int(heads), C.int(variant),
	)
	if status != 0 {
		return fmt.Errorf("compute: GLM KDA Vulkan dispatch failed: status %d", int(status))
	}
	return nil
}

var _ VulkanGLMKDAStepper = (*vulkanBackend)(nil)
