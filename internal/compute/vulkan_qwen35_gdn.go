//go:build vulkan && windows

package compute

/*
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

const qwen35GDNVulkanPath = "vulkan/qwen35-gdn-ssm-decode-v1"

func (v *vulkanBackend) Qwen35GDNPath() string { return qwen35GDNVulkanPath }

// Qwen35GDNPreprojected runs the causal convolution and recurrent GDN panel on
// Vulkan-resident tensors. Both auxiliary states are updated in place; the
// operation performs no host readback and has no CPU fallback.
// Qwen35GDNDecode composes the four input projections, the device-resident
// recurrent seam, and the output projection without making any host readback.
func (v *vulkanBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState Tensor, err error) {
	_, _, _, _, err = validateQwen35GDNGeometry(
		normalizedInput,
		inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj,
		convState, recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel,
		rmsNormEpsilon,
	)
	if err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	operands := qwen35GDNOperands(normalizedInput, inProjQKV, inProjZ, inProjB, inProjA, conv1D, aLog, dtBias, norm, outProj, convState, recurrentState)
	for _, operand := range operands {
		if _, ok := operand.t.buf.(*vulkanBuf); !ok {
			return Tensor{}, Tensor{}, Tensor{}, fmt.Errorf("compute: vulkan Qwen GDN %s is not Vulkan-resident; no host fallback", operand.name)
		}
	}

	mixed := v.MatMul(inProjQKV, normalizedInput)
	z := v.MatMul(inProjZ, normalizedInput)
	beta := v.MatMul(inProjB, normalizedInput)
	alpha := v.MatMul(inProjA, normalizedInput)
	core, err := v.Qwen35GDNPreprojected(
		mixed, z, beta, alpha, conv1D, aLog, dtBias, norm, convState, recurrentState,
		1, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon,
	)
	if err != nil {
		return Tensor{}, Tensor{}, Tensor{}, err
	}
	output = v.MatMul(outProj, core)
	return output, convState, recurrentState, nil
}
func (v *vulkanBackend) Qwen35GDNPreprojected(
	mixed, z, beta, alpha, conv1D, aLog, dtBias, norm, convState, recurrentState Tensor,
	tokens, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	eps float32,
) (Tensor, error) {
	if tokens <= 0 || numKeyHeads <= 0 || numValueHeads <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || valueHeadDim > 1024 || convKernel <= 0 || numValueHeads%numKeyHeads != 0 {
		return Tensor{}, fmt.Errorf("compute: vulkan Qwen GDN invalid geometry")
	}
	convDim := 2*numKeyHeads*keyHeadDim + numValueHeads*valueHeadDim
	valueDim := numValueHeads * valueHeadDim
	want := func(name string, t Tensor, n int) error {
		if t.Dtype != F32 || t.Numel() != n {
			return fmt.Errorf("compute: vulkan Qwen GDN %s elements/dtype=%d/%s, want %d/F32", name, t.Numel(), t.Dtype, n)
		}
		if _, ok := t.buf.(*vulkanBuf); !ok {
			return fmt.Errorf("compute: vulkan Qwen GDN %s is not Vulkan-resident", name)
		}
		return nil
	}
	checks := []error{
		want("mixed", mixed, tokens*convDim), want("z", z, tokens*valueDim),
		want("beta", beta, tokens*numValueHeads), want("alpha", alpha, tokens*numValueHeads),
		want("conv1d", conv1D, convDim*convKernel), want("a_log", aLog, numValueHeads),
		want("dt_bias", dtBias, numValueHeads), want("norm", norm, valueHeadDim),
		want("conv_state", convState, (convKernel-1)*convDim),
		want("recurrent_state", recurrentState, numValueHeads*keyHeadDim*valueHeadDim),
	}
	for _, err := range checks {
		if err != nil {
			return Tensor{}, err
		}
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	buf := v.dallocTransient(tokens * valueDim * F32.Bytes())
	out := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tokens, valueDim}, buf: buf, be: v}
	status := int(C.fvk_qwen35_gdn_preprojected_f32(
		v.vp(mixed), v.vp(z), v.vp(beta), v.vp(alpha), v.vp(conv1D), v.vp(aLog), v.vp(dtBias), v.vp(norm),
		v.vp(convState), v.vp(recurrentState), v.vp(out), C.int(tokens), C.int(convDim), C.int(numKeyHeads), C.int(numValueHeads),
		C.int(keyHeadDim), C.int(valueHeadDim), C.int(convKernel), C.float(eps)))
	if status != 0 {
		C.fvk_free(v.vp(out))
		return Tensor{}, fmt.Errorf("compute: vulkan Qwen GDN kernel failed closed (code %d); no CPU fallback", status)
	}
	v.transient = append(v.transient, buf)
	return out, nil
}
