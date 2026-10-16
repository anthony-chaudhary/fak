//go:build vulkan && (windows || linux) && cgo

package compute

func (v *vulkanBackend) ConfigureQ4KFusion(rmsnorm2, swigluAdd, forceScalar bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.haveQ4KFusedRMSNormMatMul2 = rmsnorm2
	v.haveQ4KFusedSwiGLUMatMulAdd = swigluAdd
	v.forceScalarQ4K = forceScalar
}

func (v *vulkanBackend) Q4KFusionStatus() (haveRMSNorm2, haveSwiGLUAdd, enabled bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.haveQ4KFusedRMSNormMatMul2, v.haveQ4KFusedSwiGLUMatMulAdd, v.selectQ4KFusionLocked(1)
}

func (v *vulkanBackend) VulkanDebugQ4KFusionCalls() (fusedRMSNorm, composedRMSNorm, fusedSwiGLU, composedSwiGLU int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.q4kFusionRMSNormCalls, v.q4kComposedRMSNormCalls, v.q4kFusionSwiGLUCalls, v.q4kComposedSwiGLUCalls
}

func (v *vulkanBackend) VulkanDebugResetQ4KFusionProfile() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.q4kFusionRMSNormCalls = 0
	v.q4kComposedRMSNormCalls = 0
	v.q4kFusionSwiGLUCalls = 0
	v.q4kComposedSwiGLUCalls = 0
}
