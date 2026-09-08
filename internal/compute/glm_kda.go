package compute

import "fmt"

// GLMKDAHeadDim is the production GLM-5.3/5.4 KDA key/value head width.
// The first physical Vulkan kernel deliberately fixes this geometry so the
// shader has one statically bounded recurrent column per invocation.
const GLMKDAHeadDim = 128

// VulkanGLMKDAVariant selects one of two arithmetically equivalent physical
// kernels. The reread variant is the matched parent for the Wave32 retention
// candidate; it materializes the decay to the state buffer and rereads it for
// the rank-one update. The retention variant keeps the decayed column private
// until its sole final state write.
type VulkanGLMKDAVariant uint32

const (
	VulkanGLMKDAReread VulkanGLMKDAVariant = iota
	VulkanGLMKDAWave32Retain
)

// VulkanGLMKDAStepper is an optional, cycle-free capability for the fixed-width
// GLM KDA recurrent step. Implementations mutate state and output in place and
// must refuse unsupported geometry or residency rather than silently falling
// back to a host implementation.
type VulkanGLMKDAStepper interface {
	VulkanGLMKDAWave32Available() bool
	VulkanGLMKDAStep(state, q, k, value, alpha, beta, output Tensor, variant VulkanGLMKDAVariant) error
}

// GLMKDAContractError is returned before dispatch when the fixed-width Vulkan
// operation cannot prove its geometry, ownership, or no-alias requirements.
type GLMKDAContractError struct {
	Operand string
	Reason  string
}

func (e *GLMKDAContractError) Error() string {
	return fmt.Sprintf("compute: GLM KDA %s: %s", e.Operand, e.Reason)
}
