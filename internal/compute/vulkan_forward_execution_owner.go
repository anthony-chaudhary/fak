//go:build vulkan && (windows || linux) && cgo

package compute

import "sync"

// vulkanForwardExecutionMu is the outer cooperative owner of the process-global
// C recording domain. It must not be replaced with vulkanMu: owned model bodies
// call Vulkan operations that acquire vulkanMu themselves.
var vulkanForwardExecutionMu sync.Mutex

var _ ForwardExecutionOwnerBackend = (*vulkanBackend)(nil)

// ForwardExecutionOwner returns the same domain for every Vulkan handle. The
// method neither acquires ownership nor qualifies a shared-head arithmetic path.
// Wrappers must preserve this exact mutex. Direct backend clients remain subject
// to the cooperative contract. Native execution qualification is separate.
func (*vulkanBackend) ForwardExecutionOwner() *sync.Mutex {
	return &vulkanForwardExecutionMu
}
