//go:build cuda && cgo

package compute

import "sync"

// cudaForwardExecutionMu is the cooperative outer owner of the process-global
// CUDA recording context. Backend operations still acquire cudaMu themselves.
var cudaForwardExecutionMu sync.Mutex

var _ ForwardExecutionOwnerBackend = (*cudaBackend)(nil)

// ForwardExecutionOwner returns the same recording domain for every CUDA handle
// and promoted wrapper. It does not acquire ownership or qualify GPU execution.
func (*cudaBackend) ForwardExecutionOwner() *sync.Mutex {
	return &cudaForwardExecutionMu
}
