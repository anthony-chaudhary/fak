package compute

import "sync"

// ForwardExecutionOwnerBackend identifies the complete device recording domain.
// Every backend instance and wrapper sharing a command context must return the
// same non-nil mutex. It is separate from the backend's per-operation mutex.
// Model entrypoints acquire it before recording, transfer or lifecycle work.
// This cooperative contract does not make direct, unaudited Backend calls safe
// and does not qualify a shared-head arithmetic or device capability.
// Providers join the complete owned model execution and lifecycle surface.
// Source or CPU protocol evidence does not establish native execution safety.
type ForwardExecutionOwnerBackend interface {
	ForwardExecutionOwner() *sync.Mutex
}
