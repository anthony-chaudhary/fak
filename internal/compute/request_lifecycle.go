package compute

import "sync"

// RequestLifetime owns the backend resources created for one request. Retire is
// safe to defer on every exit path, including cancellation and panic unwinding.
type RequestLifetime struct {
	backend Backend
	once    sync.Once
}

// BeginRequest starts a backend-neutral request lifetime. A nil backend is
// permitted so callers can install the cleanup defer before backend selection.
func BeginRequest(backend Backend) *RequestLifetime {
	return &RequestLifetime{backend: backend}
}

// Retire releases request-owned resources exactly once.
func (r *RequestLifetime) Retire() {
	if r == nil {
		return
	}
	r.once.Do(func() {
		if r.backend != nil {
			RetireBackendRequestResources(r.backend)
		}
	})
}
