package compute

// V41DeferredStatusProvider supplies a separate deferred operator while the
// original Backend remains the identity and storage owner of every tensor.
// An explicit refusal is authoritative, even if the backend also implements
// the legacy interface. Providers must remain stable while work is in flight;
// in particular, a sticky execution fault must not revoke the cleanup reader.
type V41DeferredStatusProvider interface {
	V41DeferredStatusForModel() (V41DeferredStatusBackend, bool)
}

// ResolveV41DeferredStatus is the sole model admission path. Legacy Vulkan and
// other direct implementations retain their behavior when no provider exists.
func ResolveV41DeferredStatus(backend Backend) (V41DeferredStatusBackend, bool) {
	if backend == nil {
		return nil, false
	}
	if provider, ok := backend.(V41DeferredStatusProvider); ok {
		deferred, admitted := provider.V41DeferredStatusForModel()
		if !admitted || deferred == nil {
			return nil, false
		}
		return deferred, true
	}
	deferred, ok := backend.(V41DeferredStatusBackend)
	return deferred, ok
}

// V41DeferredStatusCanceler optionally takes ownership of issued status records
// on an early abort before V41ReadDeferred has taken them. It must preserve a
// non-nil primary error, settle accepted records exactly once, and retain native
// allocations on unknown completion. Callers must not Free those records afterward.
// completed is checked stream completion, not numerical output readiness. Only
// completed=true permits callers to free their own outputs; false retains them.
type V41DeferredStatusCanceler interface {
	V41CancelDeferred(statuses []V41DeferredStatus, primary error) (completed bool, err error)
}

// V41RoutedDeferredStatusProvider supplies full producer operations only to the
// two routed model paths that implement checked cancellation ownership. It
// does not widen the operator exposed to unrelated deferred consumers.
type V41RoutedDeferredStatusProvider interface {
	V41RoutedDeferredStatusForModel() (V41DeferredStatusBackend, bool)
}

// ResolveV41RoutedDeferredStatus preserves legacy selection unless an explicit
// routed provider exists. Its refusal is authoritative, never a generic fallback.
func ResolveV41RoutedDeferredStatus(backend Backend) (V41DeferredStatusBackend, bool) {
	if backend == nil {
		return nil, false
	}
	if provider, ok := backend.(V41RoutedDeferredStatusProvider); ok {
		deferred, admitted := provider.V41RoutedDeferredStatusForModel()
		if !admitted || deferred == nil {
			return nil, false
		}
		return deferred, true
	}
	return ResolveV41DeferredStatus(backend)
}
