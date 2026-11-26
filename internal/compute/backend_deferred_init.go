package compute

import (
	"slices"
	"sync"
)

// deferredBackend is a backend whose device bring-up is too slow for package init.
// Its place in registration order is reserved at init, so the registry reads the
// same once the device is up as if the backend had registered eagerly.
type deferredBackend struct {
	slot  int
	start func() Backend
}

var (
	registryMu       sync.RWMutex
	deferredBackends []deferredBackend
	deferredOnce     sync.Once
)

// registerDeferred reserves the current registration slot for a backend whose start
// brings its device up and returns it, or returns nil when no device is reachable.
// start must not read the registry: it runs inside the join every reader waits on.
func registerDeferred(start func() Backend) {
	registryMu.Lock()
	defer registryMu.Unlock()
	deferredBackends = append(deferredBackends, deferredBackend{slot: len(registry), start: start})
}

// StartBackendInit begins deferred device bring-up in the background so it overlaps
// host work that does not need a device. Every registry read joins the bring-up, so
// calling this only moves when the work happens, never whether it does.
// Bring-up reads its env knobs (FAK_CUDA_*, FAK_INKLING2_PREFILL_NVFP4_W4A4) when it
// runs, so set them before calling this or before the first registry read.
func StartBackendInit() { go joinDeferredBackends() }

// joinDeferredBackends runs every reserved backend's start once and splices the
// backends that came up into their reserved slots. A backend registered under the
// same name after the reservation keeps its place, as a later Register would.
func joinDeferredBackends() {
	deferredOnce.Do(func() {
		registryMu.RLock()
		pending := deferredBackends
		registryMu.RUnlock()
		started := make([]Backend, len(pending))
		for i, d := range pending {
			started[i] = d.start()
		}
		registryMu.Lock()
		defer registryMu.Unlock()
		deferredBackends = nil
		for i := len(pending) - 1; i >= 0; i-- {
			b := started[i]
			if b == nil || registeredLocked(b.Name()) {
				continue
			}
			registry = slices.Insert(registry, min(pending[i].slot, len(registry)), b)
		}
	})
}

func registeredLocked(name string) bool {
	for _, b := range registry {
		if b.Name() == name {
			return true
		}
	}
	return false
}
