package compute

import (
	"fmt"
	"sync/atomic"
)

// DeviceSyncContractEnv selects how a device-sync contract violation is
// reported when a hot path performs a scalar host/device synchronization.
//
// The default ("") records the violation and returns a typed error. Setting the
// environment to "error" makes the violation fatal to the caller's operation so
// a hot path that regressed into a scalar readback cannot be mistaken for a
// merely slower one.
const DeviceSyncContractEnv = "FAK_DEVICE_SYNC_CONTRACT"

// DeviceSyncViolationError is the fail-closed refusal raised when a callable
// governed by a DeviceSyncGuard performed a scalar host/device synchronization.
// The device-sync-error mode is the moral equivalent of torch's
// set_sync_debug_mode("error"): the point is to convert an invisible latency
// regression into a named failure at the seam that introduced it.
type DeviceSyncViolationError struct {
	// Path is the caller-supplied label of the guarded hot path.
	Path string
	// Site names the synchronization seam that fired (e.g. "scalar-readback").
	Site string
	// Detail carries the caller-provided description of what synchronized.
	Detail string
}

func (e *DeviceSyncViolationError) Error() string {
	if e == nil {
		return "compute: nil device-sync contract violation"
	}
	detail := e.Detail
	if detail == "" {
		detail = "a scalar host/device synchronization ran on a hot path"
	}
	return fmt.Sprintf("compute: device-sync contract violation on %q at %s: %s", e.Path, e.Site, detail)
}

// DeviceSyncGuard is the installed device-sync-error mode. While armed, any
// Synced call records a violation; the guard is the single source of truth for
// "did this forward synchronize?", so a test can assert it did not without
// threading a boolean through the callable.
//
// A guard is NOT safe to share across goroutines that both arm and disarm it;
// arm one per guarded callable, exactly as the upstream contract does.
type DeviceSyncGuard struct {
	path     string
	armed    atomic.Bool
	observed atomic.Bool
	site     atomic.Value // string
	detail   atomic.Value // string
}

// NewDeviceSyncGuard returns an unarmed guard for the named hot path.
func NewDeviceSyncGuard(path string) *DeviceSyncGuard {
	return &DeviceSyncGuard{path: path}
}

// Path returns the guarded hot path label.
func (g *DeviceSyncGuard) Path() string { return g.path }

// Armed reports whether the guard is currently watching for synchronization.
func (g *DeviceSyncGuard) Armed() bool { return g.armed.Load() }

// Synced records a scalar host/device synchronization observed by the guarded
// callable. It is the single seam a backend hot path calls when it reads a
// scalar host-ward (a `.item()`/`.cpu()` equivalent). Calling Synced while the
// guard is disarmed is a no-op, so instrumented production paths carry no
// report when the contract test is not running.
func (g *DeviceSyncGuard) Synced(site, detail string) {
	if !g.armed.Load() {
		return
	}
	if site == "" {
		site = "scalar-readback"
	}
	g.site.Store(site)
	g.detail.Store(detail)
	g.observed.Store(true)
}

// Violated reports whether the guarded callable synchronized since the guard
// was last armed.
func (g *DeviceSyncGuard) Violated() bool { return g.observed.Load() }

// Violation returns the typed refusal for the observed synchronization, or nil
// when the guarded callable stayed sync-free.
func (g *DeviceSyncGuard) Violation() error {
	if !g.observed.Load() {
		return nil
	}
	site, _ := g.site.Load().(string)
	detail, _ := g.detail.Load().(string)
	return &DeviceSyncViolationError{Path: g.path, Site: site, Detail: detail}
}

// WithDeviceSyncContract arms the guard, runs fn, disarms it, and returns the
// typed violation (if any) alongside fn's own error. fn always runs with the
// guard armed, and the guard is always disarmed afterwards even when fn panics
// or returns an error, so a failed assertion cannot leak an armed guard into
// the next callable.
//
// The contract is deliberately a wrapper rather than a global mode: a device
// backend can install its hot-path guard at exactly the entry point under test
// without a process-wide switch that could mask a sync in an unrelated path.
func WithDeviceSyncContract(g *DeviceSyncGuard, fn func()) (violation error, err error) {
	if g == nil {
		return nil, fmt.Errorf("compute: nil device-sync guard")
	}
	g.armed.Store(true)
	g.observed.Store(false)
	g.site.Store("")
	g.detail.Store("")
	defer g.armed.Store(false)

	defer func() {
		if r := recover(); r != nil {
			// fn panicked: still surface the disarmed guard and the observed
			// violation, then re-panic so the caller's own recovery is intact.
			violation = g.Violation()
			panic(r)
		}
	}()

	err = runDeviceSyncCallable(fn)
	violation = g.Violation()
	return violation, err
}

// runDeviceSyncCallable isolates fn so WithDeviceSyncContract's deferred
// bookkeeping (disarm + violation capture) runs on every exit path.
func runDeviceSyncCallable(fn func()) error {
	fn()
	return nil
}
