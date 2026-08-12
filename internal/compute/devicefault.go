package compute

import (
	"errors"
	"strconv"
	"sync"
)

// The device fault boundary (#6412).
//
// A CUDA execution fault — NVIDIA Xid 31 `FAULT_PDE ACCESS_TYPE_VIRT_READ`, or the
// `cuda_kernels.cu: an illegal memory access was encountered` the shim prints — poisons the
// whole CUDA CONTEXT, not one buffer. Every later kernel launched on that context is undefined,
// yet the launch itself keeps "succeeding": the live L4 serve kept answering HTTP 200 with
// prompt-independent repeated punctuation for hours after the fault, while fresh sibling
// containers on the same GPU, image, binary, and weights answered correctly. Only reconstructing
// the process/CUDA context restored correctness (0/10 correct before, 10/10 after).
//
// The pre-existing poison in this package is BUFFER-scoped (cudaBuf.invalid, set when an
// in-place Qwen GDN op reports a launch/execution failure) — it protects one tensor's contents.
// Nothing recorded that the SESSION the buffer belongs to is no longer trustworthy, so a fault
// on one request could not stop the next one from being served as though it had succeeded.
//
// DeviceFaultLatch is that missing session/backend-scoped state. It is deliberately plain Go in
// an untagged file — the same rationale as DeviceAllocError in capacity.go — so the fault
// boundary can be observed, gated, recovered, and regression-tested with no GPU, no nvcc, and no
// cuda build tag, and so packages that import compute only for its types can errors.As the
// refusal.
//
// The contract is fail-CLOSED and one-way until recovery is PROVEN: once a fault is observed,
// Admit refuses every subsequent operation. A refusal is the point — an error the serving
// boundary can render is strictly better than a successful-looking generation off suspect state.

// DeviceFaultClass names what the backend observed. It is fak's own classification of a device
// status, never upstream content, so it is safe to render in an operator-facing message.
type DeviceFaultClass string

const (
	// DeviceFaultUnknown is the conservative default: something failed on the device and the
	// backend could not attribute it. It poisons exactly like a known class — an unattributed
	// device failure is never evidence that the context is still sound.
	DeviceFaultUnknown DeviceFaultClass = "unknown"
	// DeviceFaultLaunch is a kernel launch the driver rejected before execution.
	DeviceFaultLaunch DeviceFaultClass = "launch"
	// DeviceFaultExecution is an asynchronous execution error surfaced at a stream fence —
	// the illegal-memory-access class that leaves the context undefined.
	DeviceFaultExecution DeviceFaultClass = "execution"
	// DeviceFaultContext is a context/device-level fault (Xid, device reset, ECC fallout).
	// Nothing on the device survives it.
	DeviceFaultContext DeviceFaultClass = "context"
	// DeviceFaultAllocation is an allocation that returned nil because a PRIOR async fault
	// already poisoned the context, as opposed to a genuine capacity miss.
	DeviceFaultAllocation DeviceFaultClass = "allocation"
)

// DeviceHealth is the latch's state. The zero value is DeviceHealthy so an unfaulted latch and a
// nil latch behave identically — a backend that never reports a fault is never gated.
type DeviceHealth string

const (
	// DeviceHealthy admits operations.
	DeviceHealthy DeviceHealth = "healthy"
	// DevicePoisoned refuses operations but may still be reconstructed.
	DevicePoisoned DeviceHealth = "poisoned"
	// DeviceUnrecoverable refuses operations permanently: reconstruction was attempted up to
	// the latch's budget and never validated. The session must be torn down by its owner.
	DeviceUnrecoverable DeviceHealth = "unrecoverable"
)

// Refusing reports whether this health state denies device work.
func (h DeviceHealth) Refusing() bool {
	return h == DevicePoisoned || h == DeviceUnrecoverable
}

// DeviceFaultError is the typed fail-closed refusal. It is returned by Admit for every operation
// attempted on a poisoned session, and by Reconstruct when recovery did not validate. Callers
// errors.As it to distinguish "this device session is untrustworthy" from an ordinary error, and
// to decide between retrying elsewhere and failing the request.
//
// Every field is fak's own value — a class name, a call site, and the flat C ABI status integer —
// so rendering it can never leak upstream content.
type DeviceFaultError struct {
	Backend  string           // backend that owns the poisoned session, e.g. "cuda"
	Site     string           // choke point that observed or was refused, e.g. "qwen35-gdn-decode"
	Class    DeviceFaultClass // what was observed
	Code     int              // flat C ABI status preserved for device-side diagnosis (0 = none)
	Health   DeviceHealth     // latch state at the moment of the refusal
	Attempts int              // reconstruction attempts made so far
}

func (e *DeviceFaultError) Error() string {
	if e == nil {
		return "compute: nil device fault error"
	}
	backend := e.Backend
	if backend == "" {
		backend = "device"
	}
	class := e.Class
	if class == "" {
		class = DeviceFaultUnknown
	}
	msg := "compute: " + backend + " session failed closed after a " + string(class) + " fault"
	if e.Site != "" {
		msg += " at " + e.Site
	}
	if e.Code != 0 {
		msg += " (code " + strconv.Itoa(e.Code) + ")"
	}
	if e.Health == DeviceUnrecoverable {
		msg += "; reconstruction failed " + strconv.Itoa(e.Attempts) + "x and the session is unrecoverable"
		return msg
	}
	msg += "; the device context is suspect and no result may be returned from it until" +
		" reconstruction validates"
	return msg
}

// Unrecoverable reports whether the owner must tear the session down rather than retry.
func (e *DeviceFaultError) Unrecoverable() bool {
	return e != nil && e.Health == DeviceUnrecoverable
}

// DeviceFaultTransition is one recorded edge of the latch's state machine. The sequence is
// monotonic per latch so operator telemetry can order poison and recovery against the requests
// that surrounded them without depending on a clock.
type DeviceFaultTransition struct {
	Seq   uint64           // 1-based, monotonic within one latch
	From  DeviceHealth     // state before this edge
	To    DeviceHealth     // state after this edge
	Class DeviceFaultClass // fault class that drove the edge ("" on a successful recovery)
	Site  string           // choke point that drove the edge
	Code  int              // flat C ABI status (0 = none)
}

// DeviceFaultSnapshot is the telemetry read of a latch: enough to identify the poison/recovery
// transition and relate it to the affected backend and call site. It is a value copy, safe to
// render or serialize after the latch has moved on.
type DeviceFaultSnapshot struct {
	Backend      string                  // backend that owns the session
	Health       DeviceHealth            // current state
	Faults       int                     // faults observed since construction
	Attempts     int                     // reconstruction attempts made
	Recoveries   int                     // reconstructions that rebuilt AND validated
	MaxAttempts  int                     // reconstruction budget (0 = unlimited)
	LastClass    DeviceFaultClass        // class of the most recent fault
	LastSite     string                  // site of the most recent fault
	LastCode     int                     // flat C ABI status of the most recent fault
	Transitions  []DeviceFaultTransition // every edge, oldest first
	RefusedCalls int                     // operations Admit refused while poisoned
}

// Refusing reports whether the snapshot was taken while the session denied device work.
func (s DeviceFaultSnapshot) Refusing() bool { return s.Health.Refusing() }

// DeviceFaultLatch is the session/backend-scoped fault boundary. The zero value is not usable;
// construct with NewDeviceFaultLatch. A nil *DeviceFaultLatch is a valid inert latch: it admits
// everything and records nothing, so a backend with no fault reporting is unaffected.
//
// It is safe for concurrent use. The serving path drives it from multiple request goroutines,
// and the whole point is that a fault observed by one request is seen by every other.
type DeviceFaultLatch struct {
	backend     string
	maxAttempts int

	mu           sync.Mutex
	health       DeviceHealth
	faults       int
	attempts     int
	recoveries   int
	refusedCalls int
	lastClass    DeviceFaultClass
	lastSite     string
	lastCode     int
	seq          uint64
	transitions  []DeviceFaultTransition
}

// NewDeviceFaultLatch builds a healthy latch for a named backend. maxAttempts bounds how many
// times Reconstruct may try before the session is declared unrecoverable; <= 0 means unlimited,
// which never converts a poisoned session into a serving one — it only leaves the owner free to
// keep retrying instead of being forced to tear down.
func NewDeviceFaultLatch(backend string, maxAttempts int) *DeviceFaultLatch {
	if maxAttempts < 0 {
		maxAttempts = 0
	}
	return &DeviceFaultLatch{backend: backend, maxAttempts: maxAttempts, health: DeviceHealthy}
}

// Backend names the backend this latch guards.
func (l *DeviceFaultLatch) Backend() string {
	if l == nil {
		return ""
	}
	return l.backend
}

// Health reports the current state. A nil latch is healthy.
func (l *DeviceFaultLatch) Health() DeviceHealth {
	if l == nil {
		return DeviceHealthy
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.health
}

// Observe records a device fault and poisons the session. It returns the typed refusal for the
// operation that observed the fault, so the observing call site can both poison and fail in one
// step. Observing again while already poisoned counts the fault and updates the attribution but
// never un-poisons, and never resets an unrecoverable session back to merely poisoned.
//
// Observe is the ONLY way into the refusing states. It is deliberately not conditional on the
// class: an unattributed device failure poisons exactly like an Xid, because "we could not tell
// what broke" is not evidence that nothing did.
func (l *DeviceFaultLatch) Observe(class DeviceFaultClass, site string, code int) *DeviceFaultError {
	if l == nil {
		return nil
	}
	if class == "" {
		class = DeviceFaultUnknown
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	from := l.health
	l.faults++
	l.lastClass, l.lastSite, l.lastCode = class, site, code
	if l.health != DeviceUnrecoverable {
		l.health = DevicePoisoned
	}
	l.record(from, l.health, class, site, code)
	return l.faultErrorLocked(site)
}

// ObserveError classifies an error this package already raises and poisons on it. It exists so
// the observation points do not each re-derive the mapping: a GDN kernel failure is a launch or
// asynchronous execution fault, and an allocation that returned nil may be a context already
// poisoned by a prior async fault rather than a capacity miss. It returns nil (and does not
// poison) for a nil error or an error that is not device-fault evidence — a geometry, residency,
// or pre-flight capacity refusal is fak refusing on purpose, with the context intact.
func (l *DeviceFaultLatch) ObserveError(err error, site string) *DeviceFaultError {
	if l == nil || err == nil {
		return nil
	}
	var kernelErr *Qwen35GDNKernelError
	if errors.As(err, &kernelErr) {
		if site == "" {
			site = kernelErr.Stage
		}
		return l.Observe(DeviceFaultExecution, site, kernelErr.Code)
	}
	var invalidState *Qwen35GDNInvalidStateError
	if errors.As(err, &invalidState) {
		if site == "" {
			site = invalidState.Operand
		}
		return l.Observe(DeviceFaultExecution, site, 0)
	}
	var allocErr *DeviceAllocError
	if errors.As(err, &allocErr) {
		if site == "" {
			site = allocErr.Site
		}
		return l.Observe(DeviceFaultAllocation, site, 0)
	}
	return nil
}

// Admit is the gate every device operation passes before it runs. It returns nil while the
// session is healthy and the typed refusal once a fault has been observed, counting the refusal
// for telemetry. This is the call that converts "kept serving HTTP 200 off a poisoned context"
// into "returned an explicit device-fault error".
func (l *DeviceFaultLatch) Admit(site string) error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.health.Refusing() {
		return nil
	}
	l.refusedCalls++
	return l.faultErrorLocked(site)
}

// Reconstruct attempts recovery of a poisoned session: rebuild tears down and rebuilds the device
// context, then validate proves the rebuilt context actually computes — the reconstruction is only
// believed when something is checked on it, because a context that rebuilds and still returns
// garbage is the exact failure this boundary exists to catch.
//
// It returns nil only when BOTH succeeded and the session is serving again. Any other outcome
// leaves the session refusing and returns the typed refusal, so a caller that ignores the error
// still cannot serve: Admit keeps refusing regardless. Once the attempt budget is spent the
// session becomes DeviceUnrecoverable and further calls refuse without invoking rebuild at all.
//
// Calling Reconstruct on a healthy session is a no-op that returns nil: recovery is idempotent,
// so a racing second request does not tear down the context the first one just rebuilt.
func (l *DeviceFaultLatch) Reconstruct(rebuild func() error, validate func() error) error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	if !l.health.Refusing() {
		l.mu.Unlock()
		return nil
	}
	if l.health == DeviceUnrecoverable {
		defer l.mu.Unlock()
		return l.faultErrorLocked(l.lastSite)
	}
	if l.maxAttempts > 0 && l.attempts >= l.maxAttempts {
		defer l.mu.Unlock()
		l.exhaustLocked()
		return l.faultErrorLocked(l.lastSite)
	}
	l.attempts++
	attemptSite := l.lastSite
	l.mu.Unlock()

	// rebuild/validate run OUTSIDE the lock: they tear down and re-create a device context,
	// which is slow and may itself call back into the latch. The session stays poisoned for
	// their whole duration, so a concurrent request is refused rather than admitted onto a
	// half-rebuilt context.
	err := runDeviceRecoveryStep(rebuild)
	if err == nil {
		err = runDeviceRecoveryStep(validate)
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	if err != nil {
		// A failed reconstruction is itself fault evidence, but it must not consume the
		// budget twice, so it records the edge without going through Observe.
		l.lastSite = attemptSite
		if l.maxAttempts > 0 && l.attempts >= l.maxAttempts {
			l.exhaustLocked()
		} else {
			l.record(l.health, l.health, l.lastClass, attemptSite, l.lastCode)
		}
		return l.faultErrorLocked(attemptSite)
	}
	if l.health == DeviceUnrecoverable {
		// A fault observed during the rebuild window exhausted the budget; the validated
		// context cannot un-condemn a session another observation already gave up on.
		return l.faultErrorLocked(attemptSite)
	}
	from := l.health
	l.health = DeviceHealthy
	l.recoveries++
	l.record(from, DeviceHealthy, "", attemptSite, 0)
	return nil
}

// runDeviceRecoveryStep runs one recovery closure, converting a panic into an error. The CUDA
// teardown/rebuild path sits below a CGO boundary and raises DeviceAllocError as a panic; a
// recovery step that panicked must leave the session poisoned, not unwind through the serving
// goroutine. A nil step is a no-op success, so a caller with nothing to validate is explicit
// rather than silently unchecked.
func runDeviceRecoveryStep(step func() error) (err error) {
	if step == nil {
		return nil
	}
	defer func() {
		if r := recover(); r != nil {
			if panicErr, ok := r.(error); ok {
				err = panicErr
				return
			}
			err = errors.New("compute: device recovery step panicked")
		}
	}()
	return step()
}

// Snapshot copies the latch's telemetry. Callers hold the returned value freely: the transition
// slice is a fresh copy, not the latch's live backing array.
func (l *DeviceFaultLatch) Snapshot() DeviceFaultSnapshot {
	if l == nil {
		return DeviceFaultSnapshot{Health: DeviceHealthy}
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return DeviceFaultSnapshot{
		Backend:      l.backend,
		Health:       l.health,
		Faults:       l.faults,
		Attempts:     l.attempts,
		Recoveries:   l.recoveries,
		MaxAttempts:  l.maxAttempts,
		LastClass:    l.lastClass,
		LastSite:     l.lastSite,
		LastCode:     l.lastCode,
		Transitions:  append([]DeviceFaultTransition(nil), l.transitions...),
		RefusedCalls: l.refusedCalls,
	}
}

// exhaustLocked condemns a session whose reconstruction budget is spent. Caller holds l.mu.
func (l *DeviceFaultLatch) exhaustLocked() {
	if l.health == DeviceUnrecoverable {
		return
	}
	from := l.health
	l.health = DeviceUnrecoverable
	l.record(from, DeviceUnrecoverable, l.lastClass, l.lastSite, l.lastCode)
}

// faultErrorLocked builds the typed refusal for the current state. Caller holds l.mu.
func (l *DeviceFaultLatch) faultErrorLocked(site string) *DeviceFaultError {
	if site == "" {
		site = l.lastSite
	}
	return &DeviceFaultError{
		Backend:  l.backend,
		Site:     site,
		Class:    l.lastClass,
		Code:     l.lastCode,
		Health:   l.health,
		Attempts: l.attempts,
	}
}

// record appends one state-machine edge. Caller holds l.mu.
func (l *DeviceFaultLatch) record(from, to DeviceHealth, class DeviceFaultClass, site string, code int) {
	l.seq++
	l.transitions = append(l.transitions, DeviceFaultTransition{
		Seq:   l.seq,
		From:  from,
		To:    to,
		Class: class,
		Site:  site,
		Code:  code,
	})
}

// DeviceFaultReporter is the optional Backend capability for a backend that maintains a fault
// latch. It is an extension interface — like the other optional capabilities in this package —
// so the CPU reference backend and every non-device backend stay untouched and unguarded.
type DeviceFaultReporter interface {
	// DeviceFaultLatch returns the backend's session-scoped latch, or nil if it keeps none.
	DeviceFaultLatch() *DeviceFaultLatch
}

// AdmitDevice is the backend-neutral gate for a serving path that holds a Backend rather than a
// latch. It returns nil for any backend that reports no latch, so wiring it in is behavior-
// preserving everywhere until a backend starts observing faults.
func AdmitDevice(b Backend, site string) error {
	return BackendFaultLatch(b).Admit(site)
}

// BackendFaultLatch extracts a backend's fault latch, or nil when it keeps none.
func BackendFaultLatch(b Backend) *DeviceFaultLatch {
	reporter, ok := b.(DeviceFaultReporter)
	if !ok {
		return nil
	}
	return reporter.DeviceFaultLatch()
}
