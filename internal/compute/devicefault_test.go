package compute

import (
	"errors"
	"strings"
	"sync"
	"testing"
)

// The #6412 fault-boundary witness.
//
// It reproduces the live L4 defect from #6406 as a deterministic, GPU-free simulation: a device
// that suffers an execution fault (Xid 31 / `an illegal memory access was encountered`) and then
// keeps REPORTING SUCCESS while returning prompt-independent repeated punctuation. The captured
// live behavior was 0/10 correct after the fault and 10/10 correct only after the CUDA context
// was reconstructed, so that is exactly what faultyDevice models.
//
// The witness runs the same simulated device twice: once unguarded (the pre-fix serving path,
// which reproduces the defect — post-fault requests still come back "ok") and once behind a
// DeviceFaultLatch (which must refuse instead). No test here needs a GPU, nvcc, or the cuda build
// tag, so the fault boundary stays regression-covered in the default `go test ./internal/compute`.

const (
	faultyDeviceGood       = "fak-night-ok" // the correct answer for the fixed prompt
	faultyDeviceDegenerate = "!!!!!!!!!!!!" // what the poisoned live CUDA context returned
	faultyDeviceSite       = "qwen35-gdn-decode"
	faultyDeviceCode       = 31 // Xid 31, preserved as the flat C ABI status
)

// faultyDevice simulates one CUDA session across a fault boundary. Before the fault it computes
// correctly; after it, generation still "succeeds" but the content is degenerate — the defect.
// Reconstructing it clears the fault, matching the live restart recovery.
type faultyDevice struct {
	mu          sync.Mutex
	faulted     bool
	rebuilds    int
	rebuildFail int  // fail this many rebuild attempts before one succeeds
	validateBad bool // rebuild succeeds but the rebuilt context still computes garbage
	generations int
}

// fault injects the device fault at a chosen point in the request stream.
func (d *faultyDevice) fault() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.faulted = true
}

// generate is the device's own generation call. It mirrors the live defect: it never reports an
// error, so status-only monitoring cannot see the corruption.
func (d *faultyDevice) generate() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.generations++
	if d.faulted {
		return faultyDeviceDegenerate
	}
	return faultyDeviceGood
}

// rebuild tears down and re-creates the device context, the simulated process/CUDA-context
// reconstruction that recovered the live serve.
func (d *faultyDevice) rebuild() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.rebuilds++
	if d.rebuildFail > 0 {
		d.rebuildFail--
		return errors.New("simulated: device context rebuild failed")
	}
	if !d.validateBad {
		d.faulted = false
	}
	return nil
}

// validate proves the rebuilt context actually computes. A rebuild that does not validate must
// leave the session refusing — a context that rebuilds and still returns garbage is precisely
// what this boundary exists to catch.
func (d *faultyDevice) validate() error {
	if got := d.generate(); got != faultyDeviceGood {
		return errors.New("simulated: rebuilt context still returns degenerate output")
	}
	return nil
}

// serve is the request boundary under test: it is the "HTTP 200" seam. It admits through the
// latch first and only reaches the device when the latch says the session is trustworthy, so a
// refusal here is exactly "no post-fault success was emitted".
func serveThroughLatch(latch *DeviceFaultLatch, device *faultyDevice) (content string, err error) {
	if err := latch.Admit(faultyDeviceSite); err != nil {
		return "", err
	}
	return device.generate(), nil
}

// TestDeviceFaultUnguardedServeReproducesPostFaultSuccess is the REPRODUCTION half of the
// witness. With no fault boundary — the pre-fix serving path — a post-fault request still comes
// back reporting success, carrying the degenerate content. This is the defect #6406 captured
// live and #6412 exists to close; if this test ever stops reproducing, the simulation has drifted
// away from the behavior the fix is verified against.
func TestDeviceFaultUnguardedServeReproducesPostFaultSuccess(t *testing.T) {
	device := &faultyDevice{}
	if got := device.generate(); got != faultyDeviceGood {
		t.Fatalf("pre-fault generation = %q, want %q", got, faultyDeviceGood)
	}

	device.fault()

	// No latch: nil is the inert latch, which is byte-for-byte the unguarded path.
	for i := 0; i < 10; i++ {
		content, err := serveThroughLatch(nil, device)
		if err != nil {
			t.Fatalf("unguarded serve %d returned an error %v; the defect is that it does NOT", i, err)
		}
		if content != faultyDeviceDegenerate {
			t.Fatalf("unguarded serve %d content = %q, want the degenerate %q", i, content, faultyDeviceDegenerate)
		}
	}
}

// TestDeviceFaultLatchRefusesEveryPostFaultGeneration is the FIX half: the same device, same
// fault, but behind the latch. Done condition 1 and 3 — the fault poisons the session and no
// successful generation is emitted from suspect state.
func TestDeviceFaultLatchRefusesEveryPostFaultGeneration(t *testing.T) {
	device := &faultyDevice{}
	latch := NewDeviceFaultLatch("cuda", 3)

	content, err := serveThroughLatch(latch, device)
	if err != nil {
		t.Fatalf("pre-fault serve failed: %v", err)
	}
	if content != faultyDeviceGood {
		t.Fatalf("pre-fault content = %q, want %q", content, faultyDeviceGood)
	}

	device.fault()
	latch.Observe(DeviceFaultExecution, faultyDeviceSite, faultyDeviceCode)

	before := device.generations
	for i := 0; i < 10; i++ {
		content, err := serveThroughLatch(latch, device)
		if err == nil {
			t.Fatalf("post-fault serve %d succeeded with %q; a poisoned session must fail closed", i, content)
		}
		if content != "" {
			t.Fatalf("post-fault serve %d returned content %q alongside a refusal", i, content)
		}
		var faultErr *DeviceFaultError
		if !errors.As(err, &faultErr) {
			t.Fatalf("post-fault serve %d error %v is not a *DeviceFaultError", i, err)
		}
		if faultErr.Class != DeviceFaultExecution {
			t.Fatalf("refusal class = %q, want %q", faultErr.Class, DeviceFaultExecution)
		}
		if faultErr.Code != faultyDeviceCode {
			t.Fatalf("refusal code = %d, want %d", faultErr.Code, faultyDeviceCode)
		}
		if faultErr.Health != DevicePoisoned {
			t.Fatalf("refusal health = %q, want %q", faultErr.Health, DevicePoisoned)
		}
	}
	if device.generations != before {
		t.Fatalf("the device was reached %d times while poisoned; a refused request must never run a kernel",
			device.generations-before)
	}
}

// TestDeviceFaultLatchServesAgainOnlyAfterValidatedReconstruction covers done condition 2: the
// server reconstructs AND validates before it serves again. The recovery mirrors the live
// restart, where the identical endpoint went from 0/10 to 10/10 correct only once the context was
// rebuilt.
func TestDeviceFaultLatchServesAgainOnlyAfterValidatedReconstruction(t *testing.T) {
	device := &faultyDevice{}
	latch := NewDeviceFaultLatch("cuda", 3)
	device.fault()
	latch.Observe(DeviceFaultExecution, faultyDeviceSite, faultyDeviceCode)

	if _, err := serveThroughLatch(latch, device); err == nil {
		t.Fatal("serve succeeded before reconstruction")
	}
	if err := latch.Reconstruct(device.rebuild, device.validate); err != nil {
		t.Fatalf("reconstruction failed: %v", err)
	}
	if latch.Health() != DeviceHealthy {
		t.Fatalf("health after validated reconstruction = %q, want %q", latch.Health(), DeviceHealthy)
	}
	for i := 0; i < 10; i++ {
		content, err := serveThroughLatch(latch, device)
		if err != nil {
			t.Fatalf("post-recovery serve %d failed: %v", i, err)
		}
		if content != faultyDeviceGood {
			t.Fatalf("post-recovery serve %d content = %q, want %q", i, content, faultyDeviceGood)
		}
	}
}

// TestDeviceFaultLatchKeepsRefusingWhenReconstructionDoesNotValidate is the case that separates
// a real boundary from a hopeful one: the rebuild SUCCEEDS but the rebuilt context still computes
// garbage. Serving must stay closed, because "we restarted it" is not evidence that it works.
func TestDeviceFaultLatchKeepsRefusingWhenReconstructionDoesNotValidate(t *testing.T) {
	device := &faultyDevice{validateBad: true}
	latch := NewDeviceFaultLatch("cuda", 0) // unlimited attempts: isolate validation from the budget
	device.fault()
	latch.Observe(DeviceFaultContext, faultyDeviceSite, faultyDeviceCode)

	err := latch.Reconstruct(device.rebuild, device.validate)
	if err == nil {
		t.Fatal("reconstruction reported success even though validation failed")
	}
	var faultErr *DeviceFaultError
	if !errors.As(err, &faultErr) {
		t.Fatalf("reconstruction error %v is not a *DeviceFaultError", err)
	}
	if device.rebuilds != 1 {
		t.Fatalf("rebuilds = %d, want 1", device.rebuilds)
	}
	if latch.Health() != DevicePoisoned {
		t.Fatalf("health = %q, want %q after an unvalidated reconstruction", latch.Health(), DevicePoisoned)
	}
	if _, err := serveThroughLatch(latch, device); err == nil {
		t.Fatal("serve succeeded after an unvalidated reconstruction")
	}
}

// TestDeviceFaultLatchCondemnsSessionAfterAttemptBudget proves the budget terminates: a session
// that never recovers becomes unrecoverable and stops calling rebuild, so a hopeless context
// cannot be retried forever on the serving path.
func TestDeviceFaultLatchCondemnsSessionAfterAttemptBudget(t *testing.T) {
	device := &faultyDevice{rebuildFail: 99}
	latch := NewDeviceFaultLatch("cuda", 2)
	device.fault()
	latch.Observe(DeviceFaultContext, faultyDeviceSite, faultyDeviceCode)

	for i := 0; i < 2; i++ {
		if err := latch.Reconstruct(device.rebuild, device.validate); err == nil {
			t.Fatalf("reconstruction attempt %d reported success", i)
		}
	}
	if latch.Health() != DeviceUnrecoverable {
		t.Fatalf("health = %q, want %q after the attempt budget is spent", latch.Health(), DeviceUnrecoverable)
	}
	rebuildsAtBudget := device.rebuilds
	if rebuildsAtBudget != 2 {
		t.Fatalf("rebuilds = %d, want exactly the 2-attempt budget", rebuildsAtBudget)
	}

	err := latch.Reconstruct(device.rebuild, device.validate)
	if err == nil {
		t.Fatal("reconstruction succeeded on an unrecoverable session")
	}
	var faultErr *DeviceFaultError
	if !errors.As(err, &faultErr) || !faultErr.Unrecoverable() {
		t.Fatalf("error %v does not report an unrecoverable session", err)
	}
	if device.rebuilds != rebuildsAtBudget {
		t.Fatalf("rebuild was called again after the session was condemned (%d -> %d)",
			rebuildsAtBudget, device.rebuilds)
	}
	if _, err := serveThroughLatch(latch, device); err == nil {
		t.Fatal("serve succeeded on an unrecoverable session")
	}
}

// TestDeviceFaultSnapshotIdentifiesPoisonAndRecovery covers done condition 4: the telemetry names
// the poison/recovery transition and relates it to the affected backend and call site.
func TestDeviceFaultSnapshotIdentifiesPoisonAndRecovery(t *testing.T) {
	device := &faultyDevice{}
	latch := NewDeviceFaultLatch("cuda", 3)
	device.fault()
	latch.Observe(DeviceFaultExecution, faultyDeviceSite, faultyDeviceCode)
	for i := 0; i < 3; i++ {
		_, _ = serveThroughLatch(latch, device)
	}
	if err := latch.Reconstruct(device.rebuild, device.validate); err != nil {
		t.Fatalf("reconstruction failed: %v", err)
	}

	snap := latch.Snapshot()
	if snap.Backend != "cuda" {
		t.Fatalf("snapshot backend = %q, want %q", snap.Backend, "cuda")
	}
	if snap.Health != DeviceHealthy || snap.Refusing() {
		t.Fatalf("snapshot health = %q, want a recovered healthy session", snap.Health)
	}
	if snap.Faults != 1 || snap.Recoveries != 1 || snap.Attempts != 1 {
		t.Fatalf("snapshot counters = faults %d attempts %d recoveries %d, want 1/1/1",
			snap.Faults, snap.Attempts, snap.Recoveries)
	}
	if snap.RefusedCalls != 3 {
		t.Fatalf("snapshot RefusedCalls = %d, want the 3 refused requests", snap.RefusedCalls)
	}
	if snap.LastSite != faultyDeviceSite || snap.LastCode != faultyDeviceCode {
		t.Fatalf("snapshot attribution = %q/%d, want %q/%d",
			snap.LastSite, snap.LastCode, faultyDeviceSite, faultyDeviceCode)
	}
	if len(snap.Transitions) != 2 {
		t.Fatalf("transitions = %d, want the poison edge and the recovery edge", len(snap.Transitions))
	}
	poison, recovery := snap.Transitions[0], snap.Transitions[1]
	if poison.From != DeviceHealthy || poison.To != DevicePoisoned {
		t.Fatalf("poison edge = %q -> %q, want healthy -> poisoned", poison.From, poison.To)
	}
	if poison.Class != DeviceFaultExecution || poison.Site != faultyDeviceSite || poison.Code != faultyDeviceCode {
		t.Fatalf("poison edge attribution = %q/%q/%d", poison.Class, poison.Site, poison.Code)
	}
	if recovery.From != DevicePoisoned || recovery.To != DeviceHealthy {
		t.Fatalf("recovery edge = %q -> %q, want poisoned -> healthy", recovery.From, recovery.To)
	}
	if recovery.Seq != poison.Seq+1 {
		t.Fatalf("transition sequence is not monotonic: %d then %d", poison.Seq, recovery.Seq)
	}

	// The snapshot is a copy: mutating it must not reach back into the latch.
	snap.Transitions[0].To = DeviceHealthy
	if again := latch.Snapshot(); again.Transitions[0].To != DevicePoisoned {
		t.Fatal("Snapshot aliased the latch's live transition slice")
	}
}

// TestDeviceFaultObserveErrorClassifiesPackageErrors proves the observation points do not each
// re-derive the mapping, and — just as important — that a deliberate fak refusal (geometry,
// residency) does NOT poison a healthy context.
func TestDeviceFaultObserveErrorClassifiesPackageErrors(t *testing.T) {
	poisoning := []struct {
		name string
		err  error
		want DeviceFaultClass
		code int
	}{
		{"kernel", &Qwen35GDNKernelError{Stage: "conv", Code: 7}, DeviceFaultExecution, 7},
		{"invalid-state", &Qwen35GDNInvalidStateError{Operand: "recurrent_state"}, DeviceFaultExecution, 0},
		{"allocation", &DeviceAllocError{Bytes: 4096, Site: "dalloc", Class: MemoryWeights}, DeviceFaultAllocation, 0},
	}
	for _, tc := range poisoning {
		t.Run(tc.name, func(t *testing.T) {
			latch := NewDeviceFaultLatch("cuda", 1)
			got := latch.ObserveError(tc.err, "")
			if got == nil {
				t.Fatalf("%v did not poison the session", tc.err)
			}
			if got.Class != tc.want || got.Code != tc.code {
				t.Fatalf("classified as %q/%d, want %q/%d", got.Class, got.Code, tc.want, tc.code)
			}
			if latch.Health() != DevicePoisoned {
				t.Fatalf("health = %q, want poisoned", latch.Health())
			}
			// A wrapped error must classify identically: the serving path wraps.
			wrapped := NewDeviceFaultLatch("cuda", 1)
			if wrapped.ObserveError(wrapError(tc.err), "site") == nil {
				t.Fatalf("wrapped %v did not poison the session", tc.err)
			}
		})
	}

	intact := []struct {
		name string
		err  error
	}{
		{"nil", nil},
		{"geometry", &Qwen35GDNGeometryError{Operand: "geometry", Reason: "head counts must be positive"}},
		{"residency", &Qwen35GDNResidencyError{Operand: "conv1d", Reason: "not device resident"}},
		{"unrelated", errors.New("context canceled")},
	}
	for _, tc := range intact {
		t.Run("intact/"+tc.name, func(t *testing.T) {
			latch := NewDeviceFaultLatch("cuda", 1)
			if got := latch.ObserveError(tc.err, "site"); got != nil {
				t.Fatalf("%v poisoned the session (%v); a deliberate refusal leaves the context intact", tc.err, got)
			}
			if latch.Health() != DeviceHealthy {
				t.Fatalf("health = %q, want healthy", latch.Health())
			}
			if err := latch.Admit("site"); err != nil {
				t.Fatalf("Admit refused a healthy session: %v", err)
			}
		})
	}
}

func wrapError(err error) error { return &wrappedTestError{err} }

type wrappedTestError struct{ inner error }

func (w *wrappedTestError) Error() string { return "decode step 7: " + w.inner.Error() }
func (w *wrappedTestError) Unwrap() error { return w.inner }

// TestDeviceFaultLatchNilIsInert proves the boundary is opt-in: a backend that keeps no latch is
// byte-for-byte unaffected, so wiring the gate in cannot change behavior until a fault is
// actually observed.
func TestDeviceFaultLatchNilIsInert(t *testing.T) {
	var latch *DeviceFaultLatch
	if err := latch.Admit("site"); err != nil {
		t.Fatalf("nil latch refused: %v", err)
	}
	if latch.Health() != DeviceHealthy {
		t.Fatalf("nil latch health = %q, want healthy", latch.Health())
	}
	if got := latch.Observe(DeviceFaultContext, "site", 1); got != nil {
		t.Fatalf("nil latch Observe returned %v", got)
	}
	if got := latch.ObserveError(&Qwen35GDNKernelError{Code: 3}, "site"); got != nil {
		t.Fatalf("nil latch ObserveError returned %v", got)
	}
	rebuilt := false
	if err := latch.Reconstruct(func() error { rebuilt = true; return nil }, nil); err != nil {
		t.Fatalf("nil latch Reconstruct returned %v", err)
	}
	if rebuilt {
		t.Fatal("nil latch invoked the rebuild closure")
	}
	if snap := latch.Snapshot(); snap.Refusing() || snap.Health != DeviceHealthy {
		t.Fatalf("nil latch snapshot = %+v, want an inert healthy read", snap)
	}
	if err := AdmitDevice(nil, "site"); err != nil {
		t.Fatalf("AdmitDevice on a nil backend refused: %v", err)
	}
	if latch := BackendFaultLatch(nil); latch != nil {
		t.Fatal("BackendFaultLatch on a nil backend returned a latch")
	}
}

// TestDeviceFaultReconstructIsIdempotentOnHealthy proves a racing second request cannot tear down
// the context the first one just rebuilt.
func TestDeviceFaultReconstructIsIdempotentOnHealthy(t *testing.T) {
	device := &faultyDevice{}
	latch := NewDeviceFaultLatch("cuda", 3)
	if err := latch.Reconstruct(device.rebuild, device.validate); err != nil {
		t.Fatalf("Reconstruct on a healthy session returned %v", err)
	}
	if device.rebuilds != 0 {
		t.Fatalf("rebuilds = %d, want 0 — a healthy session must not be torn down", device.rebuilds)
	}
}

// TestDeviceFaultRecoveryStepPanicKeepsSessionClosed covers the CGO reality: the CUDA rebuild path
// raises DeviceAllocError as a PANIC. A panicking recovery step must leave the session refusing
// rather than unwind through the serving goroutine.
func TestDeviceFaultRecoveryStepPanicKeepsSessionClosed(t *testing.T) {
	latch := NewDeviceFaultLatch("cuda", 0)
	latch.Observe(DeviceFaultContext, faultyDeviceSite, faultyDeviceCode)

	err := latch.Reconstruct(func() error {
		panic(&DeviceAllocError{Bytes: 1 << 20, Site: "dalloc", Class: MemoryWeights})
	}, nil)
	if err == nil {
		t.Fatal("a panicking rebuild reported success")
	}
	if latch.Health() != DevicePoisoned {
		t.Fatalf("health = %q, want poisoned after a panicking rebuild", latch.Health())
	}

	// A non-error panic value must be handled too.
	err = latch.Reconstruct(func() error { panic("device reset timed out") }, nil)
	if err == nil {
		t.Fatal("a string-panicking rebuild reported success")
	}
	if latch.Health() != DevicePoisoned {
		t.Fatalf("health = %q, want poisoned", latch.Health())
	}
}

// TestDeviceFaultLatchIsConcurrencySafe proves the whole point of a SESSION-scoped latch: a fault
// observed by one request goroutine is seen by every other. It also runs under -race.
func TestDeviceFaultLatchIsConcurrencySafe(t *testing.T) {
	device := &faultyDevice{}
	latch := NewDeviceFaultLatch("cuda", 3)

	const workers = 16
	var wg sync.WaitGroup
	start := make(chan struct{})
	degenerate := make(chan string, workers*8)

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for j := 0; j < 8; j++ {
				content, err := serveThroughLatch(latch, device)
				if err == nil && content == faultyDeviceDegenerate {
					degenerate <- content
				}
			}
		}()
	}
	close(start)
	// Fault and poison while the workers are in flight.
	device.fault()
	latch.Observe(DeviceFaultExecution, faultyDeviceSite, faultyDeviceCode)
	wg.Wait()
	close(degenerate)

	// A request that entered the device BEFORE the poison may legitimately have observed the
	// fault's output, so the invariant asserted here is the durable one: once poisoned, the
	// latch never admits again.
	if latch.Health() != DevicePoisoned {
		t.Fatalf("health = %q, want poisoned", latch.Health())
	}
	for i := 0; i < 64; i++ {
		if content, err := serveThroughLatch(latch, device); err == nil {
			t.Fatalf("serve succeeded with %q after concurrent poisoning", content)
		}
	}
}

// TestDeviceFaultErrorMessageIsLeakFreeAndActionable checks the operator-facing render: every
// field is fak's own value, so the message can never carry upstream content.
func TestDeviceFaultErrorMessageIsLeakFreeAndActionable(t *testing.T) {
	err := &DeviceFaultError{
		Backend: "cuda",
		Site:    faultyDeviceSite,
		Class:   DeviceFaultExecution,
		Code:    faultyDeviceCode,
		Health:  DevicePoisoned,
	}
	msg := err.Error()
	for _, want := range []string{"cuda", "execution", faultyDeviceSite, "31"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("message %q does not name %q", msg, want)
		}
	}
	if err.Unrecoverable() {
		t.Fatal("a poisoned-but-recoverable session reported Unrecoverable")
	}

	condemned := &DeviceFaultError{Backend: "cuda", Class: DeviceFaultContext, Health: DeviceUnrecoverable, Attempts: 3}
	if !condemned.Unrecoverable() {
		t.Fatal("an unrecoverable session did not report Unrecoverable")
	}
	if !strings.Contains(condemned.Error(), "unrecoverable") {
		t.Fatalf("condemned message %q does not say the session is unrecoverable", condemned.Error())
	}

	var nilErr *DeviceFaultError
	if nilErr.Error() == "" {
		t.Fatal("a nil *DeviceFaultError must still render")
	}
	if nilErr.Unrecoverable() {
		t.Fatal("a nil *DeviceFaultError reported Unrecoverable")
	}
}

// TestBackendFaultLatchExtractsTheOptionalCapability proves the extension interface: a backend
// that implements DeviceFaultReporter is gated, and every other backend is not.
func TestBackendFaultLatchExtractsTheOptionalCapability(t *testing.T) {
	latch := NewDeviceFaultLatch("cuda", 1)
	reporting := &faultReportingBackend{latch: latch}

	if got := BackendFaultLatch(reporting); got != latch {
		t.Fatal("BackendFaultLatch did not return the backend's latch")
	}
	if err := AdmitDevice(reporting, "step"); err != nil {
		t.Fatalf("AdmitDevice refused a healthy reporting backend: %v", err)
	}
	latch.Observe(DeviceFaultContext, "step", 31)
	if err := AdmitDevice(reporting, "step"); err == nil {
		t.Fatal("AdmitDevice admitted a poisoned reporting backend")
	}

	// A backend that reports no latch is never gated.
	if got := BackendFaultLatch(&faultReportingBackend{}); got != nil {
		t.Fatal("a backend reporting a nil latch produced one")
	}
	if err := AdmitDevice(&faultReportingBackend{}, "step"); err != nil {
		t.Fatalf("AdmitDevice refused a backend with no latch: %v", err)
	}
}

// faultReportingBackend is a DeviceFaultReporter that is deliberately NOT a full Backend: the
// capability is extracted by type assertion from the Backend interface value, so the test asserts
// through that same seam.
type faultReportingBackend struct {
	Backend
	latch *DeviceFaultLatch
}

func (b *faultReportingBackend) DeviceFaultLatch() *DeviceFaultLatch { return b.latch }
