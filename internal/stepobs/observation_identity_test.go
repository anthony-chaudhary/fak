package stepobs

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=10ms lane=default
func TestHostInputObservationIdentity(t *testing.T) {
	computetrace.SetObserver(nil)
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	absent := CaptureHostInputObservation()
	if absent == nil || !absent.Unchanged() {
		t.Fatal("absent observer refused")
	}
	r := New()
	r.Attach()
	installed := CaptureHostInputObservation()
	if installed == nil || !installed.Unchanged() || absent.Unchanged() {
		t.Fatal("canonical metrics registration was not captured distinctly")
	}
	computetrace.Record(computetrace.Event{Kernel: "input-scope", Backend: "cpu-ref", DurationNS: 1})
	if r.Snapshot().KernelEvents != 1 || !installed.Unchanged() {
		t.Fatal("ordinary metrics emission changed identity or lost the event")
	}
	// A concurrent ordinary replacement followed by the same canonical Attach
	// cannot restore a previously captured lifetime, even if no check ran between.
	done := make(chan struct{})
	go func() {
		computetrace.SetObserver(r.ObserveKernel)
		r.Attach()
		close(done)
	}()
	<-done
	if installed.Unchanged() || CaptureHostInputObservation() == nil {
		t.Fatal("replacement/restoration revived an obsolete observation")
	}
	computetrace.RegisterObserver(func(e computetrace.Event) { r.ObserveKernel(e) })
	if CaptureHostInputObservation() != nil {
		t.Fatal("arbitrary registration or wrapper claimed canonical identity")
	}
	computetrace.SetObserver(nil)
	if CaptureHostInputObservation() == nil {
		t.Fatal("detaching the observer prevented fresh unobserved capture")
	}
	beforeTrace := CaptureHostInputObservation()
	_, disable := computetrace.Enable(0, "zero-limit", "input-scope")
	if CaptureHostInputObservation() != nil || beforeTrace.Unchanged() {
		disable()
		t.Fatal("installed zero-limit recorder retained observation authority")
	}
	disable()
	if beforeTrace.Unchanged() || CaptureHostInputObservation() == nil {
		t.Fatal("recorder install/remove did not invalidate only the old lifetime")
	}
}
