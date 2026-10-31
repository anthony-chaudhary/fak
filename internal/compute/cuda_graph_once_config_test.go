package compute

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// fak-test:runtime unit est=1ms
func TestCUDACaptureOnceConfigurationPolicy(t *testing.T) {
	// Both current states and every input combination: refusals are transactional,
	// including an idempotent request while an owner is live.
	for bits := 0; bits < 64; bits++ {
		t.Run(fmt.Sprintf("flags-%02d", bits), func(t *testing.T) {
			initial, enabled := bits&1 != 0, bits&2 != 0
			compatible, graphs := bits&4 != 0, bits&8 != 0
			profiling, live := bits&16 != 0, bits&32 != 0
			s := cudaCaptureOnceConfiguration{enabled: initial}
			var want error
			if live {
				want = ErrCUDACaptureOnceBusy
			} else if !compatible || enabled && (!graphs || profiling) {
				want = ErrCUDACaptureOnceUnsupported
			}
			if err := s.set(enabled, compatible, graphs, profiling, live); err != want {
				t.Fatalf("error = %v, want %v", err, want)
			}
			wantState := initial
			if want == nil {
				wantState = enabled
			}
			if s.enabled != wantState {
				t.Fatalf("state = %v, want %v", s.enabled, wantState)
			}
		})
	}
}

// fak-test:runtime unit est=1ms
func TestCUDACaptureOnceConfigurationExplicitOnly(t *testing.T) {
	// The retired environment spelling cannot provide runtime consent.
	t.Setenv("FAK_CUDA_CAPTURE_ONCE", "1")
	var s cudaCaptureOnceConfiguration
	if s.enabled {
		t.Fatal("zero-value configuration must be OFF")
	}
	if err := s.set(false, true, true, false, false); err != nil || s.enabled {
		t.Fatalf("explicit OFF: state=%v error=%v", s.enabled, err)
	}
	t.Setenv("FAK_CUDA_CAPTURE_ONCE", "0")
	if err := s.set(true, true, true, false, false); err != nil || !s.enabled {
		t.Fatalf("explicit ON: state=%v error=%v", s.enabled, err)
	}
	if err := s.set(false, true, false, true, false); err != nil || s.enabled {
		t.Fatalf("checked-teardown disable remains possible without graph admission: state=%v error=%v", s.enabled, err)
	}
}

// These are dispatch test doubles, not CUDA execution or replacement production
// backend types. Embedding the actual Backend makes any accidental backend call
// fail rather than silently supplying invented GPU behavior.
type captureOnceDispatchBackend struct {
	Backend
	calls []bool
	err   error
}

func (b *captureOnceDispatchBackend) configureCUDACaptureOnce(enabled bool) error {
	b.calls = append(b.calls, enabled)
	return b.err
}

// fak-test:runtime unit est=1ms
func TestConfigureCUDACaptureOnceDispatch(t *testing.T) {
	for _, b := range []Backend{nil, struct{ Backend }{}} {
		for _, enabled := range []bool{false, true} {
			if err := ConfigureCUDACaptureOnce(b, enabled); err != ErrCUDACaptureOnceUnsupported {
				t.Fatalf("unsupported backend accepted: %v", err)
			}
		}
	}
	b := &captureOnceDispatchBackend{}
	if err := ConfigureCUDACaptureOnce(b, false); err != nil {
		t.Fatal(err)
	}
	sentinel := errors.New("checked teardown still pending")
	b.err = sentinel
	if err := ConfigureCUDACaptureOnce(b, true); err != sentinel {
		t.Fatalf("backend refusal not preserved: %v", err)
	}
	if !reflect.DeepEqual(b.calls, []bool{false, true}) {
		t.Fatalf("wrong dispatch sequence: %v", b.calls)
	}
}
