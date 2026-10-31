package compute

import "errors"

var (
	ErrCUDACaptureOnceUnsupported = errors.New("compute: capture-once requires a compatible CUDA qualification backend and graph mode")
	ErrCUDACaptureOnceBusy        = errors.New("compute: capture-once configuration requires checked graph teardown first")
)

type cudaCaptureOnceConfigurer interface {
	configureCUDACaptureOnce(enabled bool) error
}

// ConfigureCUDACaptureOnce explicitly selects the qualification-only capture-once
// runtime mode. It is OFF by default; neither a qualification build nor the
// retired FAK_CUDA_CAPTURE_ONCE environment variable enables it.
//
// Only the registered CUDA backend may configure its shared stream; this is not
// an independent per-device mode. Configure before sessions, or after their
// checked capture-once teardown.
// A live capture-once owner or active capture refuses configuration without
// changing the current mode. A completed ordinary per-token graph is independent:
// this setting neither controls nor tears down its cached native exec.
// Unsupported backends/build profiles refuse rather than silently accepting an
// inert setting. This does not enable ordinary CUDA graphs or qualify hardware.
func ConfigureCUDACaptureOnce(backend Backend, enabled bool) error {
	configured, ok := backend.(cudaCaptureOnceConfigurer)
	if !ok {
		return ErrCUDACaptureOnceUnsupported
	}
	return configured.configureCUDACaptureOnce(enabled)
}

// The CUDA qualification implementation serializes this state and all readers
// with cudaMu. This pure policy is separately host-testable without a CUDA init.
type cudaCaptureOnceConfiguration struct{ enabled bool }

func (s *cudaCaptureOnceConfiguration) set(enabled, compatible, graphs, profiling, ownerLive bool) error {
	if ownerLive {
		return ErrCUDACaptureOnceBusy
	}
	if !compatible || enabled && (!graphs || profiling) {
		return ErrCUDACaptureOnceUnsupported
	}
	s.enabled = enabled
	return nil
}
