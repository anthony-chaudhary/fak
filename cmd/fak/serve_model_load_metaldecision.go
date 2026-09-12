package main

import (
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// serveMetalSkipReason is the deterministic two-state reason the Metal auto-target
// was missed: the binary has no Metal support compiled in at all, or it was built
// with Metal but this host exposes no usable device. Naming the distinction lets
// startup surfaces stamp an honest CPU-truth line instead of silently serving CPU.
type serveMetalSkipReason string

const (
	// skipReasonNotCompiled means metalgemm.Compiled() is false: this binary was
	// not built with Metal support, so CPU is the only forward it can run.
	skipReasonNotCompiled serveMetalSkipReason = "metal-not-compiled"
	// skipReasonNoDevice means metalgemm.Compiled() is true but the auto-target
	// was missed anyway: the device probe found no usable Metal device.
	skipReasonNoDevice serveMetalSkipReason = "metal-no-device"
)

// serveMetalDecision is the side-effect-free result of the Metal availability
// decision shared by the resolver and the startup stamp surfaces.
type serveMetalDecision struct {
	live           bool                 // the session runs the Metal forward
	skippedBecause serveMetalSkipReason // set only when live==false
}

// serveMetalSkipReasonFrom shades the skip reason from the pure Metal probe pair.
// It is only consulted when live==false, meaning available==false: a compiled-in
// backend with no usable device shades no-device, and an uncompiled binary shades
// not-compiled. It never calls metalgemm, so it is identical in both build
// configurations and unit-testable as a pure function. Outside the consulted
// domain (available==true) the helper still defines a deterministic answer —
// compiled input shades no-device, uncompiled shades not-compiled.
func serveMetalSkipReasonFrom(available, compiled bool) serveMetalSkipReason {
	if !compiled {
		return skipReasonNotCompiled
	}
	return skipReasonNoDevice
}

// resolveServeMetalDecision is resolveServeMetal's decision twin: it returns the
// same (live, error) contract through decision.live/err and additionally names
// why the auto-target was missed, so startup surfaces can stamp CPU-truth lines
// without re-probing. Kept side-effect free so it is unit-testable in both
// build configurations. The reason is shaded only when the resolver declined
// Metal without erroring AND the running GOOS is darwin: only the darwin
// auto-selection targets Metal (resolveServeBackendSelection), so a false there
// is a genuine Metal miss (not-compiled vs no-device). On linux/windows, and for
// the portable "cpu" opt-out or a selected compute backend, the resolver's false
// means "Metal was never the target": the reason stays empty and callers must
// not stamp a Metal line.
func resolveServeMetalDecision(flag, env bool, backendName string) (serveMetalDecision, error) {
	resolved, err := resolveServeMetal(flag, env, backendName)
	if err != nil {
		return serveMetalDecision{live: false}, err
	}
	if resolved {
		return serveMetalDecision{live: true}, nil
	}
	if runtime.GOOS != "darwin" {
		return serveMetalDecision{live: false}, nil
	}
	return serveMetalDecision{live: false, skippedBecause: serveMetalSkipReasonFrom(metalgemm.Available(), metalgemm.Compiled())}, nil
}
