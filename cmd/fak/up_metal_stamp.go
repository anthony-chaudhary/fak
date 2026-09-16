package main

import (
	"fmt"
	"io"
)

// metalStampLine renders the one-line CPU/Metal truth stamp for startup surfaces.
// live → "backend=metal"; a shaded skip reason → "backend=cpu (<reason>)";
// empty reason (Metal never targeted, non-darwin) → "" (caller prints nothing).
func metalStampLine(live bool, reason serveMetalSkipReason) string {
	if live {
		return "backend=metal"
	}
	if reason == "" {
		return ""
	}
	return "backend=cpu (" + string(reason) + ")"
}

// metalStampChatWord names the forward honestly for user-visible mock chat text:
// the session runs Metal only when the decision resolved live.
func metalStampChatWord(live bool) string {
	if live {
		return "Metal"
	}
	return "CPU"
}

// metalResidencyStamp carries the startup residency facts the banner uses to qualify a
// "Metal GPU" claim (#12875): device-resident Q8/Q6_K weight counts and whether the Q8
// band was declined at load. Without this, "Metal GPU" asserted a GPU decode even when the
// Q8-minority band never reached the device and decode silently fell back to CPU.
type metalResidencyStamp struct {
	Q8Resident     int
	Q6Resident     int
	Q8Declined     bool
	ResidencyKnown bool
}

// metalResidencyStampFrom builds the banner's residency qualification from the live report the
// server exposes. A nil report (no native model / mock path) yields known=false, which the
// banner treats as "not observed" and prints the plain line rather than a fabricated zero.
func metalResidencyStampFrom(report map[string]any) metalResidencyStamp {
	if report == nil {
		return metalResidencyStamp{}
	}
	stamp := metalResidencyStamp{ResidencyKnown: true}
	if v, ok := report["metal_live_q8_weights"].(int); ok {
		stamp.Q8Resident = v
	}
	if v, ok := report["metal_live_q6_weights"].(int); ok {
		stamp.Q6Resident = v
	}
	if s, ok := report["metal_q8_residency_error"].(string); ok && s != "" {
		stamp.Q8Declined = true
	}
	return stamp
}

// metalStampQualifiedLine renders the live backend line, qualifying "backend=metal" when the
// device-resident Q8/Q6_K weights were not observed or the Q8 band was declined at load. The
// qualification is appended, never substituted: the operator still sees the decision-time
// backend AND the live residency truth, which is what makes the line auditable.
func metalStampQualifiedLine(d serveMetalDecision, res metalResidencyStamp) string {
	base := metalStampLine(d.live, d.skippedBecause)
	if base == "" || !d.live {
		return base
	}
	switch {
	case res.Q8Declined:
		return base + ", q8-device-resident=declined"
	case res.ResidencyKnown && res.Q8Resident == 0 && res.Q6Resident == 0:
		return base + ", device-resident-weights=0 (cpu-projection-fallback)"
	case res.ResidencyKnown:
		return fmt.Sprintf("%s, device-resident-weights=q8:%d,q6k:%d", base, res.Q8Resident, res.Q6Resident)
	default:
		return base
	}
}

// printTurnkeyBackendStamp emits the deterministic one-liner naming which
// in-kernel chat forward the session will run and, on a darwin Metal miss, why
// the auto-target was skipped. It prints nothing when no Metal decision exists
// (mock/custom-server paths, non-darwin "Metal never targeted") so the stamp is
// always either a truthful Metal line or a truthful CPU-with-reason line. On a
// live Metal decision it qualifies the line with the observed device residency
// (#12875) so "Metal GPU" is never asserted unqualified when the Q8/Q6_K band is
// absent and decode is falling back to CPU.
func printTurnkeyBackendStamp(w io.Writer, d serveMetalDecision, res metalResidencyStamp) {
	stamp := metalStampQualifiedLine(d, res)
	if stamp == "" {
		return
	}
	switch {
	case d.live:
		fmt.Fprintf(w, "fak up: in-kernel chat forward: Metal GPU (%s)\n", stamp)
	case d.skippedBecause == skipReasonNotCompiled:
		fmt.Fprintf(w, "fak up: in-kernel chat forward: CPU (%s) — Metal skipped because the binary has no Metal support compiled in (requires darwin/arm64 with cgo).\n", stamp)
	case d.skippedBecause == skipReasonNoDevice:
		fmt.Fprintf(w, "fak up: in-kernel chat forward: CPU (%s) — Metal skipped because no usable Metal device is available on this host.\n", stamp)
	default:
		fmt.Fprintf(w, "fak up: in-kernel chat forward: CPU (%s)\n", stamp)
	}
}
