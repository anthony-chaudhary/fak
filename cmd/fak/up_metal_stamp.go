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

// printTurnkeyBackendStamp emits the deterministic one-liner naming which
// in-kernel chat forward the session will run and, on a darwin Metal miss, why
// the auto-target was skipped. It prints nothing when no Metal decision exists
// (mock/custom-server paths, non-darwin "Metal never targeted") so the stamp is
// always either a truthful Metal line or a truthful CPU-with-reason line.
func printTurnkeyBackendStamp(w io.Writer, d serveMetalDecision) {
	stamp := metalStampLine(d.live, d.skippedBecause)
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
