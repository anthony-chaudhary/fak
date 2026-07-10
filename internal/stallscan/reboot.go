package stallscan

// reboot.go — the reboot-threshold GATE decision for stallscan.
//
// Classify()/ClassifyWithBaseline() name a handle/thread leak as a WARNING
// (elevated, see the HandleLeakProc/ThreadLeakProc doc): a leaking terminal
// "runs for days" before a pool exhausts, so the leak-suspect lines (10k handles
// / 500 threads) deliberately never page — pinging an operator every tick for a
// box that is fine for another day is noise, not signal.
//
// But there is a HIGHER line. The runbook's manual reboot workaround triggers
// around ~30k handles on a single WindowsTerminal, and on the reference box a
// WindowsTerminal that had climbed past ~2k threads is what froze dispatch
// (2026-07-01). Past those high-water marks the box is on a documented curve to a
// freeze, and the right move is to reboot the host NOW — before it locks up — not
// to diagnose the freeze after the fact. Nothing in the tree decided that line:
// the leak verdict only ever raised the box to `elevated`, and no consumer turned
// that into a reboot page. AdviseReboot is that missing decision (issue #3668
// remedy 3: "wire the reboot-threshold page — page BEFORE the freeze").
//
// It is kept in the same pure, testable shape as Classify: sample + threshold
// policy in, advice out, no I/O and no clock. It reuses worstHandleHog /
// worstThreadHog (the same scan Classify uses for the leak-suspect axis) against
// the HIGHER high-water lines, so it is correct no matter how the leak-suspect
// line in Thresholds is configured — it never derives from the verdict's leak
// attribution, which a divergent leak line could leave unpopulated. Crossing the
// leak line stays a days-long WARNING; crossing THESE is the page.

import "fmt"

// RebootThresholds are the per-process HIGH-WATER lines past which the operator
// should be paged to reboot the host before a freeze. They sit ABOVE the
// leak-suspect lines in Thresholds (HandleLeakProc 10k / ThreadLeakProc 500):
// crossing a leak line is a calm-to-elevated warning that runs for days; crossing
// one of THESE is an actionable "reboot now." A line of 0 disables that axis.
type RebootThresholds struct {
	HandleHighWater int // per-process handle count at/above which reboot is advised (default 30000)
	ThreadHighWater int // per-process thread count at/above which reboot is advised (default 2000)
}

// DefaultRebootThresholds returns the field-calibrated high-water lines: ~30k
// handles (the runbook's WindowsTerminal reboot trigger) and ~2k threads (below
// the ~2.7k-thread count at which a WindowsTerminal froze dispatch on the
// reference box on 2026-07-01, and well above the 500-thread leak-suspect line).
func DefaultRebootThresholds() RebootThresholds {
	return RebootThresholds{HandleHighWater: 30000, ThreadHighWater: 2000}
}

// RebootAdvice is the reboot-page decision for one sample. Advised is the single
// load-bearing bit; the remaining fields name which axis crossed, the offending
// process, its count, and the line it crossed, so an operator (or a wiring
// consumer like operatorbrief) can page with the evidence attached. A calm sample
// returns the zero value (Advised=false).
type RebootAdvice struct {
	Advised   bool   `json:"advised"`
	Axis      string `json:"axis,omitempty"` // handle_high_water | thread_high_water
	Process   string `json:"process,omitempty"`
	PID       int    `json:"pid,omitempty"`
	Count     int    `json:"count,omitempty"`
	Threshold int    `json:"threshold,omitempty"`
	Reason    string `json:"reason,omitempty"`
}

// AdviseReboot decides whether the operator should be paged to reboot the host,
// reading the current sample's per-process handle/thread census against the reboot
// high-water lines. It is pure: same sample and thresholds in, same advice out,
// no I/O.
//
// The handle axis wins when both cross — handle-pool exhaustion is the failure
// closest to a hard freeze on the reference box, the same precedence Classify uses
// when it assigns the leak Cause. An empty census, or a line set to 0, yields no
// advice for that axis (worstHandleHog/worstThreadHog return ok=false below the
// line or when the threshold is disabled).
func AdviseReboot(s Sample, t RebootThresholds) RebootAdvice {
	if hog, ok := worstHandleHog(s.TopHandles, t.HandleHighWater); ok {
		return RebootAdvice{
			Advised:   true,
			Axis:      "handle_high_water",
			Process:   hog.Name,
			PID:       hog.PID,
			Count:     hog.Handles,
			Threshold: t.HandleHighWater,
			Reason: fmt.Sprintf("%s (pid %d) holds %d handles (>= %d reboot high-water) — reboot the host before it freezes",
				hog.Name, hog.PID, hog.Handles, t.HandleHighWater),
		}
	}
	if hog, ok := worstThreadHog(s.TopThreads, t.ThreadHighWater); ok {
		return RebootAdvice{
			Advised:   true,
			Axis:      "thread_high_water",
			Process:   hog.Name,
			PID:       hog.PID,
			Count:     hog.Threads,
			Threshold: t.ThreadHighWater,
			Reason: fmt.Sprintf("%s (pid %d) holds %d threads (>= %d reboot high-water) — reboot before dispatch freezes",
				hog.Name, hog.PID, hog.Threads, t.ThreadHighWater),
		}
	}
	return RebootAdvice{Advised: false}
}
