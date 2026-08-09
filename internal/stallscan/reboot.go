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
// policy in, advice out, no I/O and no clock. It scans the same per-process
// census Classify reads for the leak-suspect axis, but against
// the HIGHER high-water lines, so it is correct no matter how the leak-suspect
// line in Thresholds is configured — it never derives from the verdict's leak
// attribution, which a divergent leak line could leave unpopulated. Crossing the
// leak line stays a days-long WARNING; crossing THESE is the page.

import (
	"fmt"
	"sort"
)

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

	// Crossers is the whole set of processes that INDEPENDENTLY crossed a
	// high-water in this sample, worst first, with the headline above repeated as
	// element 0. A verdict that reports only the single max masks every other
	// driver: on 2026-07-13 a WindowsTerminal at 33,054 handles hid a TermService
	// svchost that had crossed the same line on its own, and an operator deciding
	// whether to reboot needs both (issue #4614). Each element carries its own
	// axis/process/count/reason and an empty Crossers, so the structure is exactly
	// one level deep and cannot recurse.
	//
	// Populated ONLY when more than one process crossed. A single-hog sample keeps
	// the record it always had — omitempty drops the field, so the JSON is
	// byte-for-byte what it was and every consumer that reads just the headline is
	// untouched. Read len==0 as "the headline was the only crosser."
	Crossers []RebootAdvice `json:"crossers,omitempty"`

	// Unmeasured names the ENABLED reboot axes this sample carried NO per-process
	// census for. It is the binary/source skew guard (issue #3668 remedy 1). The
	// gate reads TopHandles/TopThreads; a reading written by a `fak.exe` built
	// BEFORE an axis existed carries that array empty, and Advised=false then
	// spells that absence EXACTLY the way it spells a host that was measured and
	// found calm. That is how a stale deployed binary kept returning a clean bill
	// of health while a WindowsTerminal sat at 3,206 threads: the source had the
	// thread axis, the artifact on disk did not, and nothing in the record said so.
	// Naming the axis here is what stops "nobody measured this" from ever being
	// spelled like "measured, nothing there" — the rule arming.go enforces one
	// level up for a missing RECORD, applied to a missing AXIS inside a record that
	// is present. An empty census is never a legitimate reading: every live process
	// holds handles and threads, so zero rows means the census failed or the
	// producer predates the axis, never that no process qualified.
	//
	// An axis whose line is 0 is off ON PURPOSE and is never listed — disabled is
	// the one silent state that needs no alarm. Empty when every enabled axis had a
	// census, and omitempty drops it, so a healthy record stays byte-for-byte what
	// it was. Carried on the headline only; the Crossers elements leave it unset
	// the same way they leave Crossers unset, so the structure stays one level deep.
	Unmeasured []string `json:"unmeasured,omitempty"`
}

// Measured reports whether every ENABLED reboot axis actually had a census to
// judge. Consumers must gate on it before reading Advised=false as calm:
// `!Advised && !Measured()` is "no verdict was reachable", not "no reboot
// needed" — the difference between a quiet host and an unwatched one.
func (a RebootAdvice) Measured() bool { return len(a.Unmeasured) == 0 }

// SecondaryCrossers returns the crossers BEHIND the headline — the drivers the
// old single-max verdict masked — and nothing when the headline was the only
// process over a line. Renderers list these beneath the headline without having
// to know that Crossers repeats the headline at element 0.
func (a RebootAdvice) SecondaryCrossers() []RebootAdvice {
	if len(a.Crossers) < 2 {
		return nil
	}
	return a.Crossers[1:]
}

// AdviseReboot decides whether the operator should be paged to reboot the host,
// reading the current sample's per-process handle/thread census against the reboot
// high-water lines. It is pure: same sample and thresholds in, same advice out,
// no I/O.
//
// The handle axis wins the HEADLINE when both cross — handle-pool exhaustion is
// the failure closest to a hard freeze on the reference box, the same precedence
// Classify uses when it assigns the leak Cause. An empty census, or a line set to
// 0, yields no advice for that axis.
//
// Every OTHER process over a line rides along in Crossers rather than being
// dropped: a leak that crosses the reboot line does so on its own merits, so a
// second crosser is a second driver of the same decision, not a detail of the
// first (issue #4614). The headline fields are unchanged, and Crossers stays
// unset when only one process crossed.
//
// Any enabled axis the sample carried no census for is named in Unmeasured, on
// the page and on the no-page alike, so a stale producer's silence can never be
// read as calm (issue #3668 remedy 1 — see the field's doc).
func AdviseReboot(s Sample, t RebootThresholds) RebootAdvice {
	unmeasured := unmeasuredAxes(s, t)
	crossers := rebootCrossers(s, t)
	if len(crossers) == 0 {
		return RebootAdvice{Advised: false, Unmeasured: unmeasured}
	}
	head := crossers[0]
	if len(crossers) > 1 {
		head.Crossers = crossers
	}
	head.Unmeasured = unmeasured
	return head
}

// unmeasuredAxes names each ENABLED reboot axis whose per-process census is
// empty — the axes this sample could not have decided. A line of 0 is skipped
// because an operator turned that axis off deliberately, which is the one
// silence that carries its own explanation. Returns nil when every enabled axis
// had rows, so the field stays absent from a healthy record.
func unmeasuredAxes(s Sample, t RebootThresholds) []string {
	var out []string
	if t.HandleHighWater > 0 && len(s.TopHandles) == 0 {
		out = append(out, "handle_high_water")
	}
	if t.ThreadHighWater > 0 && len(s.TopThreads) == 0 {
		out = append(out, "thread_high_water")
	}
	return out
}

// rebootCrossers returns one advice per process at/above a reboot high-water,
// worst first: the handle-axis crossers by descending handle count, then the
// thread-axis crossers the handle axis did not already name, by descending thread
// count. Ties break on the lower PID, so the order is total and the verdict is
// reproducible for a given sample.
//
// That ordering IS the headline precedence. Element 0 is exactly the process the
// single-max predecessor returned — the worst handle hog if any crossed, else the
// worst thread hog — so keeping the handle axis ahead of the thread axis here is
// what keeps "the handle axis wins" true. For the same reason a process that
// crosses BOTH lines is listed once, on the handle axis: it is one process to
// reboot away, not two. Dedup is by PID, which identifies a process within a
// single point-in-time census.
func rebootCrossers(s Sample, t RebootThresholds) []RebootAdvice {
	var out []RebootAdvice
	named := map[int]bool{}
	if t.HandleHighWater > 0 {
		hogs := append([]ProcHandles(nil), s.TopHandles...)
		sort.SliceStable(hogs, func(i, j int) bool {
			if hogs[i].Handles != hogs[j].Handles {
				return hogs[i].Handles > hogs[j].Handles
			}
			return hogs[i].PID < hogs[j].PID
		})
		for _, p := range hogs {
			if p.Handles < t.HandleHighWater {
				continue
			}
			named[p.PID] = true
			out = append(out, handleCrossing(p, t.HandleHighWater))
		}
	}
	if t.ThreadHighWater > 0 {
		hogs := append([]ProcThreads(nil), s.TopThreads...)
		sort.SliceStable(hogs, func(i, j int) bool {
			if hogs[i].Threads != hogs[j].Threads {
				return hogs[i].Threads > hogs[j].Threads
			}
			return hogs[i].PID < hogs[j].PID
		})
		for _, p := range hogs {
			if p.Threads < t.ThreadHighWater || named[p.PID] {
				continue
			}
			out = append(out, threadCrossing(p, t.ThreadHighWater))
		}
	}
	return out
}

// handleCrossing renders one process's handle-axis crossing as advice.
func handleCrossing(p ProcHandles, line int) RebootAdvice {
	return RebootAdvice{
		Advised:   true,
		Axis:      "handle_high_water",
		Process:   p.Name,
		PID:       p.PID,
		Count:     p.Handles,
		Threshold: line,
		Reason: fmt.Sprintf("%s (pid %d) holds %d handles (>= %d reboot high-water) — reboot the host before it freezes",
			p.Name, p.PID, p.Handles, line),
	}
}

// threadCrossing renders one process's thread-axis crossing as advice.
func threadCrossing(p ProcThreads, line int) RebootAdvice {
	return RebootAdvice{
		Advised:   true,
		Axis:      "thread_high_water",
		Process:   p.Name,
		PID:       p.PID,
		Count:     p.Threads,
		Threshold: line,
		Reason: fmt.Sprintf("%s (pid %d) holds %d threads (>= %d reboot high-water) — reboot before dispatch freezes",
			p.Name, p.PID, p.Threads, line),
	}
}
