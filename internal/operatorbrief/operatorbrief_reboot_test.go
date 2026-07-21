package operatorbrief

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

// rebootPages returns the host-reboot human items surfaced in r.
func rebootPages(r Report) []Item {
	var out []Item
	for _, it := range r.Human {
		if it.Source == "host-reboot" {
			out = append(out, it)
		}
	}
	return out
}

// TestFoldSurfacesArmedRebootCrossingAsOneOperatorPage is the producer-path
// witness for issue #4613: an armed stallscan monitor whose reboot high-water is
// crossed surfaces exactly one operator page, deduplicated by (Axis, Process),
// and that page is a read-only "approve and schedule a reboot" recommendation --
// never an automatic reboot.
func TestFoldSurfacesArmedRebootCrossingAsOneOperatorPage(t *testing.T) {
	adv := stallscan.RebootAdvice{
		Advised:   true,
		Axis:      "handle_high_water",
		Process:   "WindowsTerminal.exe",
		PID:       4242,
		Count:     31000,
		Threshold: 30000,
		Reason:    "WindowsTerminal.exe (pid 4242) holds 31000 handles (>= 30000 reboot high-water) — reboot the host before it freezes",
	}
	// A sustained crossing sampled twice: the second sample sees the same process
	// under a fresh PID and a higher count. It is the SAME operator decision and
	// must not re-page -- dedup is by (Axis, Process), PID excluded.
	advAgain := adv
	advAgain.PID = 5150
	advAgain.Count = 33000

	r := Fold(Inputs{Reboot: []stallscan.RebootAdvice{adv, advAgain}})

	pages := rebootPages(r)
	if len(pages) != 1 {
		t.Fatalf("armed reboot crossing should page exactly once (dedup by axis,process), got %d: %+v", len(pages), pages)
	}
	p := pages[0]

	// Read-only recommend: the action approves/schedules a reboot; it must not
	// instruct an automatic reboot or kill.
	act := strings.ToLower(p.Action)
	if !strings.Contains(act, "approve") {
		t.Fatalf("reboot page action must recommend operator approval, got %q", p.Action)
	}
	if !strings.Contains(act, "do not reboot") {
		t.Fatalf("reboot page must be read-only (no automatic reboot), got %q", p.Action)
	}
	// The measured high-water evidence rides along so the operator pages informed.
	if !strings.Contains(p.Detail, "reboot high-water") {
		t.Fatalf("reboot page should carry the measured high-water evidence, got detail %q", p.Detail)
	}
	if !strings.Contains(p.Title, "WindowsTerminal.exe") || !strings.Contains(p.Title, "handle high-water") {
		t.Fatalf("reboot page title should name the offending process and axis, got %q", p.Title)
	}

	// The gate must page on it, and the reboot -- most time-critical -- must drive
	// NextAction ahead of the missing-report pages that empty Inputs also add.
	if code, _ := CheckGate(r); code != 1 {
		t.Fatalf("operator gate should page (exit 1) on a reboot crossing, got %d", code)
	}
	if r.NextAction != p.Action {
		t.Fatalf("reboot page should lead the human bucket (drive NextAction), got %q", r.NextAction)
	}

	// It must survive the decenter-the-human enforce gate: a host reboot is a
	// genuine operator-authority decision, not agent work the fleet auto-runs.
	enforced, _ := TriageHumanBucket(r)
	if got := len(rebootPages(enforced)); got != 1 {
		t.Fatalf("reboot page must remain HUMAN_RESIDUAL under triage enforce, got %d in human bucket", got)
	}
}

// TestFoldCalmArmedMonitorAddsNoRebootPage: an armed monitor that is below the
// high-water line (Advised=false) is a true no-op -- no page, no noise.
func TestFoldCalmArmedMonitorAddsNoRebootPage(t *testing.T) {
	r := Fold(Inputs{Reboot: []stallscan.RebootAdvice{{Advised: false}}})
	if got := rebootPages(r); len(got) != 0 {
		t.Fatalf("a below-line (calm) sample must not page, got %+v", got)
	}
}

// TestFoldDistinctRebootAxesEachPageOnce: two different crossings (a handle
// high-water on one process, a thread high-water on another) are two distinct
// operator decisions and each pages once.
func TestFoldDistinctRebootAxesEachPageOnce(t *testing.T) {
	handle := stallscan.RebootAdvice{
		Advised: true, Axis: "handle_high_water", Process: "WindowsTerminal.exe",
		PID: 10, Count: 40000, Threshold: 30000,
		Reason: "WindowsTerminal.exe holds 40000 handles (>= 30000 reboot high-water)",
	}
	thread := stallscan.RebootAdvice{
		Advised: true, Axis: "thread_high_water", Process: "node.exe",
		PID: 20, Count: 2500, Threshold: 2000,
		Reason: "node.exe holds 2500 threads (>= 2000 reboot high-water)",
	}
	r := Fold(Inputs{Reboot: []stallscan.RebootAdvice{handle, thread}})
	if got := len(rebootPages(r)); got != 2 {
		t.Fatalf("two distinct (axis,process) crossings should page twice, got %d", got)
	}
}
