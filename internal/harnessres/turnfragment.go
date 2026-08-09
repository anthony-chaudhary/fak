package harnessres

import (
	"fmt"
	"strings"
)

// TurnFragment renders the compact PER-TURN resource fragment for the guard's
// `--debug-stats` economy line (#2050) — `res: cpu=12% rss=64MB io=1.2MBr/0.3MBw`.
//
// It is the fourth renderer beside Report (the exit-summary line), PrometheusText (the
// /metrics family, #2047) and MarshalLedgerRow (the durable JSONL row), and exists for the
// same reason they all live here: one owner for the numbers, so the per-turn line, the
// `fak info` pane row and the exit summary can never disagree about what this session
// burned. The host hands the rendered string to its consumer the way `fak guard` already
// hands PrometheusText() to the gateway's metrics child — the consumer keeps no dependency
// on this package.
//
// Three deliberate shape choices:
//
//   - KERNEL HALF ONLY. The Agent half is folded from the wrapped child's EXIT state
//     (FoldChildExit), so mid-session it is structurally empty: rendering it once per turn
//     would print a fake 0 — or a n/a that never resolves — on every line of a live run.
//     The agent's numbers are honest exactly once, at exit, and that is where Report()
//     prints them.
//   - PRESENCE BITS GATE EVERY AXIS, so a platform that cannot read one omits it rather
//     than reporting a fabricated 0 (the package's standing rule). CPU% additionally needs
//     positive elapsed wall time before an average means anything.
//   - EMPTY MEANS SILENT. With nothing observed it returns "", so the caller appends
//     nothing at all — matching the posture of the per-turn line's other optional fields
//     (safety=, nudge=, harness_rewrite=), which appear only once they have something to
//     say. A caller therefore never has to test the snapshot itself.
//
// The returned fragment carries no leading or trailing space: the caller owns the
// separator it joins with, exactly as it does for the rest of that line.
func (s Snapshot) TurnFragment() string {
	var parts []string
	if pct, ok := s.Kernel.CPUPercentAvg(s.Elapsed); ok {
		parts = append(parts, fmt.Sprintf("cpu=%.0f%%", pct))
	}
	if s.Kernel.HaveRSS {
		parts = append(parts, "rss="+compactBytes(s.Kernel.RSSBytes))
	}
	if s.Kernel.HaveIO {
		parts = append(parts, "io="+compactBytes(s.Kernel.IOReadBytes)+"r/"+compactBytes(s.Kernel.IOWriteBytes)+"w")
	}
	if len(parts) == 0 {
		return ""
	}
	return "res: " + strings.Join(parts, " ")
}

// compactBytes renders a byte count for a single glanceable cell — no space, one unit.
// It mirrors the `fak info` pane's own byte cell (guardInfoBytesText) digit for digit,
// including its binary math under decimal-looking MB/GB labels, so the same reading is
// never spelled two ways across the two surfaces an operator watches side by side.
func compactBytes(b uint64) string {
	const mb = 1 << 20
	const gb = 1 << 30
	if b >= gb {
		return fmt.Sprintf("%.1fGB", float64(b)/gb)
	}
	if b >= 10*mb {
		return fmt.Sprintf("%.0fMB", float64(b)/mb)
	}
	return fmt.Sprintf("%.1fMB", float64(b)/mb)
}
