package session

import (
	"fmt"
	"io"
	"path/filepath"
	"time"
)

// CompactOverviewOptions records inferred defaults so the human report explains its
// own measurement scope rather than hiding it behind convenient automation.
type CompactOverviewOptions struct {
	Days      int
	Since     time.Time
	Workspace string
	Roots     int
}

// RenderCompactOverview is the concise user-facing view of deterministic rollout
// telemetry. The expert fire/regrowth report remains available through compact-audit.
func RenderCompactOverview(w io.Writer, res CompactAuditResult, opt CompactOverviewOptions) {
	a := res.Aggregate
	verdict := "NO DATA"
	detail := "no Codex token telemetry matched this workspace and window"
	if a.MeasuredFires > 0 {
		verdict = "WORKING"
		detail = fmt.Sprintf("%d measured fires shed %s resident tokens; median occupancy %s -> %s (%.1f%% remains)",
			a.MeasuredFires, humanInt(a.ResidentTokensShed), humanInt(int64(a.MedianPreTokens)),
			humanInt(int64(a.MedianPostTokens)), 100*a.MedianResidualRatio)
	} else if a.Sessions > 0 && a.VerdictCounts[VerdictNoFireAtCeiling] > 0 {
		verdict = "CHECK"
		detail = fmt.Sprintf("%d sessions reached the context ceiling without a measured compaction fire", a.VerdictCounts[VerdictNoFireAtCeiling])
	} else if a.Sessions > 0 {
		verdict = "BOUNDED"
		detail = "no measured fires were needed; observed resident context stayed bounded"
	}
	fmt.Fprintf(w, "Codex context — %s\n", verdict)
	fmt.Fprintf(w, "  %s\n", detail)
	workspace := opt.Workspace
	if workspace == "" {
		workspace = "all workspaces"
	} else {
		workspace = filepath.Clean(workspace)
	}
	fmt.Fprintf(w, "  scope: %d calendar days since %s · %s · %d profile root(s)\n", opt.Days, opt.Since.Format("2006-01-02"), workspace, opt.Roots)
	fmt.Fprintf(w, "  observed: %d sessions · %s cumulative input tokens\n", a.Sessions, humanInt(a.CumulativeInputTokens))
	if len(a.Daily) > 0 {
		fmt.Fprintln(w, "\n  daily (UTC):")
		fmt.Fprintln(w, "    date         sessions    cumulative input    fires    resident shed")
		for _, d := range a.Daily {
			fmt.Fprintf(w, "    %-10s   %8d   %17s   %5d   %13s\n", d.Date, d.Sessions, humanInt(d.CumulativeInputTokens), d.Fires, humanInt(d.ResidentTokensShed))
		}
	}
	fmt.Fprintln(w, "\n  resident shed is measured occupancy reduction, not a billable-token savings estimate.")
	fmt.Fprintln(w, "  More detail: fak session compact-audit --since <date>")
}

func humanInt(v int64) string {
	s := fmt.Sprintf("%d", v)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
