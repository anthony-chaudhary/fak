package agent

import (
	"fmt"
	"io"
	"strings"
)

// PrintReport renders the A/B run as a human-readable summary.
func PrintReport(w io.Writer, r *RunResult, trace []traceEvent, path string) {
	seam := "OFFLINE (deterministic mock planner)"
	if r.Live {
		provider := r.Provider
		if provider == "" {
			provider = "openai"
		}
		seam = "LIVE (provider " + provider + ", real model: " + r.Model + ", transcript " + r.Transcript + ")"
	}
	fmt.Fprintf(w, "== fak agent: turn-use vs now ==\n")
	fmt.Fprintf(w, "seam        : %s\n", seam)
	fmt.Fprintf(w, "task        : %s\n\n", oneLine(r.Task, 90))

	fmt.Fprintf(w, "%-26s %12s %12s\n", "metric", "now(base)", "fak")
	fmt.Fprintf(w, "%-26s %12s %12s\n", strings.Repeat("-", 26), "----------", "----------")
	row := func(label string, base, fak int) {
		fmt.Fprintf(w, "%-26s %12d %12d\n", label, base, fak)
	}
	row("model turns", r.Baseline.Turns, r.Fak.Turns)
	row("tool calls", r.Baseline.ToolCalls, r.Fak.ToolCalls)
	row("tool errors (-> retries)", r.Baseline.ToolErrors, r.Fak.ToolErrors)
	row("prompt tokens", r.Baseline.PromptTokens, r.Fak.PromptTokens)
	row("completion tokens", r.Baseline.CompletionTokens, r.Fak.CompletionTokens)
	fmt.Fprintf(w, "%-26s %12s %12d\n", "in-syscall repairs", "n/a", r.Fak.Repairs)
	fmt.Fprintf(w, "%-26s %12s %12d\n", "vDSO dedup hits", "n/a", r.Fak.VDSOHits)
	fmt.Fprintf(w, "%-26s %12s %12d\n", "adjudicator denies", "n/a", r.Fak.Denies)
	fmt.Fprintf(w, "%-26s %12s %12d\n", "MMU quarantines", "n/a", r.Fak.Quarantines)
	fmt.Fprintf(w, "%-26s %12s %12s\n", "injection in context", yn(r.Baseline.InjectionInContext), yn(r.Fak.InjectionInContext))
	fmt.Fprintf(w, "%-26s %12s %12s\n", "destructive op executed", yn(r.Baseline.DestructiveExecuted), yn(r.Fak.DestructiveExecuted))
	fmt.Fprintf(w, "%-26s %12s %12s\n", "task completed (booked)", yn(r.Baseline.TaskCompleted), yn(r.Fak.TaskCompleted))

	fmt.Fprintf(w, "\nHEADLINE\n")
	if r.BothCompleted {
		fmt.Fprintf(w, "  turns saved by fak        : %d  (%s)   [both arms completed -> comparable]\n", r.TurnsSaved, pct(r.TurnsSaved, r.Baseline.Turns))
		fmt.Fprintf(w, "  tokens saved by fak       : %d  (%s)\n", r.TokensSaved, pct(r.TokensSaved, r.Baseline.PromptTokens+r.Baseline.CompletionTokens))
	} else {
		// Turn count is NOT comparable when the arms completed different work — a
		// derailed baseline "saves" turns by failing the task. Report it honestly.
		fmt.Fprintf(w, "  turn delta NOT comparable : baseline completed=%s, fak completed=%s\n", yn(r.Baseline.TaskCompleted), yn(r.Fak.TaskCompleted))
		if r.Fak.TaskCompleted && !r.Baseline.TaskCompleted {
			fmt.Fprintf(w, "  -> the baseline FAILED the task (derailed/incomplete); fak completed it safely\n")
		}
	}
	fmt.Fprintf(w, "  poisoned result blocked   : %s\n", yn(r.Baseline.InjectionInContext && !r.Fak.InjectionInContext))
	fmt.Fprintf(w, "  destructive op prevented  : %s\n", yn(r.Baseline.DestructiveExecuted && !r.Fak.DestructiveExecuted))
	fmt.Fprintf(w, "\nreport written: %s\n", path)
}

// RenderTrace renders the per-call trace log as text.
func RenderTrace(trace []traceEvent) []byte {
	var b strings.Builder
	b.WriteString("# fak agent per-call trace (fak arm then baseline arm)\n\n")
	for _, e := range trace {
		fmt.Fprintf(&b, "[%-8s turn %d] %-22s args=%s\n", e.Arm, e.Turn, e.Tool, oneLine(e.RawArgs, 80))
		if e.Verdict != "" {
			line := "          verdict=" + e.Verdict
			if e.By != "" {
				line += " by=" + e.By
			}
			if e.Reason != "" {
				line += " reason=" + e.Reason
			}
			if e.Disposition != "" {
				line += " disposition=" + e.Disposition
			}
			fmt.Fprintf(&b, "%s\n", line)
		}
		if e.Note != "" {
			fmt.Fprintf(&b, "          note=%s\n", e.Note)
		}
	}
	return []byte(b.String())
}

func yn(b bool) string {
	if b {
		return "YES"
	}
	return "no"
}

func pct(part, whole int) string {
	if whole == 0 {
		return "0%"
	}
	return fmt.Sprintf("%.0f%%", 100*float64(part)/float64(whole))
}

func oneLine(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
