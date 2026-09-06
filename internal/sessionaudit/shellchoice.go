package sessionaudit

// The shell-choice KPI (#3227): which shell the fleet actually reaches for, and how
// often each one comes back broken.
//
// A hand-read of a 7-day window found agents choosing Bash over the documented
// "primary" Windows shell by roughly 6:1, and PowerShell erroring about seven times
// more often per call (194 calls / 5 errors = 2.6% vs 33 / 6 = 18.2%). Both halves
// were ALREADY in the artifact — Session.Tools counts the calls, Behavior.ToolErrors
// counts the errored results — but only as two tables nobody had joined, so the
// friction stayed an anecdote someone had to re-derive by eye. Joining them in the
// scope rollup turns it into one number a gate can watch and a fix can move.
//
// Two deliberate choices about the denominators:
//
//   - Calls are tool_use blocks; errors are errored tool_results. A call whose result
//     never arrived (session killed mid-call) counts as a call and NOT as an error.
//     The rate answers "how often did a shell call I made come back broken", which is
//     the friction the agent actually feels.
//   - CallShare is per-shell calls over SHELL calls, not over every tool call. The
//     question is which shell an agent picks once it has already decided to run a
//     command; dividing by every Read and Grep in the window would move the number
//     whenever unrelated tool use moved.
//
// Rates are pointers because this package distinguishes "the answer is zero" from
// "there is no answer" everywhere else it reports a ratio (ReadOnlyFrac, IORatio,
// SelfHostedShare). A window with no shell calls has NO shell error rate — reporting
// 0% there would read as "the shells were flawless", a much stronger claim.

// ShellTools is the closed shell-tool vocabulary this KPI reports, in report order.
// behavior.go's shellTools set is built from it so the two cannot drift apart when a
// third shell shows up.
var ShellTools = []string{"Bash", "PowerShell"}

func shellToolSet() map[string]bool {
	set := make(map[string]bool, len(ShellTools))
	for _, name := range ShellTools {
		set[name] = true
	}
	return set
}

// ShellStat is one shell's row of the KPI.
type ShellStat struct {
	Tool   string `json:"tool"`
	Calls  int64  `json:"calls"`
	Errors int64  `json:"errors"`
	// CallShare is this shell's share of all shell calls — the "choice" half.
	CallShare *float64 `json:"call_share"`
	// ErrorRate is errored results over calls — the "does it work" half.
	ErrorRate *float64 `json:"error_rate"`
}

// ShellChoice is the shell-choice fold of a tool-mix rollup joined to a tool-error
// rollup: one row per known shell plus the all-shell totals.
type ShellChoice struct {
	Shells []ShellStat `json:"shells"`
	Calls  int64       `json:"calls"`
	Errors int64       `json:"errors"`
	// ErrorRate is the all-shell error rate: the single number the KPI drives down.
	ErrorRate *float64 `json:"error_rate"`
	// Preferred names the most-called shell (ties broken by ShellTools order), so the
	// "agents avoid the primary shell" half is legible without reading the table.
	// Empty when no shell was called at all.
	Preferred string `json:"preferred,omitempty"`
}

// FoldShellChoice joins a tool-call rollup (Aggregate.ToolMix or Session.Tools) to a
// tool-error rollup (Behavior.ToolErrors, summed) over the shell vocabulary.
//
// Every shell in ShellTools gets a row even at zero calls: a shell the fleet stopped
// reaching for entirely is exactly the signal this KPI exists to show, and a
// disappearing row would read as "no data" instead of "nobody picked it".
func FoldShellChoice(toolCalls, toolErrors map[string]int64) ShellChoice {
	sc := ShellChoice{Shells: make([]ShellStat, 0, len(ShellTools))}
	for _, name := range ShellTools {
		sc.Calls += toolCalls[name]
		sc.Errors += toolErrors[name]
	}
	var topCalls int64
	for _, name := range ShellTools {
		calls, errs := toolCalls[name], toolErrors[name]
		sc.Shells = append(sc.Shells, ShellStat{
			Tool:      name,
			Calls:     calls,
			Errors:    errs,
			CallShare: ratio(calls, sc.Calls),
			ErrorRate: ratio(errs, calls),
		})
		if calls > topCalls {
			sc.Preferred, topCalls = name, calls
		}
	}
	sc.ErrorRate = ratio(sc.Errors, sc.Calls)
	return sc
}

// SessionShellErrorRate is one session's all-shell error rate — the per-session value
// the distribution block summarizes, so an outlier session is visible rather than
// averaged away by the corpus-wide rate.
//
// nil for a session that ran no shell command at all; those sessions are absent from
// the distribution rather than counted as a clean 0%.
func SessionShellErrorRate(s Session) *float64 {
	sc := FoldShellChoice(s.Tools, s.Behavior.ToolErrors)
	return sc.ErrorRate
}

// CompactShellChoice folds the shell-choice KPI (#3227, #6523) into the compact report:
// per-shell calls, call share, errors, error rate, preferred shell, and the per-session
// shell-error-rate distribution.
type CompactShellChoice struct {
	ShellChoice
	ShellErrorRate StatSet `json:"shell_error_rate"`
}

// buildCompactShellChoice folds the shell-choice KPI into the compact report format,
// joining agg.ShellChoice (or folding from sessions) with the per-session shell error
// rate distribution. An empty window reports UNKNOWN for Preferred and nil for ErrorRate.
func buildCompactShellChoice(sessions []Session, agg Aggregate) CompactShellChoice {
	sc := agg.ShellChoice
	if len(sc.Shells) == 0 {
		toolCalls := map[string]int64{}
		toolErrors := map[string]int64{}
		for _, s := range sessions {
			addMap(toolCalls, s.Tools)
			addMap(toolErrors, s.Behavior.ToolErrors)
		}
		sc = FoldShellChoice(toolCalls, toolErrors)
	}
	if sc.Calls == 0 {
		sc.Preferred = "UNKNOWN"
	}
	dist := agg.Distributions.ShellErrorRate
	if dist.Median == nil && len(sessions) > 0 {
		var shellErrs []float64
		for _, s := range sessions {
			if r := SessionShellErrorRate(s); r != nil {
				shellErrs = append(shellErrs, *r)
			}
		}
		if len(shellErrs) > 0 {
			dist = stat(shellErrs, false, false, true)
		}
	}
	return CompactShellChoice{
		ShellChoice:    sc,
		ShellErrorRate: dist,
	}
}
