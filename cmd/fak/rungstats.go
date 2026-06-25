package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/rungobs"
)

// rungStatsProbes is the FIXED probe set `fak rungstats` folds through the real
// adjudicator chain. Each tool's verdict under the built-in DefaultPolicy is stable:
// three affirmatively-allowed tools, then one tool per distinct deny reason
// (POLICY_BLOCK / SECRET_EXFIL / DEFAULT_DENY). So the printed distribution exercises
// the ALLOW bucket and all three structural-deny buckets deterministically, with no
// gateway and no engine. It is the offline counterpart of the live /metrics row
// fak_kernel_decisions_total{rung,kind,reason} (internal/gateway/metrics.go).
var rungStatsProbes = []string{
	"get_user_details", // exact allow
	"read_config",      // read_ prefix allow
	"calculate",        // exact allow
	"shell_rm_rf",      // deny: POLICY_BLOCK
	"exfiltrate",       // deny: SECRET_EXFIL
	"frobnicate_xyz",   // deny: DEFAULT_DENY (unknown tool)
}

// cmdRungStats is the os.Exit-wrapping entry point dispatched by main. The exit code
// is produced by runRungStats so the dispatch and the test share one code path
// (mirroring cmdAccounts/runAccounts).
func cmdRungStats(argv []string) { os.Exit(runRungStats(os.Stdout, argv)) }

// runRungStats registers a PASSIVE rungobs observer, folds the fixed probe set through
// the real adjudicator chain via kernel.Decide (pure adjudication — no fast path,
// engine, or network), and prints the per-(rung,kind,reason) decision distribution.
// It is the OFFLINE read-out of the same rungobs.Observer the gateway exposes live on
// /metrics, so the distribution is inspectable without a running gateway. It returns a
// process exit code (0 on success, 2 on a flag error) and never mutates kernel state —
// the observer only reads the verdict stream the fold already emits.
func runRungStats(w io.Writer, argv []string) int {
	fs := flag.NewFlagSet("rungstats", flag.ContinueOnError)
	fs.SetOutput(w)
	fs.Usage = func() {
		fmt.Fprintln(w, "usage: fak rungstats   (fold a fixed probe set through the real adjudicator chain and print the rung-decision distribution)")
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	// The observer is a global Emitter: kernel.Decide fans its EvDecide/EvDeny events
	// out through the process registry, so it must be registered before the probes run.
	obs := rungobs.New()
	abi.RegisterEmitter(obs)

	k := kernel.New("inkernel")
	res := abi.ActiveResolver()
	for _, tool := range rungStatsProbes {
		ref, err := res.Put(ctx(), []byte("{}"))
		if err != nil {
			fmt.Fprintln(w, "fak rungstats:", err)
			return 1
		}
		// Decide folds ONLY the adjudicator chain and emits the decision the observer
		// folds — no engine dispatch, so the read-out is fully offline and deterministic.
		k.Decide(ctx(), &abi.ToolCall{Tool: tool, Args: ref})
	}

	printRungStats(w, obs.Snapshot(), obs.Total(), len(rungStatsProbes))
	return 0
}

// printRungStats renders the rung-decision distribution as an aligned table. Rows
// arrive already sorted by (rung, kind, reason) from Snapshot; an empty reason (an
// allow carries none) renders as "-" so the column never collapses.
func printRungStats(w io.Writer, rows []rungobs.DecisionRow, total int64, probes int) {
	fmt.Fprintf(w, "fak rungstats — passive rung-decision distribution (%d probes, real fold, no gateway)\n\n", probes)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "RUNG\tKIND\tREASON\tCOUNT")
	for _, r := range rows {
		reason := r.Reason
		if reason == "" {
			reason = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\n", r.Rung, r.Kind, reason, r.Count)
	}
	fmt.Fprintf(tw, "\t\tTOTAL\t%d\n", total)
	_ = tw.Flush()
	fmt.Fprintln(w, "\ndrill down on one call with: fak preflight --tool <name> --explain")
}
