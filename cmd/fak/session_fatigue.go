package main

// session_fatigue.go — `fak session gate-fatigue`, the READ-ONLY operator surface
// over the confirm-fatigue detector (#4427, part of #2753).
//
// It folds the existing fak.guard-stop.v1 ledger — the same file `fak guard-stops`
// tallies — into a per-gate approval-without-inspection rate, and names the gates
// that have crossed into rubber-stamp territory. Naming is all it does: coarsening
// a gate is the regime mechanism (#2389/#2405) and the autonomy dial (#2759), not
// this verb. Nothing here writes an event, mutates policy, or changes a gate.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// runFatigue is the testable core of `fak session gate-fatigue`: it returns the
// process exit code (0 ok, 1 an unreadable ledger, 2 a usage error) and takes its
// streams explicitly so a test can assert the rendered output.
func runFatigue(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session gate-fatigue", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak session gate-fatigue [--ledger PATH] [--trace ID] [--threshold F] [--min-fires N] [--json]")
		fmt.Fprintln(stderr, "  Fold the guard-stop stream into a per-gate approval-without-inspection")
		fmt.Fprintln(stderr, "  rate and name the rubber-stamped gates worth coarsening. Read-only.")
		fs.PrintDefaults()
	}
	ledgerFlag := fs.String("ledger", "", "path to the guard stops JSONL ledger (default: $FAK_GUARD_STOPS_LEDGER or the repo default)")
	traceFlag := fs.String("trace", "", "fold only this session id (default: every session in the ledger)")
	thresholdFlag := fs.Float64("threshold", sessionctl.DefaultFatigueThreshold, "soft fatigue bar a gate must reach to be flagged")
	minFiresFlag := fs.Int("min-fires", sessionctl.DefaultFatigueMinFires, "minimum firings before a gate can be flagged")
	jsonFlag := fs.Bool("json", false, "emit the report as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session gate-fatigue: unexpected argument %q (this verb takes flags only)\n", fs.Arg(0))
		return 2
	}

	ledger := strings.TrimSpace(*ledgerFlag)
	if ledger == "" {
		ledger = guardStopsLedgerResolved()
	}
	if ledger == "" {
		fmt.Fprintln(stderr, "fak session gate-fatigue: no ledger path (pass --ledger, set $FAK_GUARD_STOPS_LEDGER, or run inside a repo)")
		return 1
	}
	content, err := readGuardStopsLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak session gate-fatigue: read %s: %v\n", ledger, err)
		return 1
	}

	rep := sessionctl.FoldFatigue(sessionctl.ParseFatigueEvents(content), sessionctl.FatigueOptions{
		Threshold: *thresholdFlag,
		MinFires:  *minFiresFlag,
		Session:   *traceFlag,
	})
	if *jsonFlag {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak session gate-fatigue: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, formatFatigue(ledger, rep))
	return 0
}

// formatFatigue renders the report for a human reader: one line per gate identity,
// most fatigued first, then the flagged worklist.
func formatFatigue(ledger string, rep sessionctl.FatigueReport) string {
	var b strings.Builder
	if rep.Events == 0 {
		fmt.Fprintf(&b, "fak session gate-fatigue: no guard-stop events recorded yet.\n")
		fmt.Fprintf(&b, "  ledger: %s\n", ledger)
		b.WriteString("  The Stop hook records one row per turn-end decision once a `fak guard` session runs.")
		return b.String()
	}
	fmt.Fprintf(&b, "fak session gate-fatigue: %d event(s) over %d gate(s)", rep.Events, len(rep.Rows))
	if rep.Session != "" {
		fmt.Fprintf(&b, " (session %s)", rep.Session)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "  ledger: %s\n", ledger)
	fmt.Fprintf(&b, "  bar: fatigue >= %.2f with >= %d fires\n", rep.Threshold, rep.MinFires)
	fmt.Fprintf(&b, "  %-40s %6s %9s %10s %8s  %s\n", "GATE (stage/kind/disposition)", "FIRES", "APPROVED", "NO-INSPECT", "FATIGUE", "FLAG")
	for _, r := range rep.Rows {
		flag := ""
		if r.RubberStamped {
			flag = r.Flag
		}
		fmt.Fprintf(&b, "  %-40s %6d %9d %10d %8.2f  %s\n", r.Key, r.Fires, r.Approved, r.ApprovedWithoutInspection, r.Rate, flag)
	}
	if len(rep.Flagged) == 0 {
		b.WriteString("  → no rubber-stamped gate: every gate is either inspected or below the sample floor.")
		return b.String()
	}
	fmt.Fprintf(&b, "  → %d rubber-stamped gate(s): %s\n", len(rep.Flagged), strings.Join(rep.Flagged, ", "))
	b.WriteString("    These fire per call and are waved through without inspection — the approval-fatigue\n")
	b.WriteString("    failure mode. Coarsen them into a witnessed regime (#2389) or an autonomy level\n")
	b.WriteString("    (#2759) rather than leaving a prompt that trains blanket-approval. This verb only\n")
	b.WriteString("    names them; it never changes a gate.")
	return strings.TrimRight(b.String(), "\n")
}
