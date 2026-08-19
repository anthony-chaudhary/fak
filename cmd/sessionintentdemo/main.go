package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionintent"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("sessionintentdemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	objective := fs.String("objective", "research structured session needs", "session objective")
	minimum := fs.Duration("min-active", 0, "minimum active work before a normal stop is eligible")
	target := fs.Duration("target-active", 0, "planning target for active work")
	maximum := fs.Duration("max-elapsed", 0, "hard elapsed-time ceiling")
	selfcheck := fs.Bool("selfcheck", false, "run the canonical 2h minimum / 10h maximum witness")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "unexpected positional arguments")
		return 2
	}
	if *selfcheck {
		*objective, *minimum, *target, *maximum = "research structured session needs", 2*time.Hour, 4*time.Hour, 10*time.Hour
	}
	i := sessionintent.Intent{Version: "fak.session-intent/v1alpha1", Objective: *objective, Trigger: sessionintent.Trigger{Kind: sessionintent.TriggerImmediate}}
	if *minimum > 0 {
		i.Effort = append(i.Effort, sessionintent.EffortBound{Kind: sessionintent.BoundMinimum, Clock: sessionintent.ClockActive, Duration: *minimum})
	}
	if *target > 0 {
		i.Effort = append(i.Effort, sessionintent.EffortBound{Kind: sessionintent.BoundTarget, Clock: sessionintent.ClockActive, Duration: *target})
	}
	if *maximum > 0 {
		i.Effort = append(i.Effort, sessionintent.EffortBound{Kind: sessionintent.BoundMaximum, Clock: sessionintent.ClockElapsed, Duration: *maximum})
	}
	if err := i.Validate(); err != nil {
		fmt.Fprintf(stderr, "INVALID: %v\n", err)
		return 1
	}
	payload, err := json.MarshalIndent(i, "", "  ")
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintln(stdout, string(payload))
	if *selfcheck {
		tooEarly := i.DecideStop(sessionintent.Progress{Active: time.Hour, Elapsed: 3 * time.Hour})
		eligible := i.DecideStop(sessionintent.Progress{Active: 2 * time.Hour, Elapsed: 3 * time.Hour})
		timedOut := i.DecideStop(sessionintent.Progress{Active: 6 * time.Hour, Elapsed: 10 * time.Hour})
		fmt.Fprintf(stdout, "DECISIONS: 1h_active=%s 2h_active=%s 10h_elapsed=%s\n", tooEarly.State, eligible.State, timedOut.State)
		fmt.Fprintln(stdout, "SELFCHECK PASS: minimum and target govern stop eligibility/planning; maximum governs forced timeout")
	}
	return 0
}
