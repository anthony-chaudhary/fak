package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/superstream"
)

func cmdSuperstream(argv []string) { os.Exit(runSuperstream(os.Stdout, os.Stderr, argv)) }

func runSuperstream(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		superstreamUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "plan":
		return runSuperstreamPlan(stdout, stderr, argv[1:])
	case "step":
		return runSuperstreamStep(stdout, stderr, argv[1:])
	case "carryover":
		return runSuperstreamCarryover(stdout, stderr, argv[1:])
	case "status":
		return runSuperstreamStatus(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		superstreamUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak superstream: unknown subcommand %q\n", argv[0])
		superstreamUsage(stderr)
		return 2
	}
}

func superstreamUsage(w io.Writer) {
	fmt.Fprint(w, `fak superstream — high-level workstream coordinator: queue-ordered tasks,
lane leases, and long-turn context safety

Usage:
  fak superstream plan [--spec <path>] [--json]
  fak superstream step [--spec <path>] [--holder <name>] [--apply] [--result <result.json>] [--json]
  fak superstream carryover [--spec <path>] [--json]
  fak superstream status [--spec <path>] [--all] [--json]

Subcommands:
  plan       inspect the queue, target lane leases, and context budgets
  step       evaluate the single next state machine step (acquire, execute, witness, release, advance)
  carryover  render the compact O(1) state handoff seed for context-safe turn boundaries
  status     render current queue progression, active leases, and context safety verdict
`)
}

func sampleStreamSpec() superstream.StreamSpec {
	return superstream.StreamSpec{
		ID:     "stream-sample",
		Intent: "sample-workstream-progression",
		BasePins: []string{
			"maintain green test suite",
			"commit strictly by explicit paths",
			"release lane lease immediately upon task completion",
		},
		MaxTurnsTotal:   40,
		MaxTokensTotal:  200000,
		MaxTurnsPerItem: 8,
		Queue: []superstream.WorkItem{
			{
				ID:       "task-gateway-parity",
				Title:    "verify gateway governance parity",
				Lane:     "gateway",
				Tree:     []string{"internal/gateway/**"},
				MaxTurns: 8,
				Witness:  "go test ./internal/gateway/...",
			},
			{
				ID:       "task-docs-refresh",
				Title:    "refresh integration documentation",
				Lane:     "docs",
				Tree:     []string{"docs/integrations/**"},
				MaxTurns: 6,
				Witness:  "fak validate --mine docs/integrations/**",
			},
			{
				ID:       "task-engine-audit",
				Title:    "audit model engine memory demand",
				Lane:     "engine",
				Tree:     []string{"internal/engine/**"},
				MaxTurns: 10,
				Witness:  "go test ./internal/engine/...",
			},
		},
	}
}

func loadStreamSpec(path string) (superstream.StreamSpec, error) {
	if path == "" {
		return sampleStreamSpec(), nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return superstream.StreamSpec{}, fmt.Errorf("read spec %s: %w", path, err)
	}
	var spec superstream.StreamSpec
	if err := json.Unmarshal(b, &spec); err != nil {
		return superstream.StreamSpec{}, fmt.Errorf("parse spec %s: %w", path, err)
	}
	if err := spec.Validate(); err != nil {
		return superstream.StreamSpec{}, fmt.Errorf("spec %s: %w", path, err)
	}
	return spec, nil
}

// loadStreamState loads the durable state persisted next to specPath,
// reconciled against spec. A missing state file yields a fresh state; an
// unreadable, corrupt, or misplaced state is an error (fail closed rather than
// silently restarting at 0 completed). The built-in sample spec has no path and
// is therefore always stateless.
func loadStreamState(specPath string, spec superstream.StreamSpec) (superstream.StreamState, bool, error) {
	if specPath == "" {
		return superstream.NewStreamState(spec.NormalizedSpec()), false, nil
	}
	return superstream.LoadState(specPath, spec)
}

func runSuperstreamPlan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superstream plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	specPath := fs.String("spec", "", "path to superstream spec JSON (defaults to sample spec)")
	asJSON := fs.Bool("json", false, "emit plan as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}

	spec, err := loadStreamSpec(*specPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream plan: %v\n", err)
		return 1
	}
	norm := spec.NormalizedSpec()
	state, _, err := loadStreamState(*specPath, spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream plan: %v\n", err)
		return 1
	}
	safety := superstream.EvaluateContextSafety(norm, state)

	if *asJSON {
		payload := struct {
			Schema string                           `json:"schema"`
			Spec   superstream.StreamSpec           `json:"spec"`
			State  superstream.StreamState          `json:"state"`
			Safety superstream.ContextSafetyVerdict `json:"safety"`
		}{
			Schema: superstream.ReportSchema,
			Spec:   norm,
			State:  state,
			Safety: safety,
		}
		return encodeJSONOrFail(stdout, stderr, payload, "fak superstream plan")
	}

	fmt.Fprintf(stdout, "Super Workstream: %s (Intent: %s)\n", norm.ID, norm.Intent)
	fmt.Fprintf(stdout, "Budget: %d max total turns, %d max tokens, %d turns/item\n",
		norm.MaxTurnsTotal, norm.MaxTokensTotal, norm.MaxTurnsPerItem)
	fmt.Fprintf(stdout, "Context Safety: %s (%s)\n\n", safety.Status, safety.Reason)

	tw := tabwriter.NewWriter(stdout, 0, 8, 2, ' ', 0)
	fmt.Fprintln(tw, "#\tTASK ID\tLANE\tTURNS\tSTATUS\tWITNESS")
	for i, it := range norm.Queue {
		fmt.Fprintf(tw, "%d\t%s\t%s\t%d\t%s\t%s\n",
			i+1, it.ID, it.Lane, it.MaxTurns, it.Status, it.Witness)
	}
	tw.Flush()

	return 0
}

func runSuperstreamStep(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superstream step", flag.ContinueOnError)
	fs.SetOutput(stderr)
	specPath := fs.String("spec", "", "path to superstream spec JSON")
	holder := fs.String("holder", "superstream-agent", "holder identifier for lane lease checks")
	apply := fs.Bool("apply", false, "apply the decision to the durable state and refresh the registry")
	resultPath := fs.String("result", "", "path to a JSON file of the witnessed superstream.StepResult")
	asJSON := fs.Bool("json", false, "emit decision as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}

	spec, err := loadStreamSpec(*specPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream step: %v\n", err)
		return 1
	}
	norm := spec.NormalizedSpec()
	state, _, err := loadStreamState(*specPath, spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream step: %v\n", err)
		return 1
	}

	var result superstream.StepResult
	if *resultPath != "" {
		b, err := os.ReadFile(*resultPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak superstream step: read result %s: %v\n", *resultPath, err)
			return 1
		}
		if err := json.Unmarshal(b, &result); err != nil {
			fmt.Fprintf(stderr, "fak superstream step: parse result %s: %v\n", *resultPath, err)
			return 1
		}
	}

	tax := laneadmit.Taxonomy{Loaded: true}
	var liveLeases []laneadmit.Lease // offline/local check

	dec := superstream.DecideStep(norm, state, *holder, liveLeases, tax)

	persisted := false
	if *apply {
		now := time.Now()
		superstream.ApplyStep(norm, &state, dec, result, now)
		if *specPath != "" {
			if err := superstream.SaveState(*specPath, state, now); err != nil {
				fmt.Fprintf(stderr, "fak superstream step: %v\n", err)
				return 1
			}
			if err := superstream.UpsertRegistryEntry(superstream.SummaryFor(*specPath, norm, state, now), now); err != nil {
				fmt.Fprintf(stderr, "fak superstream step: %v\n", err)
				return 1
			}
			persisted = true
		} else {
			fmt.Fprintln(stderr, "fak superstream step: sample spec is stateless; --spec <path> required to persist")
		}
	}

	if *asJSON {
		if *apply {
			payload := struct {
				superstream.StepDecision
				Applied   bool                    `json:"applied"`
				Persisted bool                    `json:"persisted"`
				State     superstream.StreamState `json:"state"`
			}{StepDecision: dec, Applied: true, Persisted: persisted, State: state}
			return encodeJSONOrFail(stdout, stderr, payload, "fak superstream step")
		}
		return encodeJSONOrFail(stdout, stderr, dec, "fak superstream step")
	}

	fmt.Fprintf(stdout, "Super Workstream Step Decision: %s\n", dec.Action)
	fmt.Fprintf(stdout, "Reason: %s\n", dec.Reason)
	fmt.Fprintf(stdout, "Next Safe Step: %s\n", dec.NextSafeStep)
	if dec.Item != nil {
		fmt.Fprintf(stdout, "Target Item: %s (Lane: %s, Status: %s)\n", dec.Item.ID, dec.Item.Lane, dec.Item.Status)
	}
	if dec.LeaseRequest != nil {
		fmt.Fprintf(stdout, "Lease ID: %s (Tree: %v)\n", dec.LeaseRequest.LeaseID, dec.LeaseRequest.Tree)
	}
	fmt.Fprintf(stdout, "Context Safety: %s (turns remaining item: %d, total: %d)\n",
		dec.Safety.Status, dec.Safety.TurnsRemainingItem, dec.Safety.TurnsRemainingAll)
	if *apply {
		if persisted {
			fmt.Fprintf(stdout, "Persisted: %s (completed=%d failed=%d yielded=%d)\n",
				superstream.StateFilePath(*specPath), state.CompletedCount, state.FailedCount, state.YieldedCount)
		} else {
			fmt.Fprintf(stdout, "Persisted: none (sample spec is stateless; completed=%d failed=%d yielded=%d)\n",
				state.CompletedCount, state.FailedCount, state.YieldedCount)
		}
	}

	return 0
}

func runSuperstreamCarryover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superstream carryover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	specPath := fs.String("spec", "", "path to superstream spec JSON")
	asJSON := fs.Bool("json", false, "emit carryover as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}

	spec, err := loadStreamSpec(*specPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream carryover: %v\n", err)
		return 1
	}
	norm := spec.NormalizedSpec()
	state, _, err := loadStreamState(*specPath, spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream carryover: %v\n", err)
		return 1
	}

	seed := superstream.BuildCarryoverSeed(norm, state)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, seed, "fak superstream carryover")
	}

	fmt.Fprintf(stdout, "Stream Carryover Seed (O(1) Context Handshake)\n")
	fmt.Fprintf(stdout, "Stream ID: %s (Intent: %s)\n", seed.StreamID, seed.Intent)
	fmt.Fprintf(stdout, "Completed Items: %d / %d\n", len(seed.CompletedItems), seed.TotalItems)
	if seed.CurrentItem != nil {
		fmt.Fprintf(stdout, "Active Task: %s (%s, Lane: %s)\n", seed.CurrentItem.ID, seed.CurrentItem.Title, seed.CurrentItem.Lane)
	}
	fmt.Fprintf(stdout, "Remaining Budget: %d turns, %d tokens\n", seed.TurnsRemaining, seed.TokensRemain)
	fmt.Fprintf(stdout, "Stream Pins: %d defined\n", len(seed.StreamPins))
	for _, p := range seed.StreamPins {
		fmt.Fprintf(stdout, "  - %s\n", p)
	}

	return 0
}

func runSuperstreamStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superstream status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	specPath := fs.String("spec", "", "path to superstream spec JSON")
	all := fs.Bool("all", false, "serve the registry snapshot over every known stream instead of one spec")
	asJSON := fs.Bool("json", false, "emit status as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *all {
		return runSuperstreamStatusAll(stdout, stderr, *asJSON)
	}

	spec, err := loadStreamSpec(*specPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream status: %v\n", err)
		return 1
	}
	norm := spec.NormalizedSpec()
	state, _, err := loadStreamState(*specPath, spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream status: %v\n", err)
		return 1
	}
	safety := superstream.EvaluateContextSafety(norm, state)

	if *asJSON {
		payload := struct {
			State  superstream.StreamState          `json:"state"`
			Safety superstream.ContextSafetyVerdict `json:"safety"`
		}{
			State:  state,
			Safety: safety,
		}
		return encodeJSONOrFail(stdout, stderr, payload, "fak superstream status")
	}

	fmt.Fprintf(stdout, "Super Workstream Status: %s\n", state.StreamID)
	fmt.Fprintf(stdout, "Intent: %s\n", state.Intent)
	fmt.Fprintf(stdout, "Progress: %d completed, %d failed, %d yielded (total queue: %d)\n",
		state.CompletedCount, state.FailedCount, state.YieldedCount, len(state.Queue))
	if active := state.ActiveItem(); active != nil {
		fmt.Fprintf(stdout, "Active Item: %s (%s)\n", active.ID, active.Status)
	} else {
		fmt.Fprintf(stdout, "Active Item: none\n")
	}
	fmt.Fprintf(stdout, "Turns Spent: %d / %d total (current item: %d)\n",
		state.TotalTurnsSpent, norm.MaxTurnsTotal, state.CurrentItemTurns)
	fmt.Fprintf(stdout, "Context Safety: %s (%s)\n", safety.Status, safety.Reason)
	if state.CurrentLease != nil {
		fmt.Fprintf(stdout, "Active Lease: %s (Lane: %s, Holder: %s)\n",
			state.CurrentLease.LeaseID, state.CurrentLease.Lane, state.CurrentLease.Holder)
	} else {
		fmt.Fprintf(stdout, "Active Lease: none (lane free)\n")
	}

	return 0
}

// runSuperstreamStatusAll serves the registry snapshot: one file read, one
// row per stream, and never a spec or state file read. That is the O(1)-served
// navigation answer to "which of my W streams is blocked?".
func runSuperstreamStatusAll(stdout, stderr io.Writer, asJSON bool) int {
	entries, err := superstream.LoadRegistry()
	if err != nil {
		fmt.Fprintf(stderr, "fak superstream status: %v\n", err)
		return 1
	}

	if asJSON {
		if entries == nil {
			entries = []superstream.RegistrySummary{}
		}
		payload := struct {
			Schema  string                        `json:"schema"`
			Streams []superstream.RegistrySummary `json:"streams"`
		}{Schema: superstream.RegistrySchema, Streams: entries}
		return encodeJSONOrFail(stdout, stderr, payload, "fak superstream status")
	}

	if len(entries) == 0 {
		fmt.Fprintf(stdout, "No stream registry at %s; run 'fak superstream step --apply --spec <path>' to create entries.\n",
			superstream.RegistryFilePath())
		return 0
	}

	tw := tabwriter.NewWriter(stdout, 0, 8, 2, ' ', 0)
	fmt.Fprintln(tw, "STREAM\tSPEC\tCOMPLETED/TOTAL\tFAILED\tYIELDED\tTURNS\tCLOSED")
	for _, e := range entries {
		closed := "no"
		if e.Closed {
			closed = "yes"
		}
		fmt.Fprintf(tw, "%s\t%s\t%d/%d\t%d\t%d\t%d\t%s\n",
			e.StreamID, e.SpecPath, e.Completed, e.TotalItems, e.Failed, e.Yielded, e.TotalTurnsSpent, closed)
	}
	tw.Flush()

	return 0
}
