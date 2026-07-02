package main

// session_envelope.go -- issue #1573 (managed-context epic #1570, product track):
// `fak session envelope`, the CLI front end over internal/session.ParseEnvelopeFlags.
// It is the one command a user runs to STATE a managed-run budget goal in plain
// flags (token/wall-clock/turn/spend/throughput) and get back the exact
// deterministic Budget/TimeBudget/Pace the runtime will enforce -- the "inspect the
// parsed deterministic budget" half of the issue's Done condition made concrete.
//
// Like `fak session reset-diff`, this is OFFLINE by design: parsing a stated
// envelope needs no live gateway, so it is dispatched before the gateway-shaped
// arity table in runSession (which assumes every other verb talks to a
// sessionClient).
//
//	fak session envelope --tokens 50000 --wall-clock 10m --turns 25 --spend $5 --throughput 20
//	fak session envelope --tokens unbounded --json

import (
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// runSessionEnvelope is the testable shell: it returns the process exit code (0 ok,
// 2 a malformed envelope/usage error) and takes its streams explicitly, mirroring
// runSessionResetDiff.
func runSessionEnvelope(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak session envelope", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tokens := fs.String("tokens", "", "cap total output tokens (integer, or \"unbounded\"); unset = not stated")
	wallClock := fs.String("wall-clock", "", "cap real elapsed time across the run's lineage (a duration like \"10m\", \"1h30m\"); unset = not stated")
	turns := fs.String("turns", "", "cap the number of model round-trips (integer, or \"unbounded\"); unset = not stated")
	spend := fs.String("spend", "", "cap the rough dollar cost (e.g. \"$5\", \"5.25\"); unset = not stated")
	throughput := fs.String("throughput", "", "the minimum tokens/sec this run is expected to sustain; unset = not stated")
	asJSON := fs.Bool("json", false, "emit the parsed envelope as JSON instead of the human summary")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	env, err := session.ParseEnvelopeFlags(*tokens, *wallClock, *turns, *spend, *throughput)
	if err != nil {
		fmt.Fprintf(stderr, "fak session envelope: %v\n", err)
		return 2
	}
	parsed := env.Parse()

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, parsed, "fak session envelope")
	}
	fmt.Fprint(stdout, formatParsedEnvelope(parsed))
	return 0
}

// formatParsedEnvelope renders the deterministic parse as a short, scannable
// human summary -- the stated envelope axes next to the runtime Budget/TimeBudget
// values they produced, so an operator can eyeball "what I asked for" against
// "what the runtime will enforce" without reading JSON.
func formatParsedEnvelope(p session.ParsedEnvelope) string {
	return fmt.Sprintf(
		"envelope: tokens=%s wall_clock=%s turns=%s spend=%s throughput_floor=%s\n"+
			"budget:   turns_left=%s tokens_left=%s\n"+
			"time:     limit=%s bounded=%v\n",
		envelopeIntAxis(p.Envelope.Tokens),
		envelopeDurationAxis(p.Envelope.WallClock),
		envelopeIntAxis(p.Envelope.Turns),
		envelopeSpendAxis(p.Envelope.SpendCapCents),
		envelopeThroughputAxis(p.Envelope.ThroughputFloor),
		budgetAxis(p.Budget.TurnsLeft),
		budgetAxis(p.Budget.TokensLeft),
		time.Duration(p.TimeBudget.LimitNanos),
		p.TimeBudget.Bounded(),
	)
}

func envelopeIntAxis(v int) string {
	if v == 0 {
		return "(not stated)"
	}
	if v < 0 {
		return "unbounded"
	}
	return fmt.Sprintf("%d", v)
}

func envelopeDurationAxis(d time.Duration) string {
	if d == 0 {
		return "(not stated)"
	}
	return d.String()
}

func envelopeSpendAxis(cents int64) string {
	if cents == 0 {
		return "(not stated)"
	}
	return fmt.Sprintf("$%d.%02d", cents/100, cents%100)
}

func envelopeThroughputAxis(f float64) string {
	if f == 0 {
		return "(not stated)"
	}
	return fmt.Sprintf("%.2f tok/s", f)
}

func sessionEnvelopeUsage(w io.Writer) {
	fmt.Fprint(w, `fak session envelope -- parse a user budget envelope into the deterministic runtime budget

  fak session envelope [--tokens N|unbounded] [--wall-clock DUR] [--turns N|unbounded]
                        [--spend $D] [--throughput RATE] [--json]

Every flag is optional; an unset axis is "not stated" and parses to the runtime's
own unbounded default. This command is offline: it never dials a live gateway.
`)
}
