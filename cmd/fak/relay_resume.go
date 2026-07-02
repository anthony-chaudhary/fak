// relay_resume.go — rung C7 of the perpetual-session relay track (#1876, epic #1860):
// `fak relay resume`, the OFFLINE read half of the baton IO pair. A relay leg's successor
// (or an operator) points it at a `fak.relay.baton.v1` file and sees exactly what a fresh
// leg would receive — the human summary for inspection, or `--json` for the canonical
// wire bytes (relay.Marshal over the parsed value, so the output is the byte-stable
// round-trip form, not an echo of the input file).
//
// Deliberately pure read/print: no reload re-verification (that is track D —
// relay.VerifyReload re-checks the ProgressCursor against git), no resolver calls, no
// clock, no network. The one content check it DOES make is the reader contract's step-3
// schema gate: an object whose `schema` tag is not relay.Schema is refused, because
// printing a non-baton in baton clothing would mislead the very inspection this verb
// exists for. Everything printed is display-only — a baton carries pointers and cursors
// the successor re-verifies, never trusted progress (the no-`claimed` invariant).
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

func cmdRelay(argv []string) { os.Exit(runRelay(os.Stdin, os.Stdout, os.Stderr, argv)) }

// runRelay is the testable `fak relay` dispatcher: it returns the process exit code
// (0 ok, 1 a runtime error, 2 a usage error) and takes its streams explicitly so a test
// drives it without a process. stdin backs `--baton -`.
func runRelay(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		relayUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "resume":
		return runRelayResume(stdin, stdout, stderr, argv[1:])
	case "help", "-h", "--help":
		relayUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak relay: unknown subcommand %q (want resume)\n", argv[0])
		relayUsage(stderr)
		return 2
	}
}

// runRelayResume loads one baton file (or stdin with `--baton -`), gates it on the
// schema tag, and prints it — the aligned human summary by default, the canonical
// re-marshaled wire bytes with --json.
func runRelayResume(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("relay resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	batonPath := fs.String("baton", "", `path to the baton JSON file ("-" reads stdin)`)
	asJSON := fs.Bool("json", false, "emit the canonical baton wire bytes (the byte-stable round-trip form) instead of the human summary")
	// A bare positional path is accepted in either order (`fak relay resume baton.json
	// --json` or `... --json baton.json`), matching the repo's positional-leading verb
	// convention; flag.Parse stops at the first non-flag, so a leading positional must
	// be peeled off before the parse. A bare "-" (stdin) counts as a positional too.
	rest := argv
	positional := ""
	if len(rest) > 0 && (rest[0] == "-" || !strings.HasPrefix(rest[0], "-")) {
		positional = rest[0]
		rest = rest[1:]
	}
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	sources := fs.Args()
	if positional != "" {
		sources = append([]string{positional}, sources...)
	}
	if *batonPath != "" {
		sources = append([]string{*batonPath}, sources...)
	}
	if len(sources) != 1 {
		fmt.Fprintln(stderr, "fak relay resume: exactly one baton is required (--baton <path>, --baton -, or a single positional path)")
		return 2
	}
	path := sources[0]

	var data []byte
	var err error
	if path == "-" {
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak relay resume: read baton: %v\n", err)
		return 1
	}
	b, err := relay.Parse(data)
	if err != nil {
		fmt.Fprintf(stderr, "fak relay resume: %v\n", err)
		return 1
	}
	// Reader contract step 3: reject any object that does not carry the exact schema
	// tag, BEFORE trusting any other field. A zero baton (no schema, no relay id) hits
	// this same gate, so an empty `{}` cannot print as a real handoff.
	if b.Schema != relay.Schema {
		fmt.Fprintf(stderr, "fak relay resume: not a %s baton (schema %q)\n", relay.Schema, b.Schema)
		return 1
	}

	if *asJSON {
		out, err := relay.Marshal(b)
		if err != nil {
			fmt.Fprintf(stderr, "fak relay resume: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(out))
		return 0
	}
	printBaton(stdout, b)
	return 0
}

// printBaton renders the human summary in the schema doc's stable field order. Every
// field is shown — an empty list prints as "(none)" — so an operator reading the summary
// sees the WHOLE handoff a successor leg would receive, not a lossy digest of it.
func printBaton(w io.Writer, b relay.Baton) {
	fmt.Fprintf(w, "baton %s  relay=%s leg=%d\n", b.Schema, b.RelayID, b.Leg)
	fmt.Fprintf(w, "  parent_trace:   %s\n", orNone(b.ParentTrace))
	fmt.Fprintf(w, "  tombstone:      %s @ %s%s\n", orNone(b.Tombstone.Reason), orNone(b.Tombstone.AtSHA), noteSuffix(b.Tombstone.Note))
	fmt.Fprintf(w, "  objective:      %s\n", orNone(b.Objective.Text))
	fmt.Fprintf(w, "    pin:          %s digest=%s\n", orNone(b.Objective.PinID), orNone(b.Objective.Digest))
	fmt.Fprintf(w, "  done_when:      %s\n", orNone(b.DoneWhen))
	fmt.Fprintf(w, "  next_action:    %s\n", orNone(b.NextAction))
	fmt.Fprintf(w, "  progress_cursor (re-verify before trusting — never a claim):\n")
	fmt.Fprintf(w, "    start_sha:    %s\n", orNone(b.ProgressCursor.StartSHA))
	fmt.Fprintf(w, "    ledger_ref:   %s\n", orNone(b.ProgressCursor.LedgerRef))
	fmt.Fprintf(w, "    held_region:  %s\n", joinOrNone(b.ProgressCursor.HeldRegion))
	fmt.Fprintf(w, "  open_questions: %s\n", joinOrNone(b.OpenQuestions))
	if len(b.Artifacts) == 0 {
		fmt.Fprintf(w, "  artifacts:      (none)\n")
	} else {
		fmt.Fprintf(w, "  artifacts:\n")
		for _, a := range b.Artifacts {
			fmt.Fprintf(w, "    %-8s %s\n", a.Kind, a.Ref)
		}
	}
	fmt.Fprintf(w, "  do_not_rederive: %s\n", joinOrNone(b.DoNotRederive))
}

// orNone renders an optional scalar field, keeping an empty value visible rather than
// silently dropping the line.
func orNone(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(none)"
	}
	return s
}

// joinOrNone renders a list field on one line, "(none)" when empty.
func joinOrNone(ss []string) string {
	if len(ss) == 0 {
		return "(none)"
	}
	return strings.Join(ss, "  ")
}

// noteSuffix appends the tombstone's display-only note when present.
func noteSuffix(note string) string {
	if strings.TrimSpace(note) == "" {
		return ""
	}
	return "  (" + note + ")"
}

func relayUsage(w io.Writer) {
	fmt.Fprint(w, `fak relay — perpetual-session relay: inspect the baton a leg hands its successor

  fak relay resume --baton <path>|- [--json]
  fak relay resume <path>

resume loads one fak.relay.baton.v1 file (or stdin with --baton -) and prints what a
fresh relay leg would receive: the objective pin, the done_when predicate, the
re-verifiable progress cursor, the next action, and the pointer-only artifact/dead-end
lists. --json emits the canonical wire bytes (relay.Marshal over the parsed value), so
piping the output back in round-trips byte-identically. Offline: no reload
re-verification (see relay.VerifyReload), no resolver calls, no network.

example (inspect a baton written by a rotated-out leg):
  fak relay resume --baton .fak/relay/RID-2026-07-01-a.baton.json

example (canonical round-trip):
  fak relay resume --baton leg.json --json | fak relay resume --baton - --json
`)
}
