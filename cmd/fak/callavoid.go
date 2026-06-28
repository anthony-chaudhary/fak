package main

// fak callavoid -- the operator-facing CLI surface over internal/callavoid, the pure
// avoided-call economics leaf. It exposes the package's two artifacts as JSON-first
// subcommands so an operator (or a CI line) can audit the economics without writing Go:
//
//   fak callavoid prove-memo  [--in FILE] [--json]   < MemoInput JSON   -> MemoProof
//   fak callavoid account     [--in FILE] [--json] [--gate] < Tally JSON -> TurnReport
//
// The arithmetic stays pure and deterministic (it is internal/callavoid's, verbatim);
// the CLI only marshals. It exits 0 on a valid decision, 2 on malformed input, and --
// only in the explicit --gate mode -- 1 when the window is REGRESSING (avoidance is a
// net loss). Input is read from --in or stdin; field names are the Go struct fields,
// matched case-insensitively (e.g. {"accesses":8,"validate_cost":0.02} for prove-memo,
// {"memo_hit":9,"execute":1} for account). Documented in docs/cli-reference.md.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/callavoid"
)

func cmdCallavoid(argv []string) { os.Exit(runCallavoid(os.Stdin, os.Stdout, os.Stderr, argv)) }

func runCallavoid(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, callavoidUsage)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "prove-memo":
		return runCallavoidProveMemo(stdin, stdout, stderr, rest)
	case "account":
		return runCallavoidAccount(stdin, stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, callavoidUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak callavoid: unknown subcommand %q\n%s\n", sub, callavoidUsage)
		return 2
	}
}

const callavoidUsage = `fak callavoid - avoided-call economics (over internal/callavoid)

  fak callavoid prove-memo [--in FILE] [--json]
      Read a MemoInput JSON (stdin or --in) and emit the break-even MemoProof:
      is memoizing this exact pure call net-positive? Fields (case-insensitive):
        accesses (int, k>=1), validate_cost (v), mutation_rate (m, [0,1]), capture_cost (c).

  fak callavoid account [--in FILE] [--json] [--gate]
      Read a Tally JSON (stdin or --in) and emit the amplification TurnReport for a
      window. Fields: execute, memo_hit, repair, stale_miss, hard_deny (ints),
      redirects ([]int), validate_cost, capture_cost (floats), max_redirect_fanout (int).
      --gate exits 1 when the window is REGRESSING (avoidance was a net loss).

Exit: 0 on a valid decision, 2 on malformed input, 1 only under --gate on a regressing window.`

func runCallavoidProveMemo(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak callavoid prove-memo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	inPath := fs.String("in", "", "read MemoInput JSON from this file (default: stdin)")
	asJSON := fs.Bool("json", true, "emit the proof as JSON (the default; --json=false for human text)")
	if code, done := parseCallavoidFlags(fs, argv, stderr); done {
		return code
	}

	raw, code := readCallavoidInput(stdin, stderr, *inPath, "prove-memo")
	if code != 0 {
		return code
	}
	var in callavoid.MemoInput
	if err := strictUnmarshal(raw, &in); err != nil {
		fmt.Fprintf(stderr, "fak callavoid prove-memo: invalid MemoInput JSON: %v\n", err)
		return 2
	}
	proof := callavoid.ProveMemo(in)
	if *asJSON {
		return emitCallavoidJSON(stdout, stderr, proof, "prove-memo")
	}
	fmt.Fprintf(stdout, "%s (%s): %s\n", proof.Status, proof.Decision, proof.Reason)
	fmt.Fprintf(stdout, "  break-even accesses: %d  per-reuse net gain: %.4f  saved: %.4f (%.1f%%)\n",
		proof.BreakEvenAccesses, proof.PerReuseNetGain, proof.SavedCost, proof.SavedPct)
	return 0
}

func runCallavoidAccount(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak callavoid account", flag.ContinueOnError)
	fs.SetOutput(stderr)
	inPath := fs.String("in", "", "read Tally JSON from this file (default: stdin)")
	asJSON := fs.Bool("json", true, "emit the report as JSON (the default; --json=false for human text)")
	gate := fs.Bool("gate", false, "exit 1 when the window is regressing (avoidance was a net loss)")
	if code, done := parseCallavoidFlags(fs, argv, stderr); done {
		return code
	}

	raw, code := readCallavoidInput(stdin, stderr, *inPath, "account")
	if code != 0 {
		return code
	}
	var t callavoid.Tally
	if err := strictUnmarshal(raw, &t); err != nil {
		fmt.Fprintf(stderr, "fak callavoid account: invalid Tally JSON: %v\n", err)
		return 2
	}
	rep := callavoid.Account(t)
	if *asJSON {
		if code := emitCallavoidJSON(stdout, stderr, rep, "account"); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "%s (grade %s): amplification %.4g over %d raw turns\n",
			rep.Status, rep.Grade, rep.Amplification, rep.RawTurns)
		fmt.Fprintf(stdout, "  effective %.4g / executed %.4g  (avoided %.4g)\n",
			rep.EffectiveTurns, rep.ExecutedTurns, rep.AvoidedTurns)
	}
	if *gate && rep.Status == "regressing" {
		fmt.Fprintln(stderr, "GATE: window is regressing (avoidance was a net loss)")
		return 1
	}
	return 0
}

// parseCallavoidFlags parses argv, returning (code, done): done=true means the caller
// should return code (a help request -> 0, a parse error -> 2). A trailing positional
// is rejected so a stray arg never silently no-ops.
func parseCallavoidFlags(fs *flag.FlagSet, argv []string, stderr io.Writer) (int, bool) {
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0, true
		}
		return 2, true
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", fs.Name(), fs.Arg(0))
		return 2, true
	}
	return 0, false
}

func readCallavoidInput(stdin io.Reader, stderr io.Writer, inPath, sub string) ([]byte, int) {
	if inPath != "" {
		raw, err := os.ReadFile(inPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak callavoid %s: read input: %v\n", sub, err)
			return nil, 2
		}
		return raw, 0
	}
	raw, err := io.ReadAll(stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak callavoid %s: read stdin: %v\n", sub, err)
		return nil, 2
	}
	if len(bytes.TrimSpace(raw)) == 0 {
		fmt.Fprintf(stderr, "fak callavoid %s: no input on stdin (pass JSON or --in FILE)\n", sub)
		return nil, 2
	}
	return raw, 0
}

// strictUnmarshal rejects unknown fields so a typo in a hand-authored input (e.g.
// "memohit" for "memo_hit") fails loudly instead of silently zeroing the field.
func strictUnmarshal(raw []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	return dec.Decode(v)
}

func emitCallavoidJSON(stdout, stderr io.Writer, v any, sub string) int {
	return encodeJSONOrFail(stdout, stderr, v, "fak callavoid "+sub)
}
