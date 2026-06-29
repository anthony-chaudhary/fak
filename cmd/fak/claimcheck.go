package main

// fak claim-check — the operator-facing CLI shell over internal/claimcheck, the
// named net-true follow-on (cross-cutting T14 of epic #1147, issue #1171). The
// net-true-value standard (docs/standards/net-true-value.md) is a lens over existing
// sticks; this verb is the single grader the standard's own honest fence named but
// did not build there: it takes a claim + baseline + witness and returns one of
// net-true / strawman / not-yet against the six questions a real gain must answer.
//
//	fak claim-check --self-test [--json]   -> grade the built-in honest+strawman corpus
//	fak claim-check --file claim.json      -> grade a Claim read as JSON ("-" = stdin)
//	fak claim-check --statement "..." \     -> grade a claim built from flags
//	    --baseline real|strawman|none [--baseline-desc D] --net \
//	    --scope "..." --provenance WITNESSED|OBSERVED|MODELED|SIMULATED \
//	    --witness "..." [--realized=false --gate-reason "..."]
//
// Exit codes are designed for `fak claim-check ... && ship`: 0 = net-true (a real
// gain at the stated scope), 3 = graded but NOT a pass (strawman or not-yet), 2 =
// usage, 1 = internal error. --self-test exits 0 iff every fixture case grades to
// its labeled verdict, so the grader's own behavior is a re-derivable witness (Q5
// turned on the verb itself). The pure grader is internal/claimcheck; the graded
// self-tests are internal/claimcheck/claimcheck_test.go.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/claimcheck"
)

func cmdClaimCheck(argv []string) { os.Exit(runClaimCheck(os.Stdout, os.Stderr, os.Stdin, argv)) }

func runClaimCheck(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("claim-check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		selfTest   = fs.Bool("self-test", false, "grade the built-in honest+strawman fixture corpus and exit 0 iff every case grades as labeled")
		asJSON     = fs.Bool("json", false, "emit the result(s) as JSON")
		file       = fs.String("file", "", `read a Claim as JSON from this path ("-" = stdin)`)
		statement  = fs.String("statement", "", "the claim itself, e.g. \"fleet reuse is 4.1× vs a tuned warm-cache stack\"")
		baseline   = fs.String("baseline", "", "Q1: what it is measured against — real | strawman | none")
		baseDesc   = fs.String("baseline-desc", "", "Q1: a description of the baseline, e.g. \"tuned warm-cache stack\"")
		net        = fs.Bool("net", false, "Q2: the gain is stated net of the cost the change itself adds")
		scope      = fs.String("scope", "", "Q3: the conditions it holds under AND the ones it vanishes under")
		provenance = fs.String("provenance", "", "Q4: the closed label on the number — WITNESSED | OBSERVED | MODELED | SIMULATED")
		witness    = fs.String("witness", "", "Q5: how a third party re-derives it (a test name, an artifact + reproduce command)")
		realized   = fs.Bool("realized", true, "Q6: on by default (true), or off (false) — pair --realized=false with --gate-reason")
		gateReason = fs.String("gate-reason", "", "Q6: the stated reason a gain ships OFF by default (makes it an honest gate, not a seam)")
	)
	fs.Usage = func() { fmt.Fprint(stderr, claimCheckUsage) }
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *selfTest {
		return runClaimCheckSelfTest(stdout, *asJSON)
	}

	// Build the claim to grade: from a JSON file/stdin, or from the flags.
	var c claimcheck.Claim
	switch {
	case *file != "":
		raw, err := readClaimSource(stdin, *file)
		if err != nil {
			fmt.Fprintf(stderr, "fak claim-check: %v\n", err)
			return 1
		}
		if err := json.Unmarshal(raw, &c); err != nil {
			fmt.Fprintf(stderr, "fak claim-check: --file is not a valid Claim JSON: %v\n", err)
			return 2
		}
	case *statement != "" || *baseline != "" || *scope != "" || *provenance != "" || *witness != "":
		bk, err := parseBaselineKind(*baseline)
		if err != nil {
			fmt.Fprintf(stderr, "fak claim-check: %v\n%s", err, claimCheckUsage)
			return 2
		}
		c = claimcheck.Claim{
			Statement:  *statement,
			Baseline:   claimcheck.Baseline{Kind: bk, Description: *baseDesc},
			Net:        *net,
			Scope:      *scope,
			Provenance: claimcheck.Provenance(*provenance),
			Witness:    *witness,
			Realized:   claimcheck.Realized{OnByDefault: *realized, GateReason: *gateReason},
		}
	default:
		fmt.Fprint(stderr, claimCheckUsage)
		return 2
	}

	res := claimcheck.Grade(c)
	if *asJSON {
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak claim-check: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, res.String())
	}
	return exitForVerdict(res.Verdict)
}

// runClaimCheckSelfTest grades the built-in corpus and reports whether every case
// landed on its labeled verdict. Exit 0 iff all pass — the verb's own witness.
func runClaimCheckSelfTest(stdout io.Writer, asJSON bool) int {
	cases, passed := claimcheck.RunFixture()
	allOK := passed == len(cases)
	if asJSON {
		_ = writeIndentedJSON(stdout, map[string]any{
			"cases":  cases,
			"passed": passed,
			"total":  len(cases),
			"ok":     allOK,
		})
	} else {
		for _, c := range cases {
			mark := "ok"
			if !c.OK {
				mark = "MISMATCH"
			}
			fmt.Fprintf(stdout, "  %-28s expect=%-9s got=%-9s %s\n", c.Name, c.Expect, c.Got, mark)
		}
		fmt.Fprintf(stdout, "claim-check self-test: %d/%d graded as expected\n", passed, len(cases))
	}
	if allOK {
		return 0
	}
	return 1
}

// parseBaselineKind maps the --baseline flag word to a BaselineKind. An empty value
// is the explicit "none" (no "vs what" stated) so a claim built from flags without a
// baseline honestly fails Q1 rather than erroring.
func parseBaselineKind(s string) (claimcheck.BaselineKind, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "none":
		return claimcheck.BaselineNone, nil
	case "strawman", "naive":
		return claimcheck.BaselineStrawman, nil
	case "real", "tuned":
		return claimcheck.BaselineReal, nil
	default:
		return claimcheck.BaselineNone, fmt.Errorf("--baseline %q is not one of real | strawman | none", s)
	}
}

// readClaimSource reads the --file source: stdin when path is "-", else the file.
func readClaimSource(stdin io.Reader, path string) ([]byte, error) {
	if path == "-" {
		return io.ReadAll(stdin)
	}
	return os.ReadFile(path)
}

// exitForVerdict maps a verdict to the process exit code: 0 only for net-true, so a
// CI line can gate a ship on a real gain (`fak claim-check ... && ship`).
func exitForVerdict(v claimcheck.Verdict) int {
	if v == claimcheck.NetTrue {
		return 0
	}
	return 3
}

const claimCheckUsage = `fak claim-check — grade an efficiency/perf claim against the net-true-value six questions.

usage:
  fak claim-check --self-test [--json]
  fak claim-check --file claim.json [--json]            ("-" reads the Claim JSON from stdin)
  fak claim-check --statement S --baseline real|strawman|none [--baseline-desc D]
                  [--net] --scope S --provenance WITNESSED|OBSERVED|MODELED|SIMULATED
                  --witness S [--realized=false --gate-reason R] [--json]

the six questions (docs/standards/net-true-value.md):
  Q1 baseline    measured against the real (tuned) alternative, not a strawman
  Q2 net         stated net of the cost the change itself adds (--net)
  Q3 scope       the conditions it holds under AND the ones it vanishes under
  Q4 provenance  labeled WITNESSED / OBSERVED / MODELED / SIMULATED
  Q5 witness     a third party can re-derive it (a test, an artifact + command)
  Q6 realized    on by default, or off with a stated reason (--gate-reason)

verdict & exit code:
  net-true (0)   all six answered against the real baseline
  strawman (3)   measured against the naive floor, not the tuned alternative (Q1)
  not-yet  (3)   one of the six is unanswered — the missing ones are named
  usage    (2)   a bad flag; internal error (1)
`
