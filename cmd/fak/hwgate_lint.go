package main

// fak hwgate-lint — the operator-facing shell over internal/hwgatelint, the dual
// of `fak headless-lint`. Where headless-lint catches an agent that stopped to ask
// a HUMAN, hwgate-lint catches an agent that stopped for lack of LOCAL hardware —
// "no GPU on this host", "can't run the CUDA witness without a device", "not
// reproducible on this laptop". Each hit types into a closed anti-pattern Class
// (NO_LOCAL_GPU / NO_LOCAL_RUNTIME / LOCAL_BOUNDARY) and carries the fixed
// remediation: dispatch to a sanctioned compute node (GCP fak-realmodel, the DGX
// via dgxbridge, da33, the nightrun pipeline), or hand the operator the exact
// ready-to-run command. This host is the control point, not the compute boundary.
//
//	fak hwgate-lint --self-test [--json]     -> scan the built-in corpus; exit 0 iff every case scans as labeled
//	fak hwgate-lint --file out.txt [--json]  -> scan a file ("-" = stdin)
//	fak hwgate-lint "…text…" [--json]        -> scan literal text from the args
//	… | fak hwgate-lint [--json]             -> scan stdin
//
// Exit codes suit `fak hwgate-lint … && ship`: 0 = clean (fleet-safe), 1 =
// hardware-gate notes found (or a failed self-test), 2 = usage. The pure scanner is
// internal/hwgatelint; its self-test corpus is asserted by
// internal/hwgatelint/hwgatelint_test.go.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hwgatelint"
)

func cmdHwGateLint(argv []string) {
	os.Exit(runHwGateLint(os.Stdout, os.Stderr, os.Stdin, argv))
}

func runHwGateLint(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("hwgate-lint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		selfTest = fs.Bool("self-test", false, "scan the built-in corpus and exit 0 iff every case scans as labeled")
		asJSON   = fs.Bool("json", false, "emit the Report as JSON")
		file     = fs.String("file", "", `read the text to scan from this path ("-" = stdin)`)
	)
	fs.Usage = func() { fmt.Fprint(stderr, hwGateLintUsage) }
	if !parseFlags(fs, argv) {
		return 2
	}

	if *selfTest {
		return runHwGateSelfTest(stdout, *asJSON)
	}

	text, err := readHeadlessSource(stdin, *file, fs.Args())
	if err != nil {
		fmt.Fprintf(stderr, "fak hwgate-lint: %v\n", err)
		return 1
	}
	if strings.TrimSpace(text) == "" {
		fmt.Fprint(stderr, hwGateLintUsage)
		return 2
	}

	rep := hwgatelint.Scan(text)
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak hwgate-lint: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderHwGateReport(rep))
	}
	if rep.Count > 0 {
		return 1
	}
	return 0
}

func runHwGateSelfTest(stdout io.Writer, asJSON bool) int {
	cases, passed := hwgatelint.RunFixture()
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
			exp := "dirty"
			if c.Clean {
				exp = "clean"
			}
			fmt.Fprintf(stdout, "  %-26s expect=%-6s got=%-15s %s\n", c.Name, exp, c.Got, mark)
		}
		fmt.Fprintf(stdout, "hwgate-lint self-test: %d/%d scanned as labeled\n", passed, len(cases))
	}
	if allOK {
		return 0
	}
	return 1
}

// renderHwGateReport is the human-readable view: one block per finding with the
// offending line, why stopping there strands work, and the node to route to. A
// suppressed scan (the turn already named the lab) is called out as clean.
func renderHwGateReport(rep hwgatelint.Report) string {
	if rep.Count == 0 {
		if rep.Suppressed {
			return "fak hwgate-lint: clean — the turn already names a sanctioned route (dispatched to the fleet or handed the operator a command); fleet-safe.\n"
		}
		return "fak hwgate-lint: clean — no local-hardware blocker declared as terminal; fleet-safe.\n"
	}
	var b strings.Builder
	plural := "note"
	if rep.Count != 1 {
		plural = "notes"
	}
	fmt.Fprintf(&b, "fak hwgate-lint: %d hardware-gate %s — this host is the control point, not the compute boundary; dispatch the work, never stop for lack of local hardware.\n\n", rep.Count, plural)
	for _, f := range rep.Findings {
		fmt.Fprintf(&b, "  line %-4d [%s]  %q\n", f.Line, f.Class, f.Match)
		fmt.Fprintf(&b, "           why:   %s\n", f.Reason)
		fmt.Fprintf(&b, "           route: %s\n\n", f.Node)
	}
	fmt.Fprintf(&b, "verdict: %s (%d)\n", rep.Verdict, rep.Count)
	fmt.Fprintf(&b, "\ninstead: %s\n", rep.Redirect)
	return b.String()
}

const hwGateLintUsage = `fak hwgate-lint — scan agent output for "local machine is the compute boundary" stops.

usage:
  fak hwgate-lint --self-test [--json]
  fak hwgate-lint --file out.txt [--json]      ("-" reads the text from stdin)
  fak hwgate-lint "…text…" [--json]
  … | fak hwgate-lint [--json]

the closed anti-pattern vocabulary (Class -> the local resource declared missing):
  NO_LOCAL_GPU       a GPU/CUDA/accelerator device declared missing as the blocker
  NO_LOCAL_RUNTIME   a non-Go runtime (Node.js/npm) declared missing as the blocker
  LOCAL_BOUNDARY     THIS machine treated as the limit (no naming of a device)

  every Class shares one remedy: dispatch the work to a sanctioned compute node
  (GCP fak-realmodel, the DGX via dgxbridge, da33, the nightrun pipeline) — or, if
  the credential/bridge session is missing, hand the operator the ready-to-run
  command. A turn that already names the lab route scans clean (suppressed).

verdict & exit code:
  clean (0)           no local-hardware blocker declared as terminal — fleet-safe
  hardware_gated (1)  at least one stop treats the local host as the compute boundary
  usage (2)           a bad flag or empty input
`
