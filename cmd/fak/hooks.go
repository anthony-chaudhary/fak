package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// cmd/fak/hooks.go — `fak hooks <pre-commit|commit-msg>`: run the repo's commit-boundary gates
// IN ONE PROCESS instead of spawning a Python interpreter per gate. The shell hooks call this
// once; it reads the staged diff once and runs every gate over it. Exit codes mirror the gate
// contract so the shell wrapper can fall back to Python: 0 = clean/pass, 1 = a block gate fired,
// 2 = could-not-run (the wrapper then runs the Python path — fail-open).

func cmdHooks(argv []string) { os.Exit(runHooks(os.Stdout, os.Stderr, argv)) }

func runHooks(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak hooks: subcommand required (pre-commit | commit-msg <file> | lane-audit)")
		return 2
	}
	switch argv[0] {
	case "pre-commit":
		return runHooksPreCommit(stdout, stderr, argv[1:])
	case "commit-msg":
		return runHooksCommitMsg(stdout, stderr, argv[1:])
	case "lane-audit":
		return runHooksLaneAudit(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak hooks: unknown subcommand %q (pre-commit | commit-msg | lane-audit)\n", argv[0])
		return 2
	}
}

// runHooksLaneAudit reports every internal/<leaf> Go package with no declared dos.toml lane —
// the standing, whole-tree form of the per-commit leaf check the commit-msg stamp lint runs. Each
// such leaf's `(fak <leaf>)` ship-stamp binds to a phantom unit and the arbiter cannot protect its
// edits. --gate N exits 1 when the count EXCEEDS N, so a ratchet can drive the drift to zero
// without reding the trunk on day one. Exit 0 report/at-or-under-gate, 1 over-gate, 2 could-not-run.
func runHooksLaneAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks lane-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit the undeclared leaves as JSON")
	gate := fs.Int("gate", -1, "exit 1 if the count of undeclared-lane leaves exceeds this threshold (-1 = report only)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2
	}
	gaps, err := hooks.UndeclaredLeaves(r)
	if err != nil {
		fmt.Fprintf(stderr, "fak hooks lane-audit: %v\n", err)
		return 2
	}
	if *asJSON {
		if encErr := writeIndentedJSON(stdout, map[string]any{"undeclared": gaps, "count": len(gaps)}); encErr != nil {
			fmt.Fprintf(stderr, "fak hooks lane-audit: %v\n", encErr)
			return 2
		}
	} else {
		fmt.Fprintf(stdout, "lane-audit: %d internal leaf(s) with a real Go package but NO declared dos.toml lane\n", len(gaps))
		for _, g := range gaps {
			fmt.Fprintf(stdout, "  - %s/%s — a `(fak %s)` stamp binds to a phantom unit; declare lane `%s` in dos.toml\n", g.Base, g.Leaf, g.Leaf, g.Leaf)
		}
		if len(gaps) == 0 {
			fmt.Fprintln(stdout, "  (every internal leaf has a declared lane)")
		}
	}
	if *gate >= 0 && len(gaps) > *gate {
		fmt.Fprintf(stderr, "lane-audit: %d undeclared leaves exceeds gate %d\n", len(gaps), *gate)
		return 1
	}
	return 0
}

// gateMode resolves a gate's FLEET_<NAME>_GUARD env to block (default) / warn / off, and its
// one-shot escape env. Identical semantics to the shell run_gate.
func gateMode(modeEnv, escapeEnv string) (mode string, escaped bool) {
	mode = strings.TrimSpace(os.Getenv(modeEnv))
	if mode == "" {
		mode = "block"
	}
	return mode, strings.TrimSpace(os.Getenv(escapeEnv)) == "1"
}

func runHooksPreCommit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks pre-commit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2 // not in a repo => could-not-run => fall through to python
	}

	d, err := hooks.ReadStagedDiff(r)
	if err != nil {
		// could-not-run: never block. The shell wrapper treats exit 2 as "fall back to python".
		return 2
	}

	var allFindings []hooks.Finding
	blocked := false
	for _, g := range hooks.PreCommitGates() {
		mode, escaped := gateMode(g.ModeEnv, g.EscapeEnv)
		if mode == "off" || escaped {
			continue
		}
		findings, gerr := g.Check(d)
		if gerr != nil {
			// a single gate that could-not-run is skipped (fail-open), the others still run.
			continue
		}
		if len(findings) == 0 {
			continue
		}
		allFindings = append(allFindings, findings...)
		if mode == "block" {
			blocked = true
			if !*asJSON {
				printGateFindings(stderr, g.Name, findings)
			}
		} else if !*asJSON { // warn
			printGateFindings(stderr, g.Name+" (advisory)", findings)
		}
	}

	if *asJSON {
		emitFindingsJSON(stdout, stderr, allFindings)
	}
	if blocked {
		if !*asJSON {
			fmt.Fprintln(stderr, "")
			fmt.Fprintln(stderr, "commit refused by a fleet pre-commit gate (above).")
			fmt.Fprintln(stderr, "  soften one gate: FLEET_<NAME>_GUARD=warn (or =off), or use its one-shot escape.")
		}
		return 1
	}
	return 0
}

func runHooksCommitMsg(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks commit-msg", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	rest := fs.Args()
	if len(rest) == 0 {
		fmt.Fprintln(stderr, "fak hooks commit-msg: message file required")
		return 2
	}
	msgBytes, err := os.ReadFile(rest[0])
	if err != nil {
		// could-not-read => fail open (exit 2), never wedge a commit over the message check.
		return 2
	}
	msg := string(msgBytes)
	r := resolveRoot(*root)

	// PUBLIC_LEAK over the message (block by default) — same needle list as pre-commit.
	// This is the fast PRE-block: a secret in the message is caught here without spawning
	// Python. A hit returns 1 (the shell short-circuits and blocks).
	leakMode, leakEscaped := gateMode("FLEET_SCRUB_GUARD", "FLEET_ALLOW_LEAK")
	if leakMode != "off" && !leakEscaped {
		if leaks := hooks.ScanMessageNeedles(msg, r); len(leaks) > 0 {
			for _, f := range leaks {
				fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
			}
			if leakMode == "block" {
				fmt.Fprintln(stderr, "PUBLIC_LEAK: the commit MESSAGE carries a redact needle (above).")
				return 1
			}
			fmt.Fprintln(stderr, "PUBLIC_LEAK (advisory): redact needle in the commit message (above).")
		}
	}

	// FALL THROUGH to the Python commit-msg gates (exit 2). The Go path owns ONLY the fast
	// PUBLIC_LEAK pre-block above; the HARDWARE_TELL gate (a private node label in the
	// subject/body) and the COMMIT_MSG subject-shape advisory live in tools/*.py, which the
	// shell hook runs when this returns 2. Returning 0 here would make the shell exit before
	// those gates ever run — which is exactly how a `dgx3` rode into history. There is no Go
	// port of the hardware tells (the dgxN regex needs lookahead RE2 lacks), so 2 = "Go did
	// its fast check; let Python finish." (Python COMMIT_MSG owns the subject-shape advisory,
	// so it is not duplicated here.)
	return 2
}

func formatFinding(f hooks.Finding) string {
	loc := f.File
	if f.Line > 0 {
		if loc == "" {
			loc = fmt.Sprintf("msg:%d", f.Line)
		} else {
			loc = fmt.Sprintf("%s:%d", f.File, f.Line)
		}
	}
	if loc == "" {
		return f.Detail
	}
	return loc + ": " + f.Detail
}

// printGateFindings writes a gate's findings to w under "<label>: N finding(s):" —
// the human render the pre-commit and hygiene gate loops share. label carries the gate
// name plus any " (advisory)" suffix.
func printGateFindings(w io.Writer, label string, findings []hooks.Finding) {
	fmt.Fprintf(w, "%s: %d finding(s):\n", label, len(findings))
	for _, f := range findings {
		fmt.Fprintf(w, "  %s\n", formatFinding(f))
	}
}

func emitFindingsJSON(stdout, stderr io.Writer, findings []hooks.Finding) {
	if findings == nil {
		findings = []hooks.Finding{}
	}
	if err := writeIndentedJSON(stdout, map[string]any{"findings": findings, "count": len(findings)}); err != nil {
		fmt.Fprintf(stderr, "fak hooks: %v\n", err)
	}
}

// resolveRoot returns the explicit --root, else the git toplevel from cwd, else "".
func resolveRoot(explicit string) string {
	if strings.TrimSpace(explicit) != "" {
		return explicit
	}
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
