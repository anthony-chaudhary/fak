package main

import (
	"encoding/json"
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
		fmt.Fprintln(stderr, "fak hooks: subcommand required (pre-commit | commit-msg <file>)")
		return 2
	}
	switch argv[0] {
	case "pre-commit":
		return runHooksPreCommit(stdout, stderr, argv[1:])
	case "commit-msg":
		return runHooksCommitMsg(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak hooks: unknown subcommand %q (pre-commit | commit-msg)\n", argv[0])
		return 2
	}
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
				fmt.Fprintf(stderr, "%s: %d finding(s):\n", g.Name, len(findings))
				for _, f := range findings {
					fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
				}
			}
		} else if !*asJSON { // warn
			fmt.Fprintf(stderr, "%s (advisory): %d finding(s):\n", g.Name, len(findings))
			for _, f := range findings {
				fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
			}
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

	// COMMIT_MSG subject-shape (warn by default).
	mode, escaped := gateMode("FLEET_MSG_GUARD", "FLEET_ALLOW_MSG")
	// NOTE: the commit-msg hook defaults this gate to WARN, not BLOCK.
	if strings.TrimSpace(os.Getenv("FLEET_MSG_GUARD")) == "" {
		mode = "warn"
	}
	if mode != "off" && !escaped {
		if ok, why := hooks.CommitMsgVerdict(msg); !ok {
			fmt.Fprintf(stderr, "COMMIT_MSG: subject not witness-gradeable — %s\n", why)
			if mode == "block" {
				return 1
			}
			fmt.Fprintln(stderr, "  (advisory; FLEET_MSG_GUARD=block hard-enforces)")
		}
	}
	return 0
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

func emitFindingsJSON(stdout, stderr io.Writer, findings []hooks.Finding) {
	if findings == nil {
		findings = []hooks.Finding{}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(map[string]any{"findings": findings, "count": len(findings)}); err != nil {
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
