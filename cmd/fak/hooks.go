package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
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
	return gateModeDefault(modeEnv, escapeEnv, "block")
}

// gateModeDefault is gateMode with an explicit fallback for an UNSET ModeEnv. The historical
// default is "block"; an ADVISORY gate (Gate.DefaultMode = "warn", e.g. PRIOR_ART) passes
// "warn" so it only warns out of the box while its ModeEnv can still force "block". An empty
// def keeps the "block" default. Mirrors the commit-msg path's per-gate warn default for
// FLEET_MSG_GUARD.
func gateModeDefault(modeEnv, escapeEnv, def string) (mode string, escaped bool) {
	if def == "" {
		def = "block"
	}
	mode = strings.TrimSpace(os.Getenv(modeEnv))
	if mode == "" {
		mode = def
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
		mode, escaped := gateModeDefault(g.ModeEnv, g.EscapeEnv, g.DefaultMode)
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
				printGateHint(stderr, g.Name, findings, true)
			}
		} else if !*asJSON { // warn
			printGateFindings(stderr, g.Name+" (advisory)", findings)
			printGateHint(stderr, g.Name, findings, false)
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

	hwMode, hwEscaped := gateMode("FLEET_HW_GUARD", "FLEET_ALLOW_HW")
	if hwMode != "off" && !hwEscaped {
		if tells := hooks.ScanMessageHardwareTells(msg); len(tells) > 0 {
			fmt.Fprintf(stderr, "HARDWARE_TELL: the commit MESSAGE carries %d private hardware name(s):\n", len(tells))
			for _, f := range tells {
				fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
			}
			fmt.Fprintln(stderr, "  fix: describe the box generically (GPU server / datacenter GPU), per PUBLIC-SCRUB-POLICY.md.")
			if hwMode == "block" {
				fmt.Fprintln(stderr, "  override once: FLEET_ALLOW_HW=1 <git cmd>  (a competitor citation / a commit about the scrubber).")
				return 1
			}
			fmt.Fprintln(stderr, "  advisory only because FLEET_HW_GUARD=warn.")
		}
	}

	msgMode, msgEscaped := gateMode("FLEET_MSG_GUARD", "FLEET_ALLOW_MSG")
	if strings.TrimSpace(os.Getenv("FLEET_MSG_GUARD")) == "" {
		msgMode = "warn"
	}
	if msgMode == "off" || msgEscaped {
		return 0
	}
	if ok, why := hooks.CommitMsgVerdict(msg); !ok {
		fmt.Fprintf(stderr, "COMMIT_MSG: %s\n", why)
		if msgMode == "block" {
			fmt.Fprintln(stderr, "  (FLEET_MSG_GUARD=warn softens; =off disables; FLEET_ALLOW_MSG=1 overrides once)")
			return 1
		}
		fmt.Fprintln(stderr, "  hard-enforce with FLEET_MSG_GUARD=block.")
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

// printGateFindings writes a gate's findings to w under "<label>: N finding(s):" —
// the human render the pre-commit and hygiene gate loops share. label carries the gate
// name plus any " (advisory)" suffix.
func printGateFindings(w io.Writer, label string, findings []hooks.Finding) {
	fmt.Fprintf(w, "%s: %d finding(s):\n", label, len(findings))
	for _, f := range findings {
		fmt.Fprintf(w, "  %s\n", formatFinding(f))
	}
}

// printGateHint writes a gate-specific recovery hint after a gate's findings. Today only
// HARDWARE_TELL carries one: a prose hardware tell in staged doc CONTENT is what reds the
// trunk later in `make ci` (#1455), and the scrubber can auto-fix it — so name the offending
// file(s) and the exact `tools/scrub_hardware_names.py --apply <file>` command (the same
// recovery the lint gives), plus the one-shot override when the gate actually blocked. Without
// this, the doc-content gate refused with only the generic "soften the gate" footer, leaving
// the author no pointer to the fix.
func printGateHint(w io.Writer, gate string, findings []hooks.Finding, blocked bool) {
	if gate != "HARDWARE_TELL" {
		return
	}
	files := distinctFindingFiles(findings)
	if len(files) == 0 {
		return
	}
	fmt.Fprintf(w, "  fix: tools/scrub_hardware_names.py --apply %s\n", strings.Join(files, " "))
	fmt.Fprintln(w, "       (auto-scrubs the prose tell before it reds the trunk in make ci; see PUBLIC-SCRUB-POLICY.md)")
	if blocked {
		fmt.Fprintln(w, "  override once: FLEET_ALLOW_HW=1 <git cmd>  (a competitor citation / a commit about the scrubber).")
	}
}

// distinctFindingFiles returns the unique non-empty File fields of findings, in first-seen
// order — the file set a recovery hint names.
func distinctFindingFiles(findings []hooks.Finding) []string {
	seen := map[string]bool{}
	var files []string
	for _, f := range findings {
		if f.File == "" || seen[f.File] {
			continue
		}
		seen[f.File] = true
		files = append(files, f.File)
	}
	return files
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
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
