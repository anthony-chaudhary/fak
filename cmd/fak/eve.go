package main

// fak eve -- the impure shell over the internal/evebridge core: the Eve
// integration program's (#2600) operator verbs.
//
//	fak eve preflight connections [--manifest FILE|-] [--override CONN__OP]... [--eve-bin BIN]
//	    the connection-security preflight (#2602): reads an eve connection
//	    manifest (`eve info --json` output or a compiled discovery artifact —
//	    from --manifest, stdin with "-", or by exec'ing `eve info --json` when
//	    no manifest is given) and folds it through evebridge.Preflight. The
//	    report is JSON-first: the full typed report (schema
//	    fak-eve-connection-preflight/1, with per-diagnostic remediation text
//	    suitable for issue bodies and CI logs) always goes to stdout; a
//	    one-line verdict goes to stderr. Exit 0 = pass (stdout carries the
//	    exact tool namespace fak will admit), 3 = preflight failed closed,
//	    1 = manifest unreadable, 2 = usage.
//
// All impurity lives here (file/stdin read, the `eve info --json` exec, flag
// parsing, exit codes); the checks themselves are the pure fold in
// internal/evebridge.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"

	"github.com/anthony-chaudhary/fak/internal/evebridge"
)

func cmdEve(argv []string) {
	os.Exit(runEve(os.Stdout, os.Stderr, os.Stdin, argv))
}

func runEve(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak eve: expected a subcommand (preflight)")
		return 2
	}
	switch argv[0] {
	case "preflight":
		return runEvePreflight(stdout, stderr, stdin, argv[1:])
	case "schedules":
		return runEveSchedules(stdout, stderr, argv[1:])
	case "dispatch-receipt":
		return runEveDispatchReceipt(stdout, stderr, argv[1:])
	case "inspect":
		// #2601: compile the authored agent/ layout (or a compiled .eve/)
		// into the fak policy/mount inspect manifest. See cmd/fak/eve_inspect.go.
		return runEveInspect(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stderr, "fak eve: preflight connections  (mechanical security preflight over eve MCP/OpenAPI connections)")
		fmt.Fprintln(stderr, "         inspect [--json] [--policy-draft] [ROOT]  (compile an Eve app's authored shape into the fak policy/mount manifest)")
		fmt.Fprintln(stderr, "         schedules [--json] [--host HOST] [ROOT]  (project Eve schedules into the fak recurring-job ledger)")
		return 0
	default:
		fmt.Fprintf(stderr, "fak eve: unknown subcommand %q (want preflight|inspect|schedules|dispatch-receipt)\n", argv[0])
		return 2
	}
}

func runEvePreflight(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	if len(argv) == 0 || argv[0] != "connections" {
		fmt.Fprintln(stderr, "fak eve preflight: expected a target (connections)")
		return 2
	}
	return runEvePreflightConnections(stdout, stderr, stdin, argv[1:])
}

// evePreflightFailed is the fail-closed exit: the manifest was readable and
// the preflight ran, and at least one typed diagnostic gates admission.
const evePreflightFailed = 3

func runEvePreflightConnections(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("eve preflight connections", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifest := fs.String("manifest", "", `manifest source: a file path, "-" for stdin, or empty to exec 'eve info --json'`)
	eveBin := fs.String("eve-bin", "eve", "eve binary to exec for 'eve info --json' when no --manifest is given")
	var overrides stringList
	fs.Var(&overrides, "override", "fak policy override: an exact generated tool name (<connection>__<operation>) allowed to mutate without a connection approval policy (repeatable)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak eve preflight connections: unexpected argument %q (flags only)\n", fs.Arg(0))
		return 2
	}

	data, src, err := readEveManifest(*manifest, *eveBin, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak eve preflight connections: %v\n", err)
		return 1
	}
	m, err := evebridge.ParseManifest(data)
	if err != nil {
		fmt.Fprintf(stderr, "fak eve preflight connections: %s: %v\n", src, err)
		return 1
	}

	report := evebridge.Preflight(m, evebridge.Options{Overrides: []string(overrides)})
	if code := knownBadEmitJSON(stdout, stderr, report); code != 0 {
		return code
	}
	if !report.OK {
		fails := 0
		for _, d := range report.Diagnostics {
			if d.Severity == evebridge.SeverityFail {
				fails++
			}
		}
		fmt.Fprintf(stderr, "eve preflight connections: FAILED closed — %d gating diagnostic(s) over %d connection(s); no tools admitted (see the JSON report's remediation lines)\n",
			fails, report.Connections)
		return evePreflightFailed
	}
	fmt.Fprintf(stderr, "eve preflight connections: ok — %d connection(s) checked, %d tool(s) admitted\n",
		report.Connections, len(report.AdmittedTools))
	return 0
}

// readEveManifest resolves the manifest bytes: an explicit file, stdin ("-"),
// or the live `eve info --json` when nothing is given. The returned src names
// the source for error messages.
func readEveManifest(path, eveBin string, stdin io.Reader) ([]byte, string, error) {
	switch path {
	case "-":
		data, err := io.ReadAll(stdin)
		if err != nil {
			return nil, "stdin", fmt.Errorf("read stdin: %w", err)
		}
		return data, "stdin", nil
	case "":
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, eveBin, "info", "--json")
		configureDispatchHelperCommand(cmd)
		out, err := cmd.Output()
		if err != nil {
			return nil, eveBin, fmt.Errorf("exec %s info --json: %w (pass --manifest FILE to preflight a compiled discovery artifact instead)", eveBin, err)
		}
		return out, eveBin + " info --json", nil
	default:
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, path, err
		}
		return data, path, nil
	}
}
