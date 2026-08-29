package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/devhandoff"
)

// parseVerbArgv splits os.Args into the leading verb and its argument tail without
// mutating anything, tolerating the zero- and one-arg cases (an empty verb/argv the
// dispatch preamble then rejects). Kept separate from the dispatch switch so main()
// stays at the routing table and its grandfathered ceiling.
func parseVerbArgv() (verb string, argv []string) {
	if len(os.Args) >= 2 {
		verb = os.Args[1]
	}
	if len(os.Args) > 2 {
		argv = os.Args[2:]
	}
	return
}

// recoverUsage is main()'s deferred panic boundary: it records a crash exit (2) in the
// usage journal against the verb/argv seen so far, then re-panics so the crash still
// surfaces. Taking pointers lets it observe the verb rewrite a `dev`-namespaced call
// performs before the panic.
func recoverUsage(verb *string, argv *[]string, start time.Time) {
	if r := recover(); r != nil {
		recordUsage(*verb, *argv, 2, start)
		panic(r)
	}
}

// resolveEarlyDispatch runs pre-switch gates and reports whether main should
// return: no verb prints usage, orchestration keeps its early route, and
// `fak dev ...` delegates across a process boundary to separately built fak-dev.
// Runtime fak never imports or executes development implementation packages.
func resolveEarlyDispatch(verb *string, argv *[]string, start time.Time) bool {
	if len(os.Args) < 2 {
		if maybeLaunchDefault() {
			return true
		}
		usage()
		recordUsage(*verb, *argv, 2, start)
		os.Exit(2)
	}
	if code, handled := runExactDevHandoff(os.Stdin, os.Stdout, os.Stderr, os.Args[1], os.Args[2:]); handled {
		recordUsage(*verb, *argv, code, start)
		os.Exit(code)
	}
	if devhandoff.IsCommand(os.Args[1]) {
		fmt.Fprintf(os.Stderr, "fak: %q moved to the separate fak-dev executable (DEV_COMMAND_MOVED)\n", os.Args[1])
		fmt.Fprintf(os.Stderr, "  run: fak dev %s\n", strings.Join(os.Args[1:], " "))
		recordUsage(*verb, *argv, 2, start)
		os.Exit(2)
	}
	if os.Args[1] == "harness" && len(os.Args) > 2 && os.Args[2] == "model-set" {
		code := runHarnessModelSet(os.Stdout, os.Stderr, os.Args[3:])
		recordUsage(*verb, *argv, code, start)
		os.Exit(code)
	}
	if os.Args[1] == "orchestration" {
		cmdOrchestration(os.Args[2:])
		return true
	}
	if os.Args[1] == "ultracode" {
		cmdUltracode(os.Args[2:])
		return true
	}
	if os.Args[1] == "dev" {
		code := runDevHandoff(os.Stdin, os.Stdout, os.Stderr, os.Args[2:])
		recordUsage(*verb, *argv, code, start)
		os.Exit(code)
	}
	return false
}

var executeExactDevHandoff = runDevHandoff

// runExactDevHandoff preserves `fak build` as a useful top-level product spelling
// while keeping its implementation in fak-dev. Other moved developer commands keep
// the explicit `fak dev ...` compatibility route and DEV_COMMAND_MOVED guidance.
func runExactDevHandoff(stdin io.Reader, stdout, stderr io.Writer, verb string, argv []string) (int, bool) {
	if verb != "build" {
		return 0, false
	}
	childArgv := make([]string, 1, len(argv)+1)
	childArgv[0] = verb
	childArgv = append(childArgv, argv...)
	return executeExactDevHandoff(stdin, stdout, stderr, childArgv), true
}
