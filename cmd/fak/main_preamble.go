package main

import (
	"os"
	"time"
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

// resolveEarlyDispatch runs the two pre-switch gates and returns true when it fully
// handled the call (so main() should return without entering the dispatch switch):
//
//   - no verb at all → usage() + exit 2.
//   - the `fak dev <verb>` namespace (C2 of epic #2228, #2231): it resolves BEFORE the
//     dispatch switch by rewriting os.Args to the underlying verb, so the very same case
//     arm runs — byte-identical dispatch, no re-exec — and the 200-case switch (plus the
//     devindex scanner keyed on its `switch os.Args[1]` header) stays untouched. The
//     usage journal records the composite verb ("dev commit" vs bare "commit"): the
//     bare-vs-namespaced adoption evidence the C5 enforcement flip is gated on. A
//     dev-only verb with no top-level arm is dispatched here and reported handled.
func resolveEarlyDispatch(verb *string, argv *[]string, start time.Time) bool {
	if len(os.Args) < 2 {
		if maybeLaunchDefault() {
			return true
		}
		usage()
		recordUsage(*verb, *argv, 2, start)
		os.Exit(2)
	}
	if os.Args[1] == "orchestration" {
		cmdOrchestration(os.Args[2:])
		return true
	}
	if os.Args[1] == "dev" {
		v, rest, code := resolveDevVerb(os.Args[2:], os.Stdout, os.Stderr)
		if code >= 0 {
			recordUsage(*verb, *argv, code, start)
			os.Exit(code)
		}
		*verb = "dev " + v
		*argv = rest
		if dispatchDevOnlyVerb(v, rest) {
			return true
		}
		os.Args = append([]string{os.Args[0], v}, rest...)
	}
	return false
}
