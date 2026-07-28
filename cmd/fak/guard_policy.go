// guard_policy.go is the `fak guard policy <verb>` ROUTER (#5424, epic #5170
// Track A) — the missing verb table that turns two finished floor reports into a
// surface an operator can actually type.
//
// Track A landed the reports and not the door: `policy explain` (#5172) renders
// the effective floor grouped by amendment class, and `policy diff` (#5173)
// renders the widen-drift from the shipped floor with a CI-gateable exit code —
// but neither had a call site outside its own unit test, so a complete report was
// unreachable from a real terminal. This file registers both, and it is the ONLY
// registration: delete a row here (or the `policy` peel in cmdGuard) and the
// end-to-end dispatch test fails with a usage error, because there is no second
// path by which the verbs could still be reached.
//
// Both verbs are READ-ONLY by construction — they render the floor, change no
// verdict, and write nothing — so the router owns no state and every handler is a
// pure (stdout, stderr, argv) -> exit-code function. Handlers never call os.Exit;
// cmdGuard owns the single process exit, which is what lets one subprocess drive
// the whole operator path in a test.
package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
)

// Exit contract shared by every `fak guard policy` verb, matching the sibling
// read-only guard surfaces (runGuardSessions, runHooksPopupScan): 0 = the report
// rendered and nothing is flagged, 1 = the report rendered and its GATE-ABLE
// condition is present (diff: the running floor drifted in the loosening
// direction), 2 = usage error, or the floor could not be read.
const (
	guardPolicyExitOK      = 0
	guardPolicyExitFlagged = 1
	guardPolicyExitUsage   = 2
)

// guardPolicyVerb is one row of the `fak guard policy` verb table: the word an
// operator types, the argument summary the usage screen prints beside it, a
// one-line blurb, and the handler. Run takes its OWN table row (so a handler can
// render its usage without reading the package-level registry back, which would
// be an initialization cycle), plus the argv AFTER the verb, and returns an exit
// code.
type guardPolicyVerb struct {
	Name  string
	Args  string
	Blurb string
	Run   func(v guardPolicyVerb, stdout, stderr io.Writer, argv []string) int
}

// guardPolicyVerbs is the registry. Adding a report to this table is the whole
// act of shipping it: the usage screen, the unknown-verb message, and the
// dispatch all read from here, so a verb can never be documented without being
// reachable or reachable without being documented.
var guardPolicyVerbs = []guardPolicyVerb{
	{
		Name:  "explain",
		Blurb: "the effective floor grouped by amendment class — what can change at runtime, through which authorized channel, and which layer supplied each value",
		Run:   runGuardPolicyExplainVerb,
	},
	{
		Name:  "diff",
		Args:  "[--policy FILE]",
		Blurb: "widen-drift: how the floor this host RUNS differs from the floor the binary SHIPPED, bucketed WIDENED / TIGHTENED / FROZEN-VIOLATION (exit 1 when the drift loosened the guard, so CI can gate on it)",
		Run:   runGuardPolicyDiffVerb,
	},
}

// guardPolicyVerbNames lists the registered verbs in table order, for the
// unknown-verb message.
func guardPolicyVerbNames() []string {
	names := make([]string, 0, len(guardPolicyVerbs))
	for _, v := range guardPolicyVerbs {
		names = append(names, v.Name)
	}
	return names
}

// lookupGuardPolicyVerb resolves a typed word to its registered row.
func lookupGuardPolicyVerb(name string) (guardPolicyVerb, bool) {
	for _, v := range guardPolicyVerbs {
		if v.Name == name {
			return v, true
		}
	}
	return guardPolicyVerb{}, false
}

// printGuardPolicyUsage renders the verb table from the live registry, so the
// help screen and the dispatcher can never disagree about what exists.
func printGuardPolicyUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak guard policy <verb> [flags]")
	fmt.Fprintln(w, "  read-only reports on the guard capability floor: they render the floor,")
	fmt.Fprintln(w, "  change no verdict, and write nothing.")
	fmt.Fprintln(w)
	for _, v := range guardPolicyVerbs {
		head := "  " + v.Name
		if v.Args != "" {
			head += " " + v.Args
		}
		fmt.Fprintf(w, "%-26s %s\n", head, v.Blurb)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  'fak guard policy <verb> -h' has that verb's own flags.")
	fmt.Fprintln(w, "  exit: 0 clean, 1 the verb's gate-able condition is present, 2 usage / floor unreadable.")
}

// printGuardPolicyVerbUsage renders one verb's own usage: its synopsis, its
// blurb, and its live flag defaults.
func printGuardPolicyVerbUsage(w io.Writer, v guardPolicyVerb, fs *flag.FlagSet) {
	head := "usage: fak guard policy " + v.Name
	if v.Args != "" {
		head += " " + v.Args
	}
	fmt.Fprintln(w, head)
	fmt.Fprintln(w, "  "+v.Blurb)
	fs.PrintDefaults()
}

// guardPolicyVerbFlagSet builds the ContinueOnError flag set for one verb —
// ContinueOnError, not ExitOnError, so a bad flag returns an exit code up through
// cmdGuard instead of exiting from inside the handler.
func guardPolicyVerbFlagSet(v guardPolicyVerb, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet("fak guard policy "+v.Name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { printGuardPolicyVerbUsage(stderr, v, fs) }
	return fs
}

// runGuardPolicy dispatches `fak guard policy <verb> …`. argv is everything after
// the `policy` word.
func runGuardPolicy(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintf(stderr, "fak guard policy: a verb is required (%s)\n", strings.Join(guardPolicyVerbNames(), ", "))
		printGuardPolicyUsage(stderr)
		return guardPolicyExitUsage
	}
	switch argv[0] {
	case "-h", "--help", "help":
		printGuardPolicyUsage(stdout)
		return guardPolicyExitOK
	}
	v, ok := lookupGuardPolicyVerb(argv[0])
	if !ok {
		fmt.Fprintf(stderr, "fak guard policy: unknown verb %q (known verbs: %s)\n", argv[0], strings.Join(guardPolicyVerbNames(), ", "))
		printGuardPolicyUsage(stderr)
		return guardPolicyExitUsage
	}
	return v.Run(v, stdout, stderr, argv[1:])
}

// runGuardPolicyExplainVerb is the CLI adapter for the #5172 report. It hands the
// renderer the SAME allow-layer set the floor loader reads and the same
// deny-overlay path, so the report can never describe a narrower floor than the
// one the guard actually enforces.
//
// The session scope is the load-bearing part: guardAllowOverlayPaths() alone omits
// it, which would render a session-scoped widening as absent and tell the operator
// the floor is narrower than it is. Spelled out from the committed helpers rather
// than via a wrapper so the call site depends only on symbols that exist at HEAD.
func runGuardPolicyExplainVerb(v guardPolicyVerb, stdout, stderr io.Writer, argv []string) int {
	fs := guardPolicyVerbFlagSet(v, stderr)
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	layers := guardAllowLayersWithSessionScope(guardAllowOverlayPaths())
	return runGuardPolicyExplain(stdout, stderr, layers, guardDenyOverlayPath())
}

// runGuardPolicyDiffVerb is the CLI adapter for the #5173 widen-drift report.
// --policy names the manifest to treat as the RUNNING floor, spelled exactly as
// `fak guard --policy FILE` spells it, so an operator can ask the drift question
// about the same file they would launch with; empty means the floor a bare `fak
// guard` would enforce (the embedded floor plus the launch-time overlays).
func runGuardPolicyDiffVerb(v guardPolicyVerb, stdout, stderr io.Writer, argv []string) int {
	fs := guardPolicyVerbFlagSet(v, stderr)
	// The `FILE` backquote is flag.UnquoteUsage's value-name placeholder, so the
	// rendered line reads `-policy FILE`; prose that mentions a command must NOT be
	// backquoted or it would be hijacked as the placeholder.
	policyPath := fs.String("policy", "", "capability-floor manifest `FILE` to read as the RUNNING floor (default: the floor a bare 'fak guard' enforces — the built-in floor plus the launch-time operator overlays)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	return runGuardPolicyDiff(stdout, stderr, *policyPath)
}
