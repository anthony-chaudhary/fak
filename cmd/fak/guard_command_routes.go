package main

import (
	"io"
	"os"
)

// guardCommandRoute keeps the read-only report verbs on one argv and exit-code contract.
// These rows are their only registration: removing one makes that verb unreachable instead
// of silently routing it through a different launch path.
type guardCommandRoute struct {
	verb string
	run  func(stdout, stderr io.Writer, argv []string) int
}

// routeGuardOperatorSubcommand peels operator control and report commands before guard's
// wrapped-agent FlagSet is constructed. A real program with one of these names remains
// wrappable after `--`; only an exact leading verb is handled here.
func routeGuardOperatorSubcommand(commandName string, argv []string) bool {
	// `fak guard disable` is the deliberately loud, one-child break-glass path. Peel it
	// before the wrap-a-command parser so the word "disable" can never fall through to
	// exec.LookPath as though it were an agent binary.
	if len(argv) > 0 && argv[0] == "disable" {
		os.Exit(runGuardDisable(commandName, os.Stdin, os.Stdout, os.Stderr, argv[1:]))
	}

	// `allow` maintains the always-allow overlay out-of-band from the agent. The
	// propose-only modes must run first because cmdGuardAllow rejects those flags.
	if len(argv) > 0 && argv[0] == "allow" {
		guardAllowProposalsRoute(argv[1:])
		cmdGuardAllow(argv[1:])
		return true
	}
	if len(argv) > 0 && argv[0] == "deny" {
		cmdGuardDeny(argv[1:])
		return true
	}

	for _, route := range []guardCommandRoute{
		// `policy` reports the effective capability floor and its widen-drift.
		{"policy", runGuardPolicy},
		// `compile` emits an authoring-time, review-only policy diff.
		{"compile", runGuardCompile},
		// `restart-audit` joins restart hops to carryover seeds and repairs orphans.
		{"restart-audit", runGuardRestartAudit},
	} {
		if len(argv) > 0 && argv[0] == route.verb {
			os.Exit(route.run(os.Stdout, os.Stderr, argv[1:]))
		}
	}

	if len(argv) > 0 && argv[0] == "sessions" {
		cmdGuardSessions(argv[1:])
		return true
	}
	// Both resume spellings are unambiguous before `--`; a wrapped child's own
	// `--resume` remains after the delimiter and never reaches this router.
	if len(argv) > 0 && (argv[0] == "resume" || argv[0] == "--resume") {
		cmdGuardResume(argv[1:])
		return true
	}
	return false
}
