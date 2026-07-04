package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// The `fak score <name>` parent verb groups the meta-tooling scorecards / RSI loops under one
// dispatch entry so the top-level operator surface stays the ~dozen verbs an operator actually
// drives, not the scorecards they never reach for day to day (issue #1505, parent epic #1504).
//
// This is behavior-preserving ROUTING, not deletion: every `fak score <name>` runs the SAME
// handler the top-level verb ran, forwarding argv unchanged, so `fak score conflation --json` and
// the legacy `fak conflation-scorecard --json` produce identical output. The legacy top-level
// verbs remain wired in main.go as thin aliases so no operator's muscle memory or existing caller
// breaks; the consolidation is what the help wall (usage.go) and the doc map present as the
// discoverable surface, which is what the operator-heaviness scorecard measures.

// scoreRoutes maps each `fak score <name>` subcommand to the meta-verb handler it forwards to.
// The value runs the handler with the remaining argv and never returns (each handler os.Exit's,
// or productScorecard exits with its own status). Keys are the legacy verb minus its
// -scorecard/-score/-rsi suffix, so `fak score <name>` reads as the thing being scored.
var scoreRoutes = map[string]func(argv []string){
	"conflation":          cmdConflationScorecard,
	"dogfood":             cmdDogfoodScore,
	"dojo-rsi":            cmdDojoRSI,
	"guard-rsi":           cmdGuardRSIScorecard,
	"guard-verdict-rsi":   cmdGuardVerdictRSI,
	"product":             func(argv []string) { os.Exit(runProductScorecard(os.Stdout, os.Stderr, argv)) },
	"skill-effectiveness": cmdSkillEffectivenessScorecard,
	"support-maturity":    cmdSupportMaturityScorecard,
	"token-defaults":      cmdTokenDefaultsScorecard,
	"ui-quality":          cmdUIQualityScore,
}

// cmdScore routes `fak score <name> [args...]` to the grouped meta-scorecard handler. With no
// subcommand (or `list`/`--help`), it prints the available scorecards.
func cmdScore(argv []string) {
	if len(argv) == 0 || argv[0] == "list" || argv[0] == "-h" || argv[0] == "--help" {
		printScoreList(os.Stdout)
		return
	}
	name := argv[0]
	route, ok := scoreRoutes[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "fak score: unknown scorecard %q\n", name)
		printScoreList(os.Stderr)
		os.Exit(2)
	}
	route(argv[1:])
}

// printScoreList prints the grouped scorecard names, one per line, sorted.
func printScoreList(w *os.File) {
	names := make([]string, 0, len(scoreRoutes))
	for n := range scoreRoutes {
		names = append(names, n)
	}
	sort.Strings(names)
	fmt.Fprintf(w, "usage: fak score <name> [args...]\n\nmeta-scorecards / RSI loops:\n  %s\n",
		strings.Join(names, "\n  "))
}
