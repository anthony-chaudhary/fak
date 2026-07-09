package main

// Fixture for knobcensus.Scan — NOT compiled into any build (it lives under
// testdata/). It carries one of each classification so the walker's verdicts and
// its over-match guards are pinned. Line numbers matter: the test asserts
// file:line provenance.

import (
	"flag"
	"os"
)

func sample() {
	fs := flag.NewFlagSet("sample", flag.ContinueOnError)

	// INTENT — a which-account choice the system cannot infer.
	fs.String("account", "", "which account pool to draw from")
	// HOUSEKEEPING — a TTL/eviction timing derivable from telemetry.
	fs.Duration("session-cooldown-ttl", 0, "how long a parked session stays warm")
	// strong-intent override: carries both "account" and "refresh" → INTENT wins.
	fs.Bool("account-refresh", false, "re-select the account before dispatch")

	// Excluded: pure plumbing / output — not a behavior knob.
	fs.Bool("json", false, "emit JSON")
	fs.String("root", "", "repo root")

	// A context knob #2199 already owns — folded in as HOUSEKEEPING, not
	// re-derived (the "no second context count" contract).
	fs.Int("ctx-view-budget", 0, "context view budget")

	// INTENT env — a user-set objective.
	_ = os.Getenv("FAK_GOAL_OBJECTIVE")
	// HOUSEKEEPING env — an auth-refresh timing window.
	_ = os.Getenv("FAK_GUARD_AUTO_REFRESH")
	// Excluded env — a plain address, gates no user behavior.
	_ = os.Getenv("FAK_ADDR")

	_ = fs
}
