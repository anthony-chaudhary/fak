// Package looporphanscan is the impure gather half of the duplicate-loop reaper:
// it turns a real process census into the plain Supervisor data the pure
// internal/looporphan core folds, and re-fences a PID at kill time.
//
// Tier: 2 - see internal/architest. It imports internal/procguard (the census +
// tree-kill, tier 1), internal/looprecover (the PID-reuse-safe liveness probe,
// tier 1), and internal/looporphan (the pure core, tier 1). Keeping this in its
// own package - rather than in cmd/fak - keeps the gather heuristics (which
// process is a loop supervisor, is its parent alive, does its subtree hold a live
// worker) unit-testable against a synthetic census, with no real processes and no
// syscalls in the test path.
//
// The split of concerns:
//
//   - looporphan (tier 1, pure): the keep/reap RULE over plain data.
//   - looporphanscan (tier 2, this package): the EVIDENCE - match supervisors by
//     command-line marker, parse the lane, decide parent liveness from census
//     membership + a plausibility check, count live workers in the subtree, and
//     re-probe a PID's start-time fence immediately before a kill.
//   - cmd/fak/looporphan.go: a thin CLI that calls CollectRelations, hands the
//     census here, prints the plan, and (only behind --reap) calls KillPID on the
//     re-fenced REAP set.
//
// Reap safety has two fences. The core refuses to REAP a supervisor with no
// start-time fence. This package adds the second: ConfirmReap re-reads the PID's
// current start immediately before the kill and refuses if it changed, so a PID
// recycled between the plan and the kill is never the process that dies.
package looporphanscan
