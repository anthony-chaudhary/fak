// Package waiting is the R3 waiting-on-human queue (#2272, epic #2269): a pure fold over loop-event ledgers (internal/loopmgr) that turns each blocked-on-operator notify into one kernel object with age, held resources, deadline, and the safe default that fires on expiry — babysitting inverted: the fleet files tickets on the human, with deadlines.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package waiting
