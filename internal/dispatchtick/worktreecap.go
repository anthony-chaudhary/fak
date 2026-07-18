package dispatchtick

// WorktreeABProvenVerdict mirrors the ISOLATION_POISON_FREE verdict that
// internal/scoreboard's #3180 A/B fold emits -- the ONLY outcome that earns the
// #3185 cap raise. It is mirrored as a string rather than imported so the preflight
// fold stays dependency-free and hermetically testable; TestWorktreeCapVerdictMatchesScoreboard
// pins the two spellings together so they can never drift apart silently.
const WorktreeABProvenVerdict = "ISOLATION_POISON_FREE"

// WorktreeIsolation is the OPTIONAL evidence-gated cap raise (#3185) -- the payoff of
// epic #3165. The shared-trunk fleet's observed safe ceiling is ~6 (#1333) because a
// broken build in one worker reds the others past ~4; once per-worker worktree isolation
// (#3168, FLEET_WORKER_WORKTREE) removes the shared build surface, that ceiling is
// artificial. It must NOT be removed on belief: Enabled alone raises nothing, because
// isolation being ON is necessary but not sufficient. The gate is the EVIDENCE -- a
// #3180 A/B report whose isolated arm witnessed zero build-poison at a measured peak
// concurrency. Raising on the flag alone would repeat #1334's premature-close error.
//
// The zero value (Enabled=false, no verdict, ProvenConcurrency=0) is a pure no-op, so a
// caller that sets nothing gets today's cap byte-identically.
type WorktreeIsolation struct {
	// Enabled reports whether per-worker worktree isolation (#3168) is switched on for
	// the spawn path -- cmd/fak's workerWorktreeEnabled(), reading FLEET_WORKER_WORKTREE.
	// Necessary, never sufficient: without proven evidence below it raises nothing.
	Enabled bool `json:"enabled"`
	// Verdict is the #3180 A/B report's verdict. Only WorktreeABProvenVerdict earns the
	// raise; NOT_PROVEN (or an absent report, the empty string) leaves the cap at today's.
	Verdict string `json:"verdict,omitempty"`
	// PoisonIncidents is the isolated arm's build-poison count. ANY non-zero count
	// forfeits the raise even if the verdict string says otherwise -- the count is the
	// primary witness and the verdict is a rendering of it, so they are checked
	// independently rather than trusting one to imply the other.
	PoisonIncidents int `json:"poison_incidents"`
	// ProvenConcurrency is the peak concurrency the poison-free evidence actually covers
	// (the isolated arm's PeakConcurrency). The raise is bounded BY this number, never a
	// constant: evidence at 12 workers earns 12, not "more than 6". Zero means no
	// measurement, which earns nothing.
	ProvenConcurrency int `json:"proven_concurrency"`
}

// WorktreeProvenCap is the concurrency the #3180 evidence has actually EARNED, or 0 when
// the raise is unearned. It is deliberately a conjunction of four independent conditions
// -- isolation on, a proven verdict, a zero poison count, and a positive measured peak --
// so no single mis-set field can manufacture a raise. This is the "derived, witnessed
// value, never a naked constant" the #3185 acceptance gate requires: the returned number
// is always a measurement carried in from the harness, never a literal in this file.
//
// Pure; no I/O, no env, no clock. The caller (EvaluatePreflight) further bounds the result
// by the hard physical ceiling -- host capacity (#1337) and seat inventory -- so proven
// evidence can lift the artificial build-poison ceiling but can never overbook the box.
func WorktreeProvenCap(w WorktreeIsolation) int {
	switch {
	case !w.Enabled:
		return 0
	case w.Verdict != WorktreeABProvenVerdict:
		return 0
	case w.PoisonIncidents != 0:
		return 0
	case w.ProvenConcurrency <= 0:
		return 0
	default:
		return w.ProvenConcurrency
	}
}
