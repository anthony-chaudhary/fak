package agent

// tenure.go — generational tenuring of recurring slash-commands (#848, epic #844).
//
// fak's compaction collector is purely young-generation: every turn and every
// tool-result is a fresh heap object, scored from scratch each turn. A slash-command
// run ONCE and one run EVERY loop (/conflation-score, /quality-score) therefore cost
// the same to re-derive each turn — even though the recurring one is provably
// long-lived. Borrowing the generational-GC insight: a once-invoked command is a
// YOUNG object; a command proven long-lived across loops should be TENURED — promoted
// to a compact ROLLUP so its context stops being re-derived from scratch each turn.
//
// The mechanism is two moves, both ADVISORY and DEFAULT-OFF:
//
//   - PROMOTE: a per-command recurrence counter; once a command recurs past a
//     threshold it crosses from young to tenured and gets a compact Rollup (the
//     promoted, derived-once form).
//   - DEMOTE: tenuring is not permanent. Each command carries a cachemeta.Lifecycle
//     whose per-tier TTL is the "has this command gone quiet?" clock. Record Touches
//     it (revive-on-hot, exactly cachemeta's Touch semantics); a sweep Advances it,
//     and a command whose Lifecycle expires demotes back to young and drops its
//     rollup. A tenured command that keeps recurring keeps reviving, so it stays hot.
//
// Soundness posture (the issue's hard constraints):
//   - NEVER deletes. Tenuring only promotes + demotes. Deletion stays operator-gated.
//   - A rollup is a CACHE of derived context. If absent, the planner falls back to
//     re-deriving each turn (heuristicForecast). A rollup that gated correctness would
//     be a bug — Rollup() returning (zero,false) is always a safe "re-derive" signal.
//   - DEFAULT-OFF: a nil/empty tenureTable changes nothing. A SessionPlanner with no
//     tenure table behaves byte-for-byte as before.
//
// Clock injection: every state transition takes nowMillis (the same testable posture
// as cachemeta.Lifecycle and tools/bench_plan.py's injected --now). This file calls no
// wall clock, so a workload replays deterministically.

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// DefaultTenureThreshold is the recurrence count at which a command crosses from young
// to tenured. The first invocation seeds the entry at count 1 (young); a command must
// recur to at least this many invocations to earn a rollup. 3 mirrors the "run once vs
// run every loop" distinction the issue draws — two repeats prove a loop, not a fluke.
const DefaultTenureThreshold = 3

// DefaultTenureTTLMillis is the quiet-window after which a tenured command demotes back
// to young if it has not recurred. It is the DefaultTTLMillis fed to the per-command
// Lifecycle policy; a recurrence Touches the Lifecycle and resets this clock. 0 would
// mean "never demote" (cachemeta's no-TTL semantics); a positive default keeps a
// command that stops looping from staying tenured forever.
const DefaultTenureTTLMillis int64 = 600_000 // 10 minutes of wall-clock quiet

// Rollup is the compact, promoted form of a tenured command's context — DISTINCT from
// the per-turn derived context the young path recomputes each turn. The young path
// re-derives a command's full context from scratch every turn (heuristicForecast); the
// rollup is the derived-once digest a tenured command carries so that re-derivation is
// skipped. It holds an opaque identity (the command), the digest/summary bytes the
// promotion produced, and the recurrence count that earned it — never the raw per-turn
// transcript (that is precisely what tenuring stops re-deriving).
type Rollup struct {
	Command    string // the slash-command identity this rollup stands in for
	Digest     []byte // the compact derived-once form (a summary/rollup, not the raw turns)
	Recurrence int    // invocations observed when this rollup was promoted/refreshed
}

// IsZero reports whether this is the zero Rollup (no promotion has happened). A
// zero rollup is the safe "re-derive each turn" fallback signal.
func (r Rollup) IsZero() bool { return r.Command == "" && len(r.Digest) == 0 && r.Recurrence == 0 }

// tenureEntry is one command's young/tenured lifecycle record: how many times it has
// recurred, its cachemeta.Lifecycle (the revive/demote clock), whether it has crossed
// the promotion threshold, and its rollup once promoted.
type tenureEntry struct {
	recurrence int
	life       cachemeta.Lifecycle
	tenured    bool
	rollup     Rollup
}

// tenureTable tracks per-command recurrence and tenuring. Its zero value is NOT usable
// (the map is nil) — construct with newTenureTable. A SessionPlanner holds an OPTIONAL
// *tenureTable; a nil pointer is the default-off path (Record/Rollup are nil-safe).
type tenureTable struct {
	threshold int
	policy    cachemeta.LifecyclePolicy
	tier      cachemeta.ResidencyTier
	profiles  map[cachemeta.ResidencyTier]cachemeta.TierProfile
	entries   map[string]*tenureEntry
}

// newTenureTable mints a tenure table with the given promotion threshold and quiet-TTL.
// A non-positive threshold falls back to DefaultTenureThreshold; a non-positive ttl
// falls back to DefaultTenureTTLMillis. The Lifecycle lives in a single notional tier
// (DRAM — the warm, attendable working tier) whose DefaultTTLMillis is the quiet clock.
func newTenureTable(threshold int, ttlMillis int64) *tenureTable {
	if threshold <= 0 {
		threshold = DefaultTenureThreshold
	}
	if ttlMillis <= 0 {
		ttlMillis = DefaultTenureTTLMillis
	}
	return &tenureTable{
		threshold: threshold,
		policy: cachemeta.LifecyclePolicy{
			DefaultTTLMillis: ttlMillis,
			GraceMillis:      0, // demote straight from expiring to expired; tenuring has no grace
		},
		tier:     cachemeta.TierDRAM,
		profiles: cachemeta.DefaultTierProfiles(),
		entries:  map[string]*tenureEntry{},
	}
}

// Record observes one invocation of a slash-command at nowMillis: it bumps the
// recurrence counter and Touches the command's Lifecycle (revive-on-hot — the same
// Touch that revives a cachemeta entry during its grace window, so a recurring command
// never demotes). When the recurrence count crosses the threshold the command is
// PROMOTED to tenured with a compact rollup built by deriveRollup; an already-tenured
// command refreshes its rollup's recurrence on each further hit. It returns the
// command's current Rollup and whether the command is tenured.
//
// Nil-safe: a nil *tenureTable records nothing and returns (zero, false) — the
// default-off path. A command is never deleted here; demotion happens only in Sweep.
func (tt *tenureTable) Record(command string, nowMillis int64) (Rollup, bool) {
	if tt == nil || command == "" {
		return Rollup{}, false
	}
	e, ok := tt.entries[command]
	if !ok {
		e = &tenureEntry{life: cachemeta.NewLifecycle(tt.tier, nowMillis).MarkResident(tt.profiles, nowMillis)}
		tt.entries[command] = e
	}
	e.recurrence++
	// Revive-on-hot: a recurrence renews the freshness window. Touch advances the
	// access tallies and revives an Expiring entry (cachemeta's grace-window revive);
	// for tenuring we ALSO reset the per-tier TTL clock (EnteredTierMillis) so the
	// quiet-window is measured from THIS hit, not from first admission — a command that
	// keeps looping never goes quiet. A hit on a fully-expired entry re-admits it
	// resident (young again until it re-earns tenure; Touch alone does not leave
	// Expired). This is what keeps a still-looping command tenured across a Sweep.
	if e.life.State == cachemeta.StateExpired {
		e.life = cachemeta.NewLifecycle(tt.tier, nowMillis).MarkResident(tt.profiles, nowMillis)
		e.life.Accesses = uint64(e.recurrence)
	} else {
		e.life = e.life.Touch(nowMillis)
		e.life.State = cachemeta.StateResident
		e.life.EnteredTierMillis = nowMillis // renew the per-tier TTL (quiet) clock
		e.life.StateSinceMillis = nowMillis
	}
	if !e.tenured && e.recurrence >= tt.threshold {
		e.tenured = true
		e.rollup = deriveRollup(command, e.recurrence)
	} else if e.tenured {
		// keep the rollup's recurrence current so an audit sees the live count
		e.rollup.Recurrence = e.recurrence
	}
	return e.rollup, e.tenured
}

// Sweep applies the time-driven demotion at nowMillis: it Advances each command's
// Lifecycle and demotes (drops the rollup, marks young) any command whose Lifecycle has
// EXPIRED — the command went quiet past its TTL. It returns the commands demoted this
// sweep (for an audit/metric). A still-resident or merely-expiring command is left
// tenured (it is in its grace/quiet window and a Record would revive it). Nil-safe.
//
// Sweep NEVER deletes the entry — a demoted command keeps its recurrence history and
// can re-earn tenure by recurring again; only its rollup (the cache) is dropped.
func (tt *tenureTable) Sweep(nowMillis int64) []string {
	if tt == nil {
		return nil
	}
	var demoted []string
	for cmd, e := range tt.entries {
		// Advance is a single time-driven step (Resident->Expiring, then Expiring->Expired
		// after the grace window). Drive it to a fixpoint so a command quiet well past its
		// TTL reaches Expired in ONE sweep rather than needing a sweep per transition.
		for {
			life, changed := e.life.Advance(tt.policy, nowMillis)
			e.life = life
			if !changed {
				break
			}
		}
		if e.tenured && e.life.State == cachemeta.StateExpired {
			e.tenured = false
			e.rollup = Rollup{} // drop the cache; the planner falls back to re-deriving
			demoted = append(demoted, cmd)
		}
	}
	sort.Strings(demoted)
	return demoted
}

// Rollup returns a command's compact rollup and whether it is currently tenured. A
// command that is young, unknown, or demoted returns (zero, false) — the safe
// "re-derive this turn" signal the planner falls back on. Nil-safe (default-off).
func (tt *tenureTable) Rollup(command string) (Rollup, bool) {
	if tt == nil {
		return Rollup{}, false
	}
	e, ok := tt.entries[command]
	if !ok || !e.tenured {
		return Rollup{}, false
	}
	return e.rollup, true
}

// Recurrence returns how many invocations of a command have been observed (0 for an
// unknown command). Exposed for an EXPLAIN/audit surface and the promote/demote tests.
func (tt *tenureTable) Recurrence(command string) int {
	if tt == nil {
		return 0
	}
	if e, ok := tt.entries[command]; ok {
		return e.recurrence
	}
	return 0
}

// deriveRollup builds the compact promoted form for a command. The promotion that the
// generational model skips re-deriving is represented here as a small, opaque digest
// (the command identity + the recurrence that earned tenure) — DISTINCT by construction
// from the per-turn derived context (which is the full re-scanned span set). A richer
// rollup (an actual summarized context block) is the natural extension; the contract
// that matters is that it is a derived-once CACHE, never correctness-gating.
func deriveRollup(command string, recurrence int) Rollup {
	return Rollup{
		Command:    command,
		Digest:     []byte("tenured:" + command),
		Recurrence: recurrence,
	}
}
