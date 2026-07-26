package cachevaluereport

// census.go — the fleet managed-cache posture-adoption census (#3650), one scoped leaf
// under epic #3569 (independent trust-but-verify LOOPS for managed cache).
//
// The question this fold answers is "how much of the LIVE fleet actually runs managed
// cache ACTIVE, and of those, how many ever fire an upgrade?". Posture is chosen at
// `fak accounts launch` / dispatch-worker argv build time (the fleet posture policy in
// cmd/fak, keyed on FAK_MANAGED_CACHE and FAK_GUARD_API_KEY_ENV) and published per worker
// on the gateway's /debug/vars managed_cache block
// (guardvars.ManagedCacheVars), but it was never rolled up across workers — so a fleet
// running mostly PASSIVE was invisible.
//
// It is deliberately NOT the weekly digest's PostureAdoptionPct (#3646). That number is
// INFERRED from durable exit rows: a session counts as posture-active only when its
// TTL-upgrade lever left evidence behind (an upgrade or a refusal reason), so an ACTIVE
// worker that fired nothing and logged no reason reads as passive there. This census reads
// the RESOLVED posture flag itself off the live worker, so ACTIVE-with-no-evidence and
// genuinely PASSIVE stay distinct — which is exactly the pair the epic needs to trust.
//
// The fold is PURE and deterministic (worker rows in, report out; `now` only stamps
// GeneratedAt), mirroring FoldWeeklyDigest / FoldConfiguredButInert. It is a diagnostic
// census, not a CI gate: OK stays true for every verdict.

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// CensusSchema versions the census envelope so a downstream reader can pin it
// independently of the other cache-value fold schemas.
const CensusSchema = "fak-managed-cache-adoption-census/1"

// The per-worker census states. UNKNOWN is reserved for a worker the scrape could not
// reach: it is excluded from every ratio rather than folded in as PASSIVE, so an
// unreachable slice of the fleet can never manufacture a low adoption number.
const (
	StateActive  = "ACTIVE"
	StatePassive = "PASSIVE"
	StateUnknown = "UNKNOWN"
)

// The census verdicts. This is an observability loop, so MOSTLY_PASSIVE is a finding to
// read, not a failure — OK stays true for every verdict (mirroring FoldConfiguredButInert).
// VerdictInsufficient is shared with the configured-but-inert loop.
const (
	VerdictMostlyPassive = "MOSTLY_PASSIVE"
	VerdictAdopted       = "ADOPTED"
)

// censusMajorityPct is the share of reached workers that must be ACTIVE for the fleet to
// read ADOPTED rather than MOSTLY_PASSIVE. A bare majority — the census reports the exact
// percentage either way, so this only picks the headline word.
const censusMajorityPct = 50.0

// WorkerRow is one live fleet worker's contribution to the census: the outcome of scraping
// its gateway /debug/vars and reading the managed_cache posture block.
//
// The three-way Reached/Published/Active split is what keeps the headline honest:
//
//   - Reached=false — the scrape failed (worker gone, port closed, bearer refused). Nothing
//     is known about its posture, so it lands in UNKNOWN and is excluded from both the
//     numerator and the denominator.
//   - Reached=true, Published=false — the worker answered and published NO managed_cache
//     block. Per that block's producer contract (internal/gateway.managedCacheVars) the
//     block is omitted ONLY when the lever is off and nothing was observed, so an absent
//     block is an affirmative PASSIVE witness, not a gap.
//   - Reached=true, Published=true — Active is the worker's own resolved lever state and
//     Upgraded its witnessed 1h-TTL upgrade count.
type WorkerRow struct {
	Worker    string `json:"worker"`         // worker label (trace id / handle / host:port)
	Reached   bool   `json:"reached"`        // the /debug/vars scrape succeeded
	Published bool   `json:"published"`      // the managed_cache block was present
	Active    bool   `json:"active"`         // resolved managed-cache lever state
	Upgraded  uint64 `json:"upgraded"`       // witnessed 1h-TTL upgrades this worker fired
	Wire      string `json:"wire,omitempty"` // resolved upstream provider
}

// RowFromVars builds a census row from a REACHED worker's /debug/vars managed_cache block.
// A nil block is the producer's "lever off and nothing observed" shape, so it folds as an
// affirmative PASSIVE witness rather than an unknown.
func RowFromVars(worker string, mc *guardvars.ManagedCacheVars) WorkerRow {
	row := WorkerRow{Worker: labelOrUnknown(worker), Reached: true}
	if mc == nil {
		return row
	}
	row.Published = true
	row.Active = mc.Active
	row.Upgraded = mc.Upgraded
	row.Wire = mc.Wire
	return row
}

// UnreachedRow builds the census row for a worker the scrape could not reach. It carries no
// posture claim at all — the census counts it and excludes it from every ratio.
func UnreachedRow(worker string) WorkerRow {
	return WorkerRow{Worker: labelOrUnknown(worker)}
}

// labelOrUnknown keeps every row addressable in the rendered census even when the caller
// had no worker label to supply.
func labelOrUnknown(worker string) string {
	if w := strings.TrimSpace(worker); w != "" {
		return w
	}
	return "unknown"
}

// State classifies the row for the census breakdown.
func (r WorkerRow) State() string {
	switch {
	case !r.Reached:
		return StateUnknown
	case r.Active:
		return StateActive
	}
	return StatePassive
}

// hasLever reports whether this worker's resolved wire even HAS the Anthropic 1h-TTL
// upgrade lever, delegating to the producer's own predicate so the two can never drift. On
// the OpenAI Responses (codex) wire the lever does not exist — fak's managed-cache lever
// there is the pinned prompt_cache_key — so an ACTIVE worker on that wire can never fire an
// upgrade, and counting it in the upgrade-fired denominator would report a fleet-wide
// failure that is really a wire without the lever.
func (r WorkerRow) hasLever() bool {
	return !(guardvars.ManagedCacheVars{Wire: r.Wire}).WireHasNo1hTTLLever()
}

// CensusReport is the fleet roll-up: what fraction of reached workers run managed cache
// ACTIVE, and among the ACTIVE workers whose wire has the 1h-TTL lever, what fraction ever
// fired at least one upgrade. Ratio fields are pointers so "nothing to divide by" (nil) is
// never conflated with a measured zero.
type CensusReport struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`

	// Coverage. Unreached workers are counted here and nowhere else.
	Workers   int `json:"workers"`
	Reached   int `json:"reached"`
	Unreached int `json:"unreached"`

	// Headline 1 — fraction ACTIVE, over REACHED workers.
	Active    int      `json:"active"`
	Passive   int      `json:"passive"`
	ActivePct *float64 `json:"active_pct,omitempty"` // 0..100; nil when nothing was reached

	// Headline 2 — among ACTIVE workers on a wire that HAS the 1h-TTL lever, the fraction
	// that fired at least one upgrade. ActiveLeverless holds the ACTIVE workers excluded
	// from that denominator because their wire has no such lever.
	ActiveWithLever int      `json:"active_with_lever"`
	ActiveLeverless int      `json:"active_leverless"`
	UpgradeFired    int      `json:"upgrade_fired"`
	UpgradeFiredPct *float64 `json:"upgrade_fired_pct,omitempty"` // 0..100; nil when no ACTIVE worker has the lever

	// Rows is the per-worker breakdown behind the ratios, ordered by state (ACTIVE, then
	// PASSIVE, then UNKNOWN) and worker label so the census renders deterministically.
	Rows []WorkerRow `json:"rows,omitempty"`

	Verdict    string `json:"verdict"` // ADOPTED | MOSTLY_PASSIVE | INSUFFICIENT
	Finding    string `json:"finding"`
	NextAction string `json:"next_action,omitempty"`
	OK         bool   `json:"ok"`
}

// stateRank orders the per-worker breakdown: the ACTIVE workers first (they carry the
// second headline), then PASSIVE, then the unreachable tail.
var stateRank = map[string]int{StateActive: 0, StatePassive: 1, StateUnknown: 2}

// FoldCensus folds live worker posture into the fleet adoption census. Pure and
// deterministic: rows + a caller-supplied `now` in, a report out — no clock, no I/O, no
// network. An empty fleet, or one where every scrape failed, folds to the honest
// INSUFFICIENT census rather than a fabricated 0% adoption.
func FoldCensus(rows []WorkerRow, now time.Time) CensusReport {
	rep := CensusReport{
		Schema:      CensusSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Workers:     len(rows),
		OK:          true,
	}
	for _, r := range rows {
		r.Worker = labelOrUnknown(r.Worker)
		rep.Rows = append(rep.Rows, r)
		switch r.State() {
		case StateUnknown:
			rep.Unreached++
			continue
		case StateActive:
			rep.Reached++
			rep.Active++
			if !r.hasLever() {
				rep.ActiveLeverless++
				continue
			}
			rep.ActiveWithLever++
			if r.Upgraded > 0 {
				rep.UpgradeFired++
			}
		default:
			rep.Reached++
			rep.Passive++
		}
	}
	sort.SliceStable(rep.Rows, func(i, j int) bool {
		li, lj := stateRank[rep.Rows[i].State()], stateRank[rep.Rows[j].State()]
		if li != lj {
			return li < lj
		}
		return rep.Rows[i].Worker < rep.Rows[j].Worker
	})
	if rep.Reached > 0 {
		pct := 100 * float64(rep.Active) / float64(rep.Reached)
		rep.ActivePct = &pct
	}
	if rep.ActiveWithLever > 0 {
		pct := 100 * float64(rep.UpgradeFired) / float64(rep.ActiveWithLever)
		rep.UpgradeFiredPct = &pct
	}
	rep.finalizeCensus()
	return rep
}

// finalizeCensus sets the report-contract fields from the folded totals.
func (rep *CensusReport) finalizeCensus() {
	if rep.Reached == 0 {
		rep.Verdict = VerdictInsufficient
		if rep.Workers == 0 {
			rep.Finding = "no fleet workers to census"
			rep.NextAction = "supply one row per live worker (fak guard sessions publishes each worker's gateway URL + read-scoped bearer), then re-fold"
		} else {
			rep.Finding = fmt.Sprintf("%d worker(s), none reachable — no posture observed", rep.Workers)
			rep.NextAction = "the whole fleet failed to answer /debug/vars; check the published gateway URLs and bearers before reading any adoption number"
		}
		return
	}
	if *rep.ActivePct >= censusMajorityPct {
		rep.Verdict = VerdictAdopted
	} else {
		rep.Verdict = VerdictMostlyPassive
	}
	rep.Finding = fmt.Sprintf("%s — managed cache ACTIVE on %d/%d reached worker(s) (%.0f%%); upgrade fired on %s of the ACTIVE workers whose wire has the 1h-TTL lever (%d/%d)%s",
		rep.Verdict, rep.Active, rep.Reached, *rep.ActivePct,
		pctOrNA(rep.UpgradeFiredPct), rep.UpgradeFired, rep.ActiveWithLever, rep.exclusionNote())
	rep.NextAction = rep.nextAction()
}

// exclusionNote appends the honesty qualifiers the ratios depend on: workers excluded from
// the denominators because they were unreachable, or because their wire has no 1h-TTL lever.
func (rep *CensusReport) exclusionNote() string {
	var parts []string
	if rep.Unreached > 0 {
		parts = append(parts, fmt.Sprintf("%d unreachable worker(s) excluded from every ratio", rep.Unreached))
	}
	if rep.ActiveLeverless > 0 {
		parts = append(parts, fmt.Sprintf("%d ACTIVE worker(s) on a wire with no 1h-TTL lever excluded from the upgrade ratio", rep.ActiveLeverless))
	}
	if len(parts) == 0 {
		return ""
	}
	return "; " + strings.Join(parts, "; ")
}

// nextAction names the one thing to do about this census, keyed on which headline is the
// weak one: a passive fleet is a launcher-posture question, an ACTIVE fleet that fires
// nothing is a lever-wiring question (the fleet-scale sibling of the per-session
// CONFIGURED_BUT_INERT finding).
func (rep *CensusReport) nextAction() string {
	if rep.Verdict == VerdictMostlyPassive {
		return "most reached workers run PASSIVE — check the launcher posture the fleet resolves (FAK_MANAGED_CACHE / FAK_GUARD_API_KEY_ENV); this census reports the fraction, it changes no default"
	}
	if rep.ActiveWithLever > 0 && rep.UpgradeFired == 0 {
		return "the fleet is ACTIVE but not one worker fired an upgrade — verify the 1h-TTL upgrade wiring is reached in practice before trusting the ACTIVE posture"
	}
	return "read the census; a falling ACTIVE share or a falling upgrade-fired share is the cue to re-check launcher posture and the 1h-TTL upgrade wiring"
}

// RenderCensus renders the census as a compact, deterministic terminal block.
func RenderCensus(r CensusReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "fleet managed-cache posture census — %s\n", r.Verdict)
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	if r.NextAction != "" {
		fmt.Fprintf(&sb, "  next: %s\n", r.NextAction)
	}
	fmt.Fprintf(&sb, "\n  workers %d (reached %d, unreachable %d)\n", r.Workers, r.Reached, r.Unreached)
	fmt.Fprintf(&sb, "  ACTIVE %d · PASSIVE %d · ACTIVE share %s\n", r.Active, r.Passive, pctOrNA(r.ActivePct))
	fmt.Fprintf(&sb, "  upgrade fired %d/%d ACTIVE-with-lever (%s)", r.UpgradeFired, r.ActiveWithLever, pctOrNA(r.UpgradeFiredPct))
	if r.ActiveLeverless > 0 {
		fmt.Fprintf(&sb, " · %d ACTIVE on a wire with no 1h-TTL lever", r.ActiveLeverless)
	}
	sb.WriteString("\n")
	if len(r.Rows) == 0 {
		return sb.String()
	}
	fmt.Fprintf(&sb, "\n  %-24s  %-8s  %-9s  %s\n", "worker", "state", "upgrades", "wire")
	for _, row := range r.Rows {
		fmt.Fprintf(&sb, "  %-24s  %-8s  %-9d  %s\n", row.Worker, row.State(), row.Upgraded, row.Wire)
	}
	return sb.String()
}
