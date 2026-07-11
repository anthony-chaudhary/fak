// Package skillvalue is the per-skill outcome-VALUE ledger — the value sibling
// of a usage counter (epic #2871, issue #2873).
//
// Hermes' curator (agent/curator.py) tracks use_count/view_count/patch_count and
// auto-archives *stale* skills. Staleness is not value: a heavily-used skill can
// still make outcomes worse. This package keeps a VALUE ledger instead — it
// attributes the witnessed outcome delta (pass / cost / latency) of sessions that
// LOADED a skill against MATCHED sessions of the same task class that did NOT, the
// ablation-harness "loaded vs not-loaded" arm keyed by skill id. Skills whose
// measured pass-lift is <= 0 (with enough evidence to say so) are flagged for
// auto-revert; the divergence between the loaded and baseline arms is surfaced the
// way the cache-value gross/net split is (#2797).
//
// It also carries the #2796 valuation-basis discipline: a promoted skill must name
// HOW its value was measured. The Gate flags any active skill that carries no
// valuation basis — a promotion no measurement grounds, the exact gap #2796 closed
// for $ figures, applied to skills.
//
// The fold is PURE and deterministic: it takes pre-read ledger rows and a promotion
// basis map and returns a Rollup. All git / filesystem plumbing (reading the JSONL
// ledger, sourcing the promotion map) is the caller's job — the CLI front door in
// cmd/fak/skill_value.go wires the live paths in.
package skillvalue

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// LedgerSchema tags each per-session outcome row this card folds. A reader keeps
// only rows carrying it, so a shared ledger file cannot feed the rollup a row of
// some other provenance.
const LedgerSchema = "fak-skill-value-ledger/1"

// DefaultLedgerRel is the tracked snapshot of the per-session skill-outcome feed.
// Live runtime rows belong under the gitignored .fak state root; this tracked path
// is the last published historical snapshot, mirroring the cache/memory ledgers.
const DefaultLedgerRel = "docs/nightrun/skill-value.jsonl"

// Flag names attached to a per-skill row. They are the closed vocabulary the report
// and the gate both read — never free text.
const (
	// FlagInsufficientEvidence: the skill has no comparable matched arm (no
	// same-task-class session that both did and did not load it), so its lift is
	// NOT measurable. It is reported not-yet — never auto-reverted on absence.
	FlagInsufficientEvidence = "insufficient-evidence"
	// FlagNetNegative: the skill has a comparable arm AND its measured pass-lift is
	// <= 0 — the auto-revert signal. Keying off measured lift, never load frequency,
	// is the whole point (a heavily-used skill can still be net-negative).
	FlagNetNegative = "net-negative"
	// FlagNoValuationBasis: the skill is active (loaded by >= 1 session) but names no
	// valuation basis — the #2796 gate finding. A promotion no measurement grounds.
	FlagNoValuationBasis = "no-valuation-basis"
)

// SessionRow is one witnessed session outcome that names the skills it loaded. It
// is the per-session grain the rollup matches loaded-vs-not-loaded arms over.
type SessionRow struct {
	Schema    string   `json:"schema"`
	SessionID string   `json:"session_id"`
	TaskClass string   `json:"task_class"` // the matching key: like-for-like comparison bucket
	Skills    []string `json:"skills"`     // skill ids this session loaded (may be empty)
	Pass      bool     `json:"pass"`       // did the session reach a green outcome
	CostUSD   float64  `json:"cost_usd"`
	LatencyMS float64  `json:"latency_ms"`
}

// SkillValue is one skill id's measured outcome lift: the LOADED arm (sessions that
// loaded the skill) against the MATCHED baseline (same-task-class sessions that did
// not), aggregated over task classes that carry BOTH arms and weighted by the loaded
// session count in each class.
type SkillValue struct {
	SkillID string `json:"skill_id"`

	TotalLoaded int `json:"total_loaded"` // every session that loaded the skill (context, not the arm)
	ComparableN int `json:"comparable_n"` // loaded sessions in a class that also has a baseline
	BaselineN   int `json:"baseline_n"`   // matched not-loaded sessions in those same classes
	TaskClasses int `json:"task_classes"` // number of comparable task classes contributing to the lift

	LoadedPass   float64 `json:"loaded_pass"`   // pass rate of the comparable loaded arm
	BaselinePass float64 `json:"baseline_pass"` // pass rate of the matched baseline arm
	PassLift     float64 `json:"pass_lift"`     // LoadedPass - BaselinePass (the gross benefit)
	CostDelta    float64 `json:"cost_delta"`    // mean cost loaded - baseline (positive = dearer)
	LatencyDelta float64 `json:"latency_delta"` // mean latency loaded - baseline (positive = slower)

	ValuationBasis string   `json:"valuation_basis"` // how the skill's value was measured; "" = ungrounded
	Flags          []string `json:"flags,omitempty"`
}

// HasFlag reports whether the skill row carries flag f.
func (s SkillValue) HasFlag(f string) bool {
	for _, g := range s.Flags {
		if g == f {
			return true
		}
	}
	return false
}

// Rollup is the whole-catalog value fold. Skills sort worst-lift-first (the
// auto-revert candidates lead), then by id for a stable order.
type Rollup struct {
	Schema   string       `json:"schema"`
	Sessions int          `json:"sessions"` // ledger rows folded
	Skills   []SkillValue `json:"skills"`
}

// AutoRevert returns the skill ids the ledger says should be reverted: a measured,
// comparable pass-lift <= 0. Insufficient-evidence skills are DELIBERATELY excluded
// — absence of a matched arm is not evidence of harm.
func (r Rollup) AutoRevert() []string {
	var ids []string
	for _, s := range r.Skills {
		if s.HasFlag(FlagNetNegative) {
			ids = append(ids, s.SkillID)
		}
	}
	return ids
}

// GateReport is the valuation-basis gate result (the #2796 mirror): the active
// skills promoted with no valuation basis. OK is true when none are ungrounded.
type GateReport struct {
	Ungrounded []string `json:"ungrounded"`
	OK         bool     `json:"ok"`
}

// Gate flags every active skill (loaded by at least one session) that names no
// valuation basis. It reads the same rollup the report renders, so the gate and the
// report can never disagree about which skills are grounded.
func (r Rollup) Gate() GateReport {
	rep := GateReport{OK: true}
	for _, s := range r.Skills {
		if s.HasFlag(FlagNoValuationBasis) {
			rep.Ungrounded = append(rep.Ungrounded, s.SkillID)
			rep.OK = false
		}
	}
	return rep
}

// ParseLedger scans JSONL content into the SessionRows this card owns, keeping only
// rows tagged LedgerSchema. Blank and malformed lines are skipped.
func ParseLedger(content string) []SessionRow {
	return jsonlledger.Parse(content, func(r SessionRow) bool {
		return r.Schema == LedgerSchema
	})
}

// classArm accumulates one (skill, task-class) arm as sessions are folded in.
type classArm struct {
	loadedN, loadedPass   int
	loadedCost, loadedLat float64
	baseN, basePass       int
	baseCost, baseLat     float64
}

// Compute folds session rows plus a per-skill valuation-basis map into the value
// rollup. basis[id] is the measurement basis a promoted skill carries; a missing or
// empty entry is treated as ungrounded (the strict default — nothing is grounded
// until a measurement names it). The fold is pure and order-independent.
func Compute(sessions []SessionRow, basis map[string]string) Rollup {
	// arms[skill][taskClass] accumulates the loaded and matched-baseline sides.
	arms := map[string]map[string]*classArm{}
	totalLoaded := map[string]int{}

	// Every distinct skill id that appears in any session is "active" — a candidate
	// for both the lift measurement and the valuation-basis gate.
	active := map[string]bool{}

	// First pass: learn the full active skill set (deduping repeats within a
	// session) and each skill's total load count. The baseline arm of skill X is
	// "same-task-class sessions that did NOT load X", so we can only attribute
	// baselines once the whole active set is known — hence the two passes.
	for _, s := range sessions {
		loaded := map[string]bool{}
		for _, id := range s.Skills {
			if id == "" {
				continue
			}
			loaded[id] = true
		}
		for id := range loaded {
			active[id] = true
			totalLoaded[id]++
		}
	}

	// Second pass: with the active set known, attribute each session to the loaded
	// arm of the skills it loaded and to the baseline arm of every active skill it
	// did NOT load, within its own task class.
	for _, s := range sessions {
		loaded := map[string]bool{}
		for _, id := range s.Skills {
			if id != "" {
				loaded[id] = true
			}
		}
		passN := 0
		if s.Pass {
			passN = 1
		}
		for id := range active {
			cm := arms[id]
			if cm == nil {
				cm = map[string]*classArm{}
				arms[id] = cm
			}
			a := cm[s.TaskClass]
			if a == nil {
				a = &classArm{}
				cm[s.TaskClass] = a
			}
			if loaded[id] {
				a.loadedN++
				a.loadedPass += passN
				a.loadedCost += s.CostUSD
				a.loadedLat += s.LatencyMS
			} else {
				a.baseN++
				a.basePass += passN
				a.baseCost += s.CostUSD
				a.baseLat += s.LatencyMS
			}
		}
	}

	var skills []SkillValue
	for id := range active {
		sv := SkillValue{
			SkillID:        id,
			TotalLoaded:    totalLoaded[id],
			ValuationBasis: basis[id],
		}
		// Aggregate over task classes that carry BOTH arms, weighting each class's
		// per-arm rates by that class's loaded count — the like-for-like comparison.
		var wLoadedPass, wBasePass, wLoadedCost, wBaseCost, wLoadedLat, wBaseLat float64
		var wLoaded float64
		for _, a := range arms[id] {
			if a.loadedN == 0 || a.baseN == 0 {
				continue // not a comparable class
			}
			w := float64(a.loadedN)
			wLoaded += w
			wLoadedPass += w * (float64(a.loadedPass) / float64(a.loadedN))
			wBasePass += w * (float64(a.basePass) / float64(a.baseN))
			wLoadedCost += w * (a.loadedCost / float64(a.loadedN))
			wBaseCost += w * (a.baseCost / float64(a.baseN))
			wLoadedLat += w * (a.loadedLat / float64(a.loadedN))
			wBaseLat += w * (a.baseLat / float64(a.baseN))
			sv.ComparableN += a.loadedN
			sv.BaselineN += a.baseN
			sv.TaskClasses++
		}
		if wLoaded > 0 {
			sv.LoadedPass = wLoadedPass / wLoaded
			sv.BaselinePass = wBasePass / wLoaded
			sv.PassLift = sv.LoadedPass - sv.BaselinePass
			sv.CostDelta = (wLoadedCost - wBaseCost) / wLoaded
			sv.LatencyDelta = (wLoadedLat - wBaseLat) / wLoaded
		}

		// Flags. Evidence gates the revert: a skill with no comparable arm is
		// not-yet, never reverted; a measured lift <= 0 is the revert signal.
		if sv.TaskClasses == 0 {
			sv.Flags = append(sv.Flags, FlagInsufficientEvidence)
		} else if sv.PassLift <= 0 {
			sv.Flags = append(sv.Flags, FlagNetNegative)
		}
		if b, ok := basis[id]; !ok || b == "" {
			sv.Flags = append(sv.Flags, FlagNoValuationBasis)
		}
		skills = append(skills, sv)
	}

	sort.Slice(skills, func(i, j int) bool {
		if skills[i].PassLift != skills[j].PassLift {
			return skills[i].PassLift < skills[j].PassLift // worst lift leads
		}
		return skills[i].SkillID < skills[j].SkillID
	})

	return Rollup{Schema: LedgerSchema, Sessions: len(sessions), Skills: skills}
}
