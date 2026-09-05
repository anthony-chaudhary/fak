// Package skillvalue provides outcome value accounting for skills by comparing
// loaded sessions against matched baseline sessions of the same task class.
package skillvalue

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// LedgerSchema identifies per-session outcome records in JSONL ledgers.
const LedgerSchema = "fak-skill-value-ledger/1"

// DefaultLedgerRel is the repository path for historical skill value snapshots.
const DefaultLedgerRel = "docs/nightrun/skill-value.jsonl"

// Flags attached to evaluated skill outcome records.
const (
	// FlagInsufficientEvidence marks a skill lacking a matched baseline comparison arm.
	FlagInsufficientEvidence = "insufficient-evidence"
	// FlagNetNegative marks a skill whose measured pass lift is non-positive.
	FlagNetNegative = "net-negative"
	// FlagNoValuationBasis marks an active skill lacking a declared valuation basis.
	FlagNoValuationBasis = "no-valuation-basis"
)

// SessionRow records a single session outcome and its loaded skills.
type SessionRow struct {
	Schema    string   `json:"schema"`
	SessionID string   `json:"session_id"`
	TaskClass string   `json:"task_class"`
	Skills    []string `json:"skills"`
	Pass      bool     `json:"pass"`
	CostUSD   float64  `json:"cost_usd"`
	LatencyMS float64  `json:"latency_ms"`
}

// SkillValue records the measured outcome lift of a skill across matched task classes.
type SkillValue struct {
	SkillID string `json:"skill_id"`

	TotalLoaded int `json:"total_loaded"`
	ComparableN int `json:"comparable_n"`
	BaselineN   int `json:"baseline_n"`
	TaskClasses int `json:"task_classes"`

	LoadedPass   float64 `json:"loaded_pass"`
	BaselinePass float64 `json:"baseline_pass"`
	PassLift     float64 `json:"pass_lift"`
	CostDelta    float64 `json:"cost_delta"`
	LatencyDelta float64 `json:"latency_delta"`

	ValuationBasis string   `json:"valuation_basis"`
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

// Rollup aggregates evaluated skill outcomes across all processed sessions.
type Rollup struct {
	Schema   string       `json:"schema"`
	Sessions int          `json:"sessions"`
	Skills   []SkillValue `json:"skills"`
}

// AutoRevert returns skill IDs exhibiting non-positive measured pass lift.
func (r Rollup) AutoRevert() []string {
	var ids []string
	for _, s := range r.Skills {
		if s.HasFlag(FlagNetNegative) {
			ids = append(ids, s.SkillID)
		}
	}
	return ids
}

// GateReport summarizes active skills lacking an explicit valuation basis.
type GateReport struct {
	Ungrounded []string `json:"ungrounded"`
	OK         bool     `json:"ok"`
}

// Gate identifies active skills that lack a declared valuation basis.
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

// ParseLedger extracts session rows matching LedgerSchema from JSONL data.
func ParseLedger(content string) []SessionRow {
	return jsonlledger.Parse(content, func(r SessionRow) bool {
		return r.Schema == LedgerSchema
	})
}

// classArm accumulates loaded and matched baseline metrics for a task class.
type classArm struct {
	loadedN, loadedPass   int
	loadedCost, loadedLat float64
	baseN, basePass       int
	baseCost, baseLat     float64
}

// Compute builds a value rollup comparing loaded skills against baselines.
func Compute(sessions []SessionRow, basis map[string]string) Rollup {
	arms := map[string]map[string]*classArm{}
	totalLoaded := map[string]int{}
	active := map[string]bool{}

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
		var wLoadedPass, wBasePass, wLoadedCost, wBaseCost, wLoadedLat, wBaseLat float64
		var wLoaded float64
		for _, a := range arms[id] {
			if a.loadedN == 0 || a.baseN == 0 {
				continue
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
			return skills[i].PassLift < skills[j].PassLift
		}
		return skills[i].SkillID < skills[j].SkillID
	})

	return Rollup{Schema: LedgerSchema, Sessions: len(sessions), Skills: skills}
}
