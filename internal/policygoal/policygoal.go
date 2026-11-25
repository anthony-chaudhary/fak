// Package policygoal defines the closed vocabulary of named inference policy
// goals and a deterministic inventory that maps the audited serve-config knobs
// to goals, hot/cold (live-changeable vs restart-required) classification, and
// defect-hiding flags.
//
// It answers the one question `fak serve-wiring` cannot: of every serving
// config knob, which policy goal does it serve, can it change policy live
// (hot) without a restart, and does it hide a defect that should be fixed by
// default? The inventory is a data unit: the next agent can read it without
// hand-digging the gateway.Config surface.
package policygoal

import (
	"encoding/json"
	"sort"
)

// Goal is the closed vocabulary of named inference policy goals.
type Goal string

const (
	// GoalBatch maximizes throughput under load (continuous batching, large
	// prefill batches).
	GoalBatch Goal = "batch"
	// GoalInteractive minimizes per-turn latency for a single user.
	GoalInteractive Goal = "interactive"
	// GoalReliability sheds or fails safely under pressure rather than
	// silently degrading.
	GoalReliability Goal = "reliability"
	// GoalCapacity maximizes admitted concurrent work.
	GoalCapacity Goal = "capacity"
	// GoalP50 optimizes the median latency.
	GoalP50 Goal = "p50"
	// GoalP99 optimizes the tail latency.
	GoalP99 Goal = "p99"
	// GoalPrefillDecode balances prefill versus decode work.
	GoalPrefillDecode Goal = "prefill-decode"
	// GoalContextLength governs context-window trade-offs.
	GoalContextLength Goal = "context-length"
)

// AllGoals is the closed set, in canonical order.
var AllGoals = []Goal{
	GoalBatch,
	GoalInteractive,
	GoalReliability,
	GoalCapacity,
	GoalP50,
	GoalP99,
	GoalPrefillDecode,
	GoalContextLength,
}

// Hotness is whether a knob can change policy live (hot) or requires a
// restart (cold).
type Hotness string

const (
	// Hot means the knob can change policy live without a restart.
	Hot Hotness = "hot"
	// Cold means the knob requires a restart to take effect.
	Cold Hotness = "cold"
)

// Knob is one audited serve-config knob mapped to policy goals.
type Knob struct {
	// Name is the operator flag or gateway.Config field.
	Name string `json:"name"`
	// Goals is the set of policy goals this knob affects.
	Goals []Goal `json:"goals"`
	// Hotness is whether the knob can change policy live.
	Hotness Hotness `json:"hotness"`
	// DefectHiding is true when the knob can hide a defect that should be
	// fixed by default (e.g. a budget that silences a real starvation).
	DefectHiding bool `json:"defect_hiding"`
	// Evidence is the load-bearing call site or doc reference.
	Evidence string `json:"evidence"`
}

// Inventory is a deterministic, sorted inventory of the audited serve-config
// knobs.
type Inventory struct {
	Knobs []Knob `json:"knobs"`
}

// DefaultInventory is the audited baseline. It is seeded from the
// `fak serve-wiring` rows and the gateway.Config surface.
var DefaultInventory = Inventory{Knobs: []Knob{
	{
		Name:         "--max-total-tokens",
		Goals:        []Goal{GoalCapacity, GoalContextLength, GoalReliability}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:353; upper bound on a request's total tokens",
	},
	{
		Name:         "--max-batch-prefill-tokens",
		Goals:        []Goal{GoalBatch, GoalPrefillDecode, GoalP99}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:354; upper bound on batch prefill tokens",
	},
	{
		Name:         "--tenant-rate-rps",
		Goals:        []Goal{GoalReliability, GoalCapacity, GoalP99}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:361; per-tenant request-rate admission",
	},
	{
		Name:         "--tenant-max-inflight",
		Goals:        []Goal{GoalCapacity, GoalReliability}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:363; per-tenant in-flight admission",
	},
	{
		Name:         "--stream-progress-timeout",
		Goals:        []Goal{GoalReliability, GoalP99, GoalInteractive}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: true,
		Evidence:     "cmd/fak/serve.go:129; a negative value disables the hard stall deadline, hiding a real upstream stall",
	},
	{
		Name:         "--ctx-view-budget",
		Goals:        []Goal{GoalContextLength, GoalInteractive, GoalP50}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: true,
		Evidence:     "cmd/fak/serve.go:105; 0 means raw history byte-for-byte, hiding context sprawl",
	},
	{
		Name:         "--compact-history-budget",
		Goals:        []Goal{GoalContextLength, GoalP50, GoalInteractive}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: true,
		Evidence:     "cmd/fak/serve.go:106; 0 means no compaction, hiding context sprawl",
	},
	{
		Name:         "--ctx-paging",
		Goals:        []Goal{GoalContextLength, GoalInteractive}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Hot,
		DefectHiding: true,
		Evidence:     "cmd/fak/serve.go:120; per-request X-Fak-Ctx-Paging header wins, so it can change live; OFF hides oversize tool results",
	},
	{
		Name:         "--chat-elide",
		Goals:        []Goal{GoalContextLength, GoalP50}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Hot,
		DefectHiding: true,
		Evidence:     "cmd/fak/serve.go:121; per-request header wins; OFF hides tool-result elision",
	},
	{
		Name:         "--elide-result-bytes",
		Goals:        []Goal{GoalContextLength, GoalP50}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: true,
		Evidence:     "cmd/fak/serve.go:109; 0 means no elision, hiding oversized tool results",
	},
	{
		Name:         "--speculative",
		Goals:        []Goal{GoalInteractive, GoalP50, GoalP99}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:132; structural speculative-decode selector",
	},
	{
		Name:         "--llama-slot-affinity",
		Goals:        []Goal{GoalBatch, GoalP50, GoalPrefillDecode}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:135; llama-server prefix slot pinning",
	},
	{
		Name:         "--readiness-coherence",
		Goals:        []Goal{GoalReliability},
		Hotness:      Cold,
		DefectHiding: true,
		Evidence:     "cmd/fak/serve.go:355; 0 disables the coherence probe, hiding binary/shader skew",
	},
	{
		Name:         "--reset-on-budget",
		Goals:        []Goal{GoalReliability, GoalContextLength}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:111; distills a carryover seed on budget exhaustion",
	},
	{
		Name:         "--native-max-turns",
		Goals:        []Goal{GoalInteractive, GoalReliability}, //enumlint:exempt a knob affects only the goals it names; the per-knob goal list is its footprint, not an exhaustive goal catalog
		Hotness:      Cold,
		DefectHiding: false,
		Evidence:     "cmd/fak/serve.go:96; caps the owned native serve loop's model round-trips",
	},
}}

// Sort returns a copy of the inventory with knobs sorted by Name, so the
// output is deterministic regardless of declaration order.
func (inv Inventory) Sort() Inventory {
	out := Inventory{Knobs: make([]Knob, len(inv.Knobs))}
	copy(out.Knobs, inv.Knobs)
	sort.Slice(out.Knobs, func(i, j int) bool {
		return out.Knobs[i].Name < out.Knobs[j].Name
	})
	return out
}

// GoalsFor returns the knobs that affect the named goal, in inventory order.
func (inv Inventory) GoalsFor(g Goal) []Knob {
	var out []Knob
	for _, k := range inv.Knobs {
		for _, kg := range k.Goals {
			if kg == g {
				out = append(out, k)
				break
			}
		}
	}
	return out
}

// Hot returns the knobs that can change policy live.
func (inv Inventory) Hot() []Knob {
	var out []Knob
	for _, k := range inv.Knobs {
		if k.Hotness == Hot {
			out = append(out, k)
		}
	}
	return out
}

// DefectHiding returns the knobs that can hide a defect that should be fixed
// by default.
func (inv Inventory) DefectHiding() []Knob {
	var out []Knob
	for _, k := range inv.Knobs {
		if k.DefectHiding {
			out = append(out, k)
		}
	}
	return out
}

// CountByGoal returns the number of knobs per goal.
func (inv Inventory) CountByGoal() map[Goal]int {
	out := make(map[Goal]int, len(AllGoals))
	for _, g := range AllGoals {
		out[g] = 0
	}
	for _, k := range inv.Knobs {
		for _, g := range k.Goals {
			out[g]++
		}
	}
	return out
}

// MarshalJSON renders the inventory in deterministic (sorted) order. It
// marshals a plain alias so the sorted Inventory does not re-enter this
// method (which would recurse forever).
func (inv Inventory) MarshalJSON() ([]byte, error) {
	type plain struct {
		Knobs []Knob `json:"knobs"`
	}
	return json.Marshal(plain{Knobs: inv.Sort().Knobs})
}
