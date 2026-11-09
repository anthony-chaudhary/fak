// Package benchsweep provides the engine-agnostic sweep mechanics for
// multi-parameter benchmark campaigns: deciding the ORDER in which a
// campaign's expanded combos execute so the serving process restarts as
// rarely as possible. It is a clean-room conceptual adaptation of the pinned
// Bench deployment-sweep study (parent borrow oss-port-bench-sweep-20261109,
// upstream anthony-chaudhary/Benchmark@b60f2f05e, MIT-declared): the behavior
// contracts were re-derived from the study note, and no upstream source was
// copied.
//
// Everything here is pure and stdlib-only: no server launching, no HTTP, no
// process management, no clock read. The caller owns execution; this package
// only decides ordering and bookkeeping, so a rerun of the same campaign
// always plans the same sequence.
package benchsweep

import (
	"fmt"
	"sort"
	"strings"
)

// Param is one swept axis pinned to one value inside a combo. Names come from
// the campaign's declared sweep axes; values are their string form.
type Param struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// Combo is one fully expanded point of the sweep's Cartesian space: every
// swept axis pinned to exactly one value. ServerParams are the axes whose
// change forces the serving process to restart before the next combo can run;
// WorkloadParams vary the per-combo load without touching the running server.
// Only ServerParams participate in the restart decision, which is what makes
// restart minimization possible at all.
type Combo struct {
	ID             string  `json:"id,omitempty"`
	ServerParams   []Param `json:"server_params,omitempty"`
	WorkloadParams []Param `json:"workload_params,omitempty"`
}

// ConfigKey is the hashable identity of a combo's restart-relevant server
// configuration. Two combos share a key exactly when one server process,
// started for the first, can serve the second without a restart.
type ConfigKey string

// EncodeConfigKey renders the canonical encoding of a server-parameter set:
// params sorted by name, each side length-prefixed so arbitrary values can
// never collide across field boundaries. Sorting makes the key independent of
// declaration order, so two combos describing the same server configuration
// always encode to the same key. An empty set encodes to "" (the
// no-server-config key), which keeps an empty sweep a valid, comparable plan.
func EncodeConfigKey(params []Param) ConfigKey {
	sorted := append([]Param(nil), params...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	var b strings.Builder
	for _, p := range sorted {
		fmt.Fprintf(&b, "%d:%s%d:%s;", len(p.Name), p.Name, len(p.Value), p.Value)
	}
	return ConfigKey(b.String())
}

// ConfigKey returns the restart-relevant configuration identity of c.
func (c Combo) ConfigKey() ConfigKey { return EncodeConfigKey(c.ServerParams) }

// SortCombos returns a new slice of combos stably ordered by ConfigKey:
// combos sharing a key run consecutively, and within one key the input order
// is preserved. The input slice is not mutated. Ordering is deterministic for
// any input, so a rerun of the same campaign plans the same sequence.
func SortCombos(combos []Combo) []Combo {
	out := make([]Combo, len(combos))
	copy(out, combos)
	sort.SliceStable(out, func(i, j int) bool { return out[i].ConfigKey() < out[j].ConfigKey() })
	return out
}

// ExcludedCombo is the typed record of one combo kept OUT of the restart plan
// because its config key is already known to have failed. Exclusion is
// explicit, never silent: every input combo appears either in a group or in
// this record, so a reader can always account for the whole sweep.
type ExcludedCombo struct {
	Combo Combo     `json:"combo"`
	Key   ConfigKey `json:"key"`
}

// Group is one contiguous run of combos served by a single server
// configuration. Walking a plan's groups in order, the executing loop restarts
// the server exactly when the key changes — once per group.
type Group struct {
	Key    ConfigKey `json:"key"`
	Combos []Combo   `json:"combos"`
}

// RestartPlan is the restart-minimizing execution plan over a set of combos:
// contiguous groups per distinct healthy config key, one restart per group,
// and every failed-key combo carried in Excluded.
type RestartPlan struct {
	// Groups runs in execution order; adjacent groups always differ in key.
	Groups []Group `json:"groups"`
	// Restarts is the number of distinct healthy config keys, equal to
	// len(Groups): the first group's initial launch counts as one restart
	// event, so the operator's machine count is exactly this number.
	Restarts int `json:"restarts"`
	// Excluded carries every combo whose key was pre-failed, in plan order.
	Excluded []ExcludedCombo `json:"excluded,omitempty"`
}

// PlanRestarts builds the restart-minimizing plan over combos given the set of
// config keys already known to have failed. Every combo under a failed key is
// excluded and returned in plan.Excluded (never dropped silently); the rest
// are stably grouped by key so the executing loop restarts the server only
// when the key changes.
//
// Empty input is a valid empty plan — zero groups, zero restarts, nil error —
// not a failure: a campaign whose sweep expanded to nothing still reports.
func PlanRestarts(combos []Combo, failedKeys []ConfigKey) (RestartPlan, error) {
	failed := make(map[ConfigKey]bool, len(failedKeys))
	for _, k := range failedKeys {
		failed[k] = true
	}
	plan := RestartPlan{Groups: []Group{}}
	for _, c := range SortCombos(combos) {
		key := c.ConfigKey()
		if failed[key] {
			plan.Excluded = append(plan.Excluded, ExcludedCombo{Combo: c, Key: key})
			continue
		}
		if n := len(plan.Groups); n > 0 && plan.Groups[n-1].Key == key {
			plan.Groups[n-1].Combos = append(plan.Groups[n-1].Combos, c)
		} else {
			plan.Groups = append(plan.Groups, Group{Key: key, Combos: []Combo{c}})
		}
	}
	plan.Restarts = len(plan.Groups)
	return plan, nil
}
