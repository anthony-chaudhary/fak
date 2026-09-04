// Package harnessmix combines independently resolved, mix-ready harness locks.
package harnessmix

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// Schema identifies the receipt format for mixed harness locks.
const Schema = "fak.harness-mix-receipt/v1alpha1"

// Limits defines resource ceilings enforced during harness mixing.
type Limits struct {
	ContextTokens int `json:"context_tokens,omitempty"`
	MemoryMiB     int `json:"memory_mib,omitempty"`
	Workers       int `json:"workers,omitempty"`
}

// Receipt records provenance, deduplicated components, and rebuild instructions for a mixed harness.
type Receipt struct {
	Schema       string   `json:"schema"`
	Imports      []string `json:"imports"`
	ResultID     string   `json:"result_id"`
	Deduplicated []string `json:"deduplicated_components,omitempty"`
	Rebuild      string   `json:"rebuild"`
}

// Result holds the combined harness lock and its verification receipt.
type Result struct {
	Lock    harnessresolve.Lock `json:"lock"`
	Receipt Receipt             `json:"receipt"`
}

// Mix combines two or more verified harness locks, deduplicating shared components,
// merging compatible policy assets, validating dependency graphs, and enforcing resource limits.
func Mix(imports []harnessresolve.Lock, limits Limits) (Result, error) {
	if len(imports) < 2 {
		return Result{}, fmt.Errorf("at least two verified harness imports are required")
	}
	locks := append([]harnessresolve.Lock(nil), imports...)
	for _, lock := range locks {
		if err := harnessresolve.Mixable(lock); err != nil {
			return Result{}, fmt.Errorf("import %s: %w", shortID(lock.ID), err)
		}
	}
	sort.Slice(locks, func(i, j int) bool { return locks[i].ID < locks[j].ID })
	env := locks[0].Environment
	for _, lock := range locks[1:] {
		if lock.Environment != env {
			return Result{}, fmt.Errorf("environment mismatch: %s targets %s/%s/%s, want %s/%s/%s", shortID(lock.ID), lock.Environment.OS, lock.Environment.Arch, lock.Environment.Contract, env.OS, env.Arch, env.Contract)
		}
	}
	components := map[string]harnessresolve.LockedComponent{}
	ids := map[string]string{}
	dedup := []string{}
	used := harnessresolve.Budget{}
	for _, lock := range locks {
		for _, c := range lock.Components {
			key := c.ID + "@" + c.Version
			if prior, ok := ids[c.ID]; ok && prior != c.Digest {
				return Result{}, fmt.Errorf("component conflict: %s has digests %s and %s", c.ID, prior, c.Digest)
			}
			ids[c.ID] = c.Digest
			if _, ok := components[key+"#"+c.Digest]; ok {
				dedup = append(dedup, key+"#"+c.Digest)
				continue
			}
			if !compatible(c.Compatibility, env) {
				return Result{}, fmt.Errorf("component %s incompatible with %s/%s/%s", c.ID, env.OS, env.Arch, env.Contract)
			}
			components[key+"#"+c.Digest] = c
			used = add(used, c.Cost)
		}
	}
	all := make([]harnessresolve.LockedComponent, 0, len(components))
	for _, c := range components {
		all = append(all, c)
	}
	sort.Slice(all, func(i, j int) bool { return all[i].ID < all[j].ID })
	if err := validateGraph(all); err != nil {
		return Result{}, err
	}
	if limits.ContextTokens > 0 && used.ContextTokens > limits.ContextTokens {
		return Result{}, fmt.Errorf("context budget exceeded: %d > %d", used.ContextTokens, limits.ContextTokens)
	}
	if limits.MemoryMiB > 0 && used.MemoryMiB > limits.MemoryMiB {
		return Result{}, fmt.Errorf("memory budget exceeded: %d > %d", used.MemoryMiB, limits.MemoryMiB)
	}
	if limits.Workers > 0 && used.Workers > limits.Workers {
		return Result{}, fmt.Errorf("worker budget exceeded: %d > %d", used.Workers, limits.Workers)
	}
	assets, trace, err := mixAssets(locks)
	if err != nil {
		return Result{}, err
	}
	out := harnessresolve.Lock{Schema: harnessresolve.LockSchema, Environment: env, Budget: used, Components: all, Assets: assets, AssetTrace: trace}
	if err := harnessresolve.ReidentifyLock(&out); err != nil {
		return Result{}, err
	}
	importIDs := make([]string, len(locks))
	for i := range locks {
		importIDs[i] = locks[i].ID
	}
	dedup = unique(dedup)
	return Result{Lock: out, Receipt: Receipt{Schema: Schema, Imports: importIDs, ResultID: out.ID, Deduplicated: dedup, Rebuild: "fak harness mix --import <lock> --import <lock> --output <mixed-lock>"}}, nil
}

func mixAssets(locks []harnessresolve.Lock) ([]harnesscompose.EffectiveAsset, []harnesscompose.Trace, error) {
	byKey := map[string]harnesscompose.EffectiveAsset{}
	trace := []harnesscompose.Trace{}
	for _, lock := range locks {
		for _, a := range lock.Assets {
			key := a.Kind + ":" + a.ID
			if prior, ok := byKey[key]; ok {
				if sameAsset(prior, a) {
					continue
				}
				if a.Kind == "policy" {
					if !sameStrings(prior.Grants, a.Grants) || prior.Locked != a.Locked || prior.Mandatory != a.Mandatory {
						return nil, nil, fmt.Errorf("policy floor collision on %s between %s and %s", key, prior.Source, a.Source)
					}
					prior.Denies = unique(append(prior.Denies, a.Denies...))
					prior.Source = "mix:" + prior.Source + "+" + a.Source
					byKey[key] = prior
					continue
				}
				if a.Kind == "secret" {
					return nil, nil, fmt.Errorf("duplicate secret boundary on %s between %s and %s", key, prior.Source, a.Source)
				}
				return nil, nil, fmt.Errorf("capability conflict on %s between %s and %s; choose explicitly before mixing", key, prior.Source, a.Source)
			}
			byKey[key] = a
			trace = append(trace, harnesscompose.Trace{Layer: "import:" + shortID(lock.ID), Kind: a.Kind, ID: a.ID, Action: "import", Reason: "inherited from verified harness " + lock.ID})
		}
	}
	out := make([]harnesscompose.EffectiveAsset, 0, len(byKey))
	for _, a := range byKey {
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].ID < out[j].ID
	})
	return out, trace, nil
}

func validateGraph(cs []harnessresolve.LockedComponent) error {
	provides := map[string]bool{}
	for _, c := range cs {
		for _, p := range c.Provides {
			provides[p] = true
		}
	}
	for _, c := range cs {
		for _, conflict := range c.Conflicts {
			if provides[conflict] {
				return fmt.Errorf("component conflict: %s conflicts with provided capability %s", c.ID, conflict)
			}
		}
		for _, req := range c.Requires {
			if !req.Optional && !provides[req.Capability] {
				return fmt.Errorf("unsatisfied requirement: %s requires %s %s", c.ID, req.Capability, req.Range)
			}
		}
	}
	return nil
}
func compatible(c harnessresolve.Compatibility, e harnessresolve.Environment) bool {
	return (c.Contract == "" || c.Contract == e.Contract) && (len(c.OS) == 0 || contains(c.OS, e.OS)) && (len(c.Arch) == 0 || contains(c.Arch, e.Arch))
}
func add(a, b harnessresolve.Budget) harnessresolve.Budget {
	return harnessresolve.Budget{ContextTokens: a.ContextTokens + b.ContextTokens, MemoryMiB: a.MemoryMiB + b.MemoryMiB, Workers: a.Workers + b.Workers}
}
func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
func unique(xs []string) []string {
	m := map[string]bool{}
	out := []string{}
	for _, x := range xs {
		if !m[x] {
			m[x] = true
			out = append(out, x)
		}
	}
	sort.Strings(out)
	return out
}
func sameStrings(a, b []string) bool {
	aa := unique(a)
	bb := unique(b)
	return strings.Join(aa, "\x00") == strings.Join(bb, "\x00")
}
func sameAsset(a, b harnesscompose.EffectiveAsset) bool {
	a.Source = ""
	b.Source = ""
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}
func shortID(id string) string {
	if len(id) > 19 {
		return id[:19]
	}
	return id
}
