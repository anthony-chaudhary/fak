// sweepseed.go ports aiperf's restart-stable sweep-seed derivation to Go.
//
// NOTICE: ported from ai-dynamo/aiperf, src/aiperf/sweep_controller/plan_builder.py
// (_stable_sweep_seed), pinned at commit ab610add4e8876fca223197596c728e17365f381
// (ai-dynamo/aiperf@ab610ad), Apache-2.0 licensed. The source derives a
// restart-stable stochastic-sweep seed as the first four bytes of SHA-256 of
// the sweep's apiserver identity (metadata.uid), read big-endian. This port
// applies the same derivation to the canonical identity of a sweep plan
// (axes + ranges), so the same plan always yields the same seed.
package perfledger

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sort"
	"strings"
)

// Axis is one sweep dimension: a named parameter and the ordered range of
// values the sweep varies it across.
type Axis struct {
	Name   string
	Values []string
}

// SweepPlan is the canonical description of a stochastic sweep: the axes and
// their ranges. Axis order is not part of the plan's identity: two plans with
// the same axes and ranges derive the same seed.
type SweepPlan struct {
	Axes []Axis
}

// CanonicalIdentity renders the plan as the canonical identity string the
// seed is derived from: axes sorted by name, values in declared order, each
// axis as "name=v1,v2,..." joined with ";". It is the plan analogue of the
// source's metadata.uid identity.
func (p SweepPlan) CanonicalIdentity() (string, error) {
	if len(p.Axes) == 0 {
		return "", fmt.Errorf("unseeded stochastic sweep requires at least one axis so variations remain stable across restarts")
	}
	axes := make([]Axis, len(p.Axes))
	copy(axes, p.Axes)
	sort.SliceStable(axes, func(i, j int) bool { return axes[i].Name < axes[j].Name })
	seen := make(map[string]struct{}, len(axes))
	parts := make([]string, 0, len(axes))
	for _, a := range axes {
		if a.Name == "" {
			return "", fmt.Errorf("sweep axis with empty name")
		}
		if _, dup := seen[a.Name]; dup {
			return "", fmt.Errorf("duplicate sweep axis %q", a.Name)
		}
		seen[a.Name] = struct{}{}
		parts = append(parts, a.Name+"="+strings.Join(a.Values, ","))
	}
	return strings.Join(parts, ";"), nil
}

// StableSweepSeed derives the restart-stable stochastic-sweep seed for plan,
// mirroring aiperf's _stable_sweep_seed: the first four bytes of SHA-256 of
// the plan's canonical identity, read big-endian. The same plan always yields
// the same seed; different plans yield different seeds with overwhelming
// probability.
func StableSweepSeed(plan SweepPlan) (uint32, error) {
	identity, err := plan.CanonicalIdentity()
	if err != nil {
		return 0, err
	}
	sum := sha256.Sum256([]byte(identity))
	return binary.BigEndian.Uint32(sum[:4]), nil
}
