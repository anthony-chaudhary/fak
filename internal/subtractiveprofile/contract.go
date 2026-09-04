// Package subtractiveprofile resolves capability profiles with sticky removals.
package subtractiveprofile

import (
	"fmt"
	"sort"
	"strings"
)

// Removal designates the enforcement tier applied when dropping a capability.
type Removal string

const (
	// RemovalRuntime purges execution access while retaining static metadata representations.
	RemovalRuntime Removal = "runtime"
	// RemovalStatic eliminates a capability completely across compile and runtime tiers.
	RemovalStatic Removal = "static"
)

// Capability defines a discrete unit of agent functionality with dependency constraints and visibility masks.
type Capability struct {
	ID                              string
	Aliases                         []string
	Requires                        []string
	Help, Schema, Runtime, Artifact bool
}

// Profile declares layered capability inclusions, configurations, replacements, and sticky removals.
type Profile struct {
	Include   []Capability
	Configure map[string]map[string]string
	Replace   map[string]Capability
	Remove    map[string]Removal
}

// Provenance records an audit trail entry tracking which profile operation mutated a capability.
type Provenance struct{ Capability, Operation, Source string }

// Delta quantifies resource impacts across binary size, startup latency, memory footprint, and tokens.
type Delta struct{ BinaryBytes, StartupMillis, IdleMemoryBytes, ContextTokens, SchemaBytes int64 }

// Report aggregates resource delta metrics between minimal and full capability deployments.
type Report struct{ Minimal, Full Delta }

// Effective represents the final immutable capability resolution state across all applied profiles.
type Effective struct {
	Capabilities map[string]Capability
	Config       map[string]map[string]string
	Removed      map[string]Removal
	Provenance   []Provenance
	Report       Report
}

// Invariant: subtractive profile resolution is fail-closed and sticky.
// Guard: once a capability is marked removed, subsequent profile layers,
// alias remappings, or replacement directives cannot resurrect it on any surface.
// Guard: all dependency prerequisites must resolve against active capabilities;
// references to removed or absent capabilities fail closed with an explicit error.
// Resolve applies profiles in order. Removal is sticky: aliases, replacement,
// inclusion, and dependencies cannot resurrect a removed capability.
func Resolve(profiles []Profile, report Report) (Effective, error) {
	out := Effective{Capabilities: map[string]Capability{}, Config: map[string]map[string]string{}, Removed: map[string]Removal{}, Report: report}
	aliases := map[string]string{}
	for pi, profile := range profiles {
		source := fmt.Sprintf("profile[%d]", pi)
		for id, mode := range profile.Remove {
			canonical := canonicalID(id, aliases)
			out.Removed[canonical] = mode
			delete(out.Capabilities, canonical)
			delete(out.Config, canonical)
			out.Provenance = append(out.Provenance, Provenance{canonical, "remove", source})
		}
		for id, replacement := range profile.Replace {
			canonical := canonicalID(id, aliases)
			replacement.ID = canonical
			if _, removed := out.Removed[canonical]; removed {
				continue
			}
			out.Capabilities[canonical] = replacement
			indexAliases(aliases, replacement)
			out.Provenance = append(out.Provenance, Provenance{canonical, "replace", source})
		}
		for _, capability := range profile.Include {
			canonical := canonicalID(capability.ID, aliases)
			capability.ID = canonical
			if _, removed := out.Removed[canonical]; removed {
				continue
			}
			out.Capabilities[canonical] = capability
			indexAliases(aliases, capability)
			out.Provenance = append(out.Provenance, Provenance{canonical, "include", source})
		}
		for id, values := range profile.Configure {
			canonical := canonicalID(id, aliases)
			if _, removed := out.Removed[canonical]; removed {
				continue
			}
			if _, ok := out.Capabilities[canonical]; !ok {
				return Effective{}, fmt.Errorf("configure %s: capability missing", canonical)
			}
			if out.Config[canonical] == nil {
				out.Config[canonical] = map[string]string{}
			}
			for key, value := range values {
				out.Config[canonical][key] = value
			}
			out.Provenance = append(out.Provenance, Provenance{canonical, "configure", source})
		}
	}
	for id, capability := range out.Capabilities {
		for _, dependency := range capability.Requires {
			canonical := canonicalID(dependency, aliases)
			if mode, removed := out.Removed[canonical]; removed {
				return Effective{}, fmt.Errorf("%s requires %s, removed (%s)", id, canonical, mode)
			}
			if _, ok := out.Capabilities[canonical]; !ok {
				return Effective{}, fmt.Errorf("%s requires %s, which is missing", id, canonical)
			}
		}
	}
	return out, nil
}

// Surface returns the sorted capability identifiers visible on a specific interface surface.
func (e Effective) Surface(surface string) []string {
	ids := []string{}
	for id, c := range e.Capabilities {
		visible := map[string]bool{"help": c.Help, "schema": c.Schema, "runtime": c.Runtime, "artifact": c.Artifact}[surface]
		if visible {
			if mode, removed := e.Removed[id]; !removed || !(surface == "artifact" && mode == RemovalStatic) {
				ids = append(ids, id)
			}
		}
	}
	sort.Strings(ids)
	return ids
}

// ProbeAbsent verifies that a designated capability identifier has been thoroughly purged and remains unreachable.
func (e Effective) ProbeAbsent(id string) error {
	canonical := strings.ToLower(strings.TrimSpace(id))
	if _, ok := e.Capabilities[canonical]; ok {
		return fmt.Errorf("capability %s remains reachable", canonical)
	}
	return nil
}
func canonicalID(id string, aliases map[string]string) string {
	id = strings.ToLower(strings.TrimSpace(id))
	if canonical := aliases[id]; canonical != "" {
		return canonical
	}
	return id
}
func indexAliases(aliases map[string]string, c Capability) {
	aliases[c.ID] = c.ID
	for _, alias := range c.Aliases {
		aliases[strings.ToLower(strings.TrimSpace(alias))] = c.ID
	}
}
