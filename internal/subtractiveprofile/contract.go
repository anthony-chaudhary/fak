// Package subtractiveprofile resolves capability profiles with sticky removals.
package subtractiveprofile

import (
	"fmt"
	"sort"
	"strings"
)

type Removal string

const (
	RemovalRuntime Removal = "runtime"
	RemovalStatic  Removal = "static"
)

type Capability struct {
	ID                              string
	Aliases                         []string
	Requires                        []string
	Help, Schema, Runtime, Artifact bool
}

type Profile struct {
	Include   []Capability
	Configure map[string]map[string]string
	Replace   map[string]Capability
	Remove    map[string]Removal
}

type Provenance struct{ Capability, Operation, Source string }
type Delta struct{ BinaryBytes, StartupMillis, IdleMemoryBytes, ContextTokens, SchemaBytes int64 }
type Report struct{ Minimal, Full Delta }
type Effective struct {
	Capabilities map[string]Capability
	Config       map[string]map[string]string
	Removed      map[string]Removal
	Provenance   []Provenance
	Report       Report
}

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
