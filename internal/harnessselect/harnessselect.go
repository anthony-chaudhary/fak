package harnessselect

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// Schema identifies the canonical manifest format for harness selection.
const Schema = "fak.harness-selection/v1alpha1"

// Manifest defines the layered hierarchy used to resolve active harness capabilities.
type Manifest struct {
	Schema string  `json:"schema"`
	Layers []Layer `json:"layers"`
}

// Layer specifies capability additions, removals, and immutable locks scoped to a level in the hierarchy.
type Layer struct {
	ID           string   `json:"id"`
	Scope        string   `json:"scope"`
	Priority     int      `json:"priority,omitempty"`
	When         Match    `json:"when,omitempty"`
	Capabilities []string `json:"capabilities,omitempty"`
	Remove       []string `json:"remove,omitempty"`
	Lock         []string `json:"lock,omitempty"`
}

// Match declares path and tag conditions that determine whether a layer applies to a given context.
type Match struct {
	PathPrefixes []string `json:"path_prefixes,omitempty"`
	Tags         []string `json:"tags,omitempty"`
}

// Context represents runtime invocation attributes evaluated against layer match rules.
type Context struct {
	Path string   `json:"path,omitempty"`
	Tags []string `json:"tags,omitempty"`
}

// Capability represents an effective capability granted and audited by the selection process.
type Capability struct {
	Name   string `json:"name"`
	Source string `json:"source"`
	Locked bool   `json:"locked,omitempty"`
}

// Trace captures a single decision step made during layer filtering and capability resolution.
type Trace struct {
	Layer  string `json:"layer"`
	Action string `json:"action"`
	Name   string `json:"name,omitempty"`
	Reason string `json:"reason"`
}

// Result contains the final active capabilities, applied layers, and decision trace for a context.
type Result struct {
	Schema       string       `json:"schema"`
	Context      Context      `json:"context"`
	Layers       []string     `json:"layers"`
	Capabilities []Capability `json:"capabilities"`
	Trace        []Trace      `json:"trace"`
}

var scopeRank = map[string]int{
	"company": 10,
	"team":    20,
	"person":  30,
	"repo":    40,
	"project": 50,
	"domain":  60,
	"task":    70,
}

// Parse deserializes and validates a harness selection manifest from JSON.
// It enforces the expected schema version, unique layer IDs, valid scopes, and non-empty capability names.
func Parse(raw []byte) (Manifest, error) {
	var m Manifest
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return Manifest{}, fmt.Errorf("parse manifest: %w", err)
	}
	if m.Schema != Schema {
		return Manifest{}, fmt.Errorf("schema must be %q", Schema)
	}
	seen := make(map[string]bool, len(m.Layers))
	for i, layer := range m.Layers {
		if strings.TrimSpace(layer.ID) == "" {
			return Manifest{}, fmt.Errorf("layers[%d].id is required", i)
		}
		if seen[layer.ID] {
			return Manifest{}, fmt.Errorf("duplicate layer id %q", layer.ID)
		}
		seen[layer.ID] = true
		if _, ok := scopeRank[layer.Scope]; !ok {
			return Manifest{}, fmt.Errorf("layer %q has unknown scope %q", layer.ID, layer.Scope)
		}
		for _, name := range append(append([]string{}, layer.Capabilities...), append(layer.Remove, layer.Lock...)...) {
			if strings.TrimSpace(name) == "" {
				return Manifest{}, fmt.Errorf("layer %q contains an empty capability", layer.ID)
			}
		}
	}
	return m, nil
}

// Resolve evaluates manifest layers against ctx in deterministic scope and priority order.
// It computes effective capabilities, respects locked capabilities, and records an audit trace.
func Resolve(m Manifest, ctx Context) (Result, error) {
	ctx.Path = cleanPath(ctx.Path)
	ctx.Tags = cleanSet(ctx.Tags)
	layers := make([]Layer, 0, len(m.Layers))
	trace := make([]Trace, 0)
	for _, layer := range m.Layers {
		ok, reason := matches(layer.When, ctx)
		if !ok {
			trace = append(trace, Trace{Layer: layer.ID, Action: "skip", Reason: reason})
			continue
		}
		layers = append(layers, layer)
	}
	sort.Slice(layers, func(i, j int) bool {
		ri, rj := scopeRank[layers[i].Scope], scopeRank[layers[j].Scope]
		if ri != rj {
			return ri < rj
		}
		if layers[i].Priority != layers[j].Priority {
			return layers[i].Priority < layers[j].Priority
		}
		return layers[i].ID < layers[j].ID
	})

	type state struct {
		source string
		locked bool
	}
	effective := map[string]state{}
	selected := make([]string, 0, len(layers))
	for _, layer := range layers {
		selected = append(selected, layer.ID)
		trace = append(trace, Trace{Layer: layer.ID, Action: "select", Reason: selectionReason(layer)})
		for _, name := range cleanSet(layer.Remove) {
			if prior, ok := effective[name]; ok && prior.locked {
				return Result{}, fmt.Errorf("layer %q cannot remove locked capability %q from layer %q", layer.ID, name, prior.source)
			}
			if _, ok := effective[name]; ok {
				delete(effective, name)
				trace = append(trace, Trace{Layer: layer.ID, Action: "remove", Name: name, Reason: "more-specific layer removed capability"})
			}
		}
		locks := makeSet(layer.Lock)
		for _, name := range cleanSet(append(append([]string{}, layer.Capabilities...), layer.Lock...)) {
			prior, exists := effective[name]
			if exists && prior.locked {
				trace = append(trace, Trace{Layer: layer.ID, Action: "retain", Name: name, Reason: "earlier layer locked capability"})
				continue
			}
			effective[name] = state{source: layer.ID, locked: locks[name]}
			action := "add"
			if exists {
				action = "override"
			}
			trace = append(trace, Trace{Layer: layer.ID, Action: action, Name: name, Reason: "selected layer declares capability"})
		}
	}

	caps := make([]Capability, 0, len(effective))
	for name, state := range effective {
		caps = append(caps, Capability{Name: name, Source: state.source, Locked: state.locked})
	}
	sort.Slice(caps, func(i, j int) bool { return caps[i].Name < caps[j].Name })
	return Result{Schema: Schema, Context: ctx, Layers: selected, Capabilities: caps, Trace: trace}, nil
}

func matches(w Match, ctx Context) (bool, string) {
	if len(w.PathPrefixes) > 0 {
		matched := false
		for _, prefix := range w.PathPrefixes {
			prefix = strings.TrimSuffix(cleanPath(prefix), "/")
			if ctx.Path == prefix || strings.HasPrefix(ctx.Path, prefix+"/") {
				matched = true
				break
			}
		}
		if !matched {
			return false, "path does not match any declared prefix"
		}
	}
	if len(w.Tags) > 0 {
		tags := makeSet(ctx.Tags)
		for _, tag := range w.Tags {
			if tags[strings.ToLower(strings.TrimSpace(tag))] {
				return true, "context tag matched"
			}
		}
		return false, "context does not carry any declared tag"
	}
	if len(w.PathPrefixes) > 0 {
		return true, "context path matched"
	}
	return true, "unconditional layer"
}

func selectionReason(layer Layer) string {
	parts := []string{layer.Scope + " scope"}
	if len(layer.When.PathPrefixes) > 0 {
		parts = append(parts, "path match")
	}
	if len(layer.When.Tags) > 0 {
		parts = append(parts, "tag match")
	}
	return strings.Join(parts, ", ")
}

func cleanPath(s string) string {
	// Manifests and launch contexts can cross OS boundaries. Normalize both
	// separator spellings before filepath.Clean applies host-native semantics.
	s = strings.ReplaceAll(strings.TrimSpace(s), `\`, "/")
	s = filepath.ToSlash(filepath.Clean(filepath.FromSlash(s)))
	if s == "." {
		return ""
	}
	return strings.TrimSuffix(s, "/")
}

func cleanSet(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, value := range in {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func makeSet(in []string) map[string]bool {
	out := make(map[string]bool, len(in))
	for _, value := range cleanSet(in) {
		out[value] = true
	}
	return out
}
