// Package categorybaseline defines the explicit, repository-owned boundary between a
// good-enough completed category layer and the next layer that should receive capacity.
//
// Invariant: category baseline evaluation is fail-closed and deterministic.
// Precondition: empty category or layer queries return a default Decision without hold.
// Guard: corrupt registries, missing paths, or mismatched schema versions fail closed to an empty normalized registry.
package categorybaseline

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Schema defines the canonical schema version identifier for category baselines.
// Invariant: Schema must be "fak-category-baselines/1".
// Guard: Any registry payload with a mismatched schema fails closed to an empty normalized registry.
const Schema = "fak-category-baselines/1"

const (
	// DefaultPath is the canonical repo-relative path for persisting category baseline configurations.
	// Invariant: DefaultPath points to "configs/category-baselines.json".
	// Guard: Load inspects DefaultPath first before falling back to LegacyPath.
	DefaultPath = "configs/category-baselines.json"

	// LegacyPath is the backward-compatible repo-relative path for category baselines.
	// Invariant: LegacyPath points to ".fak/category-baselines.json".
	// Guard: LegacyPath is only read when DefaultPath is absent.
	LegacyPath = ".fak/category-baselines.json"
)

// Registry holds the declared baseline categories and schema metadata.
// Invariant: Schema matches the supported Schema constant, and Categories contains unique, normalized, sorted categories.
// Guard: Deserialization or normalization drops malformed categories to fail closed.
type Registry struct {
	// Schema identifies the format version of the registry payload.
	Schema string `json:"schema"`
	// Categories lists the normalized baseline definitions ordered by name.
	Categories []Category `json:"categories"`
}

// Category represents a bounded development lane divided into sequential layers.
// Invariant: CompletedLayer and NextLayer must exist within Layers, and NextLayer must strictly follow CompletedLayer.
// Guard: Normalization rejects any category missing required fields, inverted layer ordering, or absent witnesses.
type Category struct {
	// Name is the normalized identifier of the category.
	Name string `json:"name"`
	// Layers lists the valid sequential layer names in progression order.
	Layers []string `json:"layers"`
	// CompletedLayer names the highest witnessed baseline layer already delivered.
	CompletedLayer string `json:"completed_layer"`
	// NextLayer names the immediate successor layer that should receive capacity.
	NextLayer string `json:"next_layer"`
	// Witness records the non-empty verifiable proof reference for CompletedLayer.
	Witness string `json:"witness"`
}

// Decision records the evaluation verdict for a requested category and layer.
// Invariant: Hold is true only when explicit work targets an already completed baseline layer.
// Guard: Any missing category, unknown layer, regression run, or unparsed state fails closed to Hold=false.
type Decision struct {
	// Hold indicates whether capacity should be paused on the requested layer.
	Hold bool `json:"hold"`
	// Category is the normalized target category name.
	Category string `json:"category,omitempty"`
	// Layer is the normalized layer under evaluation.
	Layer string `json:"layer,omitempty"`
	// CompletedLayer identifies the recorded completed baseline.
	CompletedLayer string `json:"completed_layer,omitempty"`
	// NextLayer identifies the next scheduled layer to receive investment.
	NextLayer string `json:"next_layer,omitempty"`
	// Witness is the recorded verification evidence for the baseline.
	Witness string `json:"witness,omitempty"`
}

// Load reads and normalizes the category baseline registry from root.
// Invariant: Always returns a valid Registry with Schema set, never returning a nil or uninitialized schema.
// Guard: Missing files, I/O errors, or JSON unmarshal failures fail closed to an empty Registry with default Schema.
func Load(root string) Registry {
	path := filepath.Join(root, filepath.FromSlash(DefaultPath))
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		b, err = os.ReadFile(filepath.Join(root, filepath.FromSlash(LegacyPath)))
	}
	if err != nil {
		return Registry{Schema: Schema}
	}
	var r Registry
	if json.Unmarshal(b, &r) != nil || r.Schema != Schema {
		return Registry{Schema: Schema}
	}
	return Normalize(r)
}

// Normalize canonicalizes names, validates layer sequencing, and sorts categories alphabetically.
// Invariant: Every retained Category has non-empty fields, valid layer indices with NextLayer after CompletedLayer, and sorted unique names.
// Guard: Malformed, duplicate, or out-of-order layer entries are discarded during normalization.
func Normalize(r Registry) Registry {
	r.Schema = Schema
	out := r.Categories[:0]
	seen := map[string]bool{}
	for _, c := range r.Categories {
		c.Name = norm(c.Name)
		c.CompletedLayer = norm(c.CompletedLayer)
		c.NextLayer = norm(c.NextLayer)
		c.Witness = strings.TrimSpace(c.Witness)
		layers := make([]string, 0, len(c.Layers))
		for _, layer := range c.Layers {
			layer = norm(layer)
			if layer != "" {
				layers = append(layers, layer)
			}
		}
		c.Layers = layers
		if c.Name == "" || c.CompletedLayer == "" || c.NextLayer == "" || c.Witness == "" || seen[c.Name] || index(c.Layers, c.CompletedLayer) < 0 || index(c.Layers, c.NextLayer) <= index(c.Layers, c.CompletedLayer) {
			continue
		}
		seen[c.Name] = true
		out = append(out, c)
	}
	r.Categories = out
	sort.Slice(r.Categories, func(i, j int) bool { return r.Categories[i].Name < r.Categories[j].Name })
	return r
}

// Evaluate inspects whether work on a category and layer should be held.
// Invariant: Work at or below a witnessed CompletedLayer yields Hold=true; work on NextLayer or subsequent layers yields Hold=false.
// Guard: Regression work, undeclared categories, or unknown layers fail closed to Hold=false without blocking progress.
func Evaluate(r Registry, category, layer string, regression bool) Decision {
	category, layer = norm(category), norm(layer)
	if regression || category == "" || layer == "" {
		return Decision{}
	}
	for _, c := range r.Categories {
		if c.Name != category {
			continue
		}
		li, ci := index(c.Layers, layer), index(c.Layers, c.CompletedLayer)
		if li < 0 || ci < 0 || li > ci {
			return Decision{}
		}
		return Decision{Hold: true, Category: category, Layer: layer, CompletedLayer: c.CompletedLayer, NextLayer: c.NextLayer, Witness: c.Witness}
	}
	return Decision{}
}

func norm(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, "`*_:#. ")
	s = strings.NewReplacer("_", "-", " ", "-").Replace(s)
	return s
}

func index(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return -1
}

// Upsert inserts or updates a category within the registry if the category is valid.
// Invariant: The resulting Registry is sorted and canonicalized; returns true if the category was valid and accepted.
// Guard: Invalid category definitions that fail normalization are rejected without mutating the registry.
func Upsert(r Registry, c Category) (Registry, bool) {
	r = Normalize(r)
	candidate := Normalize(Registry{Categories: []Category{c}})
	if len(candidate.Categories) != 1 {
		return r, false
	}
	c = candidate.Categories[0]
	found := false
	for i := range r.Categories {
		if r.Categories[i].Name == c.Name {
			r.Categories[i] = c
			found = true
		}
	}
	if !found {
		r.Categories = append(r.Categories, c)
	}
	return Normalize(r), true
}

// Remove deletes a category by name from the registry.
// Invariant: Returns a normalized Registry free of any category matching the normalized name.
// Guard: If the category does not exist, Remove safely returns the normalized registry without error.
func Remove(r Registry, name string) Registry {
	name = norm(name)
	out := r.Categories[:0]
	for _, c := range r.Categories {
		if c.Name != name {
			out = append(out, c)
		}
	}
	r.Categories = out
	return Normalize(r)
}

// Save writes the normalized registry atomically to DefaultPath under root.
// Invariant: File writes use an atomic temporary file with 0600 permissions renamed to the target destination.
// Guard: Any directory creation failure, serialization error, or write interruption halts and returns an error without corrupting target.
func Save(root string, r Registry) error {
	r = Normalize(r)
	path := filepath.Join(root, filepath.FromSlash(DefaultPath))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".category-baselines-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}
