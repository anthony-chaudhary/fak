// Package categorybaseline defines the explicit, repository-owned boundary between a
// good-enough completed category layer and the next layer that should receive capacity.
package categorybaseline

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const Schema = "fak-category-baselines/1"

const (
	DefaultPath = "configs/category-baselines.json"
	LegacyPath  = ".fak/category-baselines.json"
)

type Registry struct {
	Schema     string     `json:"schema"`
	Categories []Category `json:"categories"`
}

type Category struct {
	Name           string   `json:"name"`
	Layers         []string `json:"layers"`
	CompletedLayer string   `json:"completed_layer"`
	NextLayer      string   `json:"next_layer"`
	Witness        string   `json:"witness"`
}

type Decision struct {
	Hold           bool   `json:"hold"`
	Category       string `json:"category,omitempty"`
	Layer          string `json:"layer,omitempty"`
	CompletedLayer string `json:"completed_layer,omitempty"`
	NextLayer      string `json:"next_layer,omitempty"`
	Witness        string `json:"witness,omitempty"`
}

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

// Evaluate holds only explicitly declared work at or below a witnessed completed layer.
// Regression/fix work is never held. Missing registry/category/layer data fails open.
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
