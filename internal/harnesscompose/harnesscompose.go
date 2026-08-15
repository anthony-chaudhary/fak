package harnesscompose

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const Schema = "fak.harness-assets/v1alpha1"

type Manifest struct {
	Schema string  `json:"schema"`
	Layers []Layer `json:"layers"`
}

type Layer struct {
	ID     string  `json:"id"`
	Scope  string  `json:"scope"`
	Assets []Asset `json:"assets"`
}

type Asset struct {
	Kind      string   `json:"kind"`
	ID        string   `json:"id"`
	Operation string   `json:"operation,omitempty"`
	Value     string   `json:"value,omitempty"`
	Ref       string   `json:"ref,omitempty"`
	Boundary  string   `json:"boundary,omitempty"`
	Grants    []string `json:"grants,omitempty"`
	Denies    []string `json:"denies,omitempty"`
	Lock      bool     `json:"lock,omitempty"`
	Mandatory bool     `json:"mandatory,omitempty"`
}

type EffectiveAsset struct {
	Kind      string   `json:"kind"`
	ID        string   `json:"id"`
	Value     string   `json:"value,omitempty"`
	Ref       string   `json:"ref,omitempty"`
	Boundary  string   `json:"boundary,omitempty"`
	Grants    []string `json:"grants,omitempty"`
	Denies    []string `json:"denies,omitempty"`
	Source    string   `json:"source"`
	Locked    bool     `json:"locked,omitempty"`
	Mandatory bool     `json:"mandatory,omitempty"`
}

type Trace struct {
	Layer  string `json:"layer"`
	Kind   string `json:"kind"`
	ID     string `json:"id"`
	Action string `json:"action"`
	Reason string `json:"reason"`
}

type Result struct {
	Schema string           `json:"schema"`
	Layers []string         `json:"layers"`
	Assets []EffectiveAsset `json:"assets"`
	Trace  []Trace          `json:"trace"`
}

var kinds = map[string]bool{"instruction": true, "tool": true, "memory": true, "policy": true, "route": true, "secret": true, "workflow": true, "ui": true}
var scopes = map[string]bool{"company": true, "team": true, "person": true, "repo": true, "project": true, "domain": true, "task": true}

func Parse(raw []byte) (Manifest, error) {
	var manifest Manifest
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return Manifest{}, fmt.Errorf("parse assets: %w", err)
	}
	if manifest.Schema != Schema {
		return Manifest{}, fmt.Errorf("schema must be %q", Schema)
	}
	seen := map[string]bool{}
	for i, layer := range manifest.Layers {
		if layer.ID == "" || !scopes[layer.Scope] {
			return Manifest{}, fmt.Errorf("layers[%d] has invalid id or scope", i)
		}
		if seen[layer.ID] {
			return Manifest{}, fmt.Errorf("duplicate layer %q", layer.ID)
		}
		seen[layer.ID] = true
		for j := range layer.Assets {
			if err := validateAsset(layer, layer.Assets[j]); err != nil {
				return Manifest{}, fmt.Errorf("layer %q asset[%d]: %w", layer.ID, j, err)
			}
		}
	}
	return manifest, nil
}

func Compose(manifest Manifest, selected []string) (Result, error) {
	byLayer := make(map[string]Layer, len(manifest.Layers))
	for _, layer := range manifest.Layers {
		byLayer[layer.ID] = layer
	}
	state := map[string]EffectiveAsset{}
	trace := make([]Trace, 0)
	seenLayers := map[string]bool{}
	for _, layerID := range selected {
		if seenLayers[layerID] {
			return Result{}, fmt.Errorf("selected layer %q repeated", layerID)
		}
		seenLayers[layerID] = true
		layer, ok := byLayer[layerID]
		if !ok {
			return Result{}, fmt.Errorf("selected layer %q has no asset declaration", layerID)
		}
		for _, asset := range layer.Assets {
			key := asset.Kind + "/" + asset.ID
			prior, exists := state[key]
			op := asset.Operation
			if op == "" {
				op = "add"
			}
			if op == "remove" {
				if !exists {
					return Result{}, conflict(layer, asset, "cannot remove absent asset")
				}
				if prior.Locked || prior.Mandatory {
					return Result{}, conflict(layer, asset, "cannot remove locked or mandatory asset from "+prior.Source)
				}
				delete(state, key)
				trace = append(trace, Trace{Layer: layer.ID, Kind: asset.Kind, ID: asset.ID, Action: "remove", Reason: "explicit removal"})
				continue
			}
			if err := mergeAsset(state, key, layer, asset, prior, exists, op, &trace); err != nil {
				return Result{}, err
			}
		}
	}
	assets := make([]EffectiveAsset, 0, len(state))
	for _, asset := range state {
		asset.Grants = sortedUnique(asset.Grants)
		asset.Denies = sortedUnique(asset.Denies)
		assets = append(assets, asset)
	}
	sort.Slice(assets, func(i, j int) bool {
		if assets[i].Kind != assets[j].Kind {
			return assets[i].Kind < assets[j].Kind
		}
		return assets[i].ID < assets[j].ID
	})
	return Result{Schema: Schema, Layers: append([]string(nil), selected...), Assets: assets, Trace: trace}, nil
}

func mergeAsset(state map[string]EffectiveAsset, key string, layer Layer, asset Asset, prior EffectiveAsset, exists bool, op string, trace *[]Trace) error {
	if prior.Locked && exists {
		if equivalent(prior, asset) {
			*trace = append(*trace, Trace{Layer: layer.ID, Kind: asset.Kind, ID: asset.ID, Action: "retain", Reason: "earlier layer locked identical asset"})
			return nil
		}
		return conflict(layer, asset, "cannot change locked asset from "+prior.Source)
	}
	if asset.Kind == "policy" {
		return mergePolicy(state, key, layer, asset, prior, exists, op, trace)
	}
	if asset.Kind == "secret" && exists {
		return conflict(layer, asset, "secret references cannot be replaced by a narrower layer")
	}
	if asset.Kind == "memory" && exists && prior.Boundary != asset.Boundary {
		return conflict(layer, asset, "memory namespace crosses boundary "+prior.Boundary+" -> "+asset.Boundary)
	}
	if exists && op != "replace" {
		if equivalent(prior, asset) {
			*trace = append(*trace, Trace{Layer: layer.ID, Kind: asset.Kind, ID: asset.ID, Action: "retain", Reason: "identical declaration"})
			return nil
		}
		return conflict(layer, asset, "ambiguous duplicate requires explicit replace")
	}
	if !exists && op == "replace" {
		return conflict(layer, asset, "cannot replace absent asset")
	}
	effective := fromAsset(layer.ID, asset)
	state[key] = effective
	action := "add"
	if exists {
		action = "replace"
	}
	*trace = append(*trace, Trace{Layer: layer.ID, Kind: asset.Kind, ID: asset.ID, Action: action, Reason: kindReason(asset.Kind, action)})
	return nil
}

func mergePolicy(state map[string]EffectiveAsset, key string, layer Layer, asset Asset, prior EffectiveAsset, exists bool, op string, trace *[]Trace) error {
	if !exists {
		if op == "replace" {
			return conflict(layer, asset, "cannot replace absent policy")
		}
		state[key] = fromAsset(layer.ID, asset)
		*trace = append(*trace, Trace{Layer: layer.ID, Kind: asset.Kind, ID: asset.ID, Action: "add", Reason: "establish policy grant/deny ceiling"})
		return nil
	}
	if len(asset.Grants) > 0 {
		for _, grant := range asset.Grants {
			if contains(prior.Denies, grant) || !contains(prior.Grants, grant) {
				return conflict(layer, asset, "policy privilege widening for "+grant)
			}
		}
	}
	merged := prior
	merged.Denies = sortedUnique(append(merged.Denies, asset.Denies...))
	for _, deny := range merged.Denies {
		merged.Grants = remove(merged.Grants, deny)
	}
	merged.Source = layer.ID
	merged.Locked = prior.Locked || asset.Lock
	merged.Mandatory = prior.Mandatory || asset.Mandatory
	state[key] = merged
	*trace = append(*trace, Trace{Layer: layer.ID, Kind: asset.Kind, ID: asset.ID, Action: "narrow", Reason: "later policy may deny but not widen"})
	return nil
}

func validateAsset(layer Layer, asset Asset) error {
	if !kinds[asset.Kind] || asset.ID == "" {
		return fmt.Errorf("unknown kind or empty id")
	}
	op := asset.Operation
	if op == "" {
		op = "add"
	}
	if op != "add" && op != "replace" && op != "remove" {
		return fmt.Errorf("unknown operation %q", op)
	}
	if op == "remove" {
		return nil
	}
	switch asset.Kind {
	case "secret":
		if asset.Ref == "" || asset.Value != "" {
			return fmt.Errorf("secret must carry an opaque ref and no inline value")
		}
	case "memory":
		if asset.Boundary == "" {
			return fmt.Errorf("memory boundary is required")
		}
		if (layer.Scope == "project" || layer.Scope == "domain" || layer.Scope == "task") && asset.Boundary != layer.ID {
			return fmt.Errorf("memory boundary %q must equal owning %s layer %q", asset.Boundary, layer.Scope, layer.ID)
		}
	case "policy":
		if len(asset.Grants) == 0 && len(asset.Denies) == 0 {
			return fmt.Errorf("policy needs grants or denies")
		}
	default:
		if asset.Value == "" {
			return fmt.Errorf("%s value is required", asset.Kind)
		}
	}
	return nil
}

func fromAsset(source string, asset Asset) EffectiveAsset {
	return EffectiveAsset{Kind: asset.Kind, ID: asset.ID, Value: asset.Value, Ref: asset.Ref, Boundary: asset.Boundary, Grants: sortedUnique(asset.Grants), Denies: sortedUnique(asset.Denies), Source: source, Locked: asset.Lock, Mandatory: asset.Mandatory}
}
func conflict(layer Layer, asset Asset, reason string) error {
	return fmt.Errorf("layer %q %s/%s: %s", layer.ID, asset.Kind, asset.ID, reason)
}
func kindReason(kind, action string) string { return kind + " uses explicit " + action + " semantics" }
func equivalent(prior EffectiveAsset, asset Asset) bool {
	return prior.Value == asset.Value && prior.Ref == asset.Ref && prior.Boundary == asset.Boundary && strings.Join(sortedUnique(prior.Grants), "\x00") == strings.Join(sortedUnique(asset.Grants), "\x00") && strings.Join(sortedUnique(prior.Denies), "\x00") == strings.Join(sortedUnique(asset.Denies), "\x00")
}
func sortedUnique(values []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, v := range values {
		v = strings.TrimSpace(v)
		if v != "" && !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}
func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
func remove(values []string, deny string) []string {
	out := values[:0]
	for _, v := range values {
		if v != deny {
			out = append(out, v)
		}
	}
	return out
}
