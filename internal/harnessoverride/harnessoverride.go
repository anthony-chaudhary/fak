// Package harnessoverride produces structured, reviewable proposals to override
// changeable capabilities within a verified harness lock.
package harnessoverride

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// Schema defines the canonical schema identifier for harness override proposals.
const Schema = "fak-harness-override/1"

// Request specifies the target capability and desired changes for an override proposal.
type Request struct {
	Capability string
	Value      string
	Denies     []string
	LayerID    string
}

// Proposal holds the generated override layer, manifest, and review steps.
type Proposal struct {
	Schema     string                  `json:"schema"`
	CurrentID  string                  `json:"current_lock_id"`
	Capability string                  `json:"capability"`
	Source     string                  `json:"current_source"`
	Control    string                  `json:"control"`
	Layer      harnesscompose.Layer    `json:"layer"`
	Selection  []string                `json:"selection"`
	Manifest   harnesscompose.Manifest `json:"manifest"`
	Next       []string                `json:"next"`
}

// Propose validates an override request against an active lock and constructs a reviewable Proposal.
func Propose(lock harnessresolve.Lock, request Request) (Proposal, error) {
	kind, id, ok := strings.Cut(strings.TrimSpace(request.Capability), ":")
	if !ok || kind == "" || id == "" {
		return Proposal{}, fmt.Errorf("capability must be kind:id from harness inspect")
	}
	var current *harnesscompose.EffectiveAsset
	for i := range lock.Assets {
		if lock.Assets[i].Kind == kind && lock.Assets[i].ID == id {
			current = &lock.Assets[i]
			break
		}
	}
	if current == nil {
		return Proposal{}, fmt.Errorf("capability %q is not active in the verified lock", request.Capability)
	}
	if current.Mandatory {
		return Proposal{}, fmt.Errorf("capability %q is mandatory and cannot be overridden", request.Capability)
	}
	if current.Locked {
		return Proposal{}, fmt.Errorf("capability %q is locked by %s and cannot be overridden", request.Capability, current.Source)
	}
	layerID := strings.TrimSpace(request.LayerID)
	if layerID == "" {
		layerID = "operator-override"
	}
	asset := harnesscompose.Asset{Kind: kind, ID: id, Operation: "replace", Value: request.Value}
	if kind == "policy" {
		asset.Operation = "add"
		asset.Denies = sortedUnique(request.Denies)
		if len(asset.Denies) == 0 {
			return Proposal{}, fmt.Errorf("policy override requires at least one --deny; policy grants cannot be widened")
		}
	} else if strings.TrimSpace(request.Value) == "" {
		return Proposal{}, fmt.Errorf("%s override requires --value", kind)
	}
	layer := harnesscompose.Layer{ID: layerID, Scope: "person", Assets: []harnesscompose.Asset{asset}}
	manifest := harnesscompose.Manifest{Schema: harnesscompose.Schema, Layers: []harnesscompose.Layer{layer}}
	return Proposal{
		Schema: Schema, CurrentID: lock.ID, Capability: request.Capability, Source: current.Source,
		Control: "changeable by re-resolve", Layer: layer, Manifest: manifest, Selection: []string{layerID},
		Next: []string{
			"append this layer to the product manifest after the current source layer",
			"resolve a candidate lock with the current selection plus " + layerID,
			"review it: fak harness preview --current <current.lock.json> --candidate <candidate.lock.json>",
			"verify it: fak harness inspect --lock <candidate.lock.json>",
		},
	}, nil
}

// Render formats a Proposal into a human-readable summary with next-step instructions.
func Render(proposal Proposal) string {
	asset := proposal.Layer.Assets[0]
	var b strings.Builder
	fmt.Fprintf(&b, "HARNESS OVERRIDE | PROPOSAL\ncurrent: %s\ncapability: %s | from %s | %s\nlayer: %s | scope person\nchange: %s", proposal.CurrentID, proposal.Capability, proposal.Source, proposal.Control, proposal.Layer.ID, asset.Operation)
	if asset.Value != "" {
		fmt.Fprintf(&b, " | value %s", asset.Value)
	}
	if len(asset.Denies) > 0 {
		fmt.Fprintf(&b, " | deny %s", strings.Join(asset.Denies, ", "))
	}
	b.WriteString("\nnext:\n")
	for _, step := range proposal.Next {
		fmt.Fprintf(&b, "- %s\n", step)
	}
	return b.String()
}

func sortedUnique(values []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	sort.Strings(out)
	return out
}
