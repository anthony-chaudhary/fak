// Package harnessderive creates a verified product lock from an immutable base
// lock plus a small, typed local delta.
package harnessderive

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// Schema identifies the schema format for harness derivation receipts.
const Schema = "fak.harness-derivation/v1alpha1"

// Verifier identifies the verifier engine and version producing derivations.
const Verifier = "fak/harnessderive-v1"

// Delta specifies a mutation to apply to an active base capability.
type Delta struct {
	Capability string   `json:"capability"`
	Operation  string   `json:"operation"`
	Value      string   `json:"value,omitempty"`
	Denies     []string `json:"denies,omitempty"`
}

// Request describes a derivation operation over a base lock.
type Request struct {
	Layer  string
	Deltas []Delta
}

// Receipt provides tamper-evident cryptographic provenance for a derived lock.
type Receipt struct {
	SchemaID    string                     `json:"id"`
	Schema      string                     `json:"schema"`
	BaseID      string                     `json:"base_id"`
	ResultID    string                     `json:"result_id"`
	Environment harnessresolve.Environment `json:"environment"`
	Verifier    string                     `json:"verifier"`
	Layer       string                     `json:"layer"`
	Deltas      []Delta                    `json:"deltas"`
	Rebuild     string                     `json:"rebuild"`
}

// Result wraps the newly derived lock alongside its verification receipt.
type Result struct {
	Lock    harnessresolve.Lock `json:"lock"`
	Receipt Receipt             `json:"receipt"`
}

// Derive computes a new verified lock by applying typed deltas to an immutable base lock.
func Derive(base harnessresolve.Lock, request Request) (Result, error) {
	if err := harnessresolve.VerifyLock(base); err != nil {
		return Result{}, fmt.Errorf("verify base: %w", err)
	}
	layer := strings.TrimSpace(request.Layer)
	if layer == "" {
		layer = "local"
	}
	if len(request.Deltas) == 0 {
		return Result{}, fmt.Errorf("at least one typed delta is required")
	}
	deltas := normalizedDeltas(request.Deltas)
	seen := map[string]bool{}
	for _, delta := range deltas {
		if seen[delta.Capability] {
			return Result{}, fmt.Errorf("capability %q changed more than once", delta.Capability)
		}
		seen[delta.Capability] = true
	}

	derived := cloneLock(base)
	for _, delta := range deltas {
		if err := apply(&derived, base.ID, layer, delta); err != nil {
			return Result{}, err
		}
	}
	if err := harnessresolve.ReidentifyLock(&derived); err != nil {
		return Result{}, fmt.Errorf("identify derived lock: %w", err)
	}
	receipt := Receipt{
		Schema: Schema, BaseID: base.ID, ResultID: derived.ID,
		Environment: derived.Environment, Verifier: Verifier, Layer: layer, Deltas: deltas,
		Rebuild: rebuildCommand(base.ID, layer, deltas),
	}
	if err := identifyReceipt(&receipt); err != nil {
		return Result{}, fmt.Errorf("identify derivation receipt: %w", err)
	}
	return Result{Lock: derived, Receipt: receipt}, nil
}

func apply(lock *harnessresolve.Lock, baseID, layer string, delta Delta) error {
	kind, id, ok := strings.Cut(delta.Capability, ":")
	if !ok || strings.TrimSpace(kind) == "" || strings.TrimSpace(id) == "" {
		return fmt.Errorf("capability %q must be kind:id", delta.Capability)
	}
	index := -1
	for i := range lock.Assets {
		if lock.Assets[i].Kind == kind && lock.Assets[i].ID == id {
			index = i
			break
		}
	}
	if index < 0 {
		return fmt.Errorf("capability %q is not active in the verified base", delta.Capability)
	}
	asset := &lock.Assets[index]
	if asset.Mandatory {
		return fmt.Errorf("capability %q is mandatory and cannot be derived", delta.Capability)
	}
	if asset.Locked {
		return fmt.Errorf("capability %q is locked by %s and cannot be derived", delta.Capability, asset.Source)
	}
	original := asset.Source
	switch delta.Operation {
	case "replace":
		if kind != "instruction" {
			return fmt.Errorf("capability %q has no launch-conformant replace adapter; only instruction replacement is supported", delta.Capability)
		}
		if strings.TrimSpace(delta.Value) == "" || len(delta.Denies) != 0 {
			return fmt.Errorf("instruction replacement requires value and no denies")
		}
		asset.Value = delta.Value
	case "deny":
		if kind != "policy" {
			return fmt.Errorf("deny operation requires a policy capability")
		}
		if len(delta.Denies) == 0 || delta.Value != "" {
			return fmt.Errorf("policy narrowing requires denies and no value")
		}
		asset.Denies = sortedUnique(append(asset.Denies, delta.Denies...))
		asset.Grants = remove(asset.Grants, delta.Denies)
	default:
		return fmt.Errorf("unsupported operation %q for %s", delta.Operation, delta.Capability)
	}
	asset.Source = fmt.Sprintf("derive:%s (from %s)", layer, original)
	lock.AssetTrace = append(lock.AssetTrace, harnesscompose.Trace{
		Layer: layer, Kind: kind, ID: id, Action: delta.Operation,
		Reason: fmt.Sprintf("local delta over verified base %s; inherited from %s", baseID, original),
	})
	return nil
}

func normalizedDeltas(in []Delta) []Delta {
	out := append([]Delta(nil), in...)
	for i := range out {
		out[i].Capability = strings.TrimSpace(out[i].Capability)
		out[i].Operation = strings.TrimSpace(out[i].Operation)
		out[i].Value = strings.TrimSpace(out[i].Value)
		out[i].Denies = sortedUnique(out[i].Denies)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Capability != out[j].Capability {
			return out[i].Capability < out[j].Capability
		}
		return out[i].Operation < out[j].Operation
	})
	return out
}

func cloneLock(lock harnessresolve.Lock) harnessresolve.Lock {
	raw, _ := json.Marshal(lock)
	var out harnessresolve.Lock
	_ = json.Unmarshal(raw, &out)
	return out
}

func sortedUnique(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, value := range in {
		value = strings.TrimSpace(value)
		if value != "" && !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	sort.Strings(out)
	return out
}

func remove(in, denied []string) []string {
	blocked := map[string]bool{}
	for _, value := range denied {
		blocked[value] = true
	}
	out := make([]string, 0, len(in))
	for _, value := range in {
		if !blocked[value] {
			out = append(out, value)
		}
	}
	return sortedUnique(out)
}

func rebuildCommand(baseID, layer string, deltas []Delta) string {
	parts := []string{"fak harness derive", "--from <base-lock>", "--expect-base " + baseID, "--layer " + layer}
	for _, delta := range deltas {
		switch delta.Operation {
		case "replace":
			parts = append(parts, fmt.Sprintf("--set %s=%s", delta.Capability, delta.Value))
		case "deny":
			for _, denied := range delta.Denies {
				parts = append(parts, fmt.Sprintf("--deny %s=%s", delta.Capability, denied))
			}
		}
	}
	return strings.Join(append(parts, "--output <derived-lock>"), " ")
}

func identifyReceipt(receipt *Receipt) error {
	receipt.SchemaID = ""
	raw, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(raw)
	receipt.SchemaID = "sha256:" + hex.EncodeToString(sum[:])
	return nil
}

// VerifyReceipt validates the cryptographic integrity and digest of a derivation receipt.
func VerifyReceipt(receipt Receipt) error {
	if receipt.Schema != Schema || receipt.SchemaID == "" || receipt.BaseID == "" || receipt.ResultID == "" || receipt.Verifier != Verifier {
		return fmt.Errorf("invalid derivation receipt")
	}
	want := receipt.SchemaID
	if err := identifyReceipt(&receipt); err != nil {
		return err
	}
	if receipt.SchemaID != want {
		return fmt.Errorf("derivation receipt digest mismatch: got %s want %s", want, receipt.SchemaID)
	}
	return nil
}
