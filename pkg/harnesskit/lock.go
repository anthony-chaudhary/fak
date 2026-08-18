package harnesskit

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

const ProductLockSchema = "fak.harness-product-lock/v1alpha2"
const LegacyProductLockSchema = "fak.harness-product-lock/v1alpha1"
const LaunchReceiptSchema = "fak.harness-launch-receipt/v1alpha1"

type ProductLock struct {
	Schema      string            `json:"schema"`
	ID          string            `json:"id"`
	Environment LockEnvironment   `json:"environment"`
	Budget      LockBudget        `json:"budget"`
	Components  []LockedComponent `json:"components"`
	Assets      []LockedAsset     `json:"assets"`
	AssetTrace  json.RawMessage   `json:"asset_trace"`
	Decisions   json.RawMessage   `json:"decisions"`
}

type LockEnvironment struct {
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	Contract string `json:"contract"`
}
type LockBudget struct {
	ContextTokens int `json:"context_tokens,omitempty"`
	MemoryMiB     int `json:"memory_mib,omitempty"`
	Workers       int `json:"workers,omitempty"`
}
type LockRequirement struct {
	Capability string `json:"capability"`
	Range      string `json:"range,omitempty"`
	Optional   bool   `json:"optional,omitempty"`
}
type LockCompatibility struct {
	OS       []string `json:"os,omitempty"`
	Arch     []string `json:"arch,omitempty"`
	Contract string   `json:"contract,omitempty"`
}
type LockedComponent struct {
	ID            string            `json:"id"`
	Version       string            `json:"version"`
	Digest        string            `json:"digest"`
	Source        string            `json:"source"`
	Reason        string            `json:"reason"`
	Provider      string            `json:"provider"`
	Provides      []string          `json:"provides,omitempty"`
	Requires      []LockRequirement `json:"requires,omitempty"`
	Conflicts     []string          `json:"conflicts,omitempty"`
	Compatibility LockCompatibility `json:"compatibility,omitempty"`
	Cost          LockBudget        `json:"cost,omitempty"`
	Adapters      []string          `json:"adapters,omitempty"`
}
type LockedAsset struct {
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

type LaunchReceipt struct {
	Schema     string   `json:"schema"`
	LockID     string   `json:"lock_id"`
	ProductID  string   `json:"product_id"`
	Profile    string   `json:"profile"`
	Layers     []string `json:"layers"`
	Assets     []string `json:"assets"`
	Components []string `json:"components"`
}

func LoadProductLock(path string) (ProductLock, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return ProductLock{}, fmt.Errorf("read product lock: %w", err)
	}
	return ParseProductLock(raw)
}

func ParseProductLock(raw []byte) (ProductLock, error) {
	var lock ProductLock
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&lock); err != nil {
		return ProductLock{}, fmt.Errorf("parse product lock: %w", err)
	}
	if lock.Schema != ProductLockSchema && lock.Schema != LegacyProductLockSchema {
		return ProductLock{}, fmt.Errorf("product lock schema must be %q or legacy %q", ProductLockSchema, LegacyProductLockSchema)
	}
	if lock.ID == "" || len(lock.Components) == 0 {
		return ProductLock{}, fmt.Errorf("product lock id and components are required")
	}
	for _, component := range lock.Components {
		if component.ID == "" || component.Version == "" || !strings.HasPrefix(component.Digest, "sha256:") || component.Source == "" {
			return ProductLock{}, fmt.Errorf("locked component id/version/digest/source are required")
		}
	}
	for _, asset := range lock.Assets {
		if asset.Kind == "" || asset.ID == "" || asset.Source == "" {
			return ProductLock{}, fmt.Errorf("locked asset kind/id/source are required")
		}
		if asset.Kind == "secret" && asset.Ref == "" {
			return ProductLock{}, fmt.Errorf("locked secret %q has no opaque reference", asset.ID)
		}
	}
	want := lock.ID
	lock.ID = ""
	canonical, err := json.Marshal(lock)
	if err != nil {
		return ProductLock{}, err
	}
	sum := sha256.Sum256(canonical)
	got := "sha256:" + hex.EncodeToString(sum[:])
	if got != want {
		return ProductLock{}, fmt.Errorf("product lock digest mismatch: got %s want %s", want, got)
	}
	lock.ID = want
	return lock, nil
}

// Mixable reports whether the lock carries the v1alpha2 component evidence
// required for sound downstream composition. Legacy locks remain launchable,
// but missing compatibility facts are never guessed.
func (l ProductLock) Mixable() error {
	if l.Schema != ProductLockSchema {
		return fmt.Errorf("legacy product lock %q is launchable but not mixable; rebuild it from source as %q", l.Schema, ProductLockSchema)
	}
	for _, component := range l.Components {
		if component.Reason == "" || component.Provider == "" {
			return fmt.Errorf("component %q has incomplete selection provenance", component.ID)
		}
		if component.Compatibility.Contract == "" {
			return fmt.Errorf("component %q has no compatibility contract", component.ID)
		}
		if len(component.Adapters) == 0 {
			return fmt.Errorf("component %q has no runtime adapter conformance evidence", component.ID)
		}
	}
	return nil
}

func (l ProductLock) Profile() Profile {
	layers := uniqueSources(l.Assets)
	name := "locked"
	for _, layer := range layers {
		if layer == "legal" || layer == "coding" || layer == "integrated" {
			name = layer
			break
		}
	}
	caps := make([]Capability, 0, len(layers))
	for _, layer := range layers {
		caps = append(caps, Capability("layer:"+layer))
	}
	return Profile{ID: name, Capabilities: caps}
}

func (l ProductLock) InstructionText() string {
	values := []string{}
	for _, asset := range l.Assets {
		if asset.Kind == "instruction" && asset.Value != "" {
			values = append(values, asset.Value)
		}
	}
	sort.Strings(values)
	return strings.Join(values, "\n")
}

func (l ProductLock) LaunchReceipt(productID string) LaunchReceipt {
	assets := make([]string, 0, len(l.Assets))
	for _, asset := range l.Assets {
		assets = append(assets, asset.Kind+"/"+asset.ID+"@"+asset.Source)
	}
	sort.Strings(assets)
	components := make([]string, 0, len(l.Components))
	for _, component := range l.Components {
		components = append(components, component.ID+"@"+component.Version+"#"+component.Digest)
	}
	sort.Strings(components)
	profile := l.Profile()
	return LaunchReceipt{Schema: LaunchReceiptSchema, LockID: l.ID, ProductID: productID, Profile: profile.ID, Layers: layersFromProfile(profile), Assets: assets, Components: components}
}

func uniqueSources(assets []LockedAsset) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, asset := range assets {
		if !seen[asset.Source] {
			seen[asset.Source] = true
			out = append(out, asset.Source)
		}
	}
	return out
}

func layersFromProfile(profile Profile) []string {
	out := []string{}
	for _, capability := range profile.Capabilities {
		value := string(capability)
		if strings.HasPrefix(value, "layer:") {
			out = append(out, strings.TrimPrefix(value, "layer:"))
		}
	}
	return out
}
