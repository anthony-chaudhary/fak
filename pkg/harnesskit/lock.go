package harnesskit

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit/lockv2"
)

const ProductLockSchemaV2 = lockv2.ProductLockSchemaV2
const ProductLockSchema = "fak.harness-product-lock/v1alpha2"
const LegacyProductLockSchema = "fak.harness-product-lock/v1alpha1"
const LaunchReceiptSchema = "fak.harness-launch-receipt/v1alpha1"
const SecretPlaintextLeakError = lockv2.SecretPlaintextLeakError
const ErrSecretPlaintextLeak = SecretPlaintextLeakError

var secretRefPattern = regexp.MustCompile(`^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$`)

type PlatformRequirement = lockv2.PlatformRequirement
type SecretContract = lockv2.SecretContract
type ToolSchemaFingerprint = lockv2.ToolSchemaFingerprint
type LockEnvironment = lockv2.LockEnvironment
type LockBudget = lockv2.LockBudget
type LockRequirement = lockv2.LockRequirement
type LockCompatibility = lockv2.LockCompatibility
type LockedComponent = lockv2.LockedComponent
type LockedAsset = lockv2.LockedAsset

type ProductLock struct {
	Schema      string                `json:"schema"`
	ID          string                `json:"id"`
	Platforms   []PlatformRequirement `json:"platforms,omitempty"`
	Environment LockEnvironment       `json:"environment,omitempty"`
	Budget      LockBudget            `json:"budget"`
	Components  []LockedComponent     `json:"components"`
	Assets      []LockedAsset         `json:"assets"`
	AssetTrace  json.RawMessage       `json:"asset_trace,omitempty"`
	Decisions   json.RawMessage       `json:"decisions,omitempty"`
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
	raw = bytes.ReplaceAll(raw, []byte("\r\n"), []byte("\n"))
	var lock ProductLock
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&lock); err != nil {
		return ProductLock{}, fmt.Errorf("parse product lock: %w", err)
	}
	if lock.Schema != ProductLockSchemaV2 && lock.Schema != ProductLockSchema && lock.Schema != LegacyProductLockSchema {
		return ProductLock{}, fmt.Errorf("product lock schema must be %q, %q, or legacy %q", ProductLockSchemaV2, ProductLockSchema, LegacyProductLockSchema)
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
		if asset.Kind == "secret" {
			if asset.Value != "" {
				return ProductLock{}, fmt.Errorf("%s: locked secret %q cannot contain plaintext value", SecretPlaintextLeakError, asset.ID)
			}
			if asset.Ref == "" {
				return ProductLock{}, fmt.Errorf("locked secret %q has no opaque reference", asset.ID)
			}
			if !secretRefPattern.MatchString(asset.Ref) {
				return ProductLock{}, fmt.Errorf("locked secret %q invalid opaque reference %q", asset.ID, asset.Ref)
			}
		}
	}
	for i := range lock.Assets {
		lock.Assets[i].Value = strings.ReplaceAll(lock.Assets[i].Value, "\r\n", "\n")
	}
	want := lock.ID
	if lock.Schema == ProductLockSchemaV2 {
		v2 := lockv2.Lock(lock)
		got, err := lockv2.CanonicalID(&v2)
		if err != nil {
			return ProductLock{}, err
		}
		if got != want {
			return ProductLock{}, fmt.Errorf("product lock digest mismatch: got %s want %s", want, got)
		}
	} else {
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
	}
	lock.ID = want
	return lock, nil
}

// LockID calculates the canonical identity digest of the product lock.
func LockID(lock ProductLock) (string, error) {
	if lock.Schema == ProductLockSchemaV2 {
		v2 := lockv2.Lock(lock)
		return lockv2.CanonicalID(&v2)
	}
	copy := lock
	copy.ID = ""
	raw, err := json.Marshal(copy)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// SupportsPlatform reports whether the lock supports the target OS and CPU architecture.
func (l ProductLock) SupportsPlatform(targetOS, targetArch string) bool {
	v2 := lockv2.Lock(l)
	return v2.SupportsPlatform(targetOS, targetArch)
}

// ValidatePlatforms verifies that every component is compatible with declared platforms.
func (l ProductLock) ValidatePlatforms() error {
	v2 := lockv2.Lock(l)
	return v2.ValidatePlatforms()
}

// CanonicalID computes the RFC 8785 JCS LF SHA-256 digest of the lock.
func (l ProductLock) CanonicalID() (string, error) {
	return LockID(l)
}

// Mixable reports whether the lock carries the component evidence
// required for sound downstream composition. Legacy locks remain launchable,
// but missing compatibility facts are never guessed.
func (l ProductLock) Mixable() error {
	if l.Schema != ProductLockSchema && l.Schema != ProductLockSchemaV2 {
		return fmt.Errorf("legacy product lock %q is launchable but not mixable; rebuild it from source as %q", l.Schema, ProductLockSchemaV2)
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
	if l.Schema == ProductLockSchemaV2 {
		if len(l.Platforms) == 0 && (l.Environment.OS == "" && l.Environment.Arch == "") {
			return fmt.Errorf("lock %q has no platform or environment facts", l.ID)
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
