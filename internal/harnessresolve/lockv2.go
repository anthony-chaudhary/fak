package harnessresolve

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

// LockSchemaV2 specifies the version 2 JSON schema URI for multi-platform product locks.
const LockSchemaV2 = "fak.harness-product-lock/v2"

var secretRefPattern = regexp.MustCompile(`^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$`)

// SecretPlaintextLeakError records an illegal plaintext value embedded in a locked secret asset.
type SecretPlaintextLeakError struct {
	AssetID string
}

// Error formats the failure string containing the closed refusal token.
func (e *SecretPlaintextLeakError) Error() string {
	return fmt.Sprintf("SECRET_PLAINTEXT_LEAK: secret asset %q must not contain plaintext value", e.AssetID)
}

// LockEnvironment defines target execution platform attributes including OS, architecture, and runtime contract.
type LockEnvironment struct {
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	Contract string `json:"contract,omitempty"`
}

// PlatformMatrix specifies permitted combinations of operating systems, architectures, and contracts.
type PlatformMatrix struct {
	OS       []string `json:"os,omitempty"`
	Arch     []string `json:"arch,omitempty"`
	Contract []string `json:"contract,omitempty"`
}

// ProductLockV2 represents a multi-platform resolved harness product lock under schema v2.
type ProductLockV2 struct {
	Schema        string                          `json:"schema"`
	ID            string                          `json:"id,omitempty"`
	Platforms     []LockEnvironment               `json:"platforms,omitempty"`
	Matrix        *PlatformMatrix                 `json:"matrix,omitempty"`
	Compatibility Compatibility                   `json:"compatibility,omitempty"`
	Budget        Budget                          `json:"budget,omitempty"`
	Components    []LockedComponent               `json:"components,omitempty"`
	Assets        []harnesscompose.EffectiveAsset `json:"assets,omitempty"`
	AssetTrace    []harnesscompose.Trace          `json:"asset_trace,omitempty"`
	Decisions     []stackresolve.Decision         `json:"decisions,omitempty"`
}

// LockV2 is an alias for ProductLockV2 representing the compiled product lock model.
type LockV2 = ProductLockV2

// CanonicalizeLF converts CRLF line terminators to canonical LF bytes.
func CanonicalizeLF(data []byte) []byte {
	return bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
}

// CanonicalLockIDV2 computes the deterministic canonical content digest for a v2 product lock.
func CanonicalLockIDV2(lock ProductLockV2) (string, error) {
	copy := lock
	copy.ID = ""

	if len(copy.Platforms) == 0 && copy.Matrix != nil {
		copy.Platforms = expandMatrix(*copy.Matrix)
	}
	copy.Matrix = nil

	if len(copy.Platforms) > 1 {
		platforms := append([]LockEnvironment(nil), copy.Platforms...)
		sort.Slice(platforms, func(i, j int) bool {
			if platforms[i].OS != platforms[j].OS {
				return platforms[i].OS < platforms[j].OS
			}
			if platforms[i].Arch != platforms[j].Arch {
				return platforms[i].Arch < platforms[j].Arch
			}
			return platforms[i].Contract < platforms[j].Contract
		})
		copy.Platforms = platforms
	}

	if len(copy.Components) > 1 {
		comps := append([]LockedComponent(nil), copy.Components...)
		sort.Slice(comps, func(i, j int) bool {
			return comps[i].ID < comps[j].ID
		})
		copy.Components = comps
	}

	if len(copy.Assets) > 0 {
		assets := append([]harnesscompose.EffectiveAsset(nil), copy.Assets...)
		for i := range assets {
			assets[i].Value = strings.ReplaceAll(assets[i].Value, "\r\n", "\n")
		}
		if len(assets) > 1 {
			sort.Slice(assets, func(i, j int) bool {
				if assets[i].Kind != assets[j].Kind {
					return assets[i].Kind < assets[j].Kind
				}
				return assets[i].ID < assets[j].ID
			})
		}
		copy.Assets = assets
	}

	raw, err := json.Marshal(copy)
	if err != nil {
		return "", err
	}
	raw = CanonicalizeLF(raw)
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// CanonicalID computes the deterministic canonical content digest for the lock.
func (l ProductLockV2) CanonicalID() (string, error) {
	return CanonicalLockIDV2(l)
}

// CanonicalIDV2 parses raw product lock bytes and computes the canonical content fingerprint ID.
func CanonicalIDV2(data []byte) (string, error) {
	lock, err := ParseProductLockV2(data)
	if err != nil {
		return "", err
	}
	return CanonicalLockIDV2(lock)
}

// ParseProductLockV2 unmarshals JSON configuration bytes into a ProductLockV2 model.
func ParseProductLockV2(data []byte) (ProductLockV2, error) {
	data = CanonicalizeLF(data)
	var lock ProductLockV2
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&lock); err != nil {
		return ProductLockV2{}, fmt.Errorf("parse product lock v2: %w", err)
	}
	for i := range lock.Assets {
		lock.Assets[i].Value = strings.ReplaceAll(lock.Assets[i].Value, "\r\n", "\n")
	}
	return lock, nil
}

// ValidateProductLockV2 verifies schema conformance, platform matrix compatibility, secret boundaries, and canonical digests.
func ValidateProductLockV2(data []byte) error {
	lock, err := ParseProductLockV2(data)
	if err != nil {
		return err
	}
	if lock.Schema != LockSchemaV2 {
		return fmt.Errorf("invalid lock schema: got %q, want %q", lock.Schema, LockSchemaV2)
	}

	// Enforce secret contracts fail-closed before other structural checks
	for _, asset := range lock.Assets {
		if asset.Kind == "secret" {
			if asset.Value != "" {
				return &SecretPlaintextLeakError{AssetID: asset.ID}
			}
			if !secretRefPattern.MatchString(asset.Ref) {
				return fmt.Errorf("secret asset %q has invalid ref %q: must match ^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$", asset.ID, asset.Ref)
			}
		}
	}

	platforms := lock.Platforms
	if len(platforms) == 0 && lock.Matrix != nil {
		platforms = expandMatrix(*lock.Matrix)
	}
	if len(platforms) == 0 {
		return fmt.Errorf("platform requirements missing: platforms must not be empty")
	}

	seenPlatforms := make(map[string]bool, len(platforms))
	for _, p := range platforms {
		if p.OS == "" || p.Arch == "" {
			return fmt.Errorf("platform os and arch are required: got os=%q arch=%q", p.OS, p.Arch)
		}
		key := p.OS + "/" + p.Arch
		if p.Contract != "" {
			key += "@" + p.Contract
		}
		if seenPlatforms[key] {
			return fmt.Errorf("duplicate platform %q", key)
		}
		seenPlatforms[key] = true

		env := Environment{OS: p.OS, Arch: p.Arch, Contract: p.Contract}
		for _, comp := range lock.Components {
			if err := checkCompatibility(comp.Compatibility, env, fmt.Sprintf("component %q", comp.ID)); err != nil {
				return fmt.Errorf("platform %s/%s incompatible: %w", p.OS, p.Arch, err)
			}
		}
		if err := checkCompatibility(lock.Compatibility, env, "product lock"); err != nil {
			return fmt.Errorf("platform %s/%s incompatible: %w", p.OS, p.Arch, err)
		}
	}

	if lock.ID != "" {
		canonicalID, err := CanonicalLockIDV2(lock)
		if err != nil {
			return fmt.Errorf("compute canonical lock id: %w", err)
		}
		if lock.ID != canonicalID {
			return fmt.Errorf("lock id mismatch: got %s want canonical %s", lock.ID, canonicalID)
		}
	}

	return nil
}

// VerifyLockV2 confirms that a product lock v2 declares the v2 schema and matches its content digest.
func VerifyLockV2(lock ProductLockV2) error {
	if lock.Schema != LockSchemaV2 {
		return fmt.Errorf("lock schema must be %q", LockSchemaV2)
	}
	if lock.ID == "" {
		return fmt.Errorf("lock id is required")
	}
	want := lock.ID
	got, err := CanonicalLockIDV2(lock)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("lock digest mismatch: got %s want %s", want, got)
	}
	return nil
}

func expandMatrix(m PlatformMatrix) []LockEnvironment {
	var out []LockEnvironment
	contracts := m.Contract
	if len(contracts) == 0 {
		contracts = []string{""}
	}
	for _, osName := range m.OS {
		for _, archName := range m.Arch {
			for _, contract := range contracts {
				out = append(out, LockEnvironment{
					OS:       osName,
					Arch:     archName,
					Contract: contract,
				})
			}
		}
	}
	return out
}
