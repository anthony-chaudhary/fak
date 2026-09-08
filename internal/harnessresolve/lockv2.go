package harnessresolve

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	publiclockv2 "github.com/anthony-chaudhary/fak/pkg/harnesskit/lockv2"
)

// LockSchemaV2 specifies the version 2 JSON schema URI for multi-platform product locks.
const LockSchemaV2 = publiclockv2.ProductLockSchemaV2

// SecretPlaintextLeakError preserves the internal typed error contract while
// validation itself remains owned by pkg/harnesskit/lockv2.
type SecretPlaintextLeakError struct {
	AssetID string
	cause   error
}

func (e *SecretPlaintextLeakError) Error() string {
	if e.cause != nil {
		return e.cause.Error()
	}
	return fmt.Sprintf("%s: locked secret %q cannot contain plaintext value", publiclockv2.SecretPlaintextLeakError, e.AssetID)
}

func (e *SecretPlaintextLeakError) Unwrap() error { return e.cause }

// Compatibility aliases keep existing resolver callers source-compatible while
// making the public package the single owner of the v2 lock schema.
type LockEnvironment = publiclockv2.PlatformRequirement
type ProductLockV2 = publiclockv2.Lock
type LockV2 = publiclockv2.Lock
type LockBudgetV2 = publiclockv2.LockBudget
type LockCompatibilityV2 = publiclockv2.LockCompatibility
type LockedComponentV2 = publiclockv2.LockedComponent
type LockedAssetV2 = publiclockv2.LockedAsset

// PlatformMatrix is the legacy serialized shorthand accepted by the original
// internal v2 parser. New code should emit the public Platforms field.
type PlatformMatrix struct {
	OS       []string `json:"os,omitempty"`
	Arch     []string `json:"arch,omitempty"`
	Contract []string `json:"contract,omitempty"`
}

type legacyLockMetadata struct {
	Matrix        *PlatformMatrix
	Compatibility LockCompatibilityV2
}

// CanonicalizeLF converts CRLF line terminators to canonical LF bytes.
func CanonicalizeLF(data []byte) []byte {
	return bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
}

// CanonicalLockIDV2 delegates RFC 8785 canonicalization and hashing to the
// public lock contract.
func CanonicalLockIDV2(lock ProductLockV2) (string, error) {
	return publiclockv2.CanonicalID(&lock)
}

// CanonicalIDV2 returns the public canonical fingerprint for public-shaped
// locks and preserves the original fingerprint for legacy serialized locks.
func CanonicalIDV2(data []byte) (string, error) {
	lock, metadata, err := decodeProductLockV2(data)
	if err != nil {
		return "", err
	}
	// Matrix/top-level-compatibility locks were serialized by the original
	// internal implementation. Preserve their persisted content IDs while the
	// public shape uses the public RFC 8785 implementation exclusively.
	if metadata.Matrix != nil || !isZeroCompatibility(metadata.Compatibility) {
		return legacyCanonicalIDV2(data)
	}
	if lock.ID != "" {
		if legacyID, legacyErr := legacyCanonicalIDV2(data); legacyErr == nil && legacyID == lock.ID {
			return legacyID, nil
		}
	}
	return publiclockv2.CanonicalID(&lock)
}

// ParseProductLockV2 decodes into the public lock model. It intentionally
// retains the old parse-only contract (semantic validation is performed by
// ValidateProductLockV2) and expands legacy serialized matrices.
func ParseProductLockV2(data []byte) (ProductLockV2, error) {
	lock, _, err := decodeProductLockV2(data)
	return lock, err
}

// ValidateProductLockV2 reuses the public secret and canonical contracts, then
// applies resolver-specific platform admission and legacy-ID compatibility.
func ValidateProductLockV2(data []byte) error {
	lock, metadata, err := decodeProductLockV2(data)
	if err != nil {
		return err
	}
	if lock.Schema != LockSchemaV2 {
		return fmt.Errorf("invalid lock schema: got %q, want %q", lock.Schema, LockSchemaV2)
	}
	if err := publiclockv2.ValidateSecretContracts(&lock); err != nil {
		return preserveSecretLeakType(data, err)
	}
	if len(lock.Platforms) == 0 {
		return fmt.Errorf("platform requirements missing: platforms must not be empty")
	}

	seenPlatforms := make(map[string]bool, len(lock.Platforms))
	for _, platform := range lock.Platforms {
		if platform.OS == "" || platform.Arch == "" {
			return fmt.Errorf("platform os and arch are required: got os=%q arch=%q", platform.OS, platform.Arch)
		}
		key := platform.String()
		if seenPlatforms[key] {
			return fmt.Errorf("duplicate platform %q", key)
		}
		seenPlatforms[key] = true
	}
	for _, platform := range lock.Platforms {
		environment := Environment{OS: platform.OS, Arch: platform.Arch, Contract: platform.Contract}
		for _, component := range lock.Components {
			compatibility := Compatibility{
				OS:       component.Compatibility.OS,
				Arch:     component.Compatibility.Arch,
				Contract: component.Compatibility.Contract,
			}
			if err := checkCompatibility(compatibility, environment, fmt.Sprintf("component %q", component.ID)); err != nil {
				return fmt.Errorf("platform %s/%s incompatible: %w", platform.OS, platform.Arch, err)
			}
		}
		if err := checkCompatibility(Compatibility{
			OS:       metadata.Compatibility.OS,
			Arch:     metadata.Compatibility.Arch,
			Contract: metadata.Compatibility.Contract,
		}, environment, "product lock"); err != nil {
			return fmt.Errorf("platform %s/%s incompatible: %w", platform.OS, platform.Arch, err)
		}
	}
	if lock.ID != "" {
		canonicalID, err := publiclockv2.CanonicalID(&lock)
		if err != nil {
			return fmt.Errorf("compute canonical lock id: %w", err)
		}
		if lock.ID != canonicalID {
			legacyID, legacyErr := legacyCanonicalIDV2(data)
			if legacyErr != nil || lock.ID != legacyID {
				return fmt.Errorf("lock id mismatch: got %s want canonical %s", lock.ID, canonicalID)
			}
		}
	}
	return nil
}

// VerifyLockV2 confirms that a product lock v2 declares the public schema and
// matches the public canonical digest.
func VerifyLockV2(lock ProductLockV2) error {
	if lock.Schema != LockSchemaV2 {
		return fmt.Errorf("lock schema must be %q", LockSchemaV2)
	}
	if lock.ID == "" {
		return fmt.Errorf("lock id is required")
	}
	want := lock.ID
	got, err := publiclockv2.CanonicalID(&lock)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("lock digest mismatch: got %s want %s", want, got)
	}
	return nil
}

func preserveSecretLeakType(data []byte, err error) error {
	if err == nil || !strings.Contains(err.Error(), publiclockv2.SecretPlaintextLeakError) {
		return err
	}
	var envelope struct {
		Assets []publiclockv2.LockedAsset `json:"assets"`
	}
	if json.Unmarshal(CanonicalizeLF(data), &envelope) == nil {
		for _, asset := range envelope.Assets {
			if asset.Kind == "secret" && asset.Value != "" {
				return &SecretPlaintextLeakError{AssetID: asset.ID, cause: err}
			}
		}
	}
	return err
}

func decodeProductLockV2(data []byte) (ProductLockV2, legacyLockMetadata, error) {
	canonical := CanonicalizeLF(data)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(canonical, &fields); err != nil {
		return ProductLockV2{}, legacyLockMetadata{}, fmt.Errorf("parse product lock v2: %w", err)
	}

	var metadata legacyLockMetadata
	if raw, ok := fields["matrix"]; ok {
		var matrix PlatformMatrix
		if err := json.Unmarshal(raw, &matrix); err != nil {
			return ProductLockV2{}, legacyLockMetadata{}, fmt.Errorf("parse product lock v2 matrix: %w", err)
		}
		metadata.Matrix = &matrix
		delete(fields, "matrix")
	}
	if raw, ok := fields["compatibility"]; ok {
		if err := json.Unmarshal(raw, &metadata.Compatibility); err != nil {
			return ProductLockV2{}, legacyLockMetadata{}, fmt.Errorf("parse product lock v2 compatibility: %w", err)
		}
		delete(fields, "compatibility")
	}

	normalized, err := json.Marshal(fields)
	if err != nil {
		return ProductLockV2{}, legacyLockMetadata{}, fmt.Errorf("parse product lock v2: %w", err)
	}
	var lock ProductLockV2
	dec := json.NewDecoder(bytes.NewReader(normalized))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&lock); err != nil {
		return ProductLockV2{}, legacyLockMetadata{}, fmt.Errorf("parse product lock v2: %w", err)
	}
	if len(lock.Platforms) == 0 && metadata.Matrix != nil {
		lock.Platforms = expandPlatformMatrix(*metadata.Matrix)
	}
	for i := range lock.Assets {
		lock.Assets[i].Value = strings.ReplaceAll(lock.Assets[i].Value, "\r\n", "\n")
	}
	return lock, metadata, nil
}

func expandPlatformMatrix(matrix PlatformMatrix) []LockEnvironment {
	contracts := matrix.Contract
	if len(contracts) == 0 {
		contracts = []string{""}
	}
	platforms := make([]LockEnvironment, 0, len(matrix.OS)*len(matrix.Arch)*len(contracts))
	for _, osName := range matrix.OS {
		for _, archName := range matrix.Arch {
			for _, contract := range contracts {
				platforms = append(platforms, LockEnvironment{OS: osName, Arch: archName, Contract: contract})
			}
		}
	}
	return platforms
}

func isZeroCompatibility(compatibility LockCompatibilityV2) bool {
	return len(compatibility.OS) == 0 && len(compatibility.Arch) == 0 && compatibility.Contract == ""
}

// The original internal v2 implementation shipped concurrently with the
// public contract and used struct-order JSON hashing. Keep this decoder only as
// a read-compatibility bridge for persisted IDs; all newly constructed locks
// use publiclockv2.CanonicalID.
type legacyProductLockV2 struct {
	Schema        string                    `json:"schema"`
	ID            string                    `json:"id,omitempty"`
	Platforms     []LockEnvironment         `json:"platforms,omitempty"`
	Matrix        *PlatformMatrix           `json:"matrix,omitempty"`
	Compatibility LockCompatibilityV2       `json:"compatibility,omitempty"`
	Budget        LockBudgetV2              `json:"budget,omitempty"`
	Components    []legacyLockedComponentV2 `json:"components,omitempty"`
	Assets        []LockedAssetV2           `json:"assets,omitempty"`
	AssetTrace    json.RawMessage           `json:"asset_trace,omitempty"`
	Decisions     json.RawMessage           `json:"decisions,omitempty"`
}

type legacyLockedComponentV2 struct {
	ID            string                         `json:"id"`
	Version       string                         `json:"version"`
	Digest        string                         `json:"digest"`
	Source        string                         `json:"source"`
	Reason        string                         `json:"reason"`
	Provider      string                         `json:"provider"`
	Provides      []string                       `json:"provides,omitempty"`
	Requires      []publiclockv2.LockRequirement `json:"requires,omitempty"`
	Conflicts     []string                       `json:"conflicts,omitempty"`
	Compatibility LockCompatibilityV2            `json:"compatibility,omitempty"`
	Cost          LockBudgetV2                   `json:"cost,omitempty"`
	Adapters      []string                       `json:"adapters,omitempty"`
}

func legacyCanonicalIDV2(data []byte) (string, error) {
	var lock legacyProductLockV2
	dec := json.NewDecoder(bytes.NewReader(CanonicalizeLF(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&lock); err != nil {
		return "", fmt.Errorf("parse legacy product lock v2: %w", err)
	}
	lock.ID = ""
	if len(lock.Platforms) == 0 && lock.Matrix != nil {
		lock.Platforms = expandPlatformMatrix(*lock.Matrix)
	}
	lock.Matrix = nil
	sort.Slice(lock.Platforms, func(i, j int) bool {
		if lock.Platforms[i].OS != lock.Platforms[j].OS {
			return lock.Platforms[i].OS < lock.Platforms[j].OS
		}
		if lock.Platforms[i].Arch != lock.Platforms[j].Arch {
			return lock.Platforms[i].Arch < lock.Platforms[j].Arch
		}
		return lock.Platforms[i].Contract < lock.Platforms[j].Contract
	})
	sort.Slice(lock.Components, func(i, j int) bool { return lock.Components[i].ID < lock.Components[j].ID })
	for i := range lock.Assets {
		lock.Assets[i].Value = strings.ReplaceAll(lock.Assets[i].Value, "\r\n", "\n")
	}
	sort.Slice(lock.Assets, func(i, j int) bool {
		if lock.Assets[i].Kind != lock.Assets[j].Kind {
			return lock.Assets[i].Kind < lock.Assets[j].Kind
		}
		return lock.Assets[i].ID < lock.Assets[j].ID
	})
	raw, err := json.Marshal(lock)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(CanonicalizeLF(raw))
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
