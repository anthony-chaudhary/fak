// Package quantfixture provides deterministic, redistributable quantization
// interoperability fixtures and verifies their provenance manifest.
//
// Invariant: quant fixture verification is fail-closed and tamper-evident across all recipes.
// Contract: every artifact must match its declared SHA256 digest, size bounds, and deterministic recipe.
// Guard: unknown schemas, missing artifacts, or unapproved licenses are immediately rejected.
package quantfixture

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Schema identifies the canonical specification version for quant fixture manifests.
const Schema = "quantfixture/1"

// PublicDomainLicense defines the required permissive license for redistributable test fixtures.
const PublicDomainLicense = "CC0-1.0"

// Entry describes an individual quantization test artifact and its deterministic generation recipe.
type Entry struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"`
	License  string `json:"license"`
	MaxBytes int    `json:"max_bytes"`
	SHA256   string `json:"sha256"`
	Recipe   string `json:"recipe"`
}

// Manifest represents a complete collection of verified quantization interoperability fixtures.
type Manifest struct {
	Schema   string  `json:"schema"`
	Fixtures []Entry `json:"fixtures"`
}

// LoadAndVerify independently reads every artifact and checks its declared
// digest, redistributable license, size ceiling, and deterministic recipe.
func LoadAndVerify(dir string) (Manifest, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return Manifest{}, err
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, err
	}
	if manifest.Schema != Schema {
		return Manifest{}, fmt.Errorf("quantfixture: unknown schema %q", manifest.Schema)
	}
	if len(manifest.Fixtures) < 3 {
		return Manifest{}, fmt.Errorf("quantfixture: need at least three fixtures")
	}
	seen := map[string]bool{}
	for _, entry := range manifest.Fixtures {
		if entry.Name == "" || seen[entry.Name] {
			return Manifest{}, fmt.Errorf("quantfixture: invalid duplicate name %q", entry.Name)
		}
		seen[entry.Name] = true
		if entry.License != PublicDomainLicense {
			return Manifest{}, fmt.Errorf("quantfixture: %s license %q is not redistributable", entry.Name, entry.License)
		}
		artifact, err := os.ReadFile(filepath.Join(dir, filepath.Base(entry.Name)))
		if err != nil {
			return Manifest{}, err
		}
		if entry.MaxBytes <= 0 || len(artifact) > entry.MaxBytes {
			return Manifest{}, fmt.Errorf("quantfixture: %s exceeds %d-byte cap", entry.Name, entry.MaxBytes)
		}
		digest := sha256.Sum256(artifact)
		if hex.EncodeToString(digest[:]) != entry.SHA256 {
			return Manifest{}, fmt.Errorf("quantfixture: %s hash mismatch", entry.Name)
		}
		regenerated, err := Generate(entry.Recipe)
		if err != nil {
			return Manifest{}, fmt.Errorf("quantfixture: %s: %w", entry.Name, err)
		}
		if string(regenerated) != string(artifact) {
			return Manifest{}, fmt.Errorf("quantfixture: %s regeneration mismatch", entry.Name)
		}
	}
	return manifest, nil
}

// Generate implements every manifest recipe without network access or weights.
func Generate(recipe string) ([]byte, error) {
	switch recipe {
	case "gguf-v3-q4":
		out := append([]byte("GGUF"), 3, 0, 0, 0)
		return append(out, []byte{'F', 'A', 'K', '-', 'S', 'Y', 'N', 'T', 'H', 'E', 'T', 'I', 'C', '-', 'Q', '4', 0, 0x12, 0x34, 0x56, 0x78}...), nil
	case "safetensors-empty-metadata":
		meta := []byte(`{"__metadata__":{"format":"synthetic","license":"CC0-1.0"}}`)
		out := make([]byte, 8, 8+len(meta))
		binary.LittleEndian.PutUint64(out, uint64(len(meta)))
		return append(out, meta...), nil
	case "packed-ternary-2bit":
		return []byte{0x1b, 0xe4, 0x39, 0x93, 0x00, 0xff}, nil
	default:
		return nil, fmt.Errorf("unknown recipe %q", recipe)
	}
}
