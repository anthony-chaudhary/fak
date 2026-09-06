package lockv2

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

// ProductLockSchemaV2 is the canonical schema URI for version 2 multi-platform harness product locks.
const ProductLockSchemaV2 = "fak.harness-product-lock/v2"

// ProductLockSchemaV1Alpha2 is the schema URI for v1alpha2 locks.
const ProductLockSchemaV1Alpha2 = "fak.harness-product-lock/v1alpha2"

// ProductLockSchemaV1Alpha1 is the schema URI for legacy v1alpha1 locks.
const ProductLockSchemaV1Alpha1 = "fak.harness-product-lock/v1alpha1"

// SecretPlaintextLeakError is the typed refusal token when a secret asset contains an inline plaintext value.
const SecretPlaintextLeakError = "SECRET_PLAINTEXT_LEAK"

var secretRefPattern = regexp.MustCompile(`^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$`)

// PlatformRequirement defines a target execution platform consisting of OS, CPU architecture, and runtime contract.
type PlatformRequirement struct {
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	Contract string `json:"contract,omitempty"`
}

// String returns the platform identifier formatted as "os/arch" or "os/arch@contract".
func (p PlatformRequirement) String() string {
	base := p.OS + "/" + p.Arch
	if p.Contract != "" {
		return base + "@" + p.Contract
	}
	return base
}

// SecretContract defines opaque resolution metadata for external credentials.
type SecretContract struct {
	Ref      string `json:"ref,omitempty"`
	Provider string `json:"provider,omitempty"`
	Boundary string `json:"boundary,omitempty"`
}

// ToolSchemaFingerprint tracks structural integrity of tool call interfaces.
type ToolSchemaFingerprint struct {
	Name        string `json:"name"`
	Fingerprint string `json:"fingerprint,omitempty"`
	Digest      string `json:"digest,omitempty"`
}

// LockBudget specifies resource upper bounds for context tokens, memory, and workers.
type LockBudget struct {
	ContextTokens int `json:"context_tokens,omitempty"`
	MemoryMiB     int `json:"memory_mib,omitempty"`
	Workers       int `json:"workers,omitempty"`
}

// LockRequirement defines an individual capability dependency requirement.
type LockRequirement struct {
	Capability string `json:"capability"`
	Range      string `json:"range,omitempty"`
	Optional   bool   `json:"optional,omitempty"`
}

// LockCompatibility declares execution environment constraints for a locked component.
type LockCompatibility struct {
	OS       []string `json:"os,omitempty"`
	Arch     []string `json:"arch,omitempty"`
	Contract string   `json:"contract,omitempty"`
}

// LockEnvironment captures legacy single-environment runtime bounds.
type LockEnvironment struct {
	OS       string `json:"os,omitempty"`
	Arch     string `json:"arch,omitempty"`
	Contract string `json:"contract,omitempty"`
}

// ComponentKind classifies the execution role or protocol implemented by a locked component.
type ComponentKind string

const (
	ComponentKindRuntime ComponentKind = "runtime"
	ComponentKindMCP     ComponentKind = "mcp"
	ComponentKindLSP     ComponentKind = "lsp"
	ComponentKindTool    ComponentKind = "tool"
	ComponentKindEngine  ComponentKind = "engine"
)

// LockedLSPMetadata specifies execution, language, and feature settings for Language Server Protocol components.
type LockedLSPMetadata struct {
	Language       string          `json:"language,omitempty"`
	Extensions     []string        `json:"extensions,omitempty"`
	Transport      string          `json:"transport,omitempty"`
	Command        []string        `json:"command,omitempty"`
	Diagnostics    bool            `json:"diagnostics,omitempty"`
	Symbols        bool            `json:"symbols,omitempty"`
	Initialization json.RawMessage `json:"initialization,omitempty"`
}

// LockedMCPMetadata specifies transport, launch command, environment, and security policy for Model Context Protocol components.
type LockedMCPMetadata struct {
	Transport   string            `json:"transport,omitempty"`
	Command     []string          `json:"command,omitempty"`
	Environment map[string]string `json:"environment,omitempty"`
	Policy      string            `json:"policy,omitempty"`
}

// LockedComponent describes a bound provider component locked into the harness product.
type LockedComponent struct {
	ID            string                  `json:"id"`
	Version       string                  `json:"version"`
	Digest        string                  `json:"digest"`
	Source        string                  `json:"source"`
	Reason        string                  `json:"reason,omitempty"`
	Provider      string                  `json:"provider,omitempty"`
	Provides      []string                `json:"provides,omitempty"`
	Requires      []LockRequirement       `json:"requires,omitempty"`
	Conflicts     []string                `json:"conflicts,omitempty"`
	Compatibility LockCompatibility       `json:"compatibility,omitempty"`
	Cost          LockBudget              `json:"cost,omitempty"`
	Adapters      []string                `json:"adapters,omitempty"`
	Fingerprints  []ToolSchemaFingerprint `json:"fingerprints,omitempty"`
	Kind          ComponentKind           `json:"kind,omitempty"`
	MCP           *LockedMCPMetadata      `json:"mcp,omitempty"`
	LSP           *LockedLSPMetadata      `json:"lsp,omitempty"`
}

// IsMCP reports whether the component represents a Model Context Protocol server.
func (c LockedComponent) IsMCP() bool {
	return c.Kind == ComponentKindMCP || c.MCP != nil
}

// IsLSP reports whether the component represents a Language Server Protocol server.
func (c LockedComponent) IsLSP() bool {
	return c.Kind == ComponentKindLSP || c.LSP != nil
}

// LockedAsset captures resolved instructions, policies, tools, and secret references.
type LockedAsset struct {
	Kind      string          `json:"kind"`
	ID        string          `json:"id"`
	Value     string          `json:"value,omitempty"`
	Ref       string          `json:"ref,omitempty"`
	Boundary  string          `json:"boundary,omitempty"`
	Grants    []string        `json:"grants,omitempty"`
	Denies    []string        `json:"denies,omitempty"`
	Source    string          `json:"source"`
	Locked    bool            `json:"locked,omitempty"`
	Mandatory bool            `json:"mandatory,omitempty"`
	Secret    *SecretContract `json:"secret,omitempty"`
}

// Lock represents a version 2 multi-platform harness product lock.
type Lock struct {
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

// SupportsPlatform returns true if the lock supports the target OS and CPU architecture.
func (l *Lock) SupportsPlatform(targetOS, targetArch string) bool {
	if len(l.Platforms) == 0 {
		if l.Environment.OS != "" || l.Environment.Arch != "" {
			return (l.Environment.OS == "" || l.Environment.OS == targetOS) &&
				(l.Environment.Arch == "" || l.Environment.Arch == targetArch)
		}
		return true
	}
	for _, p := range l.Platforms {
		if (p.OS == "" || p.OS == targetOS) && (p.Arch == "" || p.Arch == targetArch) {
			return true
		}
	}
	return false
}

// ValidatePlatforms verifies that every component in the lock is compatible with every platform requirement.
func (l *Lock) ValidatePlatforms() error {
	platforms := l.Platforms
	if len(platforms) == 0 && (l.Environment.OS != "" || l.Environment.Arch != "") {
		platforms = []PlatformRequirement{{OS: l.Environment.OS, Arch: l.Environment.Arch, Contract: l.Environment.Contract}}
	}
	for _, p := range platforms {
		for _, comp := range l.Components {
			if len(comp.Compatibility.OS) > 0 {
				matched := false
				for _, os := range comp.Compatibility.OS {
					if os == p.OS {
						matched = true
						break
					}
				}
				if !matched {
					return fmt.Errorf("component %q incompatible with platform OS %q", comp.ID, p.OS)
				}
			}
			if len(comp.Compatibility.Arch) > 0 {
				matched := false
				for _, arch := range comp.Compatibility.Arch {
					if arch == p.Arch {
						matched = true
						break
					}
				}
				if !matched {
					return fmt.Errorf("component %q incompatible with platform arch %q", comp.ID, p.Arch)
				}
			}
			if comp.Compatibility.Contract != "" && p.Contract != "" && comp.Compatibility.Contract != p.Contract {
				return fmt.Errorf("component %q requires contract %q, platform has %q", comp.ID, comp.Compatibility.Contract, p.Contract)
			}
		}
	}
	return nil
}

// Mixable checks whether the lock provides necessary evidence for downstream composition.
func (l *Lock) Mixable() error {
	if l.Schema != ProductLockSchemaV2 {
		return fmt.Errorf("legacy product lock %q is launchable but not mixable; rebuild it from source as %q", l.Schema, ProductLockSchemaV2)
	}
	if len(l.Platforms) == 0 && (l.Environment.OS == "" && l.Environment.Arch == "") {
		return fmt.Errorf("lock %q has no platform or environment facts", l.ID)
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

// ValidateSecretContracts verifies that all secret assets satisfy external reference constraints and do not leak plaintext values.
func ValidateSecretContracts(lock *Lock) error {
	if lock == nil {
		return fmt.Errorf("lock is nil")
	}
	for _, asset := range lock.Assets {
		if asset.Kind == "secret" {
			if asset.Value != "" {
				return fmt.Errorf("%s: locked secret %q cannot contain plaintext value", SecretPlaintextLeakError, asset.ID)
			}
			if asset.Ref == "" || !secretRefPattern.MatchString(asset.Ref) {
				return fmt.Errorf("locked secret %q invalid or missing opaque reference %q (must match ^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$)", asset.ID, asset.Ref)
			}
		}
	}
	return nil
}

// CanonicalID computes the RFC 8785 JCS LF SHA-256 digest of the lock with its ID field cleared.
func CanonicalID(lock *Lock) (string, error) {
	if lock == nil {
		return "", fmt.Errorf("lock is nil")
	}
	copy := *lock
	copy.ID = ""
	raw, err := json.Marshal(copy)
	if err != nil {
		return "", fmt.Errorf("canonical id: %w", err)
	}
	canonicalBytes, err := CanonicalizeJSON(raw)
	if err != nil {
		return "", fmt.Errorf("canonical id: %w", err)
	}
	sum := sha256.Sum256(canonicalBytes)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// CanonicalizeJSON converts raw JSON bytes into RFC 8785 Canonical JSON with strict LF normalization.
func CanonicalizeJSON(data []byte) ([]byte, error) {
	data = bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("canonicalize json: decode: %w", err)
	}
	var buf bytes.Buffer
	if err := writeCanonicalJSON(&buf, v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writeCanonicalJSON(buf *bytes.Buffer, v any) error {
	switch val := v.(type) {
	case nil:
		buf.WriteString("null")
	case bool:
		if val {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case string:
		val = strings.ReplaceAll(val, "\r\n", "\n")
		buf.WriteByte('"')
		for _, r := range val {
			switch r {
			case '"':
				buf.WriteString(`\"`)
			case '\\':
				buf.WriteString(`\\`)
			case '\b':
				buf.WriteString(`\b`)
			case '\f':
				buf.WriteString(`\f`)
			case '\n':
				buf.WriteString(`\n`)
			case '\r':
				buf.WriteString(`\r`)
			case '\t':
				buf.WriteString(`\t`)
			default:
				if r < 0x20 {
					fmt.Fprintf(buf, `\u%04x`, r)
				} else {
					buf.WriteRune(r)
				}
			}
		}
		buf.WriteByte('"')
	case json.Number:
		str := string(val)
		if i, err := val.Int64(); err == nil && !strings.ContainsAny(str, ".eE") {
			buf.WriteString(strconv.FormatInt(i, 10))
		} else if f, err := val.Float64(); err == nil {
			buf.WriteString(formatCanonicalFloat(f))
		} else {
			buf.WriteString(str)
		}
	case float64:
		buf.WriteString(formatCanonicalFloat(val))
	case int:
		buf.WriteString(strconv.FormatInt(int64(val), 10))
	case int64:
		buf.WriteString(strconv.FormatInt(val, 10))
	case []any:
		buf.WriteByte('[')
		for i, item := range val {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writeCanonicalJSON(buf, item); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case map[string]any:
		buf.WriteByte('{')
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Slice(keys, func(i, j int) bool {
			return utf16Less(keys[i], keys[j])
		})
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writeCanonicalJSON(buf, k); err != nil {
				return err
			}
			buf.WriteByte(':')
			if err := writeCanonicalJSON(buf, val[k]); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	default:
		raw, err := json.Marshal(val)
		if err != nil {
			return err
		}
		c, err := CanonicalizeJSON(raw)
		if err != nil {
			return err
		}
		buf.Write(c)
	}
	return nil
}

func formatCanonicalFloat(f float64) string {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "null"
	}
	if f == math.Trunc(f) && math.Abs(f) < 1e21 {
		return strconv.FormatInt(int64(f), 10)
	}
	s := strconv.FormatFloat(f, 'g', -1, 64)
	s = strings.ReplaceAll(s, "e+", "e")
	return strings.ToLower(s)
}

func utf16Less(a, b string) bool {
	u1 := utf16.Encode([]rune(a))
	u2 := utf16.Encode([]rune(b))
	minLen := len(u1)
	if len(u2) < minLen {
		minLen = len(u2)
	}
	for i := 0; i < minLen; i++ {
		if u1[i] != u2[i] {
			return u1[i] < u2[i]
		}
	}
	return len(u1) < len(u2)
}

// Parse deserializes and validates a version 2 harness product lock from JSON bytes.
func Parse(data []byte) (*Lock, error) {
	data = bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	var lock Lock
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&lock); err != nil {
		return nil, fmt.Errorf("parse product lock: %w", err)
	}
	if lock.Schema != ProductLockSchemaV2 && lock.Schema != ProductLockSchemaV1Alpha2 && lock.Schema != ProductLockSchemaV1Alpha1 {
		return nil, fmt.Errorf("product lock schema must be %q, %q, or legacy %q", ProductLockSchemaV2, ProductLockSchemaV1Alpha2, ProductLockSchemaV1Alpha1)
	}
	if lock.ID == "" || len(lock.Components) == 0 {
		return nil, fmt.Errorf("product lock id and components are required")
	}
	for _, component := range lock.Components {
		if component.ID == "" || component.Version == "" || !strings.HasPrefix(component.Digest, "sha256:") || component.Source == "" {
			return nil, fmt.Errorf("locked component id/version/digest/source are required")
		}
	}
	for _, asset := range lock.Assets {
		if asset.Kind == "" || asset.ID == "" || asset.Source == "" {
			return nil, fmt.Errorf("locked asset kind/id/source are required")
		}
	}
	if err := ValidateSecretContracts(&lock); err != nil {
		return nil, err
	}
	for i := range lock.Assets {
		lock.Assets[i].Value = strings.ReplaceAll(lock.Assets[i].Value, "\r\n", "\n")
	}
	want := lock.ID
	if lock.Schema == ProductLockSchemaV2 {
		got, err := CanonicalID(&lock)
		if err != nil {
			return nil, err
		}
		if got != want {
			return nil, fmt.Errorf("product lock digest mismatch: got %s want %s", want, got)
		}
	} else {
		copy := lock
		copy.ID = ""
		canonical, err := json.Marshal(copy)
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(canonical)
		got := "sha256:" + hex.EncodeToString(sum[:])
		if got != want {
			return nil, fmt.Errorf("product lock digest mismatch: got %s want %s", want, got)
		}
	}
	lock.ID = want
	return &lock, nil
}
