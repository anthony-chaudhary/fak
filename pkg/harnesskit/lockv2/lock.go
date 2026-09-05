package lockv2

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Schema is the canonical schema identifier for v2 harness product locks.
const Schema = "fak.harness-product-lock/v2"

// SecretPlaintextLeak is the sentinel error code emitted when a secret contract leaks plaintext values.
const SecretPlaintextLeak = "SECRET_PLAINTEXT_LEAK"

// refPattern validates secret reference URIs.
var refPattern = regexp.MustCompile(`^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$`)

// Lock defines the v2 harness product lock contract.
type Lock struct {
	Schema           string                  `json:"schema"`
	ID               string                  `json:"id,omitempty"`
	Platforms        []PlatformRequirement   `json:"platforms,omitempty"`
	Budget           LockBudget              `json:"budget,omitempty"`
	Components       []LockedComponent       `json:"components,omitempty"`
	Assets           []LockedAsset           `json:"assets,omitempty"`
	Secrets          []SecretContract        `json:"secrets,omitempty"`
	ToolFingerprints []ToolSchemaFingerprint `json:"tool_fingerprints,omitempty"`
}

// PlatformRequirement defines an execution target requirement.
type PlatformRequirement struct {
	OS      string `json:"os,omitempty"`
	Arch    string `json:"arch,omitempty"`
	Variant string `json:"variant,omitempty"`
}

// LockBudget specifies token, memory, and concurrency constraints.
type LockBudget struct {
	ContextTokens int `json:"context_tokens,omitempty"`
	MemoryMiB     int `json:"memory_mib,omitempty"`
	Workers       int `json:"workers,omitempty"`
}

// LockedComponent defines a resolved, immutable component specification.
type LockedComponent struct {
	ID       string   `json:"id"`
	Version  string   `json:"version"`
	Digest   string   `json:"digest"`
	Source   string   `json:"source"`
	Provider string   `json:"provider,omitempty"`
	Provides []string `json:"provides,omitempty"`
}

// LockedAsset defines a locked configuration asset, prompt, or file reference.
type LockedAsset struct {
	Name   string `json:"name"`
	Kind   string `json:"kind"`
	Ref    string `json:"ref,omitempty"`
	Digest string `json:"digest,omitempty"`
	Value  string `json:"value,omitempty"`
}

// SecretContract defines an opaque reference to an injected secret.
type SecretContract struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"`
	Ref      string `json:"ref,omitempty"`
	Value    string `json:"value,omitempty"`
	Provider string `json:"provider,omitempty"`
}

// ToolSchemaFingerprint defines the verified schema hash for an MCP or harness tool.
type ToolSchemaFingerprint struct {
	Server     string `json:"server"`
	Tool       string `json:"tool"`
	SchemaHash string `json:"schema_hash"`
}

// Parse parses JSON data into a Lock and validates that its schema is Schema (v2).
func Parse(data []byte) (*Lock, error) {
	var l Lock
	if err := json.Unmarshal(data, &l); err != nil {
		return nil, fmt.Errorf("parse lock: %w", err)
	}
	if l.Schema != Schema {
		return nil, fmt.Errorf("unsupported lock schema %q: expected %q", l.Schema, Schema)
	}
	return &l, nil
}

// CanonicalID canonicalizes the Lock struct according to RFC 8785 (sorted keys,
// LF line endings, no CRLF) without its ID field, and returns "sha256:<hex>".
// It is deterministic and platform-neutral.
func CanonicalID(l *Lock) (string, error) {
	if l == nil {
		return "", fmt.Errorf("nil lock")
	}
	canon, err := Canonicalize(l)
	if err != nil {
		return "", fmt.Errorf("canonicalize lock: %w", err)
	}
	sum := sha256.Sum256(canon)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// ValidateSecretContracts validates that any asset or secret contract with
// Kind == "secret" has Value == "" (returning an error formatted with
// SecretPlaintextLeak if non-empty). It also validates that Ref matches
// ^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$.
func ValidateSecretContracts(l *Lock) error {
	if l == nil {
		return fmt.Errorf("nil lock")
	}
	for _, asset := range l.Assets {
		if asset.Kind == "secret" {
			if asset.Value != "" {
				return fmt.Errorf("%s: locked asset %q must not contain plaintext value", SecretPlaintextLeak, asset.Name)
			}
			if !refPattern.MatchString(asset.Ref) {
				return fmt.Errorf("locked asset %q secret ref %q does not match required URI scheme", asset.Name, asset.Ref)
			}
		}
	}
	for _, secret := range l.Secrets {
		if secret.Kind == "secret" {
			if secret.Value != "" {
				return fmt.Errorf("%s: secret contract %q must not contain plaintext value", SecretPlaintextLeak, secret.Name)
			}
		}
		if secret.Kind == "secret" || secret.Ref != "" {
			if !refPattern.MatchString(secret.Ref) {
				return fmt.Errorf("secret contract %q ref %q does not match required URI scheme", secret.Name, secret.Ref)
			}
		}
	}
	return nil
}

// Canonicalize returns RFC 8785 canonical JSON bytes for any JSON-compatible
// value or Lock struct.
func Canonicalize(v any) ([]byte, error) {
	if v == nil {
		return []byte("null"), nil
	}
	switch val := v.(type) {
	case *Lock:
		if val == nil {
			return []byte("null"), nil
		}
		norm := cloneAndNormalizeLock(val)
		raw, err := json.Marshal(norm)
		if err != nil {
			return nil, err
		}
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.UseNumber()
		var obj map[string]any
		if err := dec.Decode(&obj); err != nil {
			return nil, err
		}
		delete(obj, "id")
		return canonicalizeValue(obj)
	case Lock:
		return Canonicalize(&val)
	case []byte:
		dec := json.NewDecoder(strings.NewReader(string(val)))
		dec.UseNumber()
		var parsed any
		if err := dec.Decode(&parsed); err != nil {
			return nil, err
		}
		return canonicalizeValue(normalizeAny(parsed))
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("canonicalize marshal: %w", err)
		}
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.UseNumber()
		var parsed any
		if err := dec.Decode(&parsed); err != nil {
			return nil, fmt.Errorf("canonicalize decode: %w", err)
		}
		return canonicalizeValue(normalizeAny(parsed))
	}
}

func normalizeLF(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return s
}

func normalizeAny(v any) any {
	if v == nil {
		return nil
	}
	switch val := v.(type) {
	case string:
		return normalizeLF(val)
	case []any:
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = normalizeAny(item)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(val))
		for k, item := range val {
			out[k] = normalizeAny(item)
		}
		return out
	default:
		return val
	}
}

func cloneAndNormalizeLock(l *Lock) *Lock {
	out := &Lock{
		Schema: normalizeLF(l.Schema),
		Budget: LockBudget{
			ContextTokens: l.Budget.ContextTokens,
			MemoryMiB:     l.Budget.MemoryMiB,
			Workers:       l.Budget.Workers,
		},
	}
	if l.Platforms != nil {
		out.Platforms = make([]PlatformRequirement, len(l.Platforms))
		for i, p := range l.Platforms {
			out.Platforms[i] = PlatformRequirement{
				OS:      normalizeLF(p.OS),
				Arch:    normalizeLF(p.Arch),
				Variant: normalizeLF(p.Variant),
			}
		}
	}
	if l.Components != nil {
		out.Components = make([]LockedComponent, len(l.Components))
		for i, c := range l.Components {
			out.Components[i] = LockedComponent{
				ID:       normalizeLF(c.ID),
				Version:  normalizeLF(c.Version),
				Digest:   normalizeLF(c.Digest),
				Source:   normalizeLF(c.Source),
				Provider: normalizeLF(c.Provider),
			}
			if c.Provides != nil {
				out.Components[i].Provides = make([]string, len(c.Provides))
				for j, p := range c.Provides {
					out.Components[i].Provides[j] = normalizeLF(p)
				}
			}
		}
	}
	if l.Assets != nil {
		out.Assets = make([]LockedAsset, len(l.Assets))
		for i, a := range l.Assets {
			out.Assets[i] = LockedAsset{
				Name:   normalizeLF(a.Name),
				Kind:   normalizeLF(a.Kind),
				Ref:    normalizeLF(a.Ref),
				Digest: normalizeLF(a.Digest),
				Value:  normalizeLF(a.Value),
			}
		}
	}
	if l.Secrets != nil {
		out.Secrets = make([]SecretContract, len(l.Secrets))
		for i, s := range l.Secrets {
			out.Secrets[i] = SecretContract{
				Name:     normalizeLF(s.Name),
				Kind:     normalizeLF(s.Kind),
				Ref:      normalizeLF(s.Ref),
				Value:    normalizeLF(s.Value),
				Provider: normalizeLF(s.Provider),
			}
		}
	}
	if l.ToolFingerprints != nil {
		out.ToolFingerprints = make([]ToolSchemaFingerprint, len(l.ToolFingerprints))
		for i, tf := range l.ToolFingerprints {
			out.ToolFingerprints[i] = ToolSchemaFingerprint{
				Server:     normalizeLF(tf.Server),
				Tool:       normalizeLF(tf.Tool),
				SchemaHash: normalizeLF(tf.SchemaHash),
			}
		}
	}
	return out
}

func canonicalizeValue(v any) ([]byte, error) {
	var b strings.Builder
	if err := writeCanonicalValue(&b, v); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

func writeCanonicalValue(b *strings.Builder, v any) error {
	if v == nil {
		b.WriteString("null")
		return nil
	}
	switch val := v.(type) {
	case bool:
		if val {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
		return nil
	case string:
		canonicalizeString(val, b)
		return nil
	case json.Number:
		s := val.String()
		if s == "" {
			b.WriteString("0")
		} else {
			b.WriteString(s)
		}
		return nil
	case int:
		fmt.Fprintf(b, "%d", val)
		return nil
	case int8:
		fmt.Fprintf(b, "%d", val)
		return nil
	case int16:
		fmt.Fprintf(b, "%d", val)
		return nil
	case int32:
		fmt.Fprintf(b, "%d", val)
		return nil
	case int64:
		fmt.Fprintf(b, "%d", val)
		return nil
	case uint:
		fmt.Fprintf(b, "%d", val)
		return nil
	case uint8:
		fmt.Fprintf(b, "%d", val)
		return nil
	case uint16:
		fmt.Fprintf(b, "%d", val)
		return nil
	case uint32:
		fmt.Fprintf(b, "%d", val)
		return nil
	case uint64:
		fmt.Fprintf(b, "%d", val)
		return nil
	case float64:
		if val == float64(int64(val)) && !strings.Contains(fmt.Sprintf("%v", val), "e") {
			fmt.Fprintf(b, "%d", int64(val))
		} else {
			fmt.Fprintf(b, "%g", val)
		}
		return nil
	case []any:
		b.WriteByte('[')
		for i, elem := range val {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeCanonicalValue(b, elem); err != nil {
				return err
			}
		}
		b.WriteByte(']')
		return nil
	case map[string]any:
		b.WriteByte('{')
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			canonicalizeString(k, b)
			b.WriteByte(':')
			if err := writeCanonicalValue(b, val[k]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
		return nil
	default:
		raw, err := json.Marshal(val)
		if err != nil {
			return fmt.Errorf("canonicalize unsupported type %T: %w", v, err)
		}
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.UseNumber()
		var redecoded any
		if err := dec.Decode(&redecoded); err != nil {
			return err
		}
		return writeCanonicalValue(b, normalizeAny(redecoded))
	}
}

func canonicalizeString(s string, b *strings.Builder) {
	b.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if c < 0x20 {
				fmt.Fprintf(b, `\u00%02x`, c)
			} else {
				b.WriteByte(c)
			}
		}
	}
	b.WriteByte('"')
}
