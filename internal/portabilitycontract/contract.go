// Package portabilitycontract defines the versioned, transport-neutral contract for
// moving managed-agent state. Unknown object kinds remain opaque and inert.
package portabilitycontract

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Schema identifies the versioned portability specification wire format.
const Schema = "fak.portability/v1"

// Scope defines the visibility and lifecycle boundary of a managed object.
type Scope string

const (
	// ScopePublic indicates an object visible across all organizations and environments.
	ScopePublic Scope = "public"
	// ScopeCorporate indicates an object shared across an entire enterprise organization.
	ScopeCorporate Scope = "corporate"
	// ScopeTeam indicates an object scoped to a specific team.
	ScopeTeam Scope = "team"
	// ScopeProject indicates an object scoped to a single project or workspace.
	ScopeProject Scope = "project"
	// ScopeUser indicates an object scoped to an individual user across machines.
	ScopeUser Scope = "user"
	// ScopeMachine indicates host-local state that must not be transported across nodes.
	ScopeMachine Scope = "machine"
)

var scopeRank = map[Scope]int{ScopePublic: 0, ScopeCorporate: 1, ScopeTeam: 2, ScopeProject: 3, ScopeUser: 4, ScopeMachine: 5}

// CompareScope compares two scopes by precedence rank. It returns a positive integer
// if a has higher rank than b, negative if lower, or zero if equal.
func CompareScope(a, b Scope) int { return scopeRank[a] - scopeRank[b] }

// ResolvePrecedence returns a copy of objects sorted stably by descending scope rank,
// descending explicit precedence, and ascending stable ID.
func ResolvePrecedence(objects []Object) []Object {
	out := append([]Object(nil), objects...)
	sort.SliceStable(out, func(i, j int) bool {
		ri, rj := scopeRank[out[i].Scope], scopeRank[out[j].Scope]
		if ri != rj {
			return ri > rj
		}
		if out[i].Precedence != out[j].Precedence {
			return out[i].Precedence > out[j].Precedence
		}
		return out[i].StableID < out[j].StableID
	})
	return out
}

// Sensitivity classifies the confidentiality level of an object for redaction during transport.
type Sensitivity string

const (
	// SensitivityPublic designates non-sensitive data suitable for unconstrained transport.
	SensitivityPublic Sensitivity = "public"
	// SensitivityInternal designates data restricted to authorized organization members.
	SensitivityInternal Sensitivity = "internal"
	// SensitivityConfidential designates restricted data requiring elevated authorization.
	SensitivityConfidential Sensitivity = "confidential"
	// SensitivitySecret designates credentials and secrets that must not traverse portable exports.
	SensitivitySecret Sensitivity = "secret"
)

// Provenance tracks the authoring agent, source artifact, creation time, and derivation lineage.
type Provenance struct {
	Producer    string   `json:"producer"`
	Source      string   `json:"source,omitempty"`
	Created     string   `json:"created"`
	DerivedFrom []string `json:"derived_from,omitempty"`
}

// Dependency specifies a prerequisite object ID and whether it is optional.
type Dependency struct {
	ID        string `json:"id"`
	ContentID string `json:"content_id,omitempty"`
	Optional  bool   `json:"optional,omitempty"`
}

// Compatibility declares reader version requirements and supported runtime harnesses.
type Compatibility struct {
	MinReader string   `json:"min_reader"`
	Harnesses []string `json:"harnesses,omitempty"`
	Features  []string `json:"features,omitempty"`
}

// Signature provides cryptographic non-repudiation and content integrity verification.
type Signature struct {
	Algorithm string `json:"algorithm"`
	KeyID     string `json:"key_id"`
	Value     string `json:"value"`
}

// Migration defines version transition metadata and reversibility between schema revisions.
type Migration struct {
	From       string `json:"from"`
	To         string `json:"to"`
	ID         string `json:"id"`
	Reversible bool   `json:"reversible"`
}

// Receipt records the operational audit record for a transaction mutation.
type Receipt struct {
	TransactionID  string `json:"transaction_id"`
	IdempotencyKey string `json:"idempotency_key,omitempty"`
	Action         string `json:"action"`
	Status         string `json:"status"`
	Before         string `json:"before,omitempty"`
	After          string `json:"after,omitempty"`
	At             string `json:"at"`
}

// Extension encapsulates custom namespaced metadata, marking whether unrecognized extensions are critical.
type Extension struct {
	Critical bool            `json:"critical"`
	Data     json.RawMessage `json:"data"`
}

// Object represents an atomic unit of managed agent state, policy, or configuration.
type Object struct {
	Schema        string               `json:"schema"`
	Type          string               `json:"type"`
	TypeVersion   string               `json:"type_version"`
	StableID      string               `json:"stable_id"`
	ContentID     string               `json:"content_id"`
	Provenance    Provenance           `json:"provenance"`
	Dependencies  []Dependency         `json:"dependencies,omitempty"`
	Scope         Scope                `json:"scope"`
	Sensitivity   Sensitivity          `json:"sensitivity"`
	Compatibility Compatibility        `json:"compatibility"`
	Precedence    int                  `json:"precedence,omitempty"`
	Signatures    []Signature          `json:"signatures,omitempty"`
	Migration     *Migration           `json:"migration,omitempty"`
	Receipts      []Receipt            `json:"receipts,omitempty"`
	Extensions    map[string]Extension `json:"extensions,omitempty"`
	Payload       json.RawMessage      `json:"payload"`
}

// Known reports whether the object type is recognized by the kernel runtime.
func (o Object) Known() bool {
	switch o.Type {
	case "skill", "policy", "session", "loop", "model-binding", "instruction", "hook", "mcp-server", "plugin", "account", "secret-reference":
		return true
	}
	return false
}

// Active reports whether the object is recognized and admitted for runtime execution.
func (o Object) Active() bool { return o.Known() }

// CanonicalContentID computes the deterministic SHA-256 hash of the normalized object payload.
func (o Object) CanonicalContentID() (string, error) {
	var payload any
	if err := json.Unmarshal(o.Payload, &payload); err != nil {
		return "", err
	}
	b, err := json.Marshal(struct {
		Type, Version string
		Payload       any
	}{o.Type, o.TypeVersion, payload})
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:]), nil
}

// Validate checks schema conformance, required identifiers, scope validity, payload format,
// content digest fidelity, and critical extension namespaces.
func (o Object) Validate() error {
	if o.Schema != Schema {
		return fmt.Errorf("unsupported portability schema version: %q", o.Schema)
	}
	if o.Type == "" || o.TypeVersion == "" || o.StableID == "" {
		return errors.New("IDENTITY_REQUIRED")
	}
	if _, ok := scopeRank[o.Scope]; !ok {
		return fmt.Errorf("SCOPE_INVALID: %q", o.Scope)
	}
	if !json.Valid(o.Payload) {
		return errors.New("PAYLOAD_INVALID")
	}
	cid, e := o.CanonicalContentID()
	if e != nil {
		return e
	}
	if o.ContentID != cid {
		return fmt.Errorf("CONTENT_ID_MISMATCH: want %s", cid)
	}
	for n, x := range o.Extensions {
		if x.Critical && !strings.HasPrefix(n, "fak.") {
			return fmt.Errorf("UNKNOWN_CRITICAL_EXTENSION: %s", n)
		}
	}
	return nil
}

// Collection groups related objects that share dependencies and lifecycle scope.
type Collection struct {
	Schema       string               `json:"schema"`
	StableID     string               `json:"stable_id"`
	Version      string               `json:"version"`
	Objects      []Object             `json:"objects"`
	Dependencies []Dependency         `json:"dependencies,omitempty"`
	Extensions   map[string]Extension `json:"extensions,omitempty"`
}

// Context represents the active execution environment, mapping active collections and scope ordering.
type Context struct {
	Schema            string            `json:"schema"`
	StableID          string            `json:"stable_id"`
	Machine           string            `json:"machine,omitempty"`
	ActiveCollections []string          `json:"active_collections"`
	Selected          map[string]string `json:"selected,omitempty"`
	ScopeOrder        []Scope           `json:"scope_order"`
}

// Package encapsulates a self-contained export bundle containing collections, compatibility constraints, and signatures.
type Package struct {
	Schema        string               `json:"schema"`
	StableID      string               `json:"stable_id"`
	PackageID     string               `json:"package_id"`
	Collections   []Collection         `json:"collections"`
	Compatibility Compatibility        `json:"compatibility"`
	Signatures    []Signature          `json:"signatures,omitempty"`
	Extensions    map[string]Extension `json:"extensions,omitempty"`
	Local         map[string]any       `json:"-"`
}

// Channel defines a communication endpoint and its transport capabilities.
type Channel struct {
	Schema        string               `json:"schema"`
	StableID      string               `json:"stable_id"`
	Kind          string               `json:"kind"`
	Endpoint      string               `json:"endpoint,omitempty"`
	Capabilities  []string             `json:"capabilities"`
	Compatibility Compatibility        `json:"compatibility"`
	Extensions    map[string]Extension `json:"extensions,omitempty"`
}

// Degradation captures fidelity or capability loss encountered during harness translation.
type Degradation struct {
	ObjectID    string `json:"object_id"`
	Feature     string `json:"feature"`
	Severity    string `json:"severity"`
	MeaningLost string `json:"meaning_lost"`
	Fallback    string `json:"fallback,omitempty"`
}

// TranslationReport summarizes the outcome of cross-harness translation, indicating whether conversion was exact.
type TranslationReport struct {
	SourceHarness string        `json:"source_harness"`
	TargetHarness string        `json:"target_harness"`
	Exact         bool          `json:"exact"`
	Degradations  []Degradation `json:"degradations"`
}

// Operation represents a single state mutation step within a transaction.
type Operation struct {
	Kind         string `json:"kind"`
	CollectionID string `json:"collection_id,omitempty"`
	From         string `json:"from,omitempty"`
	To           string `json:"to,omitempty"`
	Strategy     string `json:"strategy,omitempty"`
}

// Transaction specifies an idempotent atomic state mutation with recovery and rollback guarantees.
type Transaction struct {
	Schema         string               `json:"schema"`
	StableID       string               `json:"stable_id"`
	Mode           string               `json:"mode"`
	IdempotencyKey string               `json:"idempotency_key"`
	ExpectedState  string               `json:"expected_state"`
	Operations     []Operation          `json:"operations"`
	PreviewHash    string               `json:"preview_hash"`
	Recovery       string               `json:"recovery"`
	Rollback       []Operation          `json:"rollback"`
	Status         string               `json:"status"`
	Receipts       []Receipt            `json:"receipts,omitempty"`
	Degradation    TranslationReport    `json:"degradation"`
	Extensions     map[string]Extension `json:"extensions,omitempty"`
}

// Portable returns a sanitized copy of a package, stripping local paths, signatures,
// receipts, machine-scoped objects, and secret references to ensure safe export.
func Portable(p Package) Package {
	q := p
	q.Local = nil
	q.Signatures = nil
	q.Collections = make([]Collection, 0, len(p.Collections))
	for _, c := range p.Collections {
		nc := c
		nc.Objects = nil
		for _, o := range c.Objects {
			if o.Sensitivity == SensitivitySecret || o.Scope == ScopeMachine {
				continue
			}
			o.Signatures = nil
			o.Receipts = nil
			nc.Objects = append(nc.Objects, o)
		}
		q.Collections = append(q.Collections, nc)
	}
	return q
}

// PackageIdentity computes the deterministic SHA-256 content digest of a package's portable representation.
func PackageIdentity(p Package) (string, error) {
	q := Portable(p)
	q.PackageID = ""
	q.StableID = ""
	b, e := json.Marshal(q)
	if e != nil {
		return "", e
	}
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:]), nil
}

// Validate checks the package schema version, validates all contained objects, and verifies the package identity digest.
func (p Package) Validate() error {
	if p.Schema != Schema {
		return errors.New("unsupported portability schema version")
	}
	for _, c := range p.Collections {
		for _, o := range c.Objects {
			if err := o.Validate(); err != nil {
				return fmt.Errorf("object %s: %w", o.StableID, err)
			}
		}
	}
	id, e := PackageIdentity(p)
	if e != nil {
		return e
	}
	if id != p.PackageID {
		return fmt.Errorf("PACKAGE_ID_MISMATCH: want %s", id)
	}
	return nil
}

// RoundTrip decodes package JSON and re-encodes it with consistent indentation for fidelity verification.
func RoundTrip(data []byte) ([]byte, error) {
	var p Package
	if e := json.Unmarshal(data, &p); e != nil {
		return nil, e
	}
	return json.MarshalIndent(p, "", "  ")
}

// Preview validates transaction preconditions against the current state and computes the preview hash.
func Preview(t Transaction, current string) (Transaction, error) {
	if t.Mode != "apply" && t.Mode != "switch" && t.Mode != "merge" {
		return t, errors.New("MODE_INVALID")
	}
	if t.IdempotencyKey == "" || t.Recovery == "" || len(t.Rollback) == 0 {
		return t, errors.New("RECOVERY_CONTRACT_REQUIRED")
	}
	if t.ExpectedState != current {
		return t, errors.New("STATE_CONFLICT")
	}
	cp := t
	cp.PreviewHash = ""
	cp.Status = ""
	cp.Receipts = nil
	b, _ := json.Marshal(cp)
	h := sha256.Sum256(b)
	t.PreviewHash = "sha256:" + hex.EncodeToString(h[:])
	t.Status = "previewed"
	return t, nil
}

// Store maintains state continuity and records applied transaction idempotency keys.
type Store struct {
	State   string
	Applied map[string]string
}

// Apply executes a previewed transaction against the store, verifying state continuity and idempotency.
func (s *Store) Apply(t Transaction) (Receipt, error) {
	if s.Applied == nil {
		s.Applied = map[string]string{}
	}
	if after, ok := s.Applied[t.IdempotencyKey]; ok {
		return Receipt{TransactionID: t.StableID, IdempotencyKey: t.IdempotencyKey, Action: t.Mode, Status: "already-applied", Before: s.State, After: after}, nil
	}
	p, e := Preview(t, s.State)
	if e != nil {
		return Receipt{}, e
	}
	if p.PreviewHash != t.PreviewHash {
		return Receipt{}, errors.New("PREVIEW_CHANGED")
	}
	before := s.State
	h := sha256.Sum256([]byte(t.PreviewHash + ":" + t.Mode))
	after := "state:" + hex.EncodeToString(h[:])
	s.State = after
	s.Applied[t.IdempotencyKey] = after
	return Receipt{TransactionID: t.StableID, IdempotencyKey: t.IdempotencyKey, Action: t.Mode, Status: "applied", Before: before, After: after}, nil
}

// Recover reverts the store state to the pre-transaction state specified in the receipt.
func (s *Store) Recover(r Receipt) error {
	if r.Status != "applied" {
		return nil
	}
	if s.State != r.After {
		return errors.New("RECOVERY_STATE_CONFLICT")
	}
	s.State = r.Before
	delete(s.Applied, r.IdempotencyKey)
	return nil
}

// Explain formats a human-readable diagnostic breakdown of a package, its collections, and precedence order.
func Explain(p Package) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Portability package %s\nSchema: %s\nIdentity: %s\n", p.StableID, p.Schema, p.PackageID)
	for _, c := range p.Collections {
		fmt.Fprintf(&b, "\nCollection %s (%d objects)\n", c.StableID, len(c.Objects))
		for _, o := range ResolvePrecedence(c.Objects) {
			state := "active"
			if !o.Active() {
				state = "INERT (unknown type; preserved)"
			}
			fmt.Fprintf(&b, "- %s: %s@%s, scope=%s, sensitivity=%s, precedence=%d, %s\n", o.StableID, o.Type, o.TypeVersion, o.Scope, o.Sensitivity, o.Precedence, state)
		}
	}
	fmt.Fprintf(&b, "\nTransport identity omits signatures, receipts, machine-scope and secret objects.\nPrecedence: machine > user > project > team > corporate > public; explicit precedence then stable ID break ties.\n")
	return b.String()
}

// EqualJSON reports whether two byte slices contain semantically equivalent JSON structures.
func EqualJSON(a, b []byte) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	return bytes.Equal(mustJSON(x), mustJSON(y))
}
func mustJSON(v any) []byte { b, _ := json.Marshal(v); return b }
