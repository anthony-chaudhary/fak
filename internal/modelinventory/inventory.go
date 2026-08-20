// Package modelinventory normalizes model artifact and runtime observations into
// a deterministic, credential-free candidate inventory.
package modelinventory

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const Schema = "fak.model-inventory/1"

type SourceKind string

const (
	SourceProvider SourceKind = "provider"
	SourceLocal    SourceKind = "local"
)

type WitnessKind string

const (
	EvidenceProbe               WitnessKind = "probe"
	EvidenceOperatorAttestation WitnessKind = "operator-attestation"
	EvidenceArtifactMetadata    WitnessKind = "artifact-metadata"
)

type Code string

const (
	CodeAsOfRequired          Code = "AS_OF_REQUIRED"
	CodeCredentialMaterial    Code = "CREDENTIAL_MATERIAL"
	CodeDuplicateEntry        Code = "DUPLICATE_ENTRY"
	CodeEvidenceFuture        Code = "EVIDENCE_FUTURE"
	CodeEvidenceMalformed     Code = "EVIDENCE_MALFORMED"
	CodeEvidenceScopeMismatch Code = "EVIDENCE_SCOPE_MISMATCH"
	CodeEvidenceStale         Code = "EVIDENCE_STALE"
	CodeEvidenceUnknown       Code = "EVIDENCE_UNKNOWN"
	CodeInvalidIdentity       Code = "INVALID_IDENTITY"
	CodeInvalidJSON           Code = "INVALID_JSON"
	CodeInvalidSource         Code = "INVALID_SOURCE"
	CodeMalformedFact         Code = "MALFORMED_FACT"
	CodeMissingFact           Code = "MISSING_FACT"
	CodeMissingIdentity       Code = "MISSING_IDENTITY"
	CodePlatformContradiction Code = "PLATFORM_CONTRADICTION"
	CodeSchemaMismatch        Code = "SCHEMA_MISMATCH"
)

// Value is a closed JSON scalar union. Exactly one arm must be set.
type Value struct {
	Bool    *bool   `json:"bool,omitempty"`
	Integer *int64  `json:"integer,omitempty"`
	Text    *string `json:"text,omitempty"`
}

func Bool(v bool) Value     { return Value{Bool: &v} }
func Integer(v int64) Value { return Value{Integer: &v} }
func Text(v string) Value   { return Value{Text: &v} }

// Witness binds one fact to its evidence source and freshness envelope. Source
// is a non-secret artifact/probe reference, never an authorization value.
type Witness struct {
	Kind       WitnessKind `json:"kind"`
	Source     string      `json:"source"`
	ObservedAt string      `json:"observed_at"`
	ExpiresAt  string      `json:"expires_at"`
}

type Fact struct {
	Name      string    `json:"name"`
	Value     Value     `json:"value"`
	Witnesses []Witness `json:"witnesses"`
}

// EvidenceSet keeps serving, platform, policy, and capability observations
// separate so later resolution cannot borrow a fact from the wrong domain.
type EvidenceSet struct {
	Availability Fact   `json:"availability"`
	Serving      []Fact `json:"serving"`
	Platform     []Fact `json:"platform"`
	Policy       []Fact `json:"policy"`
	Capabilities []Fact `json:"capabilities"`
}

type Identity struct {
	Source    SourceKind `json:"source"`
	Provider  string     `json:"provider,omitempty"`
	Artifact  string     `json:"artifact"`
	Revision  string     `json:"revision,omitempty"`
	Digest    string     `json:"digest"`
	Format    string     `json:"format"`
	Witnesses []Witness  `json:"witnesses"`
}

type Candidate struct {
	ID       string      `json:"id"`
	Identity Identity    `json:"identity"`
	Evidence EvidenceSet `json:"evidence"`
}

type Inventory struct {
	Schema     string      `json:"schema"`
	AsOf       string      `json:"as_of"`
	Candidates []Candidate `json:"candidates"`
}

// ProviderObservation is an immutable provider/repository revision plus facts
// witnessed independently of repository-card metadata.
type ProviderObservation struct {
	ID               string
	Provider         string
	Repository       string
	Revision         string
	Digest           string
	Format           string
	IdentityEvidence []Witness
	Evidence         EvidenceSet
}

// LocalObservation names a logical, repository-relative artifact. Its digest,
// not its mutable path, is the immutable identity.
type LocalObservation struct {
	ID               string
	Artifact         string
	Digest           string
	Format           string
	IdentityEvidence []Witness
	Evidence         EvidenceSet
}

type Observations struct {
	Providers []ProviderObservation
	Locals    []LocalObservation
}

type Diagnostic struct {
	Code           Code   `json:"code"`
	Candidate      string `json:"candidate,omitempty"`
	Field          string `json:"field"`
	EvidenceSource string `json:"evidence_source,omitempty"`
	Detail         string `json:"detail"`
	Remediation    string `json:"remediation"`
}

type Diagnostics []Diagnostic

func (ds Diagnostics) Error() string {
	ordered := ds.sorted()
	lines := make([]string, 0, len(ordered))
	for _, d := range ordered {
		where := d.Field
		if d.Candidate != "" {
			where = "candidate=" + d.Candidate + " " + where
		}
		if d.EvidenceSource != "" {
			where += " evidence=" + d.EvidenceSource
		}
		lines = append(lines, fmt.Sprintf("%s %s: %s; remediation: %s", d.Code, where, d.Detail, d.Remediation))
	}
	return strings.Join(lines, "\n")
}

func (ds Diagnostics) sorted() Diagnostics {
	out := append(Diagnostics(nil), ds...)
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Candidate != b.Candidate {
			return a.Candidate < b.Candidate
		}
		if a.Field != b.Field {
			return a.Field < b.Field
		}
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		if a.EvidenceSource != b.EvidenceSource {
			return a.EvidenceSource < b.EvidenceSource
		}
		if a.Detail != b.Detail {
			return a.Detail < b.Detail
		}
		return a.Remediation < b.Remediation
	})
	return out
}

func newDiagnostic(code Code, candidate, field, source, detail, remediation string) Diagnostic {
	if credentialLike(candidate) {
		candidate = "<redacted>"
	}
	if credentialLike(source) {
		source = ""
	}
	return Diagnostic{
		Code:           code,
		Candidate:      candidate,
		Field:          field,
		EvidenceSource: source,
		Detail:         detail,
		Remediation:    remediation,
	}
}

func canonicalTime(raw string) string {
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(raw))
	if err != nil {
		return strings.TrimSpace(raw)
	}
	return t.UTC().Format(time.RFC3339)
}
