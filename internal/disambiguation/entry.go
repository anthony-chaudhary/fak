package disambiguation

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

// EntrySchemaVersion is the only entry schema this package understands.
//
// Version 1 is fail-closed: ParseEntry rejects a different schema value instead
// of guessing that a future or historical record has the current meaning.
const EntrySchemaVersion = "fak-disambiguation-entry/1"

// FreshnessVerdict is the four-state result of checking an entry's cited public
// sources. Unknown means evidence could not be obtained; invalid means evidence
// was obtained but could not be interpreted as a valid witness.
type FreshnessVerdict string

const (
	FreshnessFresh   FreshnessVerdict = "fresh"
	FreshnessStale   FreshnessVerdict = "stale"
	FreshnessUnknown FreshnessVerdict = "unknown"
	FreshnessInvalid FreshnessVerdict = "invalid"
)

// LifecycleClass states whether the entry describes current authority or
// material retained for a narrower historical or research purpose.
type LifecycleClass string

const (
	LifecycleCurrent   LifecycleClass = "current"
	LifecycleVersioned LifecycleClass = "versioned"
	LifecycleResearch  LifecycleClass = "research"
	LifecycleArchived  LifecycleClass = "archived"
)

// ActivationMode is the normalized activation posture carried with an entry.
type ActivationMode string

const (
	RolloutOff    ActivationMode = "off"
	RolloutShadow ActivationMode = "shadow"
	RolloutOn     ActivationMode = "on"
)

// Entry is one canonical distinction in fak's disambiguation index.
//
// The groups are values rather than open maps so producers and consumers share
// one shape. ParseEntry is the persisted-wire admission seam; programmatically
// constructed entries use Validate before entering a generated index.
type Entry struct {
	Schema     string          `json:"schema"`
	Identity   Identity        `json:"identity"`
	Definition string          `json:"definition"`
	Contrasts  []Contrast      `json:"contrasts"`
	Scope      Scope           `json:"scope"`
	Owner      Owner           `json:"owner"`
	Sources    []SourceWitness `json:"sources"`
	Freshness  Freshness       `json:"freshness"`
	Lifecycle  Lifecycle       `json:"lifecycle"`
}

// Identity names the canonical term and every declared alias. Aliases is
// required on the wire even when empty so absence cannot be mistaken for a
// producer that forgot to classify aliases.
type Identity struct {
	CanonicalTerm string   `json:"canonical_term"`
	Aliases       []string `json:"aliases"`
}

// Contrast names one concept this entry must not be conflated with and explains
// the distinction. Cross-entry existence and symmetry checks belong to the
// index validator; this record validator guarantees the pair is expressible.
type Contrast struct {
	CanonicalTerm string `json:"canonical_term"`
	Explanation   string `json:"explanation"`
}

// Scope qualifies which public surface gives the canonical term this meaning.
// Kind names the dimension (for example product, package, cli, runtime,
// protocol, or operator); Value names the member within that dimension.
type Scope struct {
	Kind  string `json:"kind"`
	Value string `json:"value"`
}

// Owner is the accountable fak leaf and dispatch lane for repairing drift.
type Owner struct {
	Leaf string `json:"leaf"`
	Lane string `json:"lane"`
}

// SourceWitness identifies one public source supporting the entry. Locator is
// repository-relative or public-metadata-relative; Revision pins the source
// state. Source existence and public-safety probes are separate admission rungs.
type SourceWitness struct {
	Kind     string `json:"kind"`
	Locator  string `json:"locator"`
	Revision string `json:"revision"`
}

// Freshness records the last source-check outcome and the public probe that
// produced it. CheckedAt is RFC 3339 so JSON consumers do not have to guess a
// locale or timezone. ReasonCode is required for every verdict, including fresh.
type Freshness struct {
	Verdict    FreshnessVerdict `json:"verdict"`
	ReasonCode string           `json:"reason_code"`
	CheckedAt  string           `json:"checked_at"`
	Probe      string           `json:"probe"`
}

// Lifecycle separates the entry's authority class from its rollout posture.
type Lifecycle struct {
	Class   LifecycleClass `json:"class"`
	Rollout ActivationMode `json:"rollout"`
}

// ParseEntry decodes exactly one v1 JSON record. It rejects unknown fields at
// every nesting level, trailing documents, unknown schema versions, and entries
// that fail Validate.
func ParseEntry(data []byte) (Entry, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()

	var entry Entry
	if err := dec.Decode(&entry); err != nil {
		return Entry{}, fmt.Errorf("decode disambiguation entry: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return Entry{}, fmt.Errorf("decode disambiguation entry: %w", err)
	}
	if err := entry.Validate(); err != nil {
		return Entry{}, fmt.Errorf("invalid disambiguation entry: %w", err)
	}
	return entry, nil
}

// Validate enforces the complete v1 entry contract without consulting ambient
// repository, network, or clock state.
func (e Entry) Validate() error {
	if e.Schema != EntrySchemaVersion {
		return fmt.Errorf("unsupported schema %q (want %q)", e.Schema, EntrySchemaVersion)
	}
	if err := requireText("identity.canonical_term", e.Identity.CanonicalTerm); err != nil {
		return err
	}
	if e.Identity.Aliases == nil {
		return errors.New("identity.aliases is required (use [] when there are no aliases)")
	}
	for i, alias := range e.Identity.Aliases {
		if err := requireText(fmt.Sprintf("identity.aliases[%d]", i), alias); err != nil {
			return err
		}
	}
	if err := requireText("definition", e.Definition); err != nil {
		return err
	}
	if len(e.Contrasts) == 0 {
		return errors.New("at least one contrast is required")
	}
	for i, contrast := range e.Contrasts {
		if err := requireText(fmt.Sprintf("contrasts[%d].canonical_term", i), contrast.CanonicalTerm); err != nil {
			return err
		}
		if err := requireText(fmt.Sprintf("contrasts[%d].explanation", i), contrast.Explanation); err != nil {
			return err
		}
	}
	if err := requireText("scope.kind", e.Scope.Kind); err != nil {
		return err
	}
	if err := requireText("scope.value", e.Scope.Value); err != nil {
		return err
	}
	if err := requireText("owner.leaf", e.Owner.Leaf); err != nil {
		return err
	}
	if err := requireText("owner.lane", e.Owner.Lane); err != nil {
		return err
	}
	if len(e.Sources) == 0 {
		return errors.New("at least one source witness is required")
	}
	for i, source := range e.Sources {
		if err := requireText(fmt.Sprintf("sources[%d].kind", i), source.Kind); err != nil {
			return err
		}
		if err := requireText(fmt.Sprintf("sources[%d].locator", i), source.Locator); err != nil {
			return err
		}
		if err := requireText(fmt.Sprintf("sources[%d].revision", i), source.Revision); err != nil {
			return err
		}
	}
	switch e.Freshness.Verdict {
	case FreshnessFresh, FreshnessStale, FreshnessUnknown, FreshnessInvalid:
	default:
		return fmt.Errorf("freshness.verdict %q is not one of fresh, stale, unknown, invalid", e.Freshness.Verdict)
	}
	if err := requireText("freshness.reason_code", e.Freshness.ReasonCode); err != nil {
		return err
	}
	if err := requireText("freshness.checked_at", e.Freshness.CheckedAt); err != nil {
		return err
	}
	if _, err := time.Parse(time.RFC3339, e.Freshness.CheckedAt); err != nil {
		return fmt.Errorf("freshness.checked_at must be RFC3339: %w", err)
	}
	if err := requireText("freshness.probe", e.Freshness.Probe); err != nil {
		return err
	}
	switch e.Lifecycle.Class {
	case LifecycleCurrent, LifecycleVersioned, LifecycleResearch, LifecycleArchived:
	default:
		return fmt.Errorf("lifecycle.class %q is not one of current, versioned, research, archived", e.Lifecycle.Class)
	}
	switch e.Lifecycle.Rollout {
	case RolloutOff, RolloutShadow, RolloutOn:
	default:
		return fmt.Errorf("lifecycle.rollout %q is not one of off, shadow, on", e.Lifecycle.Rollout)
	}
	return nil
}

func requireText(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s is required", field)
	}
	if strings.TrimSpace(value) != value {
		return fmt.Errorf("%s must not have leading or trailing whitespace", field)
	}
	return nil
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	err := dec.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return errors.New("trailing JSON")
}
