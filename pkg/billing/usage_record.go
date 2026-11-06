package billing

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"
)

// SchemaVersionV1 is the only record format version this package produces
// and validates. Records evolve additively: later revisions may add
// optional fields, and consumers that decode strictly must expect them;
// breaking changes get a major version.
const SchemaVersionV1 = 1

// Sentinel errors. Every validation failure wraps ErrInvalidRecord, so
// errors.Is(err, ErrInvalidRecord) identifies any malformed record while
// the specific sentinel names the fault.
var (
	// ErrInvalidRecord is the root sentinel for every malformed UsageRecord.
	ErrInvalidRecord = errors.New("billing: invalid usage record")

	// ErrUnknownSchemaVersion marks a record whose schema_version this
	// package does not validate.
	ErrUnknownSchemaVersion = fmt.Errorf("%w: unknown schema_version", ErrInvalidRecord)

	// ErrMissingEventID marks a record without its dedup key.
	ErrMissingEventID = fmt.Errorf("%w: event_id is required", ErrInvalidRecord)

	// ErrMissingModel marks a record that names no model.
	ErrMissingModel = fmt.Errorf("%w: model is required", ErrInvalidRecord)

	// ErrMissingRequestedAt marks a record without its request timestamp.
	ErrMissingRequestedAt = fmt.Errorf("%w: requested_at is required", ErrInvalidRecord)

	// ErrUnknownClassification marks a record whose classification is not
	// a member of the closed vocabulary.
	ErrUnknownClassification = fmt.Errorf("%w: unknown usage classification", ErrInvalidRecord)

	// ErrInconsistentTokens marks token fields that contradict their
	// classification or each other.
	ErrInconsistentTokens = fmt.Errorf("%w: token fields inconsistent with classification", ErrInvalidRecord)

	// ErrMalformedPayload marks serialized input that is not exactly one
	// decodable v1 record: bad JSON, a field outside the v1 schema, or
	// trailing data.
	ErrMalformedPayload = fmt.Errorf("%w: malformed serialized payload", ErrInvalidRecord)

	// ErrAlreadyRecorded is the dedup/replay verdict: a record whose
	// event_id was already recorded arrives again. It is not a record
	// defect, so it deliberately does not wrap ErrInvalidRecord.
	ErrAlreadyRecorded = errors.New("billing: event already recorded")
)

// UsageClassification is the closed vocabulary describing how the usage
// carried by a UsageRecord was established. It is the record's lineage:
// it fixes what the token fields mean, so "unknown" can never be mistaken
// for "known".
type UsageClassification string

const (
	// UsageRecorded means the engine or upstream reported usage for this
	// attempt. At least one token field must carry a reported value.
	UsageRecorded UsageClassification = "recorded"

	// UsageMissing means no usage data is available for this attempt:
	// nothing was reported. All token fields must be absent. A consumer
	// must treat the usage as unknown; what to do about unknown usage is
	// a decision this package does not make.
	UsageMissing UsageClassification = "missing"

	// UsageZeroKnown means the usage is known to be exactly zero (for
	// example the attempt was refused before any token was consumed, and
	// the emitter knows that). Any token field that is present must be
	// zero; fields may be omitted.
	UsageZeroKnown UsageClassification = "zero_known"

	// UsageNotApplicable means the usage concept does not apply to this
	// record (for example a request class that consumes no tokens). All
	// token fields must be absent.
	UsageNotApplicable UsageClassification = "not_applicable"
)

// Classifications is the closed, ordered classification vocabulary.
var Classifications = [...]UsageClassification{
	UsageRecorded,
	UsageMissing,
	UsageZeroKnown,
	UsageNotApplicable,
}

// Valid reports whether c is a member of the closed classification
// vocabulary.
func (c UsageClassification) Valid() bool {
	for _, known := range Classifications {
		if c == known {
			return true
		}
	}
	return false
}

// UsageRecord is the neutral per-request usage record, version v1.
//
// It is produced by a serving engine (or its metering seam) after an
// attempt and consumed by a billing authority. The two sides share no
// code beyond this type.
//
// Privacy is structural: the type has no field that can hold a prompt or
// response body or a credential, and strict decoding
// (DecodeUsageRecord) refuses payloads carrying anything outside the v1
// schema.
//
// Token fields use the reported/unknown distinction: a nil pointer means
// the value was NOT reported (unknown), while a non-nil pointer means the
// value WAS reported — including an explicit zero. Which combinations are
// legal is fixed by Classification and enforced by Validate.
type UsageRecord struct {
	// SchemaVersion tags the record format. Always SchemaVersionV1 for
	// records this package produces or validates.
	SchemaVersion int `json:"schema_version"`

	// EventID is the minted identity of this usage event and the dedup
	// key for replay semantics. Required.
	EventID string `json:"event_id"`

	// RequestID joins the record to the wire-level request (for example
	// the upstream response id). Optional.
	RequestID string `json:"request_id,omitempty"`

	// TenantID is the opaque registry identifier of the tenant the
	// attempt belongs to. Opaque: this package neither mints nor
	// interprets it. Optional.
	TenantID string `json:"tenant_id,omitempty"`

	// KeyID is the opaque registry identifier of the key the attempt was
	// authenticated with. Opaque, as TenantID. Optional.
	KeyID string `json:"key_id,omitempty"`

	// Model is the requested (canonical) model name. Required: a record
	// that names no model cannot be reconciled against any price book.
	Model string `json:"model"`

	// ModelRevision is the revision of the model that actually served the
	// attempt, when the engine distinguishes it from the requested name.
	// Optional.
	ModelRevision string `json:"model_revision,omitempty"`

	// Backend names the upstream backend/provider that served the
	// attempt, when routing made that meaningful. Optional.
	Backend string `json:"backend,omitempty"`

	// Route is the engine route the attempt took, when routes are named.
	// Optional.
	Route string `json:"route,omitempty"`

	// RequestedAt is when the request was received. Required.
	RequestedAt time.Time `json:"requested_at"`

	// CompletedAt is when the attempt finished, when it did. Nil means
	// the attempt did not complete or the completion time was not
	// reported.
	CompletedAt *time.Time `json:"completed_at,omitempty"`

	// PromptTokens is the reported prompt-token count. Nil: not reported
	// (unknown); non-nil: reported, including an explicit zero.
	PromptTokens *int64 `json:"prompt_tokens,omitempty"`

	// CompletionTokens is the reported completion-token count. Same
	// reported/unknown distinction as PromptTokens.
	CompletionTokens *int64 `json:"completion_tokens,omitempty"`

	// CachedTokens is the reported cached-prompt-token count, where the
	// engine reports it. Cached tokens are a subset of the prompt tokens
	// (CachedTokens <= PromptTokens whenever both are reported). Nil: not
	// reported.
	CachedTokens *int64 `json:"cached_tokens,omitempty"`

	// Classification fixes what the token fields mean. Required, closed
	// set: see UsageClassification.
	Classification UsageClassification `json:"classification"`

	// PriceBookVersion is a POINTER to the price book in force when the
	// record was produced (for example a content stamp of the built
	// table). This package never stores, computes, or interprets amounts:
	// a record plus the price book it points at determine cost, and only
	// the consuming authority performs that determination. Optional.
	PriceBookVersion string `json:"price_book_version,omitempty"`

	// WitnessSHA is a hex-encoded SHA-256 over the record's canonical
	// fields, stamped by the emitting authority when it computes one.
	// Empty when the emitter does not. This package neither computes nor
	// verifies it. Optional.
	WitnessSHA string `json:"witness_sha,omitempty"`
}

// Validate checks the closed v1 contract: a known schema version, the
// required identity fields, a closed classification, and token fields
// consistent with that classification. On the first violated rule it
// returns a wrapped sentinel error, matchable with errors.Is against the
// specific sentinel or against ErrInvalidRecord.
func (r *UsageRecord) Validate() error {
	if r == nil {
		return ErrInvalidRecord
	}
	if r.SchemaVersion != SchemaVersionV1 {
		return fmt.Errorf("%w: have %d", ErrUnknownSchemaVersion, r.SchemaVersion)
	}
	if r.EventID == "" {
		return ErrMissingEventID
	}
	if r.Model == "" {
		return ErrMissingModel
	}
	if !r.Classification.Valid() {
		return fmt.Errorf("%w: %q", ErrUnknownClassification, r.Classification)
	}
	if r.RequestedAt.IsZero() {
		return ErrMissingRequestedAt
	}
	return r.validateTokens()
}

// validateTokens enforces the classification/token consistency rules.
func (r *UsageRecord) validateTokens() error {
	tokenFields := []*int64{r.PromptTokens, r.CompletionTokens, r.CachedTokens}

	reported := 0
	for _, n := range tokenFields {
		if n == nil {
			continue
		}
		reported++
		if *n < 0 {
			return fmt.Errorf("%w: negative token count %d", ErrInconsistentTokens, *n)
		}
	}

	switch r.Classification {
	case UsageRecorded:
		if reported == 0 {
			return fmt.Errorf("%w: classification %q requires at least one reported token field",
				ErrInconsistentTokens, r.Classification)
		}
	case UsageMissing, UsageNotApplicable:
		if reported > 0 {
			return fmt.Errorf("%w: classification %q forbids reported token fields",
				ErrInconsistentTokens, r.Classification)
		}
	case UsageZeroKnown:
		for _, n := range tokenFields {
			if n != nil && *n != 0 {
				return fmt.Errorf("%w: classification %q allows only zero token counts",
					ErrInconsistentTokens, r.Classification)
			}
		}
	default:
		// Unreachable once Valid() passed; guard the switch so a future
		// vocabulary addition cannot silently skip the consistency rules.
		return fmt.Errorf("%w: unhandled classification %q", ErrInvalidRecord, r.Classification)
	}

	if r.PromptTokens != nil && r.CachedTokens != nil && *r.CachedTokens > *r.PromptTokens {
		return fmt.Errorf("%w: cached tokens %d exceed prompt tokens %d",
			ErrInconsistentTokens, *r.CachedTokens, *r.PromptTokens)
	}
	return nil
}

// DecodeUsageRecord decodes exactly one serialized UsageRecord from data
// and refuses anything the v1 schema does not declare. Refusing unknown
// fields is the wire-side arm of the structural privacy invariant: a
// payload that tries to carry a prompt/response body or a credential
// alongside a record cannot be decoded into one. Decoding does not check
// semantic rules; call Validate on the result.
func DecodeUsageRecord(data []byte) (UsageRecord, error) {
	var rec UsageRecord
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&rec); err != nil {
		return UsageRecord{}, fmt.Errorf("%w: %v", ErrMalformedPayload, err)
	}
	if dec.More() {
		return UsageRecord{}, fmt.Errorf("%w: trailing data after record", ErrMalformedPayload)
	}
	return rec, nil
}

// Seen is a minimal in-memory dedup index over record event IDs. It
// answers exactly one question — has this event ID been recorded before?
// — and holds nothing else: no amounts, no balances, no policy. Durable
// storage and ledger mechanics live elsewhere; Seen is the replay guard
// for a single process.
//
// A zero-value Seen is ready to use. A nil *Seen is not usable: Record
// fails closed with ErrInvalidRecord rather than silently disabling dedup.
type Seen struct {
	mu  sync.Mutex
	ids map[string]struct{}
}

// NewSeen returns an empty, ready-to-use Seen.
func NewSeen() *Seen {
	return &Seen{ids: make(map[string]struct{})}
}

// Record marks rec's event ID as recorded. It returns
// ErrAlreadyRecorded when the same non-empty event ID was already
// recorded, which is the dedup/replay verdict a consumer acts on. An
// empty event ID is rejected with ErrMissingEventID. Record does not
// validate the rest of the record.
func (s *Seen) Record(rec UsageRecord) error {
	if s == nil {
		return ErrInvalidRecord
	}
	if rec.EventID == "" {
		return ErrMissingEventID
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ids == nil {
		s.ids = make(map[string]struct{})
	}
	if _, ok := s.ids[rec.EventID]; ok {
		return ErrAlreadyRecorded
	}
	s.ids[rec.EventID] = struct{}{}
	return nil
}

// Len reports how many distinct event IDs have been recorded.
func (s *Seen) Len() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.ids)
}
