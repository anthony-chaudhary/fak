package pricebook

import (
	"errors"
	"fmt"
	"regexp"
	"time"
)

// SchemaVersionV1 is the only operator-file format version this package
// produces and validates. Files evolve additively: later revisions may add
// optional fields, and consumers that decode strictly must expect them;
// breaking changes get a major version.
const SchemaVersionV1 = 1

// Sentinel errors. Every book-validation failure wraps ErrInvalidPriceBook,
// so errors.Is(err, ErrInvalidPriceBook) identifies any malformed book
// while the specific sentinel names the fault. Decode-level faults wrap
// ErrMalformedPayload: the bytes were not one decodable v1 file at all.
var (
	// ErrInvalidPriceBook is the root sentinel for every malformed
	// PriceBook.
	ErrInvalidPriceBook = errors.New("pricebook: invalid price book")

	// ErrUnknownSchemaVersion marks a file whose schema_version this
	// package does not validate.
	ErrUnknownSchemaVersion = fmt.Errorf("%w: unknown schema_version", ErrInvalidPriceBook)

	// ErrMalformedPayload marks serialized input that is not exactly one
	// decodable v1 price book: bad JSON, a field outside the v1 schema,
	// or trailing data.
	ErrMalformedPayload = fmt.Errorf("%w: malformed serialized payload", ErrInvalidPriceBook)

	// ErrMissingModelID marks a card without its model name.
	ErrMissingModelID = fmt.Errorf("%w: model id is required", ErrInvalidPriceBook)

	// ErrDuplicateModelID marks two cards pricing the same model.
	ErrDuplicateModelID = fmt.Errorf("%w: duplicate model id", ErrInvalidPriceBook)

	// ErrInvalidModelVersion marks a card whose version is below 1.
	ErrInvalidModelVersion = fmt.Errorf("%w: model version must be at least 1", ErrInvalidPriceBook)

	// ErrMissingSourceRevision marks a card without its source revision.
	ErrMissingSourceRevision = fmt.Errorf("%w: source_revision is required", ErrInvalidPriceBook)

	// ErrMissingSourceTime marks a card without its source time.
	ErrMissingSourceTime = fmt.Errorf("%w: source_time is required", ErrInvalidPriceBook)

	// ErrInvalidCurrency marks a card whose currency is not a three-letter
	// uppercase ISO-4217 style code.
	ErrInvalidCurrency = fmt.Errorf("%w: currency must be a three-letter uppercase code", ErrInvalidPriceBook)

	// ErrUnknownUnits marks a card whose units this package does not
	// validate.
	ErrUnknownUnits = fmt.Errorf("%w: unknown units", ErrInvalidPriceBook)

	// ErrInvalidRate marks a rate outside the closed contract: an unknown
	// status, a negative or non-finite value, or an unpriced rate that
	// carries a value.
	ErrInvalidRate = fmt.Errorf("%w: invalid rate", ErrInvalidPriceBook)

	// ErrInvalidCounts marks token counts that are negative. It is an
	// input defect, not a book defect, so it deliberately does not wrap
	// ErrInvalidPriceBook.
	ErrInvalidCounts = errors.New("pricebook: invalid token counts")
)

// Units is the closed vocabulary of pricing units a card may declare.
type Units string

// UnitsPerMillionTokens is the only unit this package validates: a rate
// value measures the card's currency per one million tokens of the named
// kind.
const UnitsPerMillionTokens Units = "per_million_tokens"

// tokensPerUnit is how many tokens one unit of UnitsPerMillionTokens
// covers. It is fixed by the closed units vocabulary above.
const tokensPerUnit = 1_000_000

// Valid reports whether u is a member of the closed units vocabulary.
func (u Units) Valid() bool {
	return u == UnitsPerMillionTokens
}

// ModelCard is one model's price card: its identity, provenance, and the
// four billing-line rates. Fresh input, cache reads, cache writes and
// output are separate lines with separate rates; a cache-read discount
// does not establish a write price, so CacheWriteRate is independently
// nullable (unpriced).
type ModelCard struct {
	// ID names the model this card prices (the canonical model name a
	// usage record carries). Required; unique within a book.
	ID string `json:"id"`

	// Version is the card revision. Required; at least 1.
	Version int `json:"version"`

	// SourceRevision identifies where the rates came from (for example a
	// catalog revision or an upstream tariff reference). Opaque: this
	// package neither mints nor interprets it. Required.
	SourceRevision string `json:"source_revision"`

	// SourceTime is when the source revision was observed. Required.
	SourceTime time.Time `json:"source_time"`

	// Currency is the three-letter uppercase code the rates are expressed
	// in. Required.
	Currency string `json:"currency"`

	// Units fixes what a rate value measures. Required; v1 accepts only
	// UnitsPerMillionTokens.
	Units Units `json:"units"`

	// FreshInputRate prices prompt tokens not served from cache.
	FreshInputRate Rate `json:"fresh_input_rate"`

	// CacheReadRate prices prompt tokens served from cache.
	CacheReadRate Rate `json:"cache_read_rate"`

	// CacheWriteRate prices tokens written to cache, where the engine
	// reports them. Independently nullable: a cache-read discount does
	// not establish a write price.
	CacheWriteRate Rate `json:"cache_write_rate"`

	// OutputRate prices completion tokens.
	OutputRate Rate `json:"output_rate"`
}

// PriceBook is the built, validated table of per-model price cards that an
// operator file declares.
type PriceBook struct {
	// SchemaVersion tags the file format. Always SchemaVersionV1 for
	// books this package produces or validates.
	SchemaVersion int `json:"schema_version"`

	// Models is the set of priced models. Empty is valid: a book that
	// prices nothing makes every lookup unpriced, which is the safe
	// direction.
	Models []ModelCard `json:"models"`
}

// Validate checks the closed v1 contract: a known schema version, and each
// card carrying a unique non-empty model ID, a version of at least 1,
// complete provenance, a well-formed currency, a validated unit, and four
// valid rates. On the first violated rule it returns a wrapped sentinel
// error, matchable with errors.Is against the specific sentinel or against
// ErrInvalidPriceBook.
func (b *PriceBook) Validate() error {
	if b == nil {
		return ErrInvalidPriceBook
	}
	if b.SchemaVersion != SchemaVersionV1 {
		return fmt.Errorf("%w: have %d", ErrUnknownSchemaVersion, b.SchemaVersion)
	}
	seen := make(map[string]struct{}, len(b.Models))
	for i := range b.Models {
		c := &b.Models[i]
		if c.ID == "" {
			return ErrMissingModelID
		}
		if _, dup := seen[c.ID]; dup {
			return fmt.Errorf("%w: %q", ErrDuplicateModelID, c.ID)
		}
		seen[c.ID] = struct{}{}
		if err := c.Validate(); err != nil {
			return err
		}
	}
	return nil
}

var currencyPattern = regexp.MustCompile(`^[A-Z]{3}$`)

// Validate checks the closed card contract. See PriceBook.Validate for the
// shared rules; this method enforces the per-card half.
func (c *ModelCard) Validate() error {
	if c == nil {
		return ErrInvalidPriceBook
	}
	if c.Version < 1 {
		return fmt.Errorf("%w: have %d", ErrInvalidModelVersion, c.Version)
	}
	if c.SourceRevision == "" {
		return ErrMissingSourceRevision
	}
	if c.SourceTime.IsZero() {
		return ErrMissingSourceTime
	}
	if !currencyPattern.MatchString(c.Currency) {
		return fmt.Errorf("%w: %q", ErrInvalidCurrency, c.Currency)
	}
	if !c.Units.Valid() {
		return fmt.Errorf("%w: %q", ErrUnknownUnits, c.Units)
	}
	rates := []Rate{c.FreshInputRate, c.CacheReadRate, c.CacheWriteRate, c.OutputRate}
	for _, r := range rates {
		if err := r.Valid(); err != nil {
			return err
		}
	}
	return nil
}

// Card returns the price card that prices model, if any. A nil book prices
// nothing.
func (b *PriceBook) Card(model string) (*ModelCard, bool) {
	if b == nil {
		return nil, false
	}
	for i := range b.Models {
		if b.Models[i].ID == model {
			return &b.Models[i], true
		}
	}
	return nil, false
}

// Counts is one attempt's declared token counts, split by billing line.
// Mapping engine-reported usage onto these lines is the caller's job; every
// field is a declared non-negative integer.
type Counts struct {
	// FreshInput is the prompt-token count not served from cache.
	FreshInput int64

	// CacheRead is the prompt-token count served from cache.
	CacheRead int64

	// CacheWrite is the token count written to cache, where the engine
	// reports it.
	CacheWrite int64

	// Output is the completion-token count.
	Output int64
}

// Valid checks the closed contract: no negative count.
func (c Counts) Valid() error {
	counts := []int64{c.FreshInput, c.CacheRead, c.CacheWrite, c.Output}
	for _, n := range counts {
		if n < 0 {
			return fmt.Errorf("%w: negative count %d", ErrInvalidCounts, n)
		}
	}
	return nil
}

// Valuate determines the cost of counts under this card, in the card's
// currency. The UNPRICED rule is structural:
//
//   - a declared zero count costs a known zero, whatever the rate's state
//     (a zero write count costs zero even when the write rate is
//     unpriced);
//   - a nonzero count against an unpriced rate leaves that line unknown,
//     and an unknown line leaves the total unknown;
//   - unknown is never treated as known: no missing rate is ever
//     substituted with zero.
//
// The figure is an analytical quantity, not a settlement amount. A nil
// card prices nothing and yields an unknown cost. An error is returned for
// invalid counts or a card that fails Validate.
func (c *ModelCard) Valuate(counts Counts) (Cost, error) {
	if c == nil {
		return UnknownCost(), nil
	}
	if err := c.Validate(); err != nil {
		return Cost{}, err
	}
	if err := counts.Valid(); err != nil {
		return Cost{}, err
	}
	total := ZeroCost()
	total = total.Add(lineCost(counts.FreshInput, c.FreshInputRate))
	total = total.Add(lineCost(counts.CacheRead, c.CacheReadRate))
	total = total.Add(lineCost(counts.CacheWrite, c.CacheWriteRate))
	total = total.Add(lineCost(counts.Output, c.OutputRate))
	return total, nil
}

// lineCost prices one billing line under the UNPRICED rule.
func lineCost(count int64, rate Rate) Cost {
	if count == 0 {
		return ZeroCost()
	}
	if rate.IsUnpriced() {
		return UnknownCost()
	}
	return KnownCost(float64(count) * rate.Value / tokensPerUnit)
}

// Valuate determines the cost of counts for model under this book, in the
// book's cards' currency. A model the book does not price is unpriced: it
// yields an unknown cost, never a known zero. An error is returned only for
// invalid counts or an invalid card.
func (b *PriceBook) Valuate(model string, counts Counts) (Cost, error) {
	card, ok := b.Card(model)
	if !ok {
		return UnknownCost(), nil
	}
	return card.Valuate(counts)
}
