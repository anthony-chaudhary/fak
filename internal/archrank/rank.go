// Package archrank ranks architecture observations by quality per active byte.
package archrank

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"sort"
	"strings"
)

const (
	// ActiveBytesFormula is the only active-byte accounting formula accepted by
	// this package. Keeping the spelling canonical makes fixture drift visible.
	//
	// Invariant: formula string must be "active_bytes = active_weight_bytes + state_bytes + kv_bytes_at_envelope".
	// Guard: validation fails closed with an error if this formula does not match.
	ActiveBytesFormula = "active_bytes = active_weight_bytes + state_bytes + kv_bytes_at_envelope"
	// ScoreFormula is the only ranking formula accepted by this package.
	//
	// Invariant: formula string must be "quality_per_active_byte = quality / active_bytes".
	// Guard: validation fails closed with an error if this formula does not match.
	ScoreFormula = "quality_per_active_byte = quality / active_bytes"
)

// Dataset is the versioned input to Rank.
//
// Invariant: SchemaVersion must be "archrank.candidates/v1", Formulas must be canonical, and Candidates must be non-empty with unique IDs.
// Guard: Validate fails closed on schema mismatches, unknown formulas, empty candidate slices, or invalid candidates.
type Dataset struct {
	SchemaVersion string      `json:"schema_version"`
	Formulas      Formulas    `json:"formulas"`
	Candidates    []Candidate `json:"candidates"`
}

// Formulas declares the accounting performed by the dataset.
//
// Invariant: ActiveBytes and Score must match ActiveBytesFormula and ScoreFormula respectively.
// Guard: non-matching formula strings are rejected during validation.
type Formulas struct {
	ActiveBytes string `json:"active_bytes"`
	Score       string `json:"score"`
}

// Candidate is one architecture observation or hypothesis.
//
// Invariant: Quality must be finite and non-negative; ActiveBytes must sum without uint64 overflow and exceed zero.
// Guard: accepted status requires measured provenance with a locator; estimated status requires literature provenance with an http(s) URL.
type Candidate struct {
	ID                string     `json:"id"`
	Architecture      string     `json:"architecture"`
	MigrationClass    string     `json:"migration_class"`
	EnvelopeID        string     `json:"envelope_id"`
	QualityMetric     string     `json:"quality_metric"`
	QualitySourceKind string     `json:"quality_source_kind"`
	MeasurementStatus string     `json:"measurement_status"`
	Quality           float64    `json:"quality"`
	ActiveWeightBytes uint64     `json:"active_weight_bytes"`
	StateBytes        uint64     `json:"state_bytes"`
	KVBytesAtEnvelope uint64     `json:"kv_bytes_at_envelope"`
	Provenance        Provenance `json:"provenance"`
}

// Provenance records whether a row is a measured observation or an estimate
// and where its evidence can be inspected.
//
// Invariant: Kind must be "synthetic_control_measurement" or "measured_artifact" for accepted status, or "literature_hypothesis" for estimated status.
// Guard: accepted rows require non-empty Locator; estimated rows require an absolute http(s) URL.
type Provenance struct {
	Kind    string `json:"kind"`
	URL     string `json:"url,omitempty"`
	Locator string `json:"locator,omitempty"`
}

// ComparabilityKey is deliberately exact: Rank never coerces or normalizes any
// of these fields when deciding whether two rows may be compared.
//
// Invariant: exact tuple matching over EnvelopeID, QualityMetric, and QualitySourceKind.
// Guard: rows differing in any field are placed in separate groups or marked unranked.
type ComparabilityKey struct {
	EnvelopeID        string `json:"envelope_id"`
	QualityMetric     string `json:"quality_metric"`
	QualitySourceKind string `json:"quality_source_kind"`
}

// RankedRow is a computed, comparable result.
//
// Invariant: Rank is 1-based, ActiveBytes is greater than zero, and QualityPerActiveByte is finite.
// Guard: deterministic tie-break orders rows with identical scores by ID ascending.
type RankedRow struct {
	ID                   string  `json:"id"`
	Rank                 int     `json:"rank"`
	ActiveBytes          uint64  `json:"active_bytes"`
	QualityPerActiveByte float64 `json:"quality_per_active_byte"`
}

// RankedGroup contains rows sharing one exact ComparabilityKey.
//
// Invariant: Key is shared by all rows; Rows contains at least two rows sorted descending by QualityPerActiveByte.
// Guard: keys with fewer than two rows fail closed to Unranked instead of emitting singleton groups.
type RankedGroup struct {
	Key  ComparabilityKey `json:"key"`
	Rows []RankedRow      `json:"rows"`
}

// UnrankedRow records why a valid input row did not participate in a ranking.
//
// Invariant: ID identifies the candidate and Reason explains the non-ranking cause.
// Guard: reasons distinguish estimated hypotheses from unmatched comparability keys.
type UnrankedRow struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

// Result separates comparable rankings from explicit non-ranking decisions.
//
// Invariant: every valid candidate appears in either Groups or Unranked, never both.
// Guard: candidates are never silently dropped or double-counted.
type Result struct {
	Groups   []RankedGroup `json:"groups"`
	Unranked []UnrankedRow `json:"unranked"`
}

// LoadJSON decodes and validates a candidate dataset.
//
// Invariant: stream must contain exactly one valid JSON object complying with the Dataset schema.
// Guard: unknown fields are disallowed, trailing content is rejected, and validation runs before returning.
func LoadJSON(r io.Reader) (*Dataset, error) {
	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()

	var dataset Dataset
	if err := decoder.Decode(&dataset); err != nil {
		return nil, fmt.Errorf("decode architecture candidates: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("decode architecture candidates: multiple JSON values")
		}
		return nil, fmt.Errorf("decode architecture candidates trailing data: %w", err)
	}
	if err := dataset.Validate(); err != nil {
		return nil, err
	}
	return &dataset, nil
}

// LoadFile loads and validates a candidate dataset from path.
//
// Invariant: file at path must exist, be readable, and contain a valid Dataset.
// Guard: closes the open file descriptor and fails closed on I/O or validation errors.
func LoadFile(path string) (*Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open architecture candidates: %w", err)
	}
	defer file.Close()
	return LoadJSON(file)
}

// Validate checks schema, formula, provenance, and arithmetic invariants.
//
// Invariant: SchemaVersion, Formulas, and each Candidate must satisfy integrity checks; IDs must be unique.
// Guard: returns an error on the first violated invariant to fail closed.
func (d Dataset) Validate() error {
	if d.SchemaVersion != "archrank.candidates/v1" {
		return fmt.Errorf("schema_version: got %q, want %q", d.SchemaVersion, "archrank.candidates/v1")
	}
	if d.Formulas.ActiveBytes != ActiveBytesFormula {
		return fmt.Errorf("formulas.active_bytes: got %q, want %q", d.Formulas.ActiveBytes, ActiveBytesFormula)
	}
	if d.Formulas.Score != ScoreFormula {
		return fmt.Errorf("formulas.score: got %q, want %q", d.Formulas.Score, ScoreFormula)
	}
	if len(d.Candidates) == 0 {
		return errors.New("candidates: at least one row is required")
	}

	seen := make(map[string]struct{}, len(d.Candidates))
	for i := range d.Candidates {
		candidate := &d.Candidates[i]
		if err := candidate.validate(); err != nil {
			return fmt.Errorf("candidate[%d] %q: %w", i, candidate.ID, err)
		}
		if _, exists := seen[candidate.ID]; exists {
			return fmt.Errorf("candidate[%d] %q: duplicate id", i, candidate.ID)
		}
		seen[candidate.ID] = struct{}{}
	}
	return nil
}

func (c Candidate) validate() error {
	required := map[string]string{
		"id":                  c.ID,
		"architecture":        c.Architecture,
		"migration_class":     c.MigrationClass,
		"envelope_id":         c.EnvelopeID,
		"quality_metric":      c.QualityMetric,
		"quality_source_kind": c.QualitySourceKind,
		"measurement_status":  c.MeasurementStatus,
		"provenance.kind":     c.Provenance.Kind,
	}
	for field, value := range required {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("%s is required", field)
		}
	}
	if math.IsNaN(c.Quality) || math.IsInf(c.Quality, 0) || c.Quality < 0 {
		return fmt.Errorf("quality must be finite and non-negative, got %v", c.Quality)
	}
	if _, err := c.ActiveBytes(); err != nil {
		return err
	}

	switch c.MeasurementStatus {
	case "accepted":
		if c.Provenance.Kind != "synthetic_control_measurement" && c.Provenance.Kind != "measured_artifact" {
			return fmt.Errorf("accepted row requires measured provenance, got %q", c.Provenance.Kind)
		}
		if strings.TrimSpace(c.Provenance.Locator) == "" {
			return errors.New("accepted row provenance.locator is required")
		}
	case "estimated":
		if c.Provenance.Kind != "literature_hypothesis" {
			return fmt.Errorf("estimated row requires literature_hypothesis provenance, got %q", c.Provenance.Kind)
		}
		if err := validateSourceURL(c.Provenance.URL); err != nil {
			return err
		}
	default:
		return fmt.Errorf("measurement_status must be accepted or estimated, got %q", c.MeasurementStatus)
	}
	return nil
}

func validateSourceURL(raw string) error {
	parsed, err := url.ParseRequestURI(raw)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "https" && parsed.Scheme != "http") {
		return fmt.Errorf("literature provenance.url must be an absolute http(s) URL, got %q", raw)
	}
	return nil
}

// ActiveBytes applies the canonical active-byte accounting formula.
//
// Invariant: returns active_weight_bytes + state_bytes + kv_bytes_at_envelope.
// Guard: fails closed with an error on uint64 overflow or if total active bytes equals zero.
func (c Candidate) ActiveBytes() (uint64, error) {
	if math.MaxUint64-c.ActiveWeightBytes < c.StateBytes {
		return 0, errors.New("active_bytes overflows uint64")
	}
	total := c.ActiveWeightBytes + c.StateBytes
	if math.MaxUint64-total < c.KVBytesAtEnvelope {
		return 0, errors.New("active_bytes overflows uint64")
	}
	total += c.KVBytesAtEnvelope
	if total == 0 {
		return 0, errors.New("active_bytes must be greater than zero")
	}
	return total, nil
}

// Rank ranks accepted measured rows only when at least two rows share the
// exact envelope, metric, and source-kind key. All other valid rows receive an
// explicit unranked reason.
//
// Invariant: only accepted rows sharing an exact ComparabilityKey with at least one peer are ranked.
// Guard: estimated rows and unmatched singleton keys fail closed into Unranked with explicit reasons.
func Rank(dataset Dataset) (Result, error) {
	if err := dataset.Validate(); err != nil {
		return Result{}, err
	}

	eligible := make(map[ComparabilityKey][]Candidate)
	result := Result{}
	for _, candidate := range dataset.Candidates {
		if candidate.MeasurementStatus != "accepted" {
			result.Unranked = append(result.Unranked, UnrankedRow{
				ID:     candidate.ID,
				Reason: "estimated row: ranking requires accepted measured evidence",
			})
			continue
		}
		key := ComparabilityKey{
			EnvelopeID:        candidate.EnvelopeID,
			QualityMetric:     candidate.QualityMetric,
			QualitySourceKind: candidate.QualitySourceKind,
		}
		eligible[key] = append(eligible[key], candidate)
	}

	keys := make([]ComparabilityKey, 0, len(eligible))
	for key := range eligible {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return keyLess(keys[i], keys[j]) })

	for _, key := range keys {
		candidates := eligible[key]
		if len(candidates) < 2 {
			candidate := candidates[0]
			result.Unranked = append(result.Unranked, UnrankedRow{
				ID: candidate.ID,
				Reason: fmt.Sprintf(
					"unmatched comparability key: envelope_id=%q, quality_metric=%q, quality_source_kind=%q",
					key.EnvelopeID,
					key.QualityMetric,
					key.QualitySourceKind,
				),
			})
			continue
		}

		rows := make([]RankedRow, 0, len(candidates))
		for _, candidate := range candidates {
			activeBytes, err := candidate.ActiveBytes()
			if err != nil {
				return Result{}, fmt.Errorf("candidate %q: %w", candidate.ID, err)
			}
			rows = append(rows, RankedRow{
				ID:                   candidate.ID,
				ActiveBytes:          activeBytes,
				QualityPerActiveByte: candidate.Quality / float64(activeBytes),
			})
		}
		sort.Slice(rows, func(i, j int) bool {
			if rows[i].QualityPerActiveByte == rows[j].QualityPerActiveByte {
				return rows[i].ID < rows[j].ID
			}
			return rows[i].QualityPerActiveByte > rows[j].QualityPerActiveByte
		})
		for i := range rows {
			rows[i].Rank = i + 1
		}
		result.Groups = append(result.Groups, RankedGroup{Key: key, Rows: rows})
	}

	sort.Slice(result.Unranked, func(i, j int) bool { return result.Unranked[i].ID < result.Unranked[j].ID })
	return result, nil
}

func keyLess(a, b ComparabilityKey) bool {
	if a.EnvelopeID != b.EnvelopeID {
		return a.EnvelopeID < b.EnvelopeID
	}
	if a.QualityMetric != b.QualityMetric {
		return a.QualityMetric < b.QualityMetric
	}
	return a.QualitySourceKind < b.QualitySourceKind
}
