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
	// ActiveBytesFormula specifies the required active-byte accounting formula.
	ActiveBytesFormula = "active_bytes = active_weight_bytes + state_bytes + kv_bytes_at_envelope"
	// ScoreFormula specifies the required quality per active byte ranking formula.
	ScoreFormula = "quality_per_active_byte = quality / active_bytes"
)

// Dataset holds architecture candidate observations and evaluation formulas.
type Dataset struct {
	SchemaVersion string      `json:"schema_version"`
	Formulas      Formulas    `json:"formulas"`
	Candidates    []Candidate `json:"candidates"`
}

// Formulas declares the active byte and score accounting equations.
type Formulas struct {
	ActiveBytes string `json:"active_bytes"`
	Score       string `json:"score"`
}

// Candidate models a single architecture observation or evaluation hypothesis.
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

// Provenance tracks measurement methodology and supporting artifact locators.
type Provenance struct {
	Kind    string `json:"kind"`
	URL     string `json:"url,omitempty"`
	Locator string `json:"locator,omitempty"`
}

// ComparabilityKey defines the exact dimension tuple required for fair candidate comparison.
type ComparabilityKey struct {
	EnvelopeID        string `json:"envelope_id"`
	QualityMetric     string `json:"quality_metric"`
	QualitySourceKind string `json:"quality_source_kind"`
}

// RankedRow represents a scored candidate with its rank and efficiency metrics.
type RankedRow struct {
	ID                   string  `json:"id"`
	Rank                 int     `json:"rank"`
	ActiveBytes          uint64  `json:"active_bytes"`
	QualityPerActiveByte float64 `json:"quality_per_active_byte"`
}

// RankedGroup aggregates candidates evaluated within the same comparability key.
type RankedGroup struct {
	Key  ComparabilityKey `json:"key"`
	Rows []RankedRow      `json:"rows"`
}

// UnrankedRow records why an individual candidate was excluded from comparative ranking.
type UnrankedRow struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

// Result contains partitioned groups of ranked candidates alongside unranked items.
type Result struct {
	Groups   []RankedGroup `json:"groups"`
	Unranked []UnrankedRow `json:"unranked"`
}

// LoadJSON parses and validates a candidate dataset from an input reader.
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

// LoadFile reads and validates an architecture candidate dataset from disk.
func LoadFile(path string) (*Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open architecture candidates: %w", err)
	}
	defer file.Close()
	return LoadJSON(file)
}

// Validate verifies schema conformance, formula strings, and candidate constraints.
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

// ActiveBytes calculates total memory footprint according to the active byte formula.
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

// Rank evaluates accepted candidates and sorts comparable groups by quality efficiency.
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
