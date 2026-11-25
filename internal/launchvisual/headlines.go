// Package launchvisual renders the launch charts from the published headline index
// and checks the launch docs against that index, so every number a reader sees traces
// to one receipt-backed row.
package launchvisual

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// IndexSchema names the only index layout this package reads.
const IndexSchema = "fak.benchpub.headlines.v1"

var (
	idPattern     = regexp.MustCompile(`^[a-z0-9]+(\.[a-z0-9_]+)+$`)
	shaPattern    = regexp.MustCompile(`^[0-9a-f]{64}$`)
	commitPattern = regexp.MustCompile(`^([0-9a-f]{7,40})?$`)
	dottedQuad    = regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`)
)

// validAgg lists how a row's value is formed from its samples: one sample, the mean, the
// median, or the minimum or maximum of per-prompt ratios.
var validAgg = map[string]bool{"scalar": true, "mean": true, "median": true, "min": true, "max": true}

// requiredRowKeys is the full INDEX CONTRACT row shape. Decoding alone cannot tell an
// absent nullable field from an explicit null, so presence is checked separately.
var requiredRowKeys = []string{
	"id", "kind", "metric", "unit", "higher_is_better", "samples", "agg", "value",
	"display_decimals", "comparator_id", "comparator_value", "ratio", "method",
	"bit_exact", "hw_class", "fak_commit", "campaign", "receipt_sha256",
	"receipt_pointer", "caveat",
}

// Index is the decoded headline index (docs/benchmarks/published/HEADLINES.json).
type Index struct {
	Schema    string `json:"schema"`
	Generator string `json:"generator"`
	Rows      []Row  `json:"rows"`
}

// Row is one headline. Nullable fields are pointers so a null and a missing key are
// distinguishable after decoding.
type Row struct {
	ID              string    `json:"id"`
	Kind            string    `json:"kind"`
	Metric          string    `json:"metric"`
	Unit            string    `json:"unit"`
	HigherIsBetter  bool      `json:"higher_is_better"`
	Samples         []float64 `json:"samples"`
	Agg             string    `json:"agg"`
	Value           float64   `json:"value"`
	DisplayDecimals int       `json:"display_decimals"`
	ComparatorID    *string   `json:"comparator_id"`
	ComparatorValue *float64  `json:"comparator_value"`
	Ratio           *float64  `json:"ratio"`
	Method          string    `json:"method"`
	BitExact        string    `json:"bit_exact"`
	HWClass         string    `json:"hw_class"`
	FakCommit       string    `json:"fak_commit"`
	Campaign        string    `json:"campaign"`
	ReceiptSHA256   string    `json:"receipt_sha256"`
	ReceiptPointer  string    `json:"receipt_pointer"`
	Caveat          string    `json:"caveat"`
}

// ParseIndex strictly decodes and validates an index. Unknown fields, missing row keys,
// trailing data, and any dotted quad anywhere in the bytes are refused.
func ParseIndex(b []byte) (Index, error) {
	if q := dottedQuad.Find(b); q != nil {
		return Index{}, fmt.Errorf("launch index: dotted quad %q refused", q)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var ix Index
	if err := dec.Decode(&ix); err != nil {
		return Index{}, fmt.Errorf("launch index: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return Index{}, errors.New("launch index: trailing data after the top-level object")
	}
	var shape struct {
		Rows []map[string]json.RawMessage `json:"rows"`
	}
	if err := json.Unmarshal(b, &shape); err != nil {
		return Index{}, fmt.Errorf("launch index: %w", err)
	}
	for i, m := range shape.Rows {
		for _, k := range requiredRowKeys {
			if _, ok := m[k]; !ok {
				return Index{}, fmt.Errorf("launch index row %d: missing key %q", i, k)
			}
		}
	}
	if err := ix.Validate(); err != nil {
		return Index{}, err
	}
	return ix, nil
}

// Validate checks the whole index: schema, row order, and every row's invariants.
func (ix Index) Validate() error {
	if ix.Schema != IndexSchema {
		return fmt.Errorf("launch index: schema %q, want %q", ix.Schema, IndexSchema)
	}
	if ix.Generator == "" {
		return errors.New("launch index: generator is required")
	}
	if len(ix.Rows) == 0 {
		return errors.New("launch index: no rows")
	}
	prev := ""
	for i, r := range ix.Rows {
		if err := r.validate(); err != nil {
			return fmt.Errorf("launch index row %d (%s): %w", i, r.ID, err)
		}
		if i > 0 && r.ID <= prev {
			return fmt.Errorf("launch index: rows must be unique and sorted ascending; %q follows %q", r.ID, prev)
		}
		prev = r.ID
	}
	return nil
}

// ByID maps each row id to its row. The index is validated unique, so no id is lost.
func (ix Index) ByID() map[string]Row {
	m := make(map[string]Row, len(ix.Rows))
	for _, r := range ix.Rows {
		m[r.ID] = r
	}
	return m
}

func (r Row) validate() error {
	switch {
	case !idPattern.MatchString(r.ID):
		return fmt.Errorf("bad id %q", r.ID)
	case r.Kind != "win" && r.Kind != "loss" && r.Kind != "neutral":
		return fmt.Errorf("bad kind %q", r.Kind)
	case r.Metric == "" || r.Unit == "":
		return errors.New("metric and unit are required")
	case len(r.Samples) == 0:
		return errors.New("samples must not be empty")
	case !validAgg[r.Agg]:
		return fmt.Errorf("bad agg %q", r.Agg)
	case r.DisplayDecimals < 0 || r.DisplayDecimals > 3:
		return fmt.Errorf("display_decimals %d outside 0..3", r.DisplayDecimals)
	case r.BitExact != "true" && r.BitExact != "false" && r.BitExact != "n/a":
		return fmt.Errorf("bad bit_exact %q", r.BitExact)
	case r.Method == "" || r.HWClass == "" || r.Campaign == "" || r.ReceiptPointer == "":
		return errors.New("method, hw_class, campaign and receipt_pointer are required")
	case !shaPattern.MatchString(r.ReceiptSHA256):
		return errors.New("receipt_sha256 must be 64 lowercase hex characters")
	case !commitPattern.MatchString(r.FakCommit):
		return fmt.Errorf("bad fak_commit %q", r.FakCommit)
	case math.IsNaN(r.Value) || math.IsInf(r.Value, 0):
		return errors.New("value must be finite")
	}
	if err := r.checkAggregate(); err != nil {
		return err
	}
	if err := r.checkRatio(); err != nil {
		return err
	}
	if r.needsCaveat() && r.Caveat == "" {
		return errors.New("caveat is required for bit_exact false, loss, a comparator without a ratio, or a single_run method")
	}
	return nil
}

func (r Row) checkAggregate() error {
	const tol = 0.0006 // value is stored to 3 dp; mean rounding can land on the half step
	if r.Agg == "scalar" {
		if math.Abs(r.Value-r.Samples[0]) > tol {
			return fmt.Errorf("scalar value %v differs from its first sample %v", r.Value, r.Samples[0])
		}
		return nil
	}
	if r.Agg == "min" || r.Agg == "max" {
		want := r.Samples[0]
		for _, s := range r.Samples[1:] {
			if (r.Agg == "min" && s < want) || (r.Agg == "max" && s > want) {
				want = s
			}
		}
		if math.Abs(want-r.Value) > tol {
			return fmt.Errorf("%s value %v differs from the %s of samples %v", r.Agg, r.Value, r.Agg, want)
		}
		return nil
	}
	if r.Agg == "median" {
		sorted := append([]float64(nil), r.Samples...)
		sort.Float64s(sorted)
		mid := len(sorted) / 2
		want := sorted[mid]
		if len(sorted)%2 == 0 {
			want = (sorted[mid-1] + sorted[mid]) / 2
		}
		if math.Abs(want-r.Value) > tol {
			return fmt.Errorf("median value %v differs from the median of samples %v", r.Value, want)
		}
		return nil
	}
	sum := 0.0
	for _, s := range r.Samples {
		sum += s
	}
	mean := sum / float64(len(r.Samples))
	if math.Abs(mean-r.Value) > tol {
		return fmt.Errorf("mean value %v differs from the mean of samples %v", r.Value, mean)
	}
	return nil
}

func (r Row) checkRatio() error {
	if r.ComparatorValue == nil {
		if r.Ratio != nil {
			return errors.New("ratio requires comparator_value")
		}
		return nil
	}
	if r.ComparatorID == nil || *r.ComparatorID == "" {
		return errors.New("comparator_value requires comparator_id")
	}
	cv := *r.ComparatorValue
	if cv <= 0 || r.Value <= 0 {
		return errors.New("comparator and value must be positive to form a ratio")
	}
	if r.Ratio == nil {
		return nil
	}
	if !isCents(*r.Ratio) {
		return fmt.Errorf("ratio %v is not 2 dp", *r.Ratio)
	}
	expected := r.Value / cv
	if !r.HigherIsBetter {
		expected = cv / r.Value
	}
	if want := RoundHalfUp(expected, 2); math.Abs(want-*r.Ratio) > 1e-9 {
		return fmt.Errorf("ratio %v does not match value over comparator (%v)", *r.Ratio, want)
	}
	return nil
}

func (r Row) needsCaveat() bool {
	if r.BitExact == "false" || r.Kind == "loss" || strings.Contains(r.Method, "single_run") {
		return true
	}
	return r.Ratio == nil && r.ComparatorID != nil
}

// Display renders the row's value rounded half-up to its display_decimals.
func (r Row) Display() string { return DisplayValue(r.Value, r.DisplayDecimals) }

// RatioLabel is the N.NNx form of the row's ratio, or "" when the row has none.
func (r Row) RatioLabel() string {
	if r.Ratio == nil {
		return ""
	}
	return strconv.FormatFloat(*r.Ratio, 'f', 2, 64) + "x"
}

// RoundHalfUp is the publisher's rounding (platform/benchpub roundHalfUp), mirrored here
// so a ratio or display value computed on either side agrees to the last digit.
func RoundHalfUp(v float64, decimals int) float64 {
	p := math.Pow(10, float64(decimals))
	r := math.Floor(math.Abs(v)*p+0.5+1e-9) / p
	if v < 0 {
		return -r
	}
	return r
}

// DisplayValue rounds v half-up to decimals places and formats it with exactly that
// many digits. The tiny epsilon keeps binary-float halves such as 333.65 on the upper
// side, which is what a reader expects of a printed half.
func DisplayValue(v float64, decimals int) string {
	p := math.Pow10(decimals)
	n := int64(math.Floor(math.Abs(v)*p + 0.5 + 1e-9))
	sign := ""
	if v < 0 {
		sign = "-"
	}
	whole := n / int64(p)
	if decimals == 0 {
		return sign + strconv.FormatInt(whole, 10)
	}
	frac := strconv.FormatInt(n%int64(p), 10)
	for len(frac) < decimals {
		frac = "0" + frac
	}
	return sign + strconv.FormatInt(whole, 10) + "." + frac
}

func isCents(x float64) bool {
	return math.Abs(x*100-math.Round(x*100)) < 1e-6
}
