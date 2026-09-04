package customizationindex

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// Invariant: customization index checks are fail-closed and deterministic.
// Guard: malformed schema versions, unknown fields, or missing evidence reject the entire index.

// Schema identifies the supported agent customization index schema version.
const Schema = "fak-agent-customization-index/1"

// Index contains the complete structured specification of customization layers, sources, and axes.
type Index struct {
	Schema      string      `json:"schema"`
	UpdatedAt   string      `json:"updated_at"`
	Scope       string      `json:"scope"`
	Maintenance Maintenance `json:"maintenance"`
	Layers      []Layer     `json:"layers"`
	Sources     []Source    `json:"sources"`
	Axes        []Axis      `json:"axes"`
}

// Maintenance specifies review schedules, validation keys, and accepted lifecycle status values.
type Maintenance struct {
	ReviewIntervalDays int      `json:"review_interval_days"`
	DedupeKey          string   `json:"dedupe_key"`
	SourceIdentity     string   `json:"source_identity"`
	RequiredRefresh    []string `json:"required_refresh_fields"`
	StatusValues       []string `json:"status_values"`
	DispositionValues  []string `json:"disposition_values"`
}

// Layer represents an architectural concern or customization dimension in the agent pipeline.
type Layer struct {
	ID       string `json:"id"`
	Question string `json:"question"`
}

// Source records an external or repository reference providing evidence for customization support.
type Source struct {
	ID              string   `json:"id"`
	Kind            string   `json:"kind"`
	URL             string   `json:"url"`
	ObservedAt      string   `json:"observed_at"`
	CheckedRevision string   `json:"checked_revision"`
	License         string   `json:"license"`
	Evidence        []string `json:"evidence"`
}

// Axis maps a user need within a layer to concrete evidence, tracking current disposition and status.
type Axis struct {
	ID          string   `json:"axis_id"`
	Layer       string   `json:"layer"`
	UserNeed    string   `json:"user_need"`
	Examples    []string `json:"examples"`
	Evidence    []string `json:"evidence"`
	Status      string   `json:"fak_status"`
	Disposition string   `json:"disposition"`
}

// SourceFreshness summarizes evidence review age and signals whether re-validation is due.
type SourceFreshness struct {
	ID         string `json:"id"`
	ObservedAt string `json:"observed_at"`
	AgeDays    int    `json:"age_days"`
	Due        bool   `json:"due"`
}

// Group tallies axes partitioned by layer and status for portfolio reporting.
type Group struct {
	Layer  string `json:"layer"`
	Status string `json:"status"`
	Count  int    `json:"count"`
}

// Report provides the evaluated freshness, grouping, and contract validation results for an index.
type Report struct {
	Schema     string            `json:"schema"`
	AsOf       string            `json:"as_of"`
	Valid      bool              `json:"valid"`
	Errors     []string          `json:"errors,omitempty"`
	Sources    []SourceFreshness `json:"sources"`
	Groups     []Group           `json:"groups"`
	DueSources int               `json:"due_sources"`
	Axes       int               `json:"axes"`
	ReviewDays int               `json:"review_interval_days"`
}

// Read decodes an index from JSON, enforcing strict field checking to reject unknown properties.
func Read(r io.Reader) (Index, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var index Index
	if err := dec.Decode(&index); err != nil {
		return Index{}, err
	}
	return index, nil
}

// Check audits an index for consistency, evidence validity, and freshness relative to an evaluation date.
func Check(index Index, asOf time.Time) Report {
	report := Report{Schema: Schema, AsOf: asOf.Format(time.DateOnly), Valid: true, Axes: len(index.Axes), ReviewDays: index.Maintenance.ReviewIntervalDays}
	addError := func(format string, args ...any) {
		report.Errors = append(report.Errors, fmt.Sprintf(format, args...))
		report.Valid = false
	}
	if index.Schema != Schema {
		addError("schema %q, want %q", index.Schema, Schema)
	}
	if index.Maintenance.ReviewIntervalDays <= 0 {
		addError("maintenance.review_interval_days must be positive")
	}
	layerIDs := uniqueSet(index.Layers, func(layer Layer) string { return layer.ID }, "layer", addError)
	sourceIDs := uniqueSet(index.Sources, func(source Source) string { return source.ID }, "source", addError)
	axisIDs := make(map[string]struct{})
	statusValues := stringSet(index.Maintenance.StatusValues)
	dispositionValues := stringSet(index.Maintenance.DispositionValues)
	groups := make(map[string]int)
	for _, axis := range index.Axes {
		if strings.TrimSpace(axis.ID) == "" || strings.TrimSpace(axis.UserNeed) == "" {
			addError("axis requires axis_id and user_need")
		}
		if _, exists := axisIDs[axis.ID]; exists {
			addError("duplicate axis %q", axis.ID)
		}
		axisIDs[axis.ID] = struct{}{}
		if _, ok := layerIDs[axis.Layer]; !ok {
			addError("axis %q references unknown layer %q", axis.ID, axis.Layer)
		}
		if _, ok := statusValues[axis.Status]; !ok {
			addError("axis %q has invalid status %q", axis.ID, axis.Status)
		}
		if _, ok := dispositionValues[axis.Disposition]; !ok {
			addError("axis %q has invalid disposition %q", axis.ID, axis.Disposition)
		}
		if len(axis.Evidence) == 0 {
			addError("axis %q requires evidence", axis.ID)
		}
		for _, evidence := range axis.Evidence {
			if _, ok := sourceIDs[evidence]; !ok {
				addError("axis %q references unknown source %q", axis.ID, evidence)
			}
		}
		groups[axis.Layer+"\x00"+axis.Status]++
	}
	for _, source := range index.Sources {
		observed, err := time.Parse(time.DateOnly, source.ObservedAt)
		if err != nil {
			addError("source %q has invalid observed_at %q", source.ID, source.ObservedAt)
			continue
		}
		age := int(asOf.Sub(observed).Hours() / 24)
		due := age > index.Maintenance.ReviewIntervalDays
		if due {
			report.DueSources++
		}
		report.Sources = append(report.Sources, SourceFreshness{ID: source.ID, ObservedAt: source.ObservedAt, AgeDays: age, Due: due})
	}
	sort.Slice(report.Sources, func(i, j int) bool { return report.Sources[i].ID < report.Sources[j].ID })
	for key, count := range groups {
		parts := strings.Split(key, "\x00")
		report.Groups = append(report.Groups, Group{Layer: parts[0], Status: parts[1], Count: count})
	}
	sort.Slice(report.Groups, func(i, j int) bool {
		if report.Groups[i].Layer == report.Groups[j].Layer {
			return report.Groups[i].Status < report.Groups[j].Status
		}
		return report.Groups[i].Layer < report.Groups[j].Layer
	})
	sort.Strings(report.Errors)
	return report
}

func uniqueSet[T any](values []T, id func(T) string, kind string, addError func(string, ...any)) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for _, value := range values {
		key := id(value)
		if strings.TrimSpace(key) == "" {
			addError("%s requires id", kind)
			continue
		}
		if _, exists := out[key]; exists {
			addError("duplicate %s %q", kind, key)
		}
		out[key] = struct{}{}
	}
	return out
}

func stringSet(values []string) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for _, value := range values {
		out[value] = struct{}{}
	}
	return out
}

// ParseAsOf parses a date string in YYYY-MM-DD format or falls back to the current UTC timestamp.
func ParseAsOf(value string, now time.Time) (time.Time, error) {
	if value == "" {
		return now.UTC(), nil
	}
	parsed, err := time.Parse(time.DateOnly, value)
	if err != nil {
		return time.Time{}, errors.New("as-of must be YYYY-MM-DD")
	}
	return parsed, nil
}
