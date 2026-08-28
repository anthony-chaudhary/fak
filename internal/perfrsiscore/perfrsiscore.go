package perfrsiscore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

const (
	EvidenceSchema = "fak-performance-rsi-evidence/1"
	ReportSchema   = "fak-performance-rsi-scorecard/1"
	TargetMultiple = 100.0
)

type Direction string

const (
	Higher Direction = "higher"
	Lower  Direction = "lower"
)

type Evidence struct {
	Schema           string      `json:"schema"`
	Snapshot         string      `json:"snapshot"`
	TargetMultiplier float64     `json:"target_multiplier"`
	Dimensions       []Dimension `json:"dimensions"`
}

type Dimension struct {
	ID           string    `json:"id"`
	Source       string    `json:"source"`
	Direction    Direction `json:"direction"`
	Current      *float64  `json:"current,omitempty"`
	Target       *float64  `json:"target"`
	Unit         string    `json:"unit"`
	NextAction   string    `json:"next_action"`
	EvidenceKind string    `json:"evidence_kind,omitempty"`
	Engine       string    `json:"engine,omitempty"`
}

type Result struct {
	ID              string   `json:"id"`
	Source          string   `json:"source"`
	Status          string   `json:"status"`
	Current         *float64 `json:"current"`
	Target          *float64 `json:"target"`
	Unit            string   `json:"unit"`
	NormalizedRatio *float64 `json:"normalized_ratio"`
	NextAction      string   `json:"next_action"`
}

type Delta struct {
	ID            string   `json:"id"`
	PriorStatus   string   `json:"prior_status"`
	CurrentStatus string   `json:"current_status"`
	PriorRatio    *float64 `json:"prior_ratio,omitempty"`
	CurrentRatio  *float64 `json:"current_ratio,omitempty"`
}

type Comparison struct {
	PriorSnapshot string  `json:"prior_snapshot"`
	Deltas        []Delta `json:"deltas"`
}

type Report struct {
	Schema             string      `json:"schema"`
	Snapshot           string      `json:"snapshot"`
	TargetMultiplier   float64     `json:"target_multiplier"`
	Dimensions         []Result    `json:"dimensions"`
	DominantBottleneck string      `json:"dominant_bottleneck"`
	UnknownDebt        int         `json:"unknown_debt"`
	Comparison         *Comparison `json:"comparison,omitempty"`
}

var dimensionIDs = []string{
	"cycle_time", "improvement_yield", "evaluation_latency", "receipt_coverage",
	"quality_gate_coverage", "experiment_throughput", "hypothesis_calibration", "discovery_freshness",
	"adaptation_speed", "reuse_ratio", "learning_retention", "production_transfer",
	"hardware_utilization", "attribution_quality", "automation_coverage", "compounding_rate",
}

func DimensionIDs() []string { return append([]string(nil), dimensionIDs...) }

func Load(path string) (Evidence, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Evidence{}, err
	}
	return Decode(bytes.NewReader(b))
}

func Decode(r io.Reader) (Evidence, error) {
	var e Evidence
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&e); err != nil {
		return e, fmt.Errorf("decode evidence: %w", err)
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return e, errors.New("decode evidence: trailing JSON value")
	}
	if err := validate(e); err != nil {
		return e, err
	}
	return e, nil
}

func validate(e Evidence) error {
	if e.Schema != EvidenceSchema {
		return fmt.Errorf("schema %q, want %q", e.Schema, EvidenceSchema)
	}
	if strings.TrimSpace(e.Snapshot) == "" {
		return errors.New("snapshot is required")
	}
	if !finite(e.TargetMultiplier) || e.TargetMultiplier != TargetMultiple {
		return fmt.Errorf("target_multiplier must preserve explicit unsaturated 100x target")
	}
	if len(e.Dimensions) != len(dimensionIDs) {
		return fmt.Errorf("dimensions: got %d, want exactly %d", len(e.Dimensions), len(dimensionIDs))
	}
	seen := make(map[string]bool, len(dimensionIDs))
	for _, d := range e.Dimensions {
		if seen[d.ID] {
			return fmt.Errorf("dimension %q appears more than once", d.ID)
		}
		seen[d.ID] = true
		if strings.TrimSpace(d.Source) == "" || strings.TrimSpace(d.NextAction) == "" || strings.TrimSpace(d.Unit) == "" {
			return fmt.Errorf("dimension %q requires source, unit, and next_action", d.ID)
		}
		if d.Direction != Higher && d.Direction != Lower {
			return fmt.Errorf("dimension %q has invalid direction %q", d.ID, d.Direction)
		}
		if d.Current != nil && (!finite(*d.Current) || *d.Current < 0 || (d.Direction == Lower && *d.Current == 0)) {
			return fmt.Errorf("dimension %q has invalid current", d.ID)
		}
		if d.Target != nil && (!finite(*d.Target) || *d.Target <= 0) {
			return fmt.Errorf("dimension %q has invalid target", d.ID)
		}
		engine := strings.ToLower(strings.TrimSpace(d.Engine))
		if strings.Contains(engine, "llama") {
			return fmt.Errorf("dimension %q: llama.cpp fallback is not native evidence", d.ID)
		}
		if d.EvidenceKind == "native_benchmark" {
			if !strings.HasPrefix(engine, "fak-native") {
				return fmt.Errorf("dimension %q: native benchmark evidence must name a fak-native engine", d.ID)
			}
		}
	}
	for _, id := range dimensionIDs {
		if !seen[id] {
			return fmt.Errorf("missing dimension %q", id)
		}
	}
	return nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func Score(e Evidence) Report {
	r := Report{Schema: ReportSchema, Snapshot: e.Snapshot, TargetMultiplier: e.TargetMultiplier}
	worst := math.Inf(1)
	for _, d := range e.Dimensions {
		x := Result{ID: d.ID, Source: d.Source, Current: d.Current, Target: d.Target, Unit: d.Unit, NextAction: d.NextAction, Status: "UNKNOWN"}
		debt := math.Inf(1)
		if d.Current != nil && d.Target != nil {
			ratio := 0.0
			if d.Direction == Higher {
				ratio = *d.Current / *d.Target
			} else if *d.Current == 0 {
				ratio = math.Inf(1)
			} else {
				ratio = *d.Target / *d.Current
			}
			x.NormalizedRatio = &ratio
			x.Status = "BEHIND"
			if ratio >= 1 {
				x.Status = "MET"
			}
			debt = ratio
		} else {
			r.UnknownDebt++
		}
		r.Dimensions = append(r.Dimensions, x)
		if debt < worst || (math.IsInf(debt, 1) && math.IsInf(worst, 1) && r.DominantBottleneck == "") {
			worst, r.DominantBottleneck = debt, d.ID
		}
	}
	return r
}

func Compare(current *Report, prior Report) error {
	if prior.Schema != ReportSchema {
		return fmt.Errorf("prior schema %q, want %q", prior.Schema, ReportSchema)
	}
	pm := map[string]Result{}
	for _, d := range prior.Dimensions {
		pm[d.ID] = d
	}
	c := &Comparison{PriorSnapshot: prior.Snapshot}
	for _, d := range current.Dimensions {
		p, ok := pm[d.ID]
		if !ok {
			return fmt.Errorf("prior snapshot missing dimension %q", d.ID)
		}
		c.Deltas = append(c.Deltas, Delta{ID: d.ID, PriorStatus: p.Status, CurrentStatus: d.Status, PriorRatio: p.NormalizedRatio, CurrentRatio: d.NormalizedRatio})
	}
	current.Comparison = c
	return nil
}

func DecodeReport(r io.Reader) (Report, error) {
	var p Report
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return p, err
	}
	return p, nil
}

func RenderHuman(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "performance RSI: %s | target %.0fx | UNKNOWN debt %d\n", r.Snapshot, r.TargetMultiplier, r.UnknownDebt)
	fmt.Fprintf(&b, "dominant bottleneck: %s\n", r.DominantBottleneck)
	for _, d := range r.Dimensions {
		fmt.Fprintf(&b, "%-25s %-7s current=%s target=%s ratio=%s source=%s next=%s\n", d.ID, d.Status, number(d.Current), number(d.Target), number(d.NormalizedRatio), d.Source, d.NextAction)
	}
	if r.Comparison != nil {
		fmt.Fprintf(&b, "compared with: %s\n", r.Comparison.PriorSnapshot)
	}
	return strings.TrimRight(b.String(), "\n")
}

func RenderMarkdown(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Performance RSI — %s\n\n- Explicit target: **%.0fx** (unsaturated)\n- Dominant bottleneck: `%s`\n- UNKNOWN debt: **%d**\n\n", r.Snapshot, r.TargetMultiplier, r.DominantBottleneck, r.UnknownDebt)
	b.WriteString("| Dimension | Status | Current | Target | Normalized ratio | Source | Next action |\n|---|---:|---:|---:|---:|---|---|\n")
	for _, d := range r.Dimensions {
		fmt.Fprintf(&b, "| %s | %s | %s %s | %s %s | %s | %s | %s |\n", d.ID, d.Status, number(d.Current), d.Unit, number(d.Target), d.Unit, number(d.NormalizedRatio), d.Source, d.NextAction)
	}
	if r.Comparison != nil {
		fmt.Fprintf(&b, "\nCompared with `%s`.\n", r.Comparison.PriorSnapshot)
	}
	return strings.TrimRight(b.String(), "\n")
}

func number(v *float64) string {
	if v == nil {
		return "UNKNOWN"
	}
	if math.IsInf(*v, 1) {
		return "+Inf"
	}
	return fmt.Sprintf("%.6g", *v)
}

func MarshalJSON(r Report) ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

func SortResultsForTest(rs []Result) {
	sort.Slice(rs, func(i, j int) bool { return rs[i].ID < rs[j].ID })
}
