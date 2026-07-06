// Ingest is the read-only adapter that turns a checked-in benchmark SNAPSHOT —
// a small JSON export of official Terminal-Bench, SWE-bench, or FrontierSWE
// leaderboard rows — into provenanced internal/modelscore evidence rows. It is
// the source-of-truth boundary between "a number a human copied off a
// leaderboard once" and "a row a tier policy is allowed to read": a snapshot
// row only becomes modelscore evidence if it names WHERE it came from (source
// URL), WHAT it measured (benchmark + version + metric unit), WHO earned it
// (model, plus the harness for an agent-plus-model bench), and WHEN it was
// captured (as-of date). A row missing any of those is refused, loud, at ingest
// — so a guessed or undated figure can never reach a routing decision wearing
// the costume of a measurement.
//
// WHY A SNAPSHOT, NOT A SCRAPER (out of scope, deliberately). The official
// pages change; this ingestor reads a committed fixture and stamps an
// observed_at date rather than claiming live freshness. There is no scheduled
// network fetch here and no Python — the issue is explicit that a helper, if
// needed, is a Go fak subcommand or pure package. This file is that pure
// package: stdlib + internal/modelscore only, no network, no credentials, so
// the acceptance gate runs anywhere.
//
// WHAT THE THREE BENCHMARKS ARE (and are NOT interchangeable):
//   - Terminal-Bench (2.0 / 2.1 / Hard are DISTINCT versions) scores an AGENT
//     PLUS a model, so every Terminal-Bench row must carry a harness; the model
//     alone did not earn the number. Native unit: percent resolved.
//   - SWE-bench Verified reports a percent-resolved figure — a model-plus-
//     scaffold resolved rate over the Verified split. Native unit: percent
//     resolved. Different axis from Terminal-Bench; not one shared rank.
//   - FrontierSWE is ultra-long-horizon (Mean@5 / Best@5 over 20h-budget
//     tasks). Native unit is the fixture's own metric label; never blended onto
//     a 0..100 axis. A different axis again.
//
// The confidence of a snapshot row is an ENUM about its SOURCE — official,
// vendor-reported, community, or unknown — which this adapter folds into the
// numeric [0,1] confidence modelscore stores. That enum is evidence about the
// evidence, never a score multiplier: a community row is still ingested, it is
// just marked less trusted so a downstream policy can discount it.
package benchcatalog

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelscore"
)

// IngestSchema is the on-disk schema tag every snapshot fixture must carry, so
// an older or forked snapshot shape fails loud on load rather than silently
// ingesting the wrong fields.
const IngestSchema = "fak.benchcatalog.snapshot.v1"

// SourceConfidence is the trust class of a whole snapshot's rows — WHERE the
// numbers came from, on the one axis that survives across benchmarks. It is an
// enum, not a free float, because a snapshot's trust is a category (the operator
// read it off the official leaderboard, a vendor's own report, a community
// tracker, or an unlabeled paste) and categories do not drift the way an
// eyeballed 0.83 does. The adapter maps each class to the numeric [0,1]
// confidence modelscore stores; the mapping lives in Weight below.
type SourceConfidence string

const (
	// ConfidenceOfficial is a row read off the benchmark's own official
	// leaderboard or maintainer-published results — the highest trust class.
	ConfidenceOfficial SourceConfidence = "official"
	// ConfidenceVendorReported is a number the model's own vendor published
	// (a launch post, a model card). Real, but self-reported and unaudited.
	ConfidenceVendorReported SourceConfidence = "vendor_reported"
	// ConfidenceCommunity is a row from a third-party tracker or community
	// leaderboard — useful signal, weaker provenance than an official page.
	ConfidenceCommunity SourceConfidence = "community"
	// ConfidenceUnknown is the fail-safe class for a row whose trust is not
	// stated. It is ingested (the shape is real) but weighted lowest, so an
	// unlabeled figure never poses as an official one.
	ConfidenceUnknown SourceConfidence = "unknown"
)

// Weight maps a source-confidence class to the numeric [0,1] belief modelscore
// stores. It is the ONE place the enum-to-float mapping lives; keeping it here
// (not scattered at each call site) means the trust ladder is a single, auditable
// table. An unrecognized class is treated as unknown — fail-safe, never a panic.
func (c SourceConfidence) Weight() float64 {
	switch c {
	case ConfidenceOfficial:
		return 0.95
	case ConfidenceVendorReported:
		return 0.75
	case ConfidenceCommunity:
		return 0.55
	default: // ConfidenceUnknown or any unrecognized label
		return 0.30
	}
}

// Valid reports whether c is one of the four known trust classes. The ingestor
// refuses an unrecognized confidence label rather than silently treating it as
// unknown, so a typo like "offical" fails loud instead of quietly downgrading a
// row's trust.
func (c SourceConfidence) Valid() bool {
	switch c {
	case ConfidenceOfficial, ConfidenceVendorReported, ConfidenceCommunity, ConfidenceUnknown:
		return true
	default:
		return false
	}
}

// SnapshotRow is ONE benchmark result as it appears in a committed snapshot
// fixture — the on-disk shape the ingestor validates and converts. Every field
// the Done condition names (source, version, date, model, metric) is required;
// the validator refuses a row missing any of them so an under-provenanced number
// never reaches modelscore.
type SnapshotRow struct {
	// Model is the model id the row is about (e.g. "opus", "glm-5.2"). Required:
	// a score with no model cannot say whom it measured.
	Model string `json:"model"`
	// Score is the native benchmark figure, UNBOUNDED, in Metric's unit. Kept in
	// the benchmark's own scale — never normalized, because normalization is the
	// policy blend one layer up.
	Score float64 `json:"score"`
	// Metric is the native unit/axis label (e.g. "pct-resolved", "mean@5"). This
	// is the "metric provenance" the Done condition requires: a bare number with
	// no metric label cannot be read honestly. Required.
	Metric string `json:"metric"`
	// Harness is the agent/scaffold that produced the row for an agent-plus-model
	// benchmark (Terminal-Bench). Required for Terminal-Bench (enforced in
	// convert), empty is allowed for a model-only benchmark.
	Harness string `json:"harness,omitempty"`
	// Source is the citation URL this row was read from. Required: a row with no
	// source is a guess, and the ingestor refuses it.
	Source string `json:"source"`
	// AsOf is the capture date (YYYY-MM-DD) — WHEN the number was observed off the
	// source. Required: an undated snapshot row cannot honestly be called
	// measured-as-of-anything, so it is refused.
	AsOf string `json:"as_of"`
	// Confidence is the trust class of this row's source. Required and must be one
	// of the four known classes.
	Confidence SourceConfidence `json:"confidence"`
	// Illustrative marks a fixture stand-in figure whose SHAPE is real but whose
	// number is a placeholder for a measurement not yet ingested. It is carried
	// through to modelscore so a downstream policy can discount or ignore it. fak
	// never invents a benchmark number and passes it off as measured.
	Illustrative bool `json:"illustrative,omitempty"`
}

// Snapshot is a committed export of one benchmark's leaderboard rows — the whole
// fixture the ingestor reads. It names the benchmark once (so every row inherits
// the same benchmark id + version), then lists the per-model rows.
type Snapshot struct {
	// Schema is the on-disk schema tag; must equal IngestSchema.
	Schema string `json:"schema"`
	// Benchmark is the benchmark id every row in this snapshot belongs to (e.g.
	// "terminal-bench", "swe-bench-verified", "frontier-swe"). Required.
	Benchmark string `json:"benchmark"`
	// Version is the benchmark version (e.g. "2.1", "verified"). Required: the
	// Done condition names version as mandatory provenance, and Terminal-Bench 2.0
	// / 2.1 / Hard are not interchangeable, so a versionless snapshot is refused.
	Version string `json:"version"`
	// Rows are the per-model result rows.
	Rows []SnapshotRow `json:"rows"`
}

// ParseSnapshot decodes a single snapshot fixture, rejecting unknown fields so a
// typo in a fixture fails loud rather than silently dropping a row's provenance.
// It validates only the envelope here (schema tag, benchmark id, version); the
// per-row provenance rules are enforced in Ingest so a caller who already holds a
// Snapshot value still gets them.
func ParseSnapshot(data []byte) (Snapshot, error) {
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	var s Snapshot
	if err := dec.Decode(&s); err != nil {
		return Snapshot{}, fmt.Errorf("benchcatalog: decode snapshot: %w", err)
	}
	if s.Schema != IngestSchema {
		return Snapshot{}, fmt.Errorf("benchcatalog: unexpected snapshot schema %q, want %q", s.Schema, IngestSchema)
	}
	if strings.TrimSpace(s.Benchmark) == "" {
		return Snapshot{}, fmt.Errorf("benchcatalog: snapshot benchmark id is required")
	}
	if strings.TrimSpace(s.Version) == "" {
		return Snapshot{}, fmt.Errorf("benchcatalog: snapshot %q: version is required (Terminal-Bench 2.0/2.1/Hard are not interchangeable)", s.Benchmark)
	}
	return s, nil
}

// benchmarkID is the canonical Terminal-Bench snapshot id; a row from this
// benchmark is agent-plus-model and MUST name a harness.
const benchmarkTerminalBench = "terminal-bench"

// validateRow enforces the fail-closed provenance rules the Done condition
// names: a row is refused unless it carries model, metric, source, and date, a
// known confidence class, and — for an agent-plus-model benchmark — a harness. It
// returns a precise error naming the missing field so a bad fixture is easy to
// fix.
func (s Snapshot) validateRow(i int, r SnapshotRow) error {
	where := fmt.Sprintf("benchcatalog: %s v%s row[%d]", s.Benchmark, s.Version, i)
	if strings.TrimSpace(r.Model) == "" {
		return fmt.Errorf("%s: model is required", where)
	}
	if strings.TrimSpace(r.Metric) == "" {
		return fmt.Errorf("%s (%s): metric is required (a bare score has no unit)", where, r.Model)
	}
	if strings.TrimSpace(r.Source) == "" {
		return fmt.Errorf("%s (%s): source is required (an unsourced row is a guess)", where, r.Model)
	}
	if strings.TrimSpace(r.AsOf) == "" {
		return fmt.Errorf("%s (%s): as_of date is required (an undated row cannot claim to be measured)", where, r.Model)
	}
	if !r.Confidence.Valid() {
		return fmt.Errorf("%s (%s): confidence %q is not one of official/vendor_reported/community/unknown", where, r.Model, r.Confidence)
	}
	if s.Benchmark == benchmarkTerminalBench && strings.TrimSpace(r.Harness) == "" {
		return fmt.Errorf("%s (%s): harness is required for %s (it scores an agent PLUS a model)", where, r.Model, benchmarkTerminalBench)
	}
	return nil
}

// toBenchScore converts one validated snapshot row into a modelscore.BenchScore,
// carrying the benchmark id + version from the snapshot envelope and folding the
// source-confidence enum into the numeric [0,1] belief modelscore stores.
func (s Snapshot) toBenchScore(r SnapshotRow) modelscore.BenchScore {
	return modelscore.BenchScore{
		Benchmark: s.Benchmark,
		Version:   s.Version,
		Score:     r.Score,
		Unit:      r.Metric,
		Harness:   r.Harness,
		Provenance: modelscore.Provenance{
			Source:       r.Source,
			AsOf:         r.AsOf,
			Confidence:   r.Confidence.Weight(),
			Illustrative: r.Illustrative,
		},
	}
}

// Ingest validates every row in the given snapshots and folds them into a fresh
// modelscore.Registry, one BenchScore per row, grouped by model. It is the whole
// adapter: snapshot bytes in, provenanced evidence rows out. It fails LOUD on the
// FIRST under-provenanced row — a missing source, version, date, model, or metric
// — so a partial ingest never half-populates the registry with trustworthy and
// guessed rows side by side. On success the registry holds exactly the rows the
// snapshots described, and modelscore's own validator has re-checked each one.
func Ingest(snaps ...Snapshot) (*modelscore.Registry, error) {
	// Group rows by model across all snapshots so one model measured on three
	// benchmarks becomes one ModelEvidence with three BenchScores, in a stable
	// order.
	byModel := map[string][]modelscore.BenchScore{}
	var order []string
	for si, snap := range snaps {
		if snap.Schema != IngestSchema {
			return nil, fmt.Errorf("benchcatalog: snapshot[%d] %q: unexpected schema %q, want %q", si, snap.Benchmark, snap.Schema, IngestSchema)
		}
		if strings.TrimSpace(snap.Benchmark) == "" {
			return nil, fmt.Errorf("benchcatalog: snapshot[%d]: benchmark id is required", si)
		}
		if strings.TrimSpace(snap.Version) == "" {
			return nil, fmt.Errorf("benchcatalog: snapshot[%d] %q: version is required", si, snap.Benchmark)
		}
		for ri, row := range snap.Rows {
			if err := snap.validateRow(ri, row); err != nil {
				return nil, err
			}
			if _, seen := byModel[row.Model]; !seen {
				order = append(order, row.Model)
			}
			byModel[row.Model] = append(byModel[row.Model], snap.toBenchScore(row))
		}
	}

	reg := modelscore.NewRegistry()
	// Deterministic model order so a re-ingest is byte-stable and diff-friendly.
	sort.Strings(order)
	for _, model := range order {
		ev := modelscore.ModelEvidence{Model: model, Benchmarks: byModel[model]}
		if err := reg.Add(ev); err != nil {
			// modelscore's validator is the second, independent gate; a row that
			// passed our provenance check but trips a finite/range rule there still
			// fails loud rather than entering the registry.
			return nil, fmt.Errorf("benchcatalog: ingest %s: %w", model, err)
		}
	}
	return reg, nil
}

// IngestBytes is the convenience door for the CLI: it parses each snapshot's raw
// bytes (envelope-validating as it goes) and then ingests them, so a caller with
// three files on disk needs one call. The names slice, when non-empty, labels
// each blob in error messages so a bad fixture is named by file, not by index.
func IngestBytes(names []string, blobs ...[]byte) (*modelscore.Registry, error) {
	snaps := make([]Snapshot, 0, len(blobs))
	for i, blob := range blobs {
		s, err := ParseSnapshot(blob)
		if err != nil {
			if i < len(names) && names[i] != "" {
				return nil, fmt.Errorf("%s: %w", names[i], err)
			}
			return nil, err
		}
		snaps = append(snaps, s)
	}
	return Ingest(snaps...)
}
