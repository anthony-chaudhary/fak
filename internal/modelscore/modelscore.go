// Package modelscore is the durable, pure registry of RAW model-capability
// evidence — the source-of-truth score shape that a tier policy (C3) and a
// dispatch chooser (C5) read, but that this package deliberately does NOT
// interpret. It replaces the brittle hard-coded model-name heuristic in
// internal/fleetaccounts (modelTierFromName) with rows that carry Terminal-Bench,
// SWE-bench, FrontierSWE, cost, and context as SEPARATE, provenanced evidence.
//
// ---------------------------------------------------------------------------
// THE ONE LAW: EVIDENCE, NEVER A DECISION.
// ---------------------------------------------------------------------------
//
// A registry row records what was MEASURED about a model — a benchmark result in
// the benchmark's own native units, a rough price, a context window — each with
// the source it came from and a rough confidence. A row is NOT a routing
// decision and, on its own, MUST NOT admit a weaker model to harder work. The
// blend that turns evidence into a tier is a POLICY VIEW that lives one layer up
// (C3, internal/modelroute); this package hands that policy the raw rows and
// stops there.
//
// Two honesty rules make that separation real, and the validator and the Profile
// query below enforce them:
//
//   - UNBOUNDED, NATIVE UNITS. Each BenchScore keeps its benchmark's own scale
//     and unit — a percent-resolved on SWE-bench, a percent on Terminal-Bench, a
//     long-horizon FrontierSWE figure. We never normalize them onto a shared
//     0..100 axis, because that normalization IS the policy blend, and a native
//     score may exceed any prior maximum as the field moves. Terminal-Bench rows
//     are agent-PLUS-model results, so a BenchScore carries the Harness that
//     produced it; the model alone did not earn the number.
//
//   - SCORE, COST, TRUST, AND RISK STAY DISTINCT. We do not collapse them into
//     one figure. A Profile returns the raw benchmark rows, the raw cost, and the
//     raw context side by side; it computes no blended ranking, so a high
//     benchmark score can never silently buy past a capability floor downstream.
//
// PURITY (lane rule): stdlib-only, imports nothing internal, off the hot path.
// Every query is in-memory — no network, no credentials — so the acceptance gate
// runs anywhere.
package modelscore

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Schema is the on-disk schema tag for a serialized registry snapshot. It is
// written and checked on load so an older or forked shape fails loud rather than
// silently dropping fields.
const Schema = "fak.modelscore-registry.v1"

// Provenance records WHERE a piece of evidence came from and how much to trust
// it — the difference between a measured row and a guessed one. Every evidence
// value (a benchmark score, a price, a context window) carries its own
// Provenance; the registry refuses a row whose evidence has no source.
type Provenance struct {
	SubmissionTruthTier   SubmissionTruthTier `json:"submission_truth_tier"`
	SubmissionSource      string              `json:"submission_source,omitempty"`
	AuthorSource          string              `json:"author_source,omitempty"`
	ReconstructionWitness string              `json:"reconstruction_witness,omitempty"`
	// Source is the citation the value came from — a leaderboard URL, a repo, a
	// published price page. Required: an evidence value with no source is a guess,
	// and the validator refuses it.
	Source string `json:"source"`
	// AsOf is the capture date (YYYY-MM-DD) so a stale row is visible as stale.
	// Optional: a missing date is a soft gap, not a hard refusal.
	AsOf string `json:"as_of,omitempty"`
	// Confidence is a rough 0..1 belief in this value — sample-size, harness
	// match, and freshness folded into one hand-set number. It is EVIDENCE about
	// the evidence, never a score multiplier here. Must be within [0,1].
	Confidence float64 `json:"confidence"`
	// Illustrative marks a fixture/placeholder value that stands in for a real
	// measurement not yet ingested. fak never invents a benchmark number and
	// passes it off as measured; a true here says "shape is real, this figure is
	// a stand-in" so a downstream policy can discount or ignore it.
	Illustrative bool `json:"illustrative,omitempty"`
}

// BenchScore is ONE benchmark result for ONE model, in the benchmark's OWN
// native units — never normalized. It is the atom of unbounded evidence.
type BenchScore struct {
	// Benchmark is the benchmark id, e.g. "terminal-bench", "swe-bench-verified",
	// "frontier-swe". Required.
	Benchmark string `json:"benchmark"`
	// Version is the benchmark version, e.g. "2.1". Optional but recommended: a
	// score without a version cannot be compared across benchmark revisions.
	Version string `json:"version,omitempty"`
	// Score is the native score, in Unit, UNBOUNDED — it may exceed any prior
	// maximum. Must be a finite number.
	Score float64 `json:"score"`
	// Unit is the native unit the Score is in, e.g. "pct-resolved", "pct",
	// "tasks". Required: a bare number with no unit cannot be read honestly.
	Unit string `json:"unit"`
	// Harness is the agent/scaffold that produced the row when the benchmark
	// scores an agent PLUS a model (Terminal-Bench). Empty for a model-only
	// benchmark. Kept because the model alone did not earn an agentic number.
	Harness string `json:"harness,omitempty"`
	// Provenance is where this score came from and how much to trust it.
	Provenance Provenance `json:"provenance"`
}

// Cost is a rough $/Mtok price for a model, split input/output to match the
// repo's canonical cost convention (see internal/modelroute/cost.go). It is a
// cost LENS, never a bill; the Provenance carries the price source.
type Cost struct {
	In         float64    `json:"in"`  // $/Mtok input
	Out        float64    `json:"out"` // $/Mtok output
	Provenance Provenance `json:"provenance"`
}

// ContextWindow is a model's usable context in tokens, with its own provenance.
type ContextWindow struct {
	Tokens     int        `json:"tokens"`
	Provenance Provenance `json:"provenance"`
}

// ModelEvidence is the durable raw evidence the registry stores for one model:
// every benchmark row plus optional cost and context, each provenanced. It holds
// NO blended score and NO tier — those are a policy view one layer up.
type ModelEvidence struct {
	// Model is the model id this evidence is about (e.g. "opus", "glm-5.2"). It
	// is the registry key. Required.
	Model string `json:"model"`
	// Benchmarks are the raw, unbounded, native-unit score rows for this model.
	Benchmarks []BenchScore `json:"benchmarks,omitempty"`
	// Cost is the rough price for this model, if known.
	Cost *Cost `json:"cost,omitempty"`
	// Context is the model's context window, if known.
	Context *ContextWindow `json:"context,omitempty"`
	// Notes is optional free text — never parsed, only surfaced to a human.
	Notes string `json:"notes,omitempty"`
}

// Registry is an in-memory set of ModelEvidence keyed by model id. It is
// append-and-query only: Add validates and stores, Profile reads back raw
// evidence. It computes no ranking and makes no routing decision.
type Registry struct {
	models map[string]ModelEvidence
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return &Registry{models: map[string]ModelEvidence{}} }

// Add validates ev and stores it under ev.Model, replacing any prior evidence
// for that model. It fails LOUD on a malformed row — a missing model id, a
// benchmark with no name/unit/source, an out-of-range confidence, or a
// non-finite number — so bad evidence never enters the registry silently.
func (r *Registry) Add(ev ModelEvidence) error {
	if err := ev.validate(); err != nil {
		return err
	}
	if r.models == nil {
		r.models = map[string]ModelEvidence{}
	}
	r.models[ev.Model] = ev
	return nil
}

// validate enforces the fail-closed shape rules for one model's evidence.
func (ev ModelEvidence) validate() error {
	if strings.TrimSpace(ev.Model) == "" {
		return fmt.Errorf("modelscore: model id is required")
	}
	for i, b := range ev.Benchmarks {
		if strings.TrimSpace(b.Benchmark) == "" {
			return fmt.Errorf("modelscore: %s benchmark[%d]: name is required", ev.Model, i)
		}
		if strings.TrimSpace(b.Unit) == "" {
			return fmt.Errorf("modelscore: %s benchmark %q: unit is required (native units, never normalized)", ev.Model, b.Benchmark)
		}
		if err := finite(b.Score, fmt.Sprintf("%s benchmark %q score", ev.Model, b.Benchmark)); err != nil {
			return err
		}
		if err := b.Provenance.validate(fmt.Sprintf("%s benchmark %q", ev.Model, b.Benchmark)); err != nil {
			return err
		}
	}
	if ev.Cost != nil {
		if err := finite(ev.Cost.In, ev.Model+" cost.in"); err != nil {
			return err
		}
		if err := finite(ev.Cost.Out, ev.Model+" cost.out"); err != nil {
			return err
		}
		if ev.Cost.In < 0 || ev.Cost.Out < 0 {
			return fmt.Errorf("modelscore: %s cost: price cannot be negative", ev.Model)
		}
		if err := ev.Cost.Provenance.validate(ev.Model + " cost"); err != nil {
			return err
		}
	}
	if ev.Context != nil {
		if ev.Context.Tokens < 0 {
			return fmt.Errorf("modelscore: %s context: tokens cannot be negative", ev.Model)
		}
		if err := ev.Context.Provenance.validate(ev.Model + " context"); err != nil {
			return err
		}
	}
	return nil
}

// validate enforces that provenance carries a source and an in-range confidence.
func (p Provenance) validate(what string) error {
	if strings.TrimSpace(p.Source) == "" {
		return fmt.Errorf("modelscore: %s: provenance source is required (evidence needs a citation)", what)
	}
	if p.Confidence < 0 || p.Confidence > 1 || math.IsNaN(p.Confidence) {
		return fmt.Errorf("modelscore: %s: confidence %v out of range [0,1]", what, p.Confidence)
	}
	return nil
}

func finite(v float64, what string) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("modelscore: %s: value must be finite, got %v", what, v)
	}
	return nil
}

// Profile is the per-model capability profile the registry returns: the RAW
// evidence for one model, side by side, with NO blended score and NO tier. It is
// a read-only view — a policy view (C3) folds these rows into a tier; this shape
// deliberately does not, so the boundary between evidence and decision is
// structural, not a matter of discipline.
type Profile struct {
	Model      string         `json:"model"`
	Benchmarks []BenchScore   `json:"benchmarks,omitempty"`
	Cost       *Cost          `json:"cost,omitempty"`
	Context    *ContextWindow `json:"context,omitempty"`
	Notes      string         `json:"notes,omitempty"`
}

// Profile returns the raw capability profile for model, or false if unknown. The
// returned slices/pointers are deep copies, so a caller cannot mutate registry
// state through the profile.
func (r *Registry) Profile(model string) (Profile, bool) {
	ev, ok := r.models[model]
	if !ok {
		return Profile{}, false
	}
	return ev.profile(), true
}

func (ev ModelEvidence) profile() Profile {
	p := Profile{Model: ev.Model, Notes: ev.Notes}
	if len(ev.Benchmarks) > 0 {
		p.Benchmarks = append([]BenchScore(nil), ev.Benchmarks...)
	}
	if ev.Cost != nil {
		c := *ev.Cost
		p.Cost = &c
	}
	if ev.Context != nil {
		c := *ev.Context
		p.Context = &c
	}
	return p
}

// Benchmark returns the raw score row for the named benchmark, or false if this
// profile has none. It is a lookup over the RAW rows — it does no normalization
// and no cross-benchmark blend; the score comes back in its native unit exactly
// as stored.
func (p Profile) Benchmark(name string) (BenchScore, bool) {
	for _, b := range p.Benchmarks {
		if b.Benchmark == name {
			return b, true
		}
	}
	return BenchScore{}, false
}

// Models returns the registry's model ids, sorted, for deterministic iteration.
func (r *Registry) Models() []string {
	out := make([]string, 0, len(r.models))
	for m := range r.models {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

// snapshot is the serialized shape: a schema tag plus the evidence rows sorted by
// model id, so a round-trip is byte-stable and diff-friendly.
type snapshot struct {
	Schema string          `json:"schema"`
	Models []ModelEvidence `json:"models"`
}

// MarshalJSON writes the registry as a schema-tagged, model-sorted snapshot.
func (r *Registry) MarshalJSON() ([]byte, error) {
	snap := snapshot{Schema: Schema, Models: make([]ModelEvidence, 0, len(r.models))}
	for _, m := range r.Models() {
		snap.Models = append(snap.Models, r.models[m])
	}
	return json.Marshal(snap)
}

// Load parses a registry snapshot, validating the schema tag and every row.
// Unknown fields are rejected so a typo in an evidence file fails loud rather
// than silently dropping evidence.
func Load(data []byte) (*Registry, error) {
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	var snap snapshot
	if err := dec.Decode(&snap); err != nil {
		return nil, fmt.Errorf("modelscore: decode snapshot: %w", err)
	}
	if snap.Schema != Schema {
		return nil, fmt.Errorf("modelscore: unexpected schema %q, want %q", snap.Schema, Schema)
	}
	r := NewRegistry()
	for _, ev := range snap.Models {
		if err := r.Add(ev); err != nil {
			return nil, err
		}
	}
	return r, nil
}
