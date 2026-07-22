// Package quality is the runnable spine of the "missing middle" validation
// ladder (epic #4509): the layer between primitive correctness tests (too local
// to catch a fluent-but-wrong decode) and end benchmarks (too coarse and late to
// localize an engine-caused regression).
//
// The spine runs one versioned prompt case through a REFERENCE path and one fak
// ENGINE path, captures request/config/token/logit/output provenance, applies at
// least one deterministic comparator and one rubric scorer, and emits a
// machine-readable result with a pass/fail verdict and — on failure — a portable,
// replay-complete failure bundle. A quality claim is never green from a single
// stochastic sample: every oracle names the baseline it compares against and
// reports the FIRST divergence so a defect localizes to a token/step rather than
// being observed only in prose.
//
// The package is deliberately stdlib-only and hermetic: the reference and engine
// paths are Runner adapters (case.go / runner.go), so the spine is CI-runnable
// without a live model, and real engine adapters wire in behind the same seam.
// The child cohort under #4509 extends this by ADDING its own oracle/runner files
// and registering them — it does not edit these cores.
package quality

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
)

// CaseSchema is the versioned envelope tag for a quality case. Bumping the
// trailing version is a breaking change to the case shape; readers reject an
// unknown major so a stale corpus can never masquerade as current.
const CaseSchema = "fak-quality-case/1"

// QualityCase is one versioned, replayable unit of quality evidence: a prompt,
// the sampling configuration it must be decoded under, the reference expectation
// it is judged against, and the named oracles to apply. It carries everything a
// runner needs to reproduce the run — nothing about a case depends on wall-clock
// or ambient state.
type QualityCase struct {
	Schema  string         `json:"schema"`
	ID      string         `json:"id"`
	Version int            `json:"version"`
	Prompt  string         `json:"prompt"`
	Params  SamplingParams `json:"params"`
	// Reference is the golden trace the engine path is compared against. For a
	// deterministic (temperature-zero / greedy) case this is an exact oracle; a
	// stochastic case carries a distribution instead (added by the sampling
	// children #4529–#4530 without touching this struct).
	Reference Trace `json:"reference"`
	// Oracles names the comparators/scorers to apply, by Oracle.Name(). An empty
	// list is a case authored without any judge — RunCase reports it rather than
	// silently passing (a case that checks nothing is not a green case).
	Oracles []string `json:"oracles"`
	// Rubric configures the rubric scorer(s) for this case (grounding phrases,
	// forbidden claims, threshold). Optional; empty means no rubric dimension.
	Rubric RubricSpec `json:"rubric,omitempty"`
	// Metadata pins the execution and baseline provenance required by canonical
	// v1 corpus fixtures. LoadCase rejects a fixture when any required lineage,
	// determinism, tier, or resource-cost field is absent.
	Metadata CaseMetadata `json:"metadata,omitempty"`
	Tags     []string     `json:"tags,omitempty"`
}

// CaseMetadata pins every external input needed to reproduce and route a case.
type CaseMetadata struct {
	Model     Revision       `json:"model"`
	Tokenizer Revision       `json:"tokenizer"`
	Engine    EngineSpec     `json:"engine"`
	Code      Revision       `json:"code"`
	Oracle    OracleEvidence `json:"oracle"`
	Tolerance ToleranceSpec  `json:"tolerance"`
	Baseline  BaselineSpec   `json:"baseline"`
	Tier      TierSpec       `json:"tier"`
	Cost      CostSpec       `json:"cost"`
	// Owner is the accountable team or person for this case — a case with no
	// owner has no one to fix it when it fails, so #4574 refuses it. Family is
	// the evidence class the suite splitter routes on (see EvidenceFamily).
	Owner  string `json:"owner"`
	Family string `json:"family"`
}

// Revision identifies immutable model, tokenizer, or code/module content.
type Revision struct {
	Name     string `json:"name"`
	Revision string `json:"revision"`
}

// EngineSpec identifies the implementation and backend plus replay-affecting flags.
type EngineSpec struct {
	Name    string            `json:"name"`
	Backend string            `json:"backend"`
	Flags   map[string]string `json:"flags,omitempty"`
}

// OracleEvidence describes deterministic evidence when a case is not seed-pinned.
type OracleEvidence struct {
	Kind     string `json:"kind"`
	Revision string `json:"revision"`
}

// ToleranceSpec names the tolerance policy and its immutable source.
type ToleranceSpec struct {
	Metric   string  `json:"metric"`
	Absolute float64 `json:"absolute,omitempty"`
	Relative float64 `json:"relative,omitempty"`
	Revision string  `json:"revision"`
}

// BaselineSpec pins the independently produced baseline artifact.
type BaselineSpec struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
}

// TierSpec routes a case to exactly one validation cadence.
type TierSpec struct {
	Name string `json:"name"`
}

// CostSpec documents expected runtime and peak resource requirements.
// TimeoutSeconds is the hard wall a run is killed at — distinct from the
// expected RuntimeSeconds — so a hung case cannot silently hold a suite's
// budget open. #4574 requires every case to declare it.
type CostSpec struct {
	RuntimeSeconds int64 `json:"runtime_seconds"`
	TimeoutSeconds int64 `json:"timeout_seconds"`
	CPU            int   `json:"cpu"`
	MemoryMiB      int64 `json:"memory_mib"`
	Accelerators   int   `json:"accelerators,omitempty"`
}

// SamplingParams is the decode configuration a case is pinned to. It is part of
// the replay contract: a result is only reproducible if the params that produced
// it travel with it. Seed is honored by stochastic runners; a temperature-zero
// case must decode identically regardless of seed (frozen by #4525).
type SamplingParams struct {
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	MaxTokens   int     `json:"max_tokens"`
	Seed        int64   `json:"seed,omitempty"`
}

// Trace is the captured output of one decode path: the emitted token sequence,
// optional per-step logits (top candidates), and the assembled text. Tokens are
// the primary differential surface — text scoring happens only after the token
// stream is proven equal or explicitly allowed to differ, because two paths can
// assemble identical-looking text from divergent tokens.
type Trace struct {
	Runner string      `json:"runner"`
	Tokens []string    `json:"tokens"`
	Logits [][]float64 `json:"logits,omitempty"`
	Text   string      `json:"text"`
}

// RubricSpec is the deterministic rubric configuration for a case: phrases the
// grounded output must contain, claims it must not contain, and the minimum
// fraction of required phrases to pass. It is intentionally a deterministic
// stand-in the report-quality children (#4550–#4565) replace with calibrated
// judges — but even the stand-in makes a "reads fine" report fail when it drops a
// material, required claim.
type RubricSpec struct {
	Required  []string `json:"required,omitempty"`
	Forbidden []string `json:"forbidden,omitempty"`
	MinScore  float64  `json:"min_score,omitempty"`
}

// Valid reports whether the case envelope is well-formed enough to run. It is the
// admission gate: a case with the wrong schema, no ID, or no oracles is refused
// with a reason rather than run to a misleading green.
func (c QualityCase) Valid() (bool, string) {
	if c.Schema != CaseSchema {
		return false, "case schema " + c.Schema + " != " + CaseSchema
	}
	if c.ID == "" {
		return false, "case has empty id"
	}
	if len(c.Oracles) == 0 {
		return false, "case declares no oracles (a case that checks nothing is not green)"
	}
	return true, ""
}

// ValidateCanonical enforces the complete v1 corpus contract. Valid remains the
// compatibility admission gate for cases constructed by older in-package helpers;
// persisted fixtures enter through LoadCase and always receive this stricter check.
func (c QualityCase) ValidateCanonical() error {
	if ok, why := c.Valid(); !ok {
		return fmt.Errorf("%s", why)
	}
	if c.Version != 1 {
		return fmt.Errorf("version must be 1")
	}
	if strings.TrimSpace(c.Prompt) == "" {
		return fmt.Errorf("prompt is empty")
	}
	if c.Params.MaxTokens <= 0 {
		return fmt.Errorf("params.max_tokens must be positive")
	}
	m := c.Metadata
	for name, rev := range map[string]Revision{"model": m.Model, "tokenizer": m.Tokenizer, "code": m.Code} {
		if strings.TrimSpace(rev.Name) == "" || strings.TrimSpace(rev.Revision) == "" {
			return fmt.Errorf("metadata.%s requires name and revision", name)
		}
	}
	if strings.TrimSpace(m.Engine.Name) == "" || strings.TrimSpace(m.Engine.Backend) == "" || len(m.Engine.Flags) == 0 {
		return fmt.Errorf("metadata.engine requires name, backend, and replay flags")
	}
	if c.Params.Seed == 0 && (strings.TrimSpace(m.Oracle.Kind) == "" || strings.TrimSpace(m.Oracle.Revision) == "") {
		return fmt.Errorf("case requires a non-zero seed or metadata.oracle kind and revision")
	}
	if strings.TrimSpace(m.Tolerance.Metric) == "" || strings.TrimSpace(m.Tolerance.Revision) == "" {
		return fmt.Errorf("metadata.tolerance requires metric and revision")
	}
	if math.IsNaN(m.Tolerance.Absolute) || math.IsNaN(m.Tolerance.Relative) || math.IsInf(m.Tolerance.Absolute, 0) || math.IsInf(m.Tolerance.Relative, 0) || m.Tolerance.Absolute < 0 || m.Tolerance.Relative < 0 {
		return fmt.Errorf("metadata.tolerance values must be finite and non-negative")
	}
	if strings.TrimSpace(m.Baseline.ID) == "" || strings.TrimSpace(m.Baseline.Revision) == "" {
		return fmt.Errorf("metadata.baseline requires id and revision")
	}
	switch m.Tier.Name {
	case "pr", "nightly", "release":
	default:
		return fmt.Errorf("metadata.tier.name must be pr, nightly, or release")
	}
	if m.Cost.RuntimeSeconds <= 0 || m.Cost.CPU <= 0 || m.Cost.MemoryMiB <= 0 || m.Cost.Accelerators < 0 {
		return fmt.Errorf("metadata.cost requires positive runtime_seconds, cpu, and memory_mib")
	}
	if m.Cost.TimeoutSeconds <= 0 {
		return fmt.Errorf("metadata.cost requires a positive timeout_seconds")
	}
	if m.Cost.TimeoutSeconds < m.Cost.RuntimeSeconds {
		return fmt.Errorf("metadata.cost.timeout_seconds must be >= runtime_seconds")
	}
	if strings.TrimSpace(m.Owner) == "" {
		return fmt.Errorf("metadata.owner is required (a case with no owner has no one to fix it)")
	}
	if !validFamily(m.Family) {
		return fmt.Errorf("metadata.family must be one of deterministic, gpu_parity, statistics, corpora, review")
	}
	return nil
}

// LoadCase decodes one canonical v1 fixture, refusing unknown fields, trailing
// documents, malformed provenance, and fixtures that cannot prove determinism.
func LoadCase(data []byte) (QualityCase, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var c QualityCase
	if err := dec.Decode(&c); err != nil {
		return QualityCase{}, fmt.Errorf("decode quality case: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return QualityCase{}, fmt.Errorf("decode quality case: trailing document")
		}
		return QualityCase{}, fmt.Errorf("decode quality case: %w", err)
	}
	if err := c.ValidateCanonical(); err != nil {
		return QualityCase{}, fmt.Errorf("invalid canonical case %q: %w", c.ID, err)
	}
	return c, nil
}
