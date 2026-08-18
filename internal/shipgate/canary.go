package shipgate

// canary.go — quality-aware canary promotion and rollback (issue #4580, epic
// #4509 middle ladder).
//
// shipgate.go decides KEEP-or-REVERT for one candidate from a witness it did not
// author. A canary is the deploy-time complement: a candidate revision serves a
// slice of traffic and is PROMOTED, HELD, or ROLLED BACK from measured quality
// deltas — the practice vLLM/SGLang-class stacks use between primitive
// correctness tests and coarse end benchmarks. The adjudicator here is a pure,
// deterministic oracle (stdlib-only): a PR-tier run needs no model call, seed,
// or GPU, and the same replay artifact re-adjudicates to the same verdict in a
// clean environment.
//
// Contract (the issue's acceptance criteria):
//   - Three outcomes: PROMOTE, HOLD (inconclusive), ROLLBACK. Missing or
//     inconclusive evidence is never pass — it HOLDs the canary at the current
//     baseline, the safe default, and never justifies an automated action.
//   - A critical slice that drops past its tolerance ROLLs BACK even when the
//     aggregate mean rises; the FIRST actionable divergence is named.
//   - PROMOTE needs full provenance (model, tokenizer, engine/backend, seed or
//     deterministic oracle, code revision, tolerance/baseline provenance), the
//     declared minimum evidence (samples per slice), every critical slice
//     within tolerance, AND a non-negative aggregate quality delta.
//   - Every non-pass emits a scrubbed replay artifact (host paths, emails, and
//     secret-shaped values redacted) that reconstructs the case byte-for-byte
//     enough to reproduce the verdict independently.
//   - Every case is pinned to a PR / nightly / release tier with a cost note.

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// CanarySchema versions the emitted replay artifact.
const CanarySchema = "fak-quality-canary/1"

// CanaryVerdict is the three-way canary decision. The zero value is HOLD so an
// unset verdict fails closed to "do not promote".
type CanaryVerdict uint8

const (
	CanaryHold     CanaryVerdict = iota // inconclusive evidence — pin the baseline
	CanaryPromote                       // fully-evidenced strict pass — promote the candidate
	CanaryRollback                      // a critical slice regressed past tolerance — roll back
)

// String renders the verdict as a stable token.
func (v CanaryVerdict) String() string {
	switch v {
	case CanaryPromote:
		return "PROMOTE"
	case CanaryRollback:
		return "ROLLBACK"
	}
	return "HOLD"
}

// CanaryTier is the CI cadence a canary case is assigned to. A case with an
// unrecognized tier is inconclusive: an unscheduled gate never runs.
type CanaryTier string

const (
	CanaryTierPR      CanaryTier = "pr"      // per-PR deterministic gate (no model call)
	CanaryTierNightly CanaryTier = "nightly" // nightly statistical / sampled gate
	CanaryTierRelease CanaryTier = "release" // release / hardware qualification gate
)

func (t CanaryTier) known() bool {
	switch t {
	case CanaryTierPR, CanaryTierNightly, CanaryTierRelease:
		return true
	}
	return false
}

// CanaryProvenance is the per-case evidence a promotion or rollback must carry
// so the decision is reproducible. A blank required field is inconclusive.
type CanaryProvenance struct {
	Model     string `json:"model"`     // model under test
	Tokenizer string `json:"tokenizer"` // tokenizer / vocab revision
	Engine    string `json:"engine"`    // engine / backend / engine-mode
	Seed      string `json:"seed"`      // RNG seed OR deterministic-oracle id
	Revision  string `json:"revision"`  // code / module revision
	Baseline  string `json:"baseline"`  // tolerance / baseline provenance
}

// missing returns the names of blank required provenance fields, in a stable
// order, so a fail-closed reason is deterministic.
func (p CanaryProvenance) missing() []string {
	var out []string
	for _, f := range []struct {
		name string
		val  string
	}{
		{"model", p.Model}, {"tokenizer", p.Tokenizer}, {"engine", p.Engine},
		{"seed", p.Seed}, {"revision", p.Revision}, {"baseline", p.Baseline},
	} {
		if strings.TrimSpace(f.val) == "" {
			out = append(out, f.name)
		}
	}
	return out
}

// scrubbed returns a copy with host paths, emails, and secret-shaped values
// redacted from every field.
func (p CanaryProvenance) scrubbed() CanaryProvenance {
	return CanaryProvenance{
		Model: canaryScrub(p.Model), Tokenizer: canaryScrub(p.Tokenizer),
		Engine: canaryScrub(p.Engine), Seed: canaryScrub(p.Seed),
		Revision: canaryScrub(p.Revision), Baseline: canaryScrub(p.Baseline),
	}
}

// QualitySlice is one measured quality cohort of the canary (a task, language,
// context-length, or engine-mode slice). Scores share one scale (higher is
// better). A critical slice is one whose loss must roll the canary back
// regardless of the aggregate.
type QualitySlice struct {
	Name      string  `json:"name"`
	Critical  bool    `json:"critical"`
	Baseline  float64 `json:"baseline"`
	Candidate float64 `json:"candidate"`
	Tolerance float64 `json:"tolerance"` // max allowed drop (>= 0) for a critical slice
	Samples   int     `json:"samples"`   // observations behind the candidate score
	Measured  bool    `json:"measured"`  // false => inconclusive evidence for this slice
}

func (s QualitySlice) delta() float64 { return s.Candidate - s.Baseline }

// CanaryCase is a fully-provenanced canary promotion case.
type CanaryCase struct {
	ID         string           `json:"id"`
	Tier       CanaryTier       `json:"tier"`
	CostNote   string           `json:"cost_note"`   // runtime / resource cost, e.g. "~1ms CPU, no GPU"
	MinSamples int              `json:"min_samples"` // minimum evidence for promotion, per slice (>= 1)
	Provenance CanaryProvenance `json:"provenance"`
	Slices     []QualitySlice   `json:"slices"`
}

// CanaryDivergence localizes the first actionable critical-slice regression.
type CanaryDivergence struct {
	Slice     string  `json:"slice"`
	Baseline  float64 `json:"baseline"`
	Candidate float64 `json:"candidate"`
	Delta     float64 `json:"delta"`
	Tolerance float64 `json:"tolerance"`
	Reason    string  `json:"reason"`
}

// CanaryReplay is the scrubbed, machine-readable bundle emitted with every
// verdict so the case can be re-adjudicated in a clean environment without
// leaking host secrets. It carries everything ReplayCase needs to reconstruct
// the case.
type CanaryReplay struct {
	Schema          string            `json:"schema"`
	CaseID          string            `json:"case_id"`
	Tier            CanaryTier        `json:"tier"`
	CostNote        string            `json:"cost_note"`
	MinSamples      int               `json:"min_samples"`
	Provenance      CanaryProvenance  `json:"provenance"`
	Slices          []QualitySlice    `json:"slices"`
	FirstDivergence *CanaryDivergence `json:"first_divergence,omitempty"`
	Scrubbed        bool              `json:"scrubbed"`
}

// ReplayCase reconstructs an adjudicable case from a replay artifact. Scrubbing
// redacts strings but never scores, flags, or counts, so a replayed case
// re-adjudicates to the same verdict — the "independently replayed" witness.
func ReplayCase(r CanaryReplay) CanaryCase {
	return CanaryCase{
		ID: r.CaseID, Tier: r.Tier, CostNote: r.CostNote,
		MinSamples: r.MinSamples, Provenance: r.Provenance,
		Slices: append([]QualitySlice(nil), r.Slices...),
	}
}

// CanaryResult is the verdict for one canary case.
type CanaryResult struct {
	Schema          string            `json:"schema"`
	CaseID          string            `json:"case_id"`
	Tier            CanaryTier        `json:"tier"`
	Verdict         string            `json:"verdict"`
	Promoted        bool              `json:"promoted"`
	Reason          string            `json:"reason"`
	BaselineMean    float64           `json:"baseline_mean"`
	CandidateMean   float64           `json:"candidate_mean"`
	FirstDivergence *CanaryDivergence `json:"first_divergence,omitempty"`
	Replay          CanaryReplay      `json:"replay"`
}

// AdjudicateCanary decides PROMOTE / HOLD / ROLLBACK for one canary case. It
// fails closed: any missing or inconclusive evidence HOLDs (never a pass, and
// never an automated rollback either — an unattributable signal is handed to a
// human, not acted on), a critical-slice loss past tolerance ROLLs BACK even
// when the aggregate mean rises, and only a fully-evidenced case with every
// critical slice within tolerance and a non-negative aggregate quality delta
// PROMOTEs.
func AdjudicateCanary(c CanaryCase) CanaryResult {
	res := CanaryResult{
		Schema: CanarySchema, CaseID: c.ID, Tier: c.Tier,
		BaselineMean:  mathx.MeanBy(c.Slices, func(s QualitySlice) float64 { return s.Baseline }),
		CandidateMean: mathx.MeanBy(c.Slices, func(s QualitySlice) float64 { return s.Candidate }),
	}

	// Fail closed on inconclusive evidence before judging any score.
	if miss := c.Provenance.missing(); len(miss) > 0 {
		return res.hold(c, fmt.Sprintf("missing provenance: %s", strings.Join(miss, ", ")))
	}
	if !c.Tier.known() {
		return res.hold(c, fmt.Sprintf("case not assigned to a pr/nightly/release tier (got %q)", c.Tier))
	}
	if strings.TrimSpace(c.CostNote) == "" {
		return res.hold(c, "case does not document its runtime/resource cost")
	}
	if c.MinSamples < 1 {
		return res.hold(c, "case declares no minimum evidence floor (min_samples < 1)")
	}
	if len(c.Slices) == 0 {
		return res.hold(c, "no quality slices measured")
	}
	nCritical := 0
	for _, s := range c.Slices {
		if !s.Measured {
			return res.hold(c, fmt.Sprintf("slice %q is unmeasured - inconclusive evidence is never pass", s.Name))
		}
		if s.Samples < c.MinSamples {
			return res.hold(c, fmt.Sprintf("slice %q has %d sample(s), below the promotion evidence floor of %d", s.Name, s.Samples, c.MinSamples))
		}
		if s.Critical {
			nCritical++
		}
	}
	if nCritical == 0 {
		return res.hold(c, "no critical slice declared - the canary cannot witness protection of a critical cohort")
	}

	// The core rule: a critical slice past tolerance rolls back even if the mean
	// rose. Iterate in caller order so the FIRST actionable divergence is named.
	for _, s := range c.Slices {
		if s.Critical && -s.delta() > s.Tolerance {
			d := &CanaryDivergence{
				Slice: s.Name, Baseline: s.Baseline, Candidate: s.Candidate,
				Delta: s.delta(), Tolerance: s.Tolerance,
				Reason: fmt.Sprintf("critical slice %q dropped %.4f (tolerance %.4f)", s.Name, -s.delta(), s.Tolerance),
			}
			res.Verdict, res.FirstDivergence = CanaryRollback.String(), d
			res.Reason = fmt.Sprintf("%s; candidate mean %.4f vs baseline %.4f does not rescue it",
				d.Reason, res.CandidateMean, res.BaselineMean)
			res.Replay = c.replay(d)
			return res
		}
	}

	// Quality-delta rule: an aggregate loss with no critical breach is ambiguous
	// evidence — held for more measurement, never promoted on ambiguity.
	if res.CandidateMean < res.BaselineMean {
		return res.hold(c, fmt.Sprintf("aggregate quality delta %.4f is negative without a critical breach - held, not promoted",
			res.CandidateMean-res.BaselineMean))
	}

	res.Verdict, res.Promoted = CanaryPromote.String(), true
	res.Reason = fmt.Sprintf("all %d critical slice(s) within tolerance at >=%d samples; mean %.4f -> %.4f",
		nCritical, c.MinSamples, res.BaselineMean, res.CandidateMean)
	res.Replay = c.replay(nil)
	return res
}

func (r CanaryResult) hold(c CanaryCase, reason string) CanaryResult {
	r.Verdict, r.Promoted, r.Reason = CanaryHold.String(), false, reason
	r.Replay = c.replay(nil)
	return r
}

// replay builds the scrubbed replay artifact for a case.
func (c CanaryCase) replay(d *CanaryDivergence) CanaryReplay {
	slices := append([]QualitySlice(nil), c.Slices...)
	for i := range slices {
		slices[i].Name = canaryScrub(slices[i].Name)
	}
	return CanaryReplay{
		Schema: CanarySchema, CaseID: canaryScrub(c.ID), Tier: c.Tier,
		CostNote: canaryScrub(c.CostNote), MinSamples: c.MinSamples,
		Provenance: c.Provenance.scrubbed(), Slices: slices,
		FirstDivergence: d, Scrubbed: true,
	}
}

// MarshalReplay renders the case's scrubbed replay artifact as indented JSON.
// It is the captured, re-runnable evidence every canary verdict carries.
func (r CanaryResult) MarshalReplay() ([]byte, error) {
	return json.MarshalIndent(r.Replay, "", "  ")
}

// SimulateCanary is the issue's simulation: three canonical, fully-provenanced
// canary cases adjudicated deterministically, covering PROMOTE,
// HOLD (inconclusive), and ROLLBACK in that fixed order. Pure CPU, no model
// call — the PR-tier cost each case's cost note documents.
func SimulateCanary() []CanaryResult {
	prov := CanaryProvenance{
		Model:     "fak-sim-7b@q4",
		Tokenizer: "fak-sim-tok@v2",
		Engine:    "fak-engine/decode@ep8",
		Seed:      "oracle:fixed-cases-v1",
		Revision:  "rev:candidate-vs-baseline",
		Baseline:  "baseline:nightly-suite-v1 tolerances:release-floor-v1",
	}
	promote := CanaryCase{
		ID: "sim-promote", Tier: CanaryTierPR, CostNote: "~1ms CPU, no GPU, no model call",
		MinSamples: 50, Provenance: prov,
		Slices: []QualitySlice{
			{Name: "safety-critical-prompts", Critical: true, Baseline: 0.94, Candidate: 0.95, Tolerance: 0.02, Samples: 200, Measured: true},
			{Name: "long-context-recall", Critical: false, Baseline: 0.81, Candidate: 0.84, Tolerance: 0.05, Samples: 120, Measured: true},
		},
	}
	hold := promote
	hold.ID = "sim-hold-inconclusive"
	hold.Slices = append([]QualitySlice(nil), promote.Slices...)
	hold.Slices[1].Samples = 10 // below the declared evidence floor — never pass
	rollback := promote
	rollback.ID = "sim-rollback"
	rollback.Slices = append([]QualitySlice(nil), promote.Slices...)
	rollback.Slices[0].Candidate = 0.90 // critical slice slips past tolerance (drop 0.04 > 0.02)
	rollback.Slices[1].Candidate = 0.99 // ...while the aggregate mean still rises

	return []CanaryResult{
		AdjudicateCanary(promote),
		AdjudicateCanary(hold),
		AdjudicateCanary(rollback),
	}
}

// canaryScrub redacts host filesystem paths, emails, and secret-shaped values
// so a replay artifact never leaks host secrets. Provenance identifiers that
// are not secrets (model names, git SHAs) are preserved for reproducibility.
var (
	reCanaryWinPath = regexp.MustCompile(`[A-Za-z]:\\[^\s"',;]+`)
	reCanaryNixPath = regexp.MustCompile(`/(?:home|Users|root)/[^\s"',;]+`)
	reCanaryEmail   = regexp.MustCompile(`[\w.+-]+@[\w-]+\.[\w.-]+`)
	reCanaryKeyVal  = regexp.MustCompile(`(?i)\b(token|secret|api[_-]?key|password|passwd|bearer)\b\s*[:=]\s*\S+`)
	reCanaryAPIKey  = regexp.MustCompile(`\b(?:sk|ghp|xoxb|AKIA)[-_][A-Za-z0-9]{12,}\b`)
)

func canaryScrub(s string) string {
	s = reCanaryWinPath.ReplaceAllString(s, "[redacted-path]")
	s = reCanaryNixPath.ReplaceAllString(s, "[redacted-path]")
	s = reCanaryAPIKey.ReplaceAllString(s, "[redacted-secret]")
	s = reCanaryEmail.ReplaceAllString(s, "[redacted-email]")
	s = reCanaryKeyVal.ReplaceAllString(s, "$1=[redacted-secret]")
	return s
}
