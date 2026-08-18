package productscorecard

// Slice-regression gate: block a severe critical-slice regression that a rising
// aggregate mean would otherwise hide (issue #4570, epic #4509 middle ladder).
//
// The productscorecard aggregate folds many KPIs into one score. That is exactly
// the failure mode leading eval stacks (lm-evaluation-harness, HELM) warn about:
// a higher overall mean can mask a catastrophic loss on a critical cohort. This
// gate is the independently verifiable middle layer between fak primitive
// correctness and coarse end-benchmark scores. It is a pure, deterministic
// oracle (stdlib-only) so a PR-tier run needs no model call, seed, or GPU.
//
// Contract:
//   - A critical slice that drops more than its tolerance FAILS the case even
//     when the candidate mean beats the baseline mean.
//   - Every case carries model / tokenizer / engine / seed-or-oracle / revision /
//     baseline provenance; a blank required field is inconclusive -> fail closed.
//   - Missing or inconclusive evidence is never a pass.
//   - A failure names the first actionable divergence and emits a scrubbed
//     replay artifact (host paths, emails, and secret-shaped values redacted).
//   - Every case is pinned to a PR / nightly / release tier with a cost note.

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// SliceRegressionSchema versions the emitted replay artifact.
const SliceRegressionSchema = "fak-slice-regression/1"

// Verdict strings for a gate result.
const (
	SliceOK           = "OK"
	SliceRegression   = "REGRESSION"
	SliceInconclusive = "INCONCLUSIVE"
)

// Tier is the CI cadence a slice-regression case is assigned to. A case with an
// unrecognized tier is inconclusive: an unscheduled gate never runs.
type Tier string

const (
	TierPR      Tier = "pr"      // per-PR deterministic gate (no model call)
	TierNightly Tier = "nightly" // nightly statistical / sampled gate
	TierRelease Tier = "release" // release / hardware qualification gate
)

func (t Tier) known() bool {
	switch t {
	case TierPR, TierNightly, TierRelease:
		return true
	}
	return false
}

// Provenance is the per-case evidence every slice-regression gate must carry so
// a divergence is reproducible. A blank required field is inconclusive evidence.
type Provenance struct {
	Model     string `json:"model"`     // model under test
	Tokenizer string `json:"tokenizer"` // tokenizer / vocab revision
	Engine    string `json:"engine"`    // engine / backend / engine-mode
	Seed      string `json:"seed"`      // RNG seed OR deterministic-oracle id
	Revision  string `json:"revision"`  // code / module revision
	Baseline  string `json:"baseline"`  // tolerance / baseline provenance
}

// missing returns the names of blank required provenance fields, in a stable
// order, so a fail-closed reason is deterministic.
func (p Provenance) missing() []string {
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
func (p Provenance) scrubbed() Provenance {
	return Provenance{
		Model: scrubText(p.Model), Tokenizer: scrubText(p.Tokenizer),
		Engine: scrubText(p.Engine), Seed: scrubText(p.Seed),
		Revision: scrubText(p.Revision), Baseline: scrubText(p.Baseline),
	}
}

// Slice is one measured cohort: a model, context-length, language, task, or
// engine-mode slice. Scores share one scale (higher is better). A critical
// slice is one whose loss must fail the case regardless of the aggregate.
type Slice struct {
	Name      string  `json:"name"`
	Critical  bool    `json:"critical"`
	Baseline  float64 `json:"baseline"`
	Candidate float64 `json:"candidate"`
	Tolerance float64 `json:"tolerance"` // max allowed drop (>= 0) for a critical slice
	Measured  bool    `json:"measured"`  // false => inconclusive evidence for this slice
}

func (s Slice) drop() float64 { return s.Baseline - s.Candidate }

// Case is a fully-provenanced slice-regression gate case.
type Case struct {
	ID         string     `json:"id"`
	Tier       Tier       `json:"tier"`
	CostNote   string     `json:"cost_note"` // runtime / resource cost, e.g. "~2s CPU, no GPU"
	Provenance Provenance `json:"provenance"`
	Slices     []Slice    `json:"slices"`
}

// Divergence localizes the first actionable critical-slice regression.
type Divergence struct {
	Slice     string  `json:"slice"`
	Baseline  float64 `json:"baseline"`
	Candidate float64 `json:"candidate"`
	Drop      float64 `json:"drop"`
	Tolerance float64 `json:"tolerance"`
	Reason    string  `json:"reason"`
}

// Replay is the scrubbed, machine-readable bundle emitted on any non-pass so the
// case can be re-run in a clean environment without leaking host secrets.
type Replay struct {
	Schema          string      `json:"schema"`
	CaseID          string      `json:"case_id"`
	Tier            Tier        `json:"tier"`
	CostNote        string      `json:"cost_note"`
	Provenance      Provenance  `json:"provenance"`
	Slices          []Slice     `json:"slices"`
	FirstDivergence *Divergence `json:"first_divergence,omitempty"`
	Scrubbed        bool        `json:"scrubbed"`
}

// GateResult is the verdict for one case.
type GateResult struct {
	Schema          string      `json:"schema"`
	CaseID          string      `json:"case_id"`
	Tier            Tier        `json:"tier"`
	Pass            bool        `json:"pass"`
	Verdict         string      `json:"verdict"`
	Reason          string      `json:"reason"`
	BaselineMean    float64     `json:"baseline_mean"`
	CandidateMean   float64     `json:"candidate_mean"`
	FirstDivergence *Divergence `json:"first_divergence,omitempty"`
	Replay          Replay      `json:"replay"`
}

// Gate adjudicates one slice-regression case. It fails closed: any missing or
// inconclusive evidence is INCONCLUSIVE (never a pass), a critical-slice loss
// past tolerance is a REGRESSION even when the candidate mean rises, and only a
// fully-evidenced case with every critical slice within tolerance is OK.
func Gate(c Case) GateResult {
	res := GateResult{
		Schema: SliceRegressionSchema, CaseID: c.ID, Tier: c.Tier,
		BaselineMean:  mathx.MeanBy(c.Slices, func(s Slice) float64 { return s.Baseline }),
		CandidateMean: mathx.MeanBy(c.Slices, func(s Slice) float64 { return s.Candidate }),
	}

	// Fail closed on inconclusive evidence before judging any score.
	if miss := c.Provenance.missing(); len(miss) > 0 {
		return res.inconclusive(c, fmt.Sprintf("missing provenance: %s", strings.Join(miss, ", ")))
	}
	if !c.Tier.known() {
		return res.inconclusive(c, fmt.Sprintf("case not assigned to a pr/nightly/release tier (got %q)", c.Tier))
	}
	if strings.TrimSpace(c.CostNote) == "" {
		return res.inconclusive(c, "case does not document its runtime/resource cost")
	}
	if len(c.Slices) == 0 {
		return res.inconclusive(c, "no slices measured")
	}
	nCritical := 0
	for _, s := range c.Slices {
		if !s.Measured {
			return res.inconclusive(c, fmt.Sprintf("slice %q is unmeasured - inconclusive evidence is never pass", s.Name))
		}
		if s.Critical {
			nCritical++
		}
	}
	if nCritical == 0 {
		return res.inconclusive(c, "no critical slice declared - the case cannot witness protection of a critical cohort")
	}

	// The core rule: a critical slice past tolerance fails even if the mean rose.
	// Iterate in caller order so the FIRST actionable divergence is reported.
	for _, s := range c.Slices {
		if s.Critical && s.drop() > s.Tolerance {
			d := &Divergence{
				Slice: s.Name, Baseline: s.Baseline, Candidate: s.Candidate,
				Drop: s.drop(), Tolerance: s.Tolerance,
				Reason: fmt.Sprintf("critical slice %q dropped %.4f (tolerance %.4f)", s.Name, s.drop(), s.Tolerance),
			}
			res.Pass, res.Verdict, res.FirstDivergence = false, SliceRegression, d
			res.Reason = fmt.Sprintf("%s; candidate mean %.4f vs baseline %.4f does not rescue it",
				d.Reason, res.CandidateMean, res.BaselineMean)
			res.Replay = c.replay(true, d)
			return res
		}
	}

	res.Pass, res.Verdict = true, SliceOK
	res.Reason = fmt.Sprintf("all %d critical slice(s) within tolerance; mean %.4f -> %.4f", nCritical, res.BaselineMean, res.CandidateMean)
	res.Replay = c.replay(true, nil)
	return res
}

func (r GateResult) inconclusive(c Case, reason string) GateResult {
	r.Pass, r.Verdict, r.Reason = false, SliceInconclusive, reason
	r.Replay = c.replay(true, nil)
	return r
}

// replay builds the scrubbed replay artifact for a case.
func (c Case) replay(scrub bool, d *Divergence) Replay {
	prov := c.Provenance
	slices := append([]Slice(nil), c.Slices...)
	if scrub {
		prov = prov.scrubbed()
		for i := range slices {
			slices[i].Name = scrubText(slices[i].Name)
		}
	}
	return Replay{
		Schema: SliceRegressionSchema, CaseID: scrubText(c.ID), Tier: c.Tier,
		CostNote: c.CostNote, Provenance: prov, Slices: slices,
		FirstDivergence: d, Scrubbed: scrub,
	}
}

// MarshalReplay renders the case's scrubbed replay artifact as indented JSON.
// It is the captured, re-runnable evidence a failed gate emits.
func (r GateResult) MarshalReplay() ([]byte, error) {
	return json.MarshalIndent(r.Replay, "", "  ")
}

// GateAll adjudicates a batch of cases and returns each result plus whether the
// whole batch passed (all cases OK). A batch with no cases fails closed.
func GateAll(cases []Case) ([]GateResult, bool) {
	out := make([]GateResult, 0, len(cases))
	pass := len(cases) > 0
	for _, c := range cases {
		r := Gate(c)
		if !r.Pass {
			pass = false
		}
		out = append(out, r)
	}
	// Stable ordering by case id keeps a batch report deterministic.
	sort.SliceStable(out, func(i, j int) bool { return out[i].CaseID < out[j].CaseID })
	return out, pass
}

// scrubText redacts host filesystem paths, emails, and secret-shaped values so a
// replay artifact never leaks host secrets. Provenance identifiers that are not
// secrets (model names, git SHAs) are preserved for reproducibility.
var (
	reWinPath = regexp.MustCompile(`[A-Za-z]:\\[^\s"',;]+`)
	reNixPath = regexp.MustCompile(`/(?:home|Users|root)/[^\s"',;]+`)
	reEmail   = regexp.MustCompile(`[\w.+-]+@[\w-]+\.[\w.-]+`)
	reKeyVal  = regexp.MustCompile(`(?i)\b(token|secret|api[_-]?key|password|passwd|bearer)\b\s*[:=]\s*\S+`)
	reAPIKey  = regexp.MustCompile(`\b(?:sk|ghp|xoxb|AKIA)[-_][A-Za-z0-9]{12,}\b`)
)

func scrubText(s string) string {
	s = reWinPath.ReplaceAllString(s, "[redacted-path]")
	s = reNixPath.ReplaceAllString(s, "[redacted-path]")
	s = reEmail.ReplaceAllString(s, "[redacted-email]")
	s = reAPIKey.ReplaceAllString(s, "[redacted-secret]")
	s = reKeyVal.ReplaceAllString(s, "$1=[redacted-secret]")
	return s
}
