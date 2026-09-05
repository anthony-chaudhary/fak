package shipgate

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// CanarySchema versions the emitted replay artifact.
const CanarySchema = "fak-quality-canary/1"

// CanaryVerdict represents the three-way canary decision.
type CanaryVerdict uint8

// CanaryVerdict constants define the possible outcomes of canary evaluation.
const (
	CanaryHold     CanaryVerdict = iota
	CanaryPromote
	CanaryRollback
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

// CanaryTier specifies CI cadence for canary evaluation.
type CanaryTier string

// CanaryTier constants define supported execution cadences for canary checks.
const (
	CanaryTierPR      CanaryTier = "pr"
	CanaryTierNightly CanaryTier = "nightly"
	CanaryTierRelease CanaryTier = "release"
)

func (t CanaryTier) known() bool {
	switch t {
	case CanaryTierPR, CanaryTierNightly, CanaryTierRelease:
		return true
	}
	return false
}

// CanaryProvenance records execution environment metadata for reproducible decisions.
type CanaryProvenance struct {
	Model     string `json:"model"`
	Tokenizer string `json:"tokenizer"`
	Engine    string `json:"engine"`
	Seed      string `json:"seed"`
	Revision  string `json:"revision"`
	Baseline  string `json:"baseline"`
}

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

func (p CanaryProvenance) scrubbed() CanaryProvenance {
	return CanaryProvenance{
		Model: canaryScrub(p.Model), Tokenizer: canaryScrub(p.Tokenizer),
		Engine: canaryScrub(p.Engine), Seed: canaryScrub(p.Seed),
		Revision: canaryScrub(p.Revision), Baseline: canaryScrub(p.Baseline),
	}
}

// QualitySlice represents one measured metric cohort in a canary evaluation.
type QualitySlice struct {
	Name      string  `json:"name"`
	Critical  bool    `json:"critical"`
	Baseline  float64 `json:"baseline"`
	Candidate float64 `json:"candidate"`
	Tolerance float64 `json:"tolerance"`
	Samples   int     `json:"samples"`
	Measured  bool    `json:"measured"`
}

func (s QualitySlice) delta() float64 { return s.Candidate - s.Baseline }

// CanaryCase specifies the full configuration and measurements for a canary run.
type CanaryCase struct {
	ID         string           `json:"id"`
	Tier       CanaryTier       `json:"tier"`
	CostNote   string           `json:"cost_note"`
	MinSamples int              `json:"min_samples"`
	Provenance CanaryProvenance `json:"provenance"`
	Slices     []QualitySlice   `json:"slices"`
}

// CanaryDivergence localizes the first detected critical-slice regression.
type CanaryDivergence struct {
	Slice     string  `json:"slice"`
	Baseline  float64 `json:"baseline"`
	Candidate float64 `json:"candidate"`
	Delta     float64 `json:"delta"`
	Tolerance float64 `json:"tolerance"`
	Reason    string  `json:"reason"`
}

// CanaryReplay holds sanitized execution data for offline reproduction.
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

// ReplayCase reconstructs an adjudicable case from a replay artifact.
func ReplayCase(r CanaryReplay) CanaryCase {
	return CanaryCase{
		ID: r.CaseID, Tier: r.Tier, CostNote: r.CostNote,
		MinSamples: r.MinSamples, Provenance: r.Provenance,
		Slices: append([]QualitySlice(nil), r.Slices...),
	}
}

// CanaryResult holds the evaluation outcome and replay evidence.
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

// AdjudicateCanary evaluates slice measurements to decide promote, hold, or rollback.
func AdjudicateCanary(c CanaryCase) CanaryResult {
	res := CanaryResult{
		Schema: CanarySchema, CaseID: c.ID, Tier: c.Tier,
		BaselineMean:  mathx.MeanBy(c.Slices, func(s QualitySlice) float64 { return s.Baseline }),
		CandidateMean: mathx.MeanBy(c.Slices, func(s QualitySlice) float64 { return s.Candidate }),
	}

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

// MarshalReplay renders the replay artifact as indented JSON.
func (r CanaryResult) MarshalReplay() ([]byte, error) {
	return json.MarshalIndent(r.Replay, "", "  ")
}

// SimulateCanary generates representative test cases covering promote, hold, and rollback.
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
	hold.Slices[1].Samples = 10
	rollback := promote
	rollback.ID = "sim-rollback"
	rollback.Slices = append([]QualitySlice(nil), promote.Slices...)
	rollback.Slices[0].Candidate = 0.90
	rollback.Slices[1].Candidate = 0.99

	return []CanaryResult{
		AdjudicateCanary(promote),
		AdjudicateCanary(hold),
		AdjudicateCanary(rollback),
	}
}

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
