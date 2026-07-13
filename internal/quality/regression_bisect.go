package quality

import (
	"fmt"
	"strconv"
	"strings"
)

// regression_bisect.go is the automated commit-and-configuration bisect child of
// the quality spine (#4583, under epic #4509): the layer that turns a run of
// per-revision Evidence into "which revision first broke this case". Primitive
// tests say a build is red; end benchmarks say a score dropped; neither localizes
// the regression to a commit. This file bisects an ordered lineage — commit
// crossed with the engine/config dimension it was evaluated under — to the FIRST
// revision whose evidence transitions from releasing to a localized divergence,
// and carries the spine's own scrubbed FailureBundle out as the replay artifact.
//
// It is additive and adjudicative in the cohort idiom: it registers no oracles and
// edits no core, consuming only the Evidence/Divergence/FailureBundle the release
// aggregator (release_gate.go) and spine (run.go) already emit. Like git-bisect it
// probes lazily — only ~log(n) points of a lineage are evaluated — so the expensive
// per-revision evaluation is the thing being economized, and the cost of the points
// actually probed is reported. Fail-closed throughout: a point whose evidence is
// missing, stale, or inconclusive is never treated as good and never fabricated
// into a culprit — it halts the bisect with a typed indeterminate outcome.

// RegressionBisectSchema is the versioned tag on a bisect verdict. Consumers pin
// the major so a schema bump is a conscious migration (the #4519 house rule), not
// a silent field drift.
const RegressionBisectSchema = "fak-quality-bisect/1"

// BisectPoint is one coordinate in an ordered regression lineage: a commit/module
// revision, crossed with the engine/config dimension and machine class it is
// evaluated under, plus the tier and per-point cost of producing its evidence. The
// sequence is the "commit × config × machine" lineage the issue bisects; a single
// sweep bisects along ONE axis (typically the commit list) with the others held
// fixed, but the type carries all three so a config- or machine-axis sweep uses the
// same contract. Points are ordered oldest→newest, so the first good→bad transition
// is the first bad revision. It carries no Evidence itself: evidence is produced
// lazily by a Probe, so a long lineage costs only the points the bisect visits.
type BisectPoint struct {
	Revision    string  `json:"revision"`          // lineage: commit / code-module revision
	Engine      string  `json:"engine"`            // lineage: engine/backend + config dimension
	Machine     string  `json:"machine,omitempty"` // lineage: machine/host class (held fixed per commit sweep)
	Tier        Tier    `json:"tier"`              // pr / nightly / release cadence this point runs at
	CostSeconds float64 `json:"cost_seconds"`      // runtime/resource cost to evaluate this point
}

// Probe produces the qualification Evidence for one lineage point — the expensive
// step (check out the revision under the engine/config, run the case, submit the
// Evidence) the bisect calls lazily at each visited point. It is the seam a real
// bisect wires a git-checkout+RunCase driver into; tests wire a deterministic
// planted-defect probe. The bisect is a pure function of (lineage, Probe): a
// deterministic Probe yields a deterministic verdict, so a bisect result replays —
// the same conditional purity RunCase has on its Runner adapters.
type Probe func(BisectPoint) Evidence

// PointState is the closed classification of one probed point. Good and Bad are the
// two monotone rungs a bisect narrows between; Indeterminate is the fail-closed
// third rung — missing, stale, mis-attributed, or inconclusive evidence that can be
// neither trusted as good nor localized as the cause. Its existence is the point:
// "missing or inconclusive evidence is never pass" (#4583 acceptance) applied to
// bisect means such a point halts the search instead of being guessed past.
type PointState string

const (
	PointGood          PointState = "good"          // releasing evidence at this revision
	PointBad           PointState = "bad"           // a localized divergence at this revision
	PointIndeterminate PointState = "indeterminate" // unclassifiable — never good, never a fabricated cause
)

// classifyPoint maps a point's produced Evidence onto a PointState. It reuses the
// release aggregator's provenance-completeness and evidence-state contracts so a
// point is classified the SAME way the release gate would adjudicate it: evidence
// with incomplete provenance, or attributed to a different revision than the point
// probed, is inconclusive (indeterminate) — you cannot bisect across evidence you
// cannot trust or attribute. Only a Pass is good; only a Fail is a localizable bad;
// every other state is indeterminate.
func classifyPoint(p BisectPoint, e Evidence) (PointState, string) {
	if ok, why := e.Provenance.complete(); !ok {
		return PointIndeterminate, "incomplete provenance: " + why
	}
	if e.Provenance.Revision != p.Revision {
		return PointIndeterminate, fmt.Sprintf("evidence attributed to revision %q != probed point %q",
			e.Provenance.Revision, p.Revision)
	}
	switch e.State {
	case StatePass:
		return PointGood, e.Detail
	case StateFail:
		return PointBad, e.Detail
	default:
		return PointIndeterminate, "evidence state " + string(e.State) + ": " + e.Detail
	}
}

// BisectOutcome is the closed outcome of a lineage bisect. Found localizes a
// good→bad transition to a first-bad revision; Clean means every probed boundary is
// good (no regression in range); Indeterminate means the bisect could not produce a
// trustworthy answer — an unclassifiable point, an already-bad oldest point, or an
// empty lineage. Indeterminate is a first-class refusal, never a silent Clean.
type BisectOutcome string

const (
	OutcomeFound         BisectOutcome = "found"
	OutcomeClean         BisectOutcome = "clean"
	OutcomeIndeterminate BisectOutcome = "indeterminate"
)

// BisectResult is the machine-readable verdict of a lineage bisect: the outcome,
// the first-bad point and its good predecessor (on Found), the first actionable
// divergence and scrubbed replay bundle folded from the culprit's evidence, and the
// number of points actually probed plus their summed cost — the bisect-efficiency
// and runtime/resource-cost documentation (#4583 acceptance). It is a pure function
// of (lineage, Probe), so a verdict replays.
type BisectResult struct {
	Schema          string         `json:"schema"`
	Outcome         BisectOutcome  `json:"outcome"`
	FirstBad        *BisectPoint   `json:"first_bad,omitempty"`
	LastGood        *BisectPoint   `json:"last_good,omitempty"`
	FirstDivergence *Divergence    `json:"first_divergence,omitempty"`
	Replay          *FailureBundle `json:"replay,omitempty"`
	Probes          int            `json:"probes"`
	CostSeconds     float64        `json:"cost_seconds"`
	Reason          string         `json:"reason"`
}

// Bisect finds the first revision at which the case regresses across an ordered
// lineage, calling probe lazily and only at the points a binary search visits. It
// assumes the lineage is monotone (a good prefix then a bad suffix) — git-bisect's
// contract — and PROVES the two claims it makes: the reported first-bad point was
// probed Bad and its immediate predecessor was probed Good, a genuine, localized
// good→bad transition. Any point that probes indeterminate halts the search with an
// indeterminate outcome rather than being guessed past; an oldest point that is
// already bad, or a newest point that is still good, are reported honestly (regression
// precedes the range / no regression in range) instead of being forced into a culprit.
func Bisect(lineage []BisectPoint, probe Probe) BisectResult {
	res := BisectResult{Schema: RegressionBisectSchema}
	n := len(lineage)
	if n == 0 {
		res.Outcome = OutcomeIndeterminate
		res.Reason = "empty lineage: nothing to bisect"
		return res
	}

	type classified struct {
		state PointState
		why   string
		ev    Evidence
	}
	cache := make(map[int]classified, n)
	eval := func(i int) classified {
		if c, ok := cache[i]; ok {
			return c
		}
		ev := probe(lineage[i])
		st, why := classifyPoint(lineage[i], ev)
		c := classified{state: st, why: why, ev: ev}
		cache[i] = c
		res.Probes++
		res.CostSeconds += lineage[i].CostSeconds
		return c
	}

	indeterminate := func(i int, why string) BisectResult {
		res.Outcome = OutcomeIndeterminate
		res.Reason = fmt.Sprintf("point %s: %s", lineage[i].Revision, why)
		return res
	}

	// Oldest boundary must be a trustworthy good to anchor the search.
	first := eval(0)
	switch first.state {
	case PointIndeterminate:
		return indeterminate(0, first.why)
	case PointBad:
		res.Outcome = OutcomeIndeterminate
		res.Reason = fmt.Sprintf("oldest point %s already bad — regression precedes the bisected range; extend the good boundary older",
			lineage[0].Revision)
		return res
	}

	// Newest boundary decides whether there is a regression in range at all.
	last := eval(n - 1)
	switch last.state {
	case PointIndeterminate:
		return indeterminate(n-1, last.why)
	case PointGood:
		res.Outcome = OutcomeClean
		res.Reason = fmt.Sprintf("no regression across %d point(s): newest revision %s still good", n, lineage[n-1].Revision)
		return res
	}

	// first is Good, last is Bad, n >= 2: binary-search the boundary. Invariant:
	// lineage[lo] is proven Good and lineage[hi] is proven Bad throughout.
	lo, hi := 0, n-1
	for hi-lo > 1 {
		mid := lo + (hi-lo)/2
		c := eval(mid)
		switch c.state {
		case PointIndeterminate:
			return indeterminate(mid, c.why)
		case PointGood:
			lo = mid
		default: // PointBad
			hi = mid
		}
	}

	bad := cache[hi]
	firstBad := lineage[hi]
	lastGood := lineage[lo]
	res.Outcome = OutcomeFound
	res.FirstBad = &firstBad
	res.LastGood = &lastGood
	res.FirstDivergence = bad.ev.FirstDivergence
	res.Replay = bad.ev.Replay
	res.Reason = fmt.Sprintf("first bad revision %s (good predecessor %s) localized in %d probe(s) over %d point(s)",
		firstBad.Revision, lastGood.Revision, res.Probes, n)
	return res
}

// ExplainBisect renders a BisectResult as an operator readout, mirroring Explain
// (run.go) and ExplainRelease (release_gate.go): REGRESSION names the first bad
// revision, its good predecessor, the localized first divergence, and the probe/cost
// budget spent; CLEAN and INDETERMINATE state their reason plainly — an
// indeterminate verdict is surfaced as a refusal, never dressed up as a clean pass.
func ExplainBisect(r BisectResult) string {
	var b strings.Builder
	switch r.Outcome {
	case OutcomeFound:
		fmt.Fprintf(&b, "REGRESSION  first bad revision %s", r.FirstBad.Revision)
		if r.FirstBad.Engine != "" {
			fmt.Fprintf(&b, " under %s", r.FirstBad.Engine)
		}
		if r.FirstBad.Machine != "" {
			fmt.Fprintf(&b, " @ %s", r.FirstBad.Machine)
		}
		b.WriteByte('\n')
		if r.LastGood != nil {
			fmt.Fprintf(&b, "  last good: %s\n", r.LastGood.Revision)
		}
		fmt.Fprintf(&b, "  %d probe(s), %.0fs evaluated (tier %s)\n", r.Probes, r.CostSeconds, r.FirstBad.Tier)
		if d := r.FirstDivergence; d != nil {
			fmt.Fprintf(&b, "  first divergence at token %d: reference %q, engine %q\n", d.Index, d.Reference, d.Engine)
		}
		if r.Replay != nil {
			fmt.Fprintf(&b, "  replay: case %s @ v%d via oracle %s\n", r.Replay.CaseID, r.Replay.Case.Version, r.Replay.FailingOracle)
		}
		fmt.Fprintf(&b, "  reason: %s\n", r.Reason)
	case OutcomeClean:
		fmt.Fprintf(&b, "CLEAN  %s (%d probe(s), %.0fs)\n", r.Reason, r.Probes, r.CostSeconds)
	default:
		fmt.Fprintf(&b, "INDETERMINATE  %s\n", r.Reason)
		fmt.Fprintf(&b, "  (missing or inconclusive evidence is never treated as good)\n")
	}
	return b.String()
}

// DemoBisectLineage builds an ordered commit×config lineage for the spine demo
// case: seven synthetic revisions r0..r6 evaluated under one fixed engine/config on
// one machine class, oldest first. It is the hermetic fixture the witness test
// sweeps over — a stand-in for a real git commit range whose per-point evidence a
// driver would produce by checking out each revision and running the case.
func DemoBisectLineage() []BisectPoint {
	pts := make([]BisectPoint, 0, 7)
	for i := 0; i < 7; i++ {
		pts = append(pts, BisectPoint{
			Revision:    fmt.Sprintf("r%d", i),
			Engine:      "engine-demo",
			Machine:     "ci-linux-x64",
			Tier:        TierPR,
			CostSeconds: 2,
		})
	}
	return pts
}

// DemoBisectProbe returns a deterministic probe that models a regression planted at
// revision index badFrom: every revision at or after badFrom decodes with the
// "decode" defect (the greedy oracle fails at token 1), every earlier revision
// decodes cleanly, and badFrom < 0 models a fully-fixed lineage where no revision is
// bad. It runs the REAL spine (RunCase over DemoCase) at each point and folds the
// Result into Evidence, so the bisect is proven against genuine first-divergence
// evidence rather than a hand-written verdict — the "planted defect fails, fix
// passes, independently replayed" witness the epic requires.
func DemoBisectProbe(badFrom int) Probe {
	oracles, _ := Lookup(DemoCase().Oracles)
	return func(p BisectPoint) Evidence {
		defect := ""
		if badFrom >= 0 && demoRevIndex(p.Revision) >= badFrom {
			defect = "decode"
		}
		res, err := RunCase(DemoCase(), ReferenceRunner{}, DemoEngine(defect), oracles)
		if err != nil {
			// A probe that cannot even run the case is inconclusive, never a pass.
			return Evidence{CaseID: DemoCase().ID, State: StateInconclusive, Detail: err.Error()}
		}
		prov := EvidenceProvenance{
			Model:     "demo-1b",
			Tokenizer: "demo-bpe",
			Engine:    p.Engine,
			Oracle:    "greedy-token-diff",
			Revision:  p.Revision,
			Baseline:  "demo-reference@1",
		}
		return EvidenceFromResult(prov, res)
	}
}

// demoRevIndex parses the trailing integer of a demo revision id ("r4" -> 4). It is
// demo-only: the synthetic lineage names revisions r0..rN, so a driver can plant a
// regression at an index. A real lineage carries opaque commit shas and the probe
// decides good/bad by running the case, not by parsing the id.
func demoRevIndex(rev string) int {
	i, err := strconv.Atoi(strings.TrimPrefix(rev, "r"))
	if err != nil {
		return -1
	}
	return i
}
