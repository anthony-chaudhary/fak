// Package dojocal is the dojo-RSI loop's PURE proposer + self-scoring rung — the
// genuinely-safe autonomous slice that MUTATES NOTHING (Phase 1 of
// docs/fak/dojo-rsi-loop.md, issue #1023). It is the twin of internal/guardrsi
// pointed at the dojo's calibration journal instead of the guard's verdict
// journal: where guardrsi folds a verdict journal and proposes the worst-bucket
// honesty hole, dojocal folds a dojo report's scored episodes (reusing
// dojo.BoardFromEpisodes) and proposes the worst-calibrated MEASURED, NON-FLOOR
// (lever, metric) cell to recalibrate.
//
// What makes the loop safe to run unattended is what it does NOT do: it never
// opens a worktree, never rewrites a claim literal, never touches a file. It
// PROPOSES a recalibration (swap the claimed number to the corpus mean) and
// SELF-SCORES it by replaying dojo.FoldCalibrable over the same episodes with the
// candidate claim swapped in — a pure, deterministic, CI-able re-fold. The KEEP
// bit is non-forgeable and mirrors guardrsi.RunIteration exactly: KEEP iff
// measured-rows>0 AND the folded calibrable metric STRICTLY drops AND a per-lever
// sample threshold is met AND an external `--witness {"ok":true}` is supplied.
//
// The structural honesty guarantee comes from dojo.FoldCalibrable, not from a
// check here: an INTENTIONAL FLOOR (false_warm_rate must stay 0.0) folds by its
// breach, not its calib_err, so "recalibrating" a floor up to its empirical rate
// RAISES the fold and is reverted. The candidate picker refuses to target a floor
// at all (it is routed, never recalibrated), and dojo.FoldCalibrable refuses to
// fold an UNMEASURED episode — so the two ways the loop could optimise itself into
// dishonesty are both closed by construction, the same way guardrsi can only ever
// repair an honesty hole and never invent a row.
package dojocal

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/mathx"
)

const (
	// ProposeSchema tags the proposal envelope.
	ProposeSchema = "fak-dojo-rsi.propose/1"
	// IterationSchema tags one replayed self-scoring iteration.
	IterationSchema = "fak-dojo-rsi/1"
	// DefaultMinSample is the per-lever sample floor a RECALIBRATE candidate must
	// clear before its proposed claim is trusted enough to KEEP: a recalibration
	// fitted to one or two boundaries is noise, not a corpus central tendency. The
	// guard's worst-bucket loop has no analogue (a journal row is its own witness);
	// the dojo's claim is a population estimate, so it owes a sample floor.
	DefaultMinSample = 3
)

// RecalKind names what a proposal asks for, mirroring the design doc's split
// (docs/fak/dojo-rsi-loop.md "The candidate"). Only RECALIBRATE is self-KEEPable
// here; the rest are ROUTED to a human/agent arm and never KEPT by this pure loop.
type RecalKind string

const (
	// RecalibrateKind re-points a genuine ESTIMATE claim at its corpus central
	// tendency — the mechanical, self-keepable win.
	RecalibrateKind RecalKind = "RECALIBRATE"
	// RouteFloor marks a candidate whose target cell is an INTENTIONAL FLOOR: a
	// guard the dojo defends, never a recalibration. A breach is a belief-code bug
	// to escalate, so the loop routes it and never proposes a claim swap.
	RouteFloor RecalKind = "ROUTE_FLOOR"
	// RouteUnmeasured marks a candidate the loop cannot score: the worst lever had
	// no measured episode (UNMEASURED is uncandidatable by construction — the floor
	// the honesty constraint routes to).
	RouteUnmeasured RecalKind = "ROUTE_UNMEASURED"
)

// Recal is one proposed recalibration targeting exactly one (lever, metric) cell
// of the worst-first board. It mirrors the design doc's Recal: the loop writes the
// proposed NewClaimed (the corpus mean) but the KEEP decision is derived by
// re-folding, never asserted here.
type Recal struct {
	Lever      string    `json:"lever"`
	Metric     string    `json:"metric"`
	Kind       RecalKind `json:"kind"`
	OldClaimed float64   `json:"old_claimed"`
	NewClaimed float64   `json:"new_claimed"`
	// MeasuredMean is the mean realized value over the cell's measured episodes —
	// the corpus central tendency a RECALIBRATE re-points the claim at.
	MeasuredMean float64 `json:"measured_mean"`
	// Sample is how many measured episodes stand behind MeasuredMean.
	Sample int `json:"sample"`
	// Verdict is the worst measured episode's verdict for this cell (OVER_CLAIM /
	// UNDER_CLAIM / CALIBRATED), carried for the operator's context.
	Verdict string `json:"verdict"`
	// CalibErr is the worst measured calib_err for the cell — the size of the gap
	// the recalibration would close.
	CalibErr float64 `json:"calib_err"`
	// IntentionalFloor is mirrored from the episode so a reader can see at a glance
	// why a ROUTE_FLOOR candidate is routed rather than recalibrated.
	IntentionalFloor bool `json:"intentional_floor"`
	// Reason is the one-line rationale (why this cell, why this kind).
	Reason string `json:"reason"`
}

// ProposePayload is the proposal envelope: the worst-first candidates, the folded
// calibrable baseline they were drawn from, and the single worst candidate the
// loop would act on first (mirroring guardrsi's WorstBucket-led FoldPayload).
type ProposePayload struct {
	Schema     string          `json:"schema"`
	Baseline   dojo.FoldResult `json:"baseline"`
	Board      dojo.Board      `json:"board"`
	Candidates []Recal         `json:"candidates"`
	Worst      Recal           `json:"worst"`
}

// cell is the per-(lever,metric) accumulator the proposer folds episodes into.
type cell struct {
	lever, metric    string
	sumRealized      float64
	measured         int
	worstCalibErr    float64
	worstVerdict     string
	oldClaimed       float64
	intentionalFloor bool
	anyMeasured      bool
}

// ProposeRecals folds a dojo report's scored episodes into the worst-first list
// of recalibration candidates — the twin of guardrsi.WorstBucket, reusing
// dojo.BoardFromEpisodes for the cross-lever board the candidates ride alongside.
//
// It is pure and total. For each MEASURED, NON-FLOOR (lever, metric) cell it
// proposes a RECALIBRATE re-pointing the claim at the cell's measured mean; an
// INTENTIONAL FLOOR cell is emitted as a ROUTE_FLOOR (never a claim swap — a
// floor breach is a bug to escalate, the design's structural anti-gaming rule);
// a lever whose episodes were ALL UNMEASURED contributes a ROUTE_UNMEASURED so the
// floor of the honesty constraint is visible rather than silently dropped.
// Candidates are sorted worst-first by calib_err (a measured recalibration always
// ahead of a routed one), then lever, then metric, for determinism.
func ProposeRecals(r dojo.Report) ProposePayload {
	cells := map[[2]string]*cell{}
	var order [][2]string
	// Track levers that produced ONLY unmeasured episodes so the honesty floor is
	// surfaced (a measured cell on the same lever supersedes it).
	leverHasMeasured := map[string]bool{}
	leverSeen := map[string]bool{}
	var leverOrder []string

	for _, e := range r.Episodes {
		if !leverSeen[e.Lever] {
			leverSeen[e.Lever] = true
			leverOrder = append(leverOrder, e.Lever)
		}
		if e.Verdict == dojo.VerdictUnmeasured {
			continue
		}
		leverHasMeasured[e.Lever] = true
		key := [2]string{e.Lever, e.Metric}
		c, ok := cells[key]
		if !ok {
			c = &cell{lever: e.Lever, metric: e.Metric, oldClaimed: e.Claimed, intentionalFloor: e.IntentionalFloor}
			cells[key] = c
			order = append(order, key)
		}
		c.anyMeasured = true
		c.measured++
		c.sumRealized += e.Realized
		if e.CalibErr >= c.worstCalibErr {
			c.worstCalibErr = e.CalibErr
			c.worstVerdict = e.Verdict
		}
	}

	var candidates []Recal
	for _, key := range order {
		c := cells[key]
		mean := mathx.Round3(c.sumRealized / float64(c.measured))
		rc := Recal{
			Lever:            c.lever,
			Metric:           c.metric,
			OldClaimed:       c.oldClaimed,
			MeasuredMean:     mean,
			Sample:           c.measured,
			Verdict:          c.worstVerdict,
			CalibErr:         mathx.Round3(c.worstCalibErr),
			IntentionalFloor: c.intentionalFloor,
		}
		if c.intentionalFloor {
			rc.Kind = RouteFloor
			rc.NewClaimed = c.oldClaimed // never swap a floor's claim
			rc.Reason = fmt.Sprintf("%s/%s is an intentional floor (claim %.3g) — a breach is a belief-code bug to escalate, never a recalibration; routed, not kept", c.lever, c.metric, c.oldClaimed)
		} else {
			rc.Kind = RecalibrateKind
			rc.NewClaimed = mean
			rc.Reason = fmt.Sprintf("re-point %s/%s estimate from %.3g toward its corpus mean %.3g (worst calib_err %.3g over %d sample(s), %s)", c.lever, c.metric, c.oldClaimed, mean, rc.CalibErr, c.measured, c.worstVerdict)
		}
		candidates = append(candidates, rc)
	}

	// Surface the honesty floor: a lever that produced episodes but NONE measured is
	// uncandidatable — emit a ROUTE_UNMEASURED so the constraint is legible.
	for _, lever := range leverOrder {
		if leverHasMeasured[lever] {
			continue
		}
		candidates = append(candidates, Recal{
			Lever:  lever,
			Kind:   RouteUnmeasured,
			Reason: fmt.Sprintf("%s produced only UNMEASURED episodes — no ground truth to recalibrate against; routed to point its scenario at a billed corpus", lever),
		})
	}

	sortCandidates(candidates)
	payload := ProposePayload{
		Schema:     ProposeSchema,
		Baseline:   dojo.FoldCalibrable(r.Episodes),
		Board:      dojo.BoardFromEpisodes(r.Episodes),
		Candidates: candidates,
	}
	if len(candidates) > 0 {
		payload.Worst = candidates[0]
	}
	return payload
}

// sortCandidates orders worst-first: a RECALIBRATE candidate (with a real
// calib_err gap) sorts ahead of any routed one; among recalibrations, the largest
// calib_err first; then lever, then metric, for a deterministic, CI-stable order.
func sortCandidates(cs []Recal) {
	sort.SliceStable(cs, func(i, j int) bool {
		ri, rj := cs[i].Kind == RecalibrateKind, cs[j].Kind == RecalibrateKind
		if ri != rj {
			return ri
		}
		if cs[i].CalibErr != cs[j].CalibErr {
			return cs[i].CalibErr > cs[j].CalibErr
		}
		if cs[i].Lever != cs[j].Lever {
			return cs[i].Lever < cs[j].Lever
		}
		return cs[i].Metric < cs[j].Metric
	})
}

// Iteration is one replayed self-scoring tick — the dojo twin of
// guardrsi.Iteration. It carries the candidate, the calibrable fold BEFORE and
// AFTER the proposed claim swap (both DERIVED by re-folding, never asserted), the
// strict measured delta, and the non-forgeable KEEP bit + reason.
type Iteration struct {
	Schema         string          `json:"schema"`
	Goal           string          `json:"goal"`
	Candidate      Recal           `json:"candidate"`
	BaselineFold   dojo.FoldResult `json:"baseline_fold"`
	ReplayedFold   dojo.FoldResult `json:"replayed_fold"`
	BaselineValue  float64         `json:"baseline_value"`
	ReplayedValue  float64         `json:"replayed_value"`
	MeasuredDelta  float64         `json:"measured_delta"` // baseline - replayed; positive = the fold dropped
	MinSample      int             `json:"min_sample"`
	Witness        map[string]any  `json:"witness,omitempty"`
	Kept           bool            `json:"kept"`
	Reason         string          `json:"reason"`
	KeepRevertRule string          `json:"keep_revert_rule"`
}

// RunIteration replays dojo.FoldCalibrable over the report's episodes with the
// candidate's claim SWAPPED IN, and decides KEEP/REVERT — the pure, deterministic,
// CI-able dojo twin of guardrsi.RunIteration, taking `--witness {"ok":true}`
// exactly as guard-verdict-rsi does.
//
// The replay re-scores every episode of the candidate's (lever, metric) cell
// against the proposed NewClaimed (using the same dojo.Score + dojo.DefaultCalibBand
// the live builders use), leaving every other episode untouched, then re-folds.
// Nothing on disk changes; the swap is a value substitution in a copy of the
// episodes. The KEEP bit mirrors guardrsi exactly:
//
//	KEEP iff measured-rows>0 AND the folded calibrable Value STRICTLY drops
//	     AND the candidate's per-lever sample >= minSample AND a green witness.
//
// A floor target can never KEEP: FoldCalibrable folds a floor by its breach, so
// closing the gap RAISES Value (negative delta) and REVERTs — the structural
// guarantee, surfaced here as a normal no-strict-gain revert. An UNMEASURED-routed
// or floor-routed candidate is refused before any replay (it carries no swap).
// minSample<=0 falls back to DefaultMinSample.
func RunIteration(r dojo.Report, candidate Recal, minSample int, witness map[string]any) Iteration {
	if minSample <= 0 {
		minSample = DefaultMinSample
	}
	base := dojo.FoldCalibrable(r.Episodes)
	it := Iteration{
		Schema:         IterationSchema,
		Goal:           "drive the dojo's folded calibrable metric down by re-pointing a genuine estimate at its corpus mean — never a floor",
		Candidate:      candidate,
		BaselineFold:   base,
		BaselineValue:  base.Value,
		MinSample:      minSample,
		Witness:        witness,
		KeepRevertRule: "KEEP iff measured-rows>0 AND replayed FoldCalibrable.Value strictly LOWER than baseline AND candidate sample >= minSample AND an external witness (suite green) confirms no regression; else REVERT. A floor target raises Value (breach folded) and an unmeasured corpus is uncandidatable — both REVERT by construction. Worst-cell-first.",
	}

	// A routed candidate carries no claim swap — refuse before any replay.
	switch candidate.Kind {
	case RouteFloor:
		it.ReplayedFold = base
		it.ReplayedValue = base.Value
		it.Reason = fmt.Sprintf("REVERT: %s/%s is an intentional floor — recalibrating it would erase the guard the dojo defends; a breach is a belief-code bug to escalate, not a recalibration", candidate.Lever, candidate.Metric)
		return it
	case RouteUnmeasured:
		it.ReplayedFold = base
		it.ReplayedValue = base.Value
		it.Reason = fmt.Sprintf("REVERT: %s had no MEASURED episode — an unmeasured corpus is uncandidatable; point the scenario at a billed corpus before it can be scored", candidate.Lever)
		return it
	}

	if base.Measured == 0 {
		it.ReplayedFold = base
		it.ReplayedValue = base.Value
		it.Reason = "REVERT: the report measured nothing — no calibrable fold to improve (point the scenario at a corpus with billed usage records)"
		return it
	}

	// Structural floor defense, independent of the candidate's Kind label: if the
	// real episodes for the target cell are an INTENTIONAL FLOOR, refuse the swap
	// outright. A floor breach is folded against the PINNED floor claim; letting a
	// caller relabel a floor as RECALIBRATE and swap its claim up to the empirical
	// rate would zero the breach (the exact auto-erasure FoldCalibrable exists to
	// forbid). The guarantee must not rest on the Kind label alone.
	if cellIsFloor(r.Episodes, candidate.Lever, candidate.Metric) {
		it.ReplayedFold = base
		it.ReplayedValue = base.Value
		it.Reason = fmt.Sprintf("REVERT: %s/%s is an intentional floor — its claim is pinned; recalibrating it to the empirical rate would erase the breach the dojo defends. A floor breach is a belief-code bug to escalate, never a recalibration", candidate.Lever, candidate.Metric)
		return it
	}

	replayed := replayWithSwap(r.Episodes, candidate)
	rf := dojo.FoldCalibrable(replayed)
	it.ReplayedFold = rf
	it.ReplayedValue = rf.Value
	delta := mathx.Round3(base.Value - rf.Value)
	it.MeasuredDelta = delta

	haveWitness := witnessOK(witness)
	strictGain := delta > 0
	sampleOK := candidate.Sample >= minSample
	it.Kept = rf.Measured > 0 && strictGain && sampleOK && haveWitness

	switch {
	case rf.Measured == 0:
		it.Reason = "REVERT: the replay measured nothing — no calibrable fold to score the swap against"
	case !strictGain:
		it.Reason = fmt.Sprintf("REVERT: no strict gain (calibrable %.4g -> %.4g, delta %.4g); swapping %s/%s to %.3g does not lower the fold (a floor breach RAISES it, an already-calibrated estimate has no gap to close)", base.Value, rf.Value, delta, candidate.Lever, candidate.Metric, candidate.NewClaimed)
	case !sampleOK:
		it.Reason = fmt.Sprintf("REVERT: estimate gained (delta +%.4g) but only %d sample(s) < min %d — a recalibration fitted to too few boundaries is noise, not a corpus central tendency", delta, candidate.Sample, minSample)
	case !haveWitness:
		it.Reason = fmt.Sprintf("REVERT: calibrable improved (delta +%.4g) but no external witness supplied; supply a green `go test ./...` / `fak dojo --check` witness to KEEP", delta)
	default:
		it.Reason = fmt.Sprintf("KEPT: re-point %s/%s %.3g -> %.3g lowered the folded calibrable %.4g -> %.4g (delta +%.4g) on %d sample(s), witness green", candidate.Lever, candidate.Metric, candidate.OldClaimed, candidate.NewClaimed, base.Value, rf.Value, delta, candidate.Sample)
	}
	return it
}

// replayWithSwap returns a copy of episodes with the candidate cell's episodes
// re-scored against the candidate's NewClaimed. It re-runs dojo.Score with the
// same band the live builders use so the replayed calib_err / verdict / floor bit
// are derived identically — the swap is a value substitution, never a hand-edited
// calib_err. Episodes outside the candidate's cell are copied unchanged.
func replayWithSwap(episodes []dojo.Episode, candidate Recal) []dojo.Episode {
	band := dojo.DefaultCalibBand()
	out := make([]dojo.Episode, len(episodes))
	for i, e := range episodes {
		// Never rewrite an intentional floor's claim: its breach is folded against
		// the pinned claim, so swapping it would zero the guard (the structural
		// no-op that makes auto-erasure impossible). Unmeasured and off-cell
		// episodes pass through unchanged.
		if e.Lever != candidate.Lever || e.Metric != candidate.Metric || e.Verdict == dojo.VerdictUnmeasured || e.IntentionalFloor {
			out[i] = e
			continue
		}
		p := dojo.Prediction{
			Lever:            e.Lever,
			Metric:           e.Metric,
			Claimed:          candidate.NewClaimed,
			Unit:             e.Unit,
			Basis:            e.Basis,
			LowerIsBetter:    e.LowerIsBetter,
			IntentionalFloor: e.IntentionalFloor,
		}
		o := dojo.Outcome{
			Realized:   e.Realized,
			Provenance: e.Provenance,
			Source:     e.Source,
			Measured:   true,
			Sample:     e.Sample,
		}
		out[i] = dojo.Score(e.Scenario, p, o, band)
	}
	return out
}

// cellIsFloor reports whether the real episodes for a (lever, metric) cell carry
// the IntentionalFloor bit — the ground-truth floor test, read from the episodes
// themselves rather than the (caller-supplied, forgeable) candidate Kind.
func cellIsFloor(episodes []dojo.Episode, lever, metric string) bool {
	for _, e := range episodes {
		if e.Lever == lever && e.Metric == metric && e.IntentionalFloor {
			return true
		}
	}
	return false
}

func witnessOK(witness map[string]any) bool {
	if witness == nil {
		return false
	}
	v, ok := witness["ok"].(bool)
	return ok && v
}

// CheckIteration re-derives the KEEP invariants from the iteration's own fields —
// the dojo twin of guardrsi.CheckIteration. A kept iteration that cannot defend
// its keep bit (no rows, no strict drop, too few samples, no witness, or a routed
// candidate) returns the violations, so a fabricated KEEP is caught at the gate.
func CheckIteration(it Iteration) []string {
	var out []string
	if it.Schema != IterationSchema {
		out = append(out, fmt.Sprintf("schema must be %q, got %q", IterationSchema, it.Schema))
	}
	if !it.Kept {
		return out
	}
	if it.Candidate.Kind != RecalibrateKind {
		out = append(out, fmt.Sprintf("kept=true on a %s candidate — only a RECALIBRATE is self-keepable; a floor/unmeasured cell is routed, never kept", it.Candidate.Kind))
	}
	if it.ReplayedFold.Measured <= 0 {
		out = append(out, "kept=true with 0 measured replayed rows — fabricated gain")
	}
	if it.MeasuredDelta <= 0 {
		out = append(out, fmt.Sprintf("kept=true but measured_delta=%v is not a strict drop in the folded calibrable", it.MeasuredDelta))
	}
	if it.ReplayedFold.FloorBreachErr > it.BaselineFold.FloorBreachErr+epsilon {
		out = append(out, fmt.Sprintf("kept=true but the floor-breach term ROSE (%.4g -> %.4g) — a kept iteration may never raise a floor breach", it.BaselineFold.FloorBreachErr, it.ReplayedFold.FloorBreachErr))
	}
	min := it.MinSample
	if min <= 0 {
		min = DefaultMinSample
	}
	if it.Candidate.Sample < min {
		out = append(out, fmt.Sprintf("kept=true with sample %d < min %d — a recalibration fitted to too few boundaries", it.Candidate.Sample, min))
	}
	if !witnessOK(it.Witness) {
		out = append(out, "kept=true with no green external witness")
	}
	return out
}

// epsilon guards the floor-breach non-regression comparison against float noise.
const epsilon = 1e-9

// Round3 is re-exported for callers that fold dojocal numbers without importing
// mathx directly. It rounds to three decimals, matching the dojo's reporting.
func Round3(v float64) float64 { return mathx.Round3(v) }
