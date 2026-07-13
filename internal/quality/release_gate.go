package quality

import (
	"fmt"
	"sort"
	"strings"
)

// This file is the release-qualification aggregator (#4578, under epic #4509): the
// layer that folds the spine's individual per-tier gates — deterministic
// (seed_replay), statistical (sampler_distribution), report (report_*), and
// hardware evidence — into a SINGLE release decision. The spine (run.go) answers
// "did this one case pass?"; this answers "may this revision ship?", and answers it
// fail-closed: a required gate whose evidence is missing, stale, inconclusive, or
// failing blocks the release. It is additive — it registers no oracles and edits no
// core, consuming only the Result/Divergence/FailureBundle the spine already emits.

// ReleaseGateSchema is the versioned tag on a release-qualification decision.
// Consumers pin the major so a schema bump is a conscious migration (the #4519
// house rule), not a silent field drift.
const ReleaseGateSchema = "fak-quality-release/1"

// Tier is the cost/cadence class a required quality case is assigned to. The
// release gate aggregates evidence per tier so a cheap PR check is never confused
// with a nightly hardware suite — the PR / nightly / release separation top stacks
// keep (#4578 scope).
type Tier string

const (
	TierPR      Tier = "pr"
	TierNightly Tier = "nightly"
	TierRelease Tier = "release"
)

// GateKind is the evidence family a required case belongs to. #4578 requires the
// four families top stacks separate: deterministic (exact oracle), statistical
// (distribution/tolerance), hardware (device-dependent), and report (rubric).
type GateKind string

const (
	KindDeterministic GateKind = "deterministic"
	KindStatistical   GateKind = "statistical"
	KindHardware      GateKind = "hardware"
	KindReport        GateKind = "report"
)

// EvidenceState is the closed set a required gate's evidence can be in. Only Pass
// releases; Fail, Missing, Stale, and Inconclusive all BLOCK — "no evidence" and
// "unclear evidence" are never a pass (#4578 acceptance: missing or inconclusive
// evidence is never pass).
type EvidenceState string

const (
	StatePass         EvidenceState = "pass"
	StateFail         EvidenceState = "fail"
	StateMissing      EvidenceState = "missing"
	StateStale        EvidenceState = "stale"
	StateInconclusive EvidenceState = "inconclusive"
)

// releases reports whether a state is the single releasing state. Everything else
// blocks — the fail-closed default is the point.
func (s EvidenceState) releases() bool { return s == StatePass }

// EvidenceProvenance is the per-case provenance #4578 requires every qualification
// record to carry: which model/tokenizer/engine produced the evidence, the seed
// (stochastic) OR deterministic oracle it was judged by, the code/module revision
// it was produced at, and the tolerance/baseline it was compared against. Absent
// provenance is treated as inconclusive — evidence you cannot attribute cannot
// release.
type EvidenceProvenance struct {
	Model     string `json:"model"`
	Tokenizer string `json:"tokenizer"`
	Engine    string `json:"engine"`
	// Seed pins a stochastic case; Oracle names the deterministic comparator. At
	// least one must be set — a case that is neither seeded nor oracle-judged is not
	// reproducible, so it cannot qualify a release.
	Seed   int64  `json:"seed,omitempty"`
	Oracle string `json:"oracle,omitempty"`
	// Revision is the code/module revision the evidence was produced at. It is the
	// staleness key: evidence produced at a different revision than the release
	// under test is stale and blocks.
	Revision string `json:"revision"`
	// Baseline is the tolerance/baseline provenance the case was judged against.
	Baseline string `json:"baseline"`
}

// complete reports whether the provenance names everything #4578 requires. The
// returned reason localizes the FIRST missing field so a gap is actionable rather
// than a bare "incomplete".
func (p EvidenceProvenance) complete() (bool, string) {
	switch {
	case p.Model == "":
		return false, "missing model"
	case p.Tokenizer == "":
		return false, "missing tokenizer"
	case p.Engine == "":
		return false, "missing engine/backend"
	case p.Seed == 0 && p.Oracle == "":
		return false, "missing seed or deterministic oracle"
	case p.Revision == "":
		return false, "missing code/module revision"
	case p.Baseline == "":
		return false, "missing tolerance/baseline provenance"
	}
	return true, ""
}

// RequiredGate is one (case, tier, kind) the release MUST have fresh, passing
// evidence for before it ships. CostSeconds documents the runtime/resource cost of
// producing the evidence (#4578: assign a tier and document cost) so an operator
// can see what a release qualification costs per tier.
type RequiredGate struct {
	CaseID      string   `json:"case_id"`
	Tier        Tier     `json:"tier"`
	Kind        GateKind `json:"kind"`
	CostSeconds float64  `json:"cost_seconds"`
}

// Evidence is a produced qualification record for one case: its state, provenance,
// and — on a non-pass — the localized first divergence and scrubbed replay bundle
// folded from the underlying quality Result. It is what a producer submits to the
// release gate; the gate never re-runs a case, it adjudicates the submitted
// evidence.
type Evidence struct {
	CaseID          string             `json:"case_id"`
	State           EvidenceState      `json:"state"`
	Provenance      EvidenceProvenance `json:"provenance"`
	FirstDivergence *Divergence        `json:"first_divergence,omitempty"`
	Replay          *FailureBundle     `json:"replay,omitempty"`
	Detail          string             `json:"detail,omitempty"`
}

// EvidenceFromResult folds a spine Result (run.go) into release Evidence under a
// given provenance. It is the seam that ties the release gate to the package's own
// gates: a RunCase pass becomes releasing evidence, and a RunCase failure becomes
// blocking Fail evidence carrying the SAME first-divergence and scrubbed
// FailureBundle the spine already localized. Provenance (model/revision/…) is
// supplied by the producer — the Result itself is deliberately host- and
// clock-free, so attribution is added here at submission time.
func EvidenceFromResult(prov EvidenceProvenance, r Result) Evidence {
	e := Evidence{CaseID: r.CaseID, Provenance: prov}
	if r.Pass {
		e.State = StatePass
		e.Detail = fmt.Sprintf("%d oracle(s) agreed with the reference", len(r.Verdicts))
		return e
	}
	e.State = StateFail
	if fb := r.FailureBundle; fb != nil {
		e.Replay = fb
		e.FirstDivergence = fb.FirstDivergence
		e.Detail = fmt.Sprintf("first failure %s (%s): %s", fb.FailingOracle, fb.FailingKind, fb.Detail)
	} else {
		e.Detail = "result failed without a bundle"
	}
	return e
}

// ReleaseBlock is one required gate that did not release, with the reason it blocked
// and — where the evidence carried it — the first actionable divergence and scrubbed
// replay artifact. The gate emits these first-actionable-first so an operator reads
// the earliest blocking gate, not an averaged summary.
type ReleaseBlock struct {
	CaseID          string         `json:"case_id"`
	Tier            Tier           `json:"tier"`
	Kind            GateKind       `json:"kind"`
	State           EvidenceState  `json:"state"`
	Reason          string         `json:"reason"`
	FirstDivergence *Divergence    `json:"first_divergence,omitempty"`
	Replay          *FailureBundle `json:"replay,omitempty"`
}

// ReleaseDecision is the machine-readable output of the release gate: whether the
// revision may ship, every gate that blocked it, and the case IDs that qualified.
// Released is true iff every required gate has fresh, passing, fully-attributed
// evidence — the fail-closed contract of #4578.
type ReleaseDecision struct {
	Schema   string         `json:"schema"`
	Revision string         `json:"revision"`
	Released bool           `json:"released"`
	Blocks   []ReleaseBlock `json:"blocks,omitempty"`
	Passed   []string       `json:"passed,omitempty"`
}

// QualifyRelease adjudicates whether revision `rev` may ship given the required
// gates and the produced evidence. A required gate blocks when its evidence is
// missing, lacks complete provenance (inconclusive), was produced at a different
// revision (stale), or did not pass. Blocks preserve the required-gate order, so
// Blocks[0] is the first actionable divergence. It is a pure function of its inputs
// — same inputs, same decision — so a release verdict replays.
func QualifyRelease(rev string, required []RequiredGate, evidence []Evidence) ReleaseDecision {
	byCase := make(map[string]Evidence, len(evidence))
	for _, e := range evidence {
		byCase[e.CaseID] = e
	}
	d := ReleaseDecision{Schema: ReleaseGateSchema, Revision: rev}
	for _, g := range required {
		e, ok := byCase[g.CaseID]
		if !ok {
			d.Blocks = append(d.Blocks, ReleaseBlock{
				CaseID: g.CaseID, Tier: g.Tier, Kind: g.Kind,
				State: StateMissing, Reason: "no evidence submitted for required case",
			})
			continue
		}
		block := ReleaseBlock{
			CaseID: g.CaseID, Tier: g.Tier, Kind: g.Kind,
			FirstDivergence: e.FirstDivergence, Replay: e.Replay,
		}
		switch {
		case !mustComplete(e.Provenance):
			_, why := e.Provenance.complete()
			block.State, block.Reason = StateInconclusive, "incomplete provenance: "+why
		case e.Provenance.Revision != rev:
			block.State = StateStale
			block.Reason = fmt.Sprintf("evidence produced at revision %q != release %q", e.Provenance.Revision, rev)
		case !e.State.releases():
			block.State = e.State
			block.Reason = "evidence state " + string(e.State)
			if e.Detail != "" {
				block.Reason += ": " + e.Detail
			}
		default:
			d.Passed = append(d.Passed, g.CaseID)
			continue
		}
		d.Blocks = append(d.Blocks, block)
	}
	sort.Strings(d.Passed)
	d.Released = len(d.Blocks) == 0
	return d
}

func mustComplete(p EvidenceProvenance) bool {
	ok, _ := p.complete()
	return ok
}

// ExplainRelease renders a ReleaseDecision as an operator readout: RELEASED with the
// qualified cases, or BLOCKED naming the first actionable divergence first and then
// every other blocking gate. It mirrors Explain (run.go) — the bridge from a machine
// verdict to "here is exactly what to fix before you can ship".
func ExplainRelease(d ReleaseDecision) string {
	var b strings.Builder
	if d.Released {
		fmt.Fprintf(&b, "RELEASED  revision %s — %d required gate(s) qualified\n", d.Revision, len(d.Passed))
		for _, id := range d.Passed {
			fmt.Fprintf(&b, "  ok   %s\n", id)
		}
		return b.String()
	}
	fmt.Fprintf(&b, "BLOCKED  revision %s — %d required gate(s) did not qualify\n", d.Revision, len(d.Blocks))
	for i, blk := range d.Blocks {
		marker := "  "
		if i == 0 {
			marker = "->" // the first actionable divergence
		}
		fmt.Fprintf(&b, "%s %-8s %-13s %-12s %s\n", marker, blk.Tier, blk.Kind, blk.State, blk.CaseID)
		fmt.Fprintf(&b, "     reason: %s\n", blk.Reason)
		if dv := blk.FirstDivergence; dv != nil {
			fmt.Fprintf(&b, "     first divergence at token %d: reference %q, engine %q\n", dv.Index, dv.Reference, dv.Engine)
		}
	}
	return b.String()
}
