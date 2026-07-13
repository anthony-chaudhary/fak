package quality

import (
	"strings"
	"testing"
)

// TestBisectFindsPlantedFirstBadRevision is the epic Witness for #4583: a lineage
// with a decode regression planted at r4 bisects to EXACTLY r4 as the first bad
// revision, names r3 as its good predecessor, and carries the spine's localized
// first divergence (token 1, "increased" -> "decreased") and scrubbed replay bundle
// out as the actionable artifact. The evidence is produced by the REAL spine
// (RunCase over DemoCase), not a hand-written verdict, so the bisect is proven
// against genuine first-divergence evidence.
func TestBisectFindsPlantedFirstBadRevision(t *testing.T) {
	const badFrom = 4
	lineage := DemoBisectLineage()
	res := Bisect(lineage, DemoBisectProbe(badFrom))

	if res.Outcome != OutcomeFound {
		t.Fatalf("outcome = %q, want found; %s", res.Outcome, ExplainBisect(res))
	}
	if res.FirstBad == nil || res.FirstBad.Revision != "r4" {
		t.Fatalf("first bad = %+v, want revision r4", res.FirstBad)
	}
	if res.LastGood == nil || res.LastGood.Revision != "r3" {
		t.Fatalf("last good = %+v, want revision r3", res.LastGood)
	}

	// The culprit must carry the localized first divergence and a replay bundle — a
	// bisect that cannot point at the actionable token is not actionable.
	d := res.FirstDivergence
	if d == nil {
		t.Fatal("found regression must localize a first divergence")
	}
	if d.Index != 1 || d.Reference != "increased" || d.Engine != "decreased" {
		t.Errorf("first divergence = {idx %d ref %q eng %q}, want {1 increased decreased}", d.Index, d.Reference, d.Engine)
	}
	if res.Replay == nil {
		t.Fatal("found regression must emit a scrubbed replay artifact")
	}
	if res.Replay.CaseID != DemoCase().ID {
		t.Errorf("replay case id = %q, want %q", res.Replay.CaseID, DemoCase().ID)
	}
}

// TestBisectProbesSublinearly witnesses the reason to bisect at all: only ~log(n)
// of the lineage's points are evaluated, not every one. A linear scan of the
// 7-point demo lineage would probe all 7; the binary search probes strictly fewer,
// and that probe count is the runtime/resource cost the result documents.
func TestBisectProbesSublinearly(t *testing.T) {
	lineage := DemoBisectLineage()
	res := Bisect(lineage, DemoBisectProbe(4))
	if res.Outcome != OutcomeFound {
		t.Fatalf("outcome = %q, want found", res.Outcome)
	}
	if res.Probes >= len(lineage) {
		t.Errorf("bisect probed %d of %d points — expected sub-linear (a binary search, not a scan)", res.Probes, len(lineage))
	}
	// Cost documented must equal the summed cost of exactly the probed points.
	if want := float64(res.Probes) * 2; res.CostSeconds != want {
		t.Errorf("documented cost = %.0fs, want %.0fs (%d probes x 2s)", res.CostSeconds, want, res.Probes)
	}
}

// TestBisectCleanLineageAfterFix is the "passes after the fix" half of the Witness:
// the SAME lineage with the planted defect removed (badFrom < 0) bisects to CLEAN —
// no regression, no fabricated culprit. Found-then-Clean across identical inputs
// with only the defect toggled is the planted-defect-fails / fix-passes proof,
// independently replayed by the pure bisect.
func TestBisectCleanLineageAfterFix(t *testing.T) {
	res := Bisect(DemoBisectLineage(), DemoBisectProbe(-1))
	if res.Outcome != OutcomeClean {
		t.Fatalf("outcome = %q, want clean; %s", res.Outcome, ExplainBisect(res))
	}
	if res.FirstBad != nil {
		t.Errorf("clean lineage must not name a first bad revision: %+v", res.FirstBad)
	}
}

// TestBisectMissingProvenanceIsNeverGood witnesses the fail-closed rule (#4583:
// missing or inconclusive evidence is never pass) at the classify seam: a point
// whose evidence says State=pass but carries NO provenance is treated as
// indeterminate, not good — an unattributable pass cannot anchor a bisect.
func TestBisectMissingProvenanceIsNeverGood(t *testing.T) {
	probe := func(p BisectPoint) Evidence {
		return Evidence{CaseID: DemoCase().ID, State: StatePass} // pass, but empty provenance
	}
	res := Bisect(DemoBisectLineage(), probe)
	if res.Outcome != OutcomeIndeterminate {
		t.Fatalf("outcome = %q, want indeterminate; %s", res.Outcome, ExplainBisect(res))
	}
	if !strings.Contains(res.Reason, "provenance") {
		t.Errorf("reason = %q, want it to name incomplete provenance", res.Reason)
	}
}

// TestBisectInconclusiveMidpointHalts witnesses that an inconclusive point mid-search
// halts with a typed indeterminate outcome naming that point — the bisect never
// guesses PAST unclassifiable evidence to a plausible-looking culprit. Endpoints are
// clean good/bad; the first probed midpoint (r3) returns a fully-attributed but
// inconclusive verdict.
func TestBisectInconclusiveMidpointHalts(t *testing.T) {
	base := DemoBisectProbe(4)
	probe := func(p BisectPoint) Evidence {
		if p.Revision == "r3" {
			return Evidence{
				CaseID: DemoCase().ID,
				State:  StateInconclusive,
				Provenance: EvidenceProvenance{
					Model: "demo-1b", Tokenizer: "demo-bpe", Engine: p.Engine,
					Oracle: "greedy-token-diff", Revision: p.Revision, Baseline: "demo-reference@1",
				},
				Detail: "oracle timed out",
			}
		}
		return base(p)
	}
	res := Bisect(DemoBisectLineage(), probe)
	if res.Outcome != OutcomeIndeterminate {
		t.Fatalf("outcome = %q, want indeterminate; %s", res.Outcome, ExplainBisect(res))
	}
	if !strings.Contains(res.Reason, "r3") {
		t.Errorf("reason = %q, want it to name the inconclusive point r3", res.Reason)
	}
}

// TestBisectMisattributedEvidenceIsIndeterminate witnesses the replay-integrity
// guard: evidence attributed to a revision OTHER than the probed point cannot be
// trusted to describe that point, so it is indeterminate — the same discipline the
// release gate applies when calling evidence stale.
func TestBisectMisattributedEvidenceIsIndeterminate(t *testing.T) {
	probe := func(p BisectPoint) Evidence {
		return Evidence{
			CaseID: DemoCase().ID,
			State:  StatePass,
			Provenance: EvidenceProvenance{
				Model: "demo-1b", Tokenizer: "demo-bpe", Engine: p.Engine,
				Oracle: "greedy-token-diff", Revision: "somewhere-else", Baseline: "demo-reference@1",
			},
		}
	}
	res := Bisect(DemoBisectLineage(), probe)
	if res.Outcome != OutcomeIndeterminate {
		t.Fatalf("outcome = %q, want indeterminate; %s", res.Outcome, ExplainBisect(res))
	}
	if !strings.Contains(res.Reason, "attributed") {
		t.Errorf("reason = %q, want it to name the mis-attribution", res.Reason)
	}
}

// TestBisectOldestAlreadyBadIsIndeterminate witnesses honesty at the boundary: if
// the oldest point is already bad there is no good predecessor inside the range to
// localize a transition against, so the bisect refuses (indeterminate) rather than
// naming r0 a culprit it cannot prove regressed.
func TestBisectOldestAlreadyBadIsIndeterminate(t *testing.T) {
	res := Bisect(DemoBisectLineage(), DemoBisectProbe(0)) // every revision bad, incl. r0
	if res.Outcome != OutcomeIndeterminate {
		t.Fatalf("outcome = %q, want indeterminate; %s", res.Outcome, ExplainBisect(res))
	}
	if !strings.Contains(res.Reason, "precedes") {
		t.Errorf("reason = %q, want it to state the regression precedes the range", res.Reason)
	}
}

// TestBisectEmptyLineageIsIndeterminate witnesses the degenerate input: an empty
// lineage bisects to indeterminate, never a silent clean pass.
func TestBisectEmptyLineageIsIndeterminate(t *testing.T) {
	res := Bisect(nil, DemoBisectProbe(0))
	if res.Outcome != OutcomeIndeterminate {
		t.Fatalf("outcome = %q, want indeterminate", res.Outcome)
	}
}

// TestBisectEvidenceRecordsRequiredProvenance witnesses acceptance criterion 2:
// every case's evidence records model, tokenizer, engine/backend, seed OR a
// deterministic oracle, code/module revision, and tolerance/baseline provenance.
// The demo probe's evidence must satisfy EvidenceProvenance.complete() for the point
// it was produced at.
func TestBisectEvidenceRecordsRequiredProvenance(t *testing.T) {
	pt := DemoBisectLineage()[2] // r2, a clean point under DemoBisectProbe(4)
	ev := DemoBisectProbe(4)(pt)
	if ok, why := ev.Provenance.complete(); !ok {
		t.Fatalf("demo evidence provenance incomplete: %s", why)
	}
	if ev.Provenance.Revision != pt.Revision {
		t.Errorf("provenance revision = %q, want %q", ev.Provenance.Revision, pt.Revision)
	}
	if ev.Provenance.Engine == "" || ev.Provenance.Oracle == "" || ev.Provenance.Baseline == "" {
		t.Errorf("provenance missing engine/oracle/baseline: %+v", ev.Provenance)
	}
}

// TestExplainBisectRendersOutcomes is a smoke test on the operator readout: the
// Found rendering names the first bad revision and its divergence; the Indeterminate
// rendering states plainly that inconclusive evidence is never treated as good.
func TestExplainBisectRendersOutcomes(t *testing.T) {
	found := ExplainBisect(Bisect(DemoBisectLineage(), DemoBisectProbe(4)))
	if !strings.Contains(found, "REGRESSION") || !strings.Contains(found, "r4") {
		t.Errorf("found readout missing regression/first-bad:\n%s", found)
	}
	indet := ExplainBisect(Bisect(DemoBisectLineage(), DemoBisectProbe(0)))
	if !strings.Contains(indet, "INDETERMINATE") || !strings.Contains(indet, "never treated as good") {
		t.Errorf("indeterminate readout missing fail-closed note:\n%s", indet)
	}
}
