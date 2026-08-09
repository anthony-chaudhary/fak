package quality

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// This is the #4584 witness. The property under test is the LIFECYCLE across
// runs, which no single-run gate can prove: a planted regression must file ONE
// issue however many nights it recurs, that issue's body must reproduce the
// defect from its own text, and it must close only after a green WINDOW — never
// on one green run, and never on silence.

const autofileRev = "git:cafef00d"

// recorder is a Filer that lands everything and remembers what it was handed. It
// stands in for the `gh issue create/comment/close` driver.
type recorder struct {
	filings []Filing
	failOn  FilingAction // when set, every filing with this action is refused
}

func (r *recorder) file(f Filing) error {
	if r.failOn != "" && f.Action == r.failOn {
		return errors.New("tracker of record refused the write")
	}
	r.filings = append(r.filings, f)
	return nil
}

func (r *recorder) last() Filing { return r.filings[len(r.filings)-1] }

// runOnce plans and applies one run of the demo case at the given defect.
func runOnce(t *testing.T, tr *Tracker, rec *recorder, runID, defect string) FilingPlan {
	t.Helper()
	run := FilingRun{ID: runID, Revision: autofileRev}
	plan := tr.Plan(run, []Observation{DemoFilingObservation(autofileRev, defect)})
	tr.Apply(plan, rec.file)
	return plan
}

func onlyFiling(t *testing.T, plan FilingPlan) Filing {
	t.Helper()
	if len(plan.Filings) != 1 {
		t.Fatalf("want exactly 1 filing, got %d:\n%s", len(plan.Filings), ExplainFilingPlan(plan))
	}
	return plan.Filings[0]
}

// TestAutoFileLifecycleOpensUpdatesThenClosesOneIssue is the headline witness:
// the SAME planted decode defect observed on two runs files ONE issue (open, then
// update on the identical marker), and the fixed engine closes it only after the
// green window is complete — one green run holds, the second closes.
func TestAutoFileLifecycleOpensUpdatesThenClosesOneIssue(t *testing.T) {
	tr := NewTracker(2)
	rec := &recorder{}

	open := onlyFiling(t, runOnce(t, tr, rec, "night-1", "decode"))
	if open.Action != ActionOpen || open.Kind != FilingRegression {
		t.Fatalf("run 1 = %s/%s, want open/regression:\n%s", open.Action, open.Kind, open.Reason)
	}
	wantKey := RegressionKey{
		CaseID: "spine-demo-exec-report", Model: "demo-1b", Backend: "cpu", Mode: "eager",
		Metric: "greedy-token-diff", FirstBad: "token:1",
	}
	if open.Key != wantKey {
		t.Errorf("key = %+v, want %+v", open.Key, wantKey)
	}
	if open.Tier != TierPR || open.Cost.RuntimeSeconds != 2 || open.Cost.TimeoutSeconds != 30 {
		t.Errorf("filing must carry the case's tier and cost: tier=%q cost=%+v", open.Tier, open.Cost)
	}

	update := onlyFiling(t, runOnce(t, tr, rec, "night-2", "decode"))
	if update.Action != ActionUpdate {
		t.Fatalf("run 2 = %s, want update (a repeated run must update ONE issue)", update.Action)
	}
	if update.Marker != open.Marker {
		t.Fatalf("run 2 marker %q != run 1 marker %q — a recurrence must key to the same issue", update.Marker, open.Marker)
	}
	if len(tr.OpenIssues()) != 1 {
		t.Fatalf("two runs of one defect left %d open issue(s), want 1", len(tr.OpenIssues()))
	}
	if got := tr.Issues[open.Marker].Occurrences; got != 2 {
		t.Errorf("occurrences = %d, want 2", got)
	}

	// The fix lands: one green run HOLDS (a single green run is as consistent
	// with a flake as with a fix), the second completes the window and closes.
	hold := onlyFiling(t, runOnce(t, tr, rec, "night-3", ""))
	if hold.Action != ActionHold {
		t.Fatalf("run 3 = %s, want hold; %s", hold.Action, hold.Reason)
	}
	if !tr.Issues[open.Marker].Open {
		t.Error("a single green run must not close the issue")
	}

	closed := onlyFiling(t, runOnce(t, tr, rec, "night-4", ""))
	if closed.Action != ActionClose {
		t.Fatalf("run 4 = %s, want close; %s", closed.Action, closed.Reason)
	}
	if tr.Issues[open.Marker].Open {
		t.Error("the issue must be closed once the green window completes")
	}
	if len(tr.OpenIssues()) != 0 {
		t.Errorf("open issues after recovery = %d, want 0", len(tr.OpenIssues()))
	}

	// A closed issue is not re-held by further green runs: nothing more to file.
	if plan := runOnce(t, tr, rec, "night-5", ""); len(plan.Filings) != 0 {
		t.Errorf("a green run after closure filed %d action(s), want 0:\n%s", len(plan.Filings), ExplainFilingPlan(plan))
	}
}

// TestFiledIssueBodyReplaysThePlantedDefect is the replay-evidence half of the
// witness: the artifact embedded in the filed issue's OWN body is cut back out,
// loaded through the package's public bundle loader, and replayed — reproducing
// the planted decode divergence from the issue text alone, with no access to the
// case, the engine, or the run that produced it.
func TestFiledIssueBodyReplaysThePlantedDefect(t *testing.T) {
	tr := NewTracker(2)
	rec := &recorder{}
	open := onlyFiling(t, runOnce(t, tr, rec, "night-1", "decode"))

	blob := fencedJSON(t, open.Body)
	b, err := LoadBundle([]byte(blob))
	if err != nil {
		t.Fatalf("the filed body's artifact must load as a bundle: %v", err)
	}
	if !b.Scrubbed {
		t.Error("a filed artifact must be scrubbed")
	}
	v := Replay(b)
	if !v.Reproduced {
		t.Fatalf("the filed issue must reproduce its own defect: %s", ExplainReplay(v))
	}
	if v.Observed == nil || v.Observed.FirstDivergence == nil {
		t.Fatal("replay must localize a first divergence")
	}
	if d := v.Observed.FirstDivergence; d.Index != 1 || d.Reference != "increased" || d.Engine != "decreased" {
		t.Errorf("replayed divergence = {idx %d ref %q eng %q}, want {1 increased decreased}", d.Index, d.Reference, d.Engine)
	}

	// The body must also carry the provenance and routing header an operator needs
	// to trust and cost the finding (#4584 acceptance 2 and 4).
	for _, want := range []string{
		open.Marker,
		"- model: `demo-1b`",
		"- tokenizer: `demo-bpe`",
		"- engine/backend: `fak/cpu`",
		"- determinism: deterministic oracle `exact-greedy-trace`",
		"- code/module revision: `" + autofileRev + "`",
		"- tolerance: `exact-token` @ `policy:v1`",
		"- baseline: `spine-demo-baseline` @ `sha256:b-demo`",
		"- tier: `pr` — runtime 2s, timeout 30s, cpu 1, memory 256MiB, accelerators 0",
		"first actionable divergence: token 1",
	} {
		if !strings.Contains(open.Body, want) {
			t.Errorf("filed body is missing %q", want)
		}
	}
}

// TestRecurrenceResetsTheGreenWindow: a defect that comes back mid-window must
// not inherit the green runs it had already banked. Without the reset, a defect
// that fails every other night would close on its second green run forever.
func TestRecurrenceResetsTheGreenWindow(t *testing.T) {
	tr := NewTracker(2)
	rec := &recorder{}
	open := onlyFiling(t, runOnce(t, tr, rec, "night-1", "decode"))
	if got := onlyFiling(t, runOnce(t, tr, rec, "night-2", "")).Action; got != ActionHold {
		t.Fatalf("run 2 = %s, want hold", got)
	}
	again := onlyFiling(t, runOnce(t, tr, rec, "night-3", "decode"))
	if again.Action != ActionUpdate {
		t.Fatalf("run 3 = %s, want update (the defect returned)", again.Action)
	}
	if got := tr.Issues[open.Marker].GreenRuns; got != 0 {
		t.Fatalf("green runs after a recurrence = %d, want 0", got)
	}
	// One green run is now NOT enough — the window restarts from zero.
	if got := onlyFiling(t, runOnce(t, tr, rec, "night-4", "")).Action; got != ActionHold {
		t.Fatalf("run 4 = %s, want hold (the window restarted)", got)
	}
	if got := onlyFiling(t, runOnce(t, tr, rec, "night-5", "")).Action; got != ActionClose {
		t.Fatalf("run 5 = %s, want close", got)
	}
}

// TestUnobservedCoordinatesNeverClose: an issue whose coordinates were not run at
// all must neither advance toward closure nor be touched. Silence is not a green
// run — the same "missing evidence is never a pass" rule, applied to absence.
func TestUnobservedCoordinatesNeverClose(t *testing.T) {
	tr := NewTracker(1) // even the most eager window must not fire on silence
	rec := &recorder{}
	open := onlyFiling(t, runOnce(t, tr, rec, "night-1", "decode"))

	for _, id := range []string{"night-2", "night-3"} {
		plan := tr.Plan(FilingRun{ID: id, Revision: autofileRev}, nil)
		if len(plan.Filings) != 0 {
			t.Fatalf("run %s with no observations filed %d action(s), want 0:\n%s", id, len(plan.Filings), ExplainFilingPlan(plan))
		}
		tr.Apply(plan, rec.file)
	}
	iss := tr.Issues[open.Marker]
	if !iss.Open {
		t.Error("an unobserved issue must stay open")
	}
	if iss.GreenRuns != 0 {
		t.Errorf("green runs from silence = %d, want 0", iss.GreenRuns)
	}
}

// TestEvidenceGapsFileAndNeverGreen walks the ways evidence can be untrustworthy.
// Each files an evidence-gap issue rather than passing, carries NO replay
// artifact (and says so), and — critically — does not exonerate the coordinates,
// so an open regression on them keeps its window at zero.
func TestEvidenceGapsFileAndNeverGreen(t *testing.T) {
	noArtifact := func() Observation {
		o := DemoFilingObservation(autofileRev, "decode")
		o.Evidence.Replay = nil
		return o
	}
	unscrubbed := func() Observation {
		o := DemoFilingObservation(autofileRev, "decode")
		fb := *o.Evidence.Replay
		fb.Scrubbed = false
		o.Evidence.Replay = &fb
		return o
	}
	incompleteProvenance := func() Observation {
		o := DemoFilingObservation(autofileRev, "") // a PASS...
		o.Evidence.Provenance.Tokenizer = ""        // ...that cannot be attributed
		return o
	}
	staleEvidence := func() Observation {
		o := DemoFilingObservation("git:0ldc0de", "")
		return o
	}
	noRoutingHeader := func() Observation {
		o := DemoFilingObservation(autofileRev, "decode")
		o.Case.Metadata.Cost.TimeoutSeconds = 0 // cannot be costed or tier-budgeted
		return o
	}

	for _, tc := range []struct {
		name  string
		obs   Observation
		cause string
	}{
		{"failure without an artifact", noArtifact(), "gap:" + causeNoArtifact},
		{"unscrubbed artifact", unscrubbed(), "gap:" + causeUnscrubbed},
		{"pass with incomplete provenance", incompleteProvenance(), "gap:" + causeProvenanceIncomplete},
		{"pass produced at another revision", staleEvidence(), "gap:" + causeStaleEvidence},
		{"case with no routing header", noRoutingHeader(), "gap:" + causeCaseHeaderIncomplete},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := NewTracker(1)
			rec := &recorder{}
			// An open regression on the same coordinates, so we can prove the gap
			// does not count as the green run that would close it.
			reg := onlyFiling(t, runOnce(t, tr, rec, "night-1", "decode"))

			plan := tr.Plan(FilingRun{ID: "night-2", Revision: autofileRev}, []Observation{tc.obs})
			tr.Apply(plan, rec.file)

			var gap *Filing
			for i := range plan.Filings {
				if plan.Filings[i].Kind == FilingEvidenceGap {
					gap = &plan.Filings[i]
				}
				if plan.Filings[i].Action == ActionClose || plan.Filings[i].Action == ActionHold {
					t.Fatalf("an evidence gap must never green the coordinates: got %s\n%s",
						plan.Filings[i].Action, ExplainFilingPlan(plan))
				}
			}
			if gap == nil {
				t.Fatalf("no evidence-gap issue was filed:\n%s", ExplainFilingPlan(plan))
			}
			if gap.Key.FirstBad != tc.cause {
				t.Errorf("gap first_bad = %q, want %q", gap.Key.FirstBad, tc.cause)
			}
			if gap.Replay != nil {
				t.Error("an evidence gap must carry no replay artifact")
			}
			if !strings.Contains(gap.Body, "ABSENCE of evidence, which is never a pass") {
				t.Errorf("gap body must say plainly that absent evidence is not a pass:\n%s", gap.Body)
			}
			if !tr.Issues[reg.Marker].Open {
				t.Error("the open regression must stay open through an evidence gap")
			}
		})
	}
}

// TestOneRunDeduplicatesRepeatedObservations: a harness that retried the same
// case in one run observed one defect twice, not two defects. The fold must file
// ONE action and count ONE occurrence.
func TestOneRunDeduplicatesRepeatedObservations(t *testing.T) {
	tr := NewTracker(2)
	rec := &recorder{}
	obs := []Observation{
		DemoFilingObservation(autofileRev, "decode"),
		DemoFilingObservation(autofileRev, "decode"),
		DemoFilingObservation(autofileRev, "decode"),
	}
	plan := tr.Plan(FilingRun{ID: "night-1", Revision: autofileRev}, obs)
	f := onlyFiling(t, plan)
	tr.Apply(plan, rec.file)
	if got := tr.Issues[f.Marker].Occurrences; got != 1 {
		t.Errorf("occurrences from one run = %d, want 1", got)
	}
}

// TestDistinctDefectsGetDistinctIssues is the ablation that proves the dedup key
// is doing work rather than merely collapsing everything: two DIFFERENT planted
// defects on the same case must not share an issue, because they are not the same
// thing to fix.
func TestDistinctDefectsGetDistinctIssues(t *testing.T) {
	tr := NewTracker(2)
	rec := &recorder{}
	plan := tr.Plan(FilingRun{ID: "night-1", Revision: autofileRev}, []Observation{
		DemoFilingObservation(autofileRev, "decode"),
		DemoFilingObservation(autofileRev, "report"),
	})
	if len(plan.Filings) != 2 {
		t.Fatalf("two distinct defects filed %d issue(s), want 2:\n%s", len(plan.Filings), ExplainFilingPlan(plan))
	}
	if plan.Filings[0].Marker == plan.Filings[1].Marker {
		t.Error("two distinct defects must not share one marker")
	}
	tr.Apply(plan, rec.file)
	if len(tr.OpenIssues()) != 2 {
		t.Errorf("open issues = %d, want 2", len(tr.OpenIssues()))
	}
}

// TestFilingThatDidNotLandIsRetried: a driver that could not write to the tracker
// of record must leave the lifecycle untouched, so the next run proposes the same
// action again. A dropped file is retried, never mistaken for filed.
func TestFilingThatDidNotLandIsRetried(t *testing.T) {
	tr := NewTracker(2)
	broken := &recorder{failOn: ActionOpen}
	run := FilingRun{ID: "night-1", Revision: autofileRev}
	plan := tr.Plan(run, []Observation{DemoFilingObservation(autofileRev, "decode")})
	rep := tr.Apply(plan, broken.file)
	if len(rep.Failed) != 1 || len(rep.Landed) != 0 {
		t.Fatalf("landed=%d failed=%d, want 0/1", len(rep.Landed), len(rep.Failed))
	}
	if len(tr.Issues) != 0 {
		t.Fatalf("a refused filing must not advance the tracker: %+v", tr.Issues)
	}
	// Next run: still an OPEN, not an update against an issue that never existed.
	working := &recorder{}
	again := onlyFiling(t, runOnce(t, tr, working, "night-2", "decode"))
	if again.Action != ActionOpen {
		t.Fatalf("retry = %s, want open", again.Action)
	}
	if got := tr.Issues[again.Marker].Occurrences; got != 1 {
		t.Errorf("occurrences after one landed filing = %d, want 1", got)
	}
}

// TestRoundTrippedPlanIsRefusedNotSilentlyAdopted: FilingPlan is json-tagged
// precisely so a driver can plan in one process and apply in another, but the
// lifecycle advance is unexported and does NOT survive that trip. Applying the
// decoded plan must therefore refuse — not adopt a zero-valued issue, which
// would blank the tracker record the filing was about (reopening a closed defect
// or losing a recurrence's history) while reporting the filing as landed.
func TestRoundTrippedPlanIsRefusedNotSilentlyAdopted(t *testing.T) {
	tr := NewTracker(2)
	rec := &recorder{}
	open := onlyFiling(t, runOnce(t, tr, rec, "night-1", "decode"))
	before := tr.Issues[open.Marker]

	blob, err := json.Marshal(tr.Plan(FilingRun{ID: "night-2", Revision: autofileRev},
		[]Observation{DemoFilingObservation(autofileRev, "decode")}))
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}
	var decoded FilingPlan
	if err := json.Unmarshal(blob, &decoded); err != nil {
		t.Fatalf("unmarshal plan: %v", err)
	}

	rep := tr.Apply(decoded, rec.file)
	if len(rep.Landed) != 0 || len(rep.Failed) != 1 {
		t.Fatalf("landed=%d failed=%d, want 0/1 — a plan with no lifecycle advance must not be adopted",
			len(rep.Landed), len(rep.Failed))
	}
	if !strings.Contains(rep.Failed[0].Error, "re-planned against the tracker") {
		t.Errorf("refusal must name the recovery step, got %q", rep.Failed[0].Error)
	}
	if got := tr.Issues[open.Marker]; got != before {
		t.Errorf("tracker record was mutated by a refused filing:\n got %+v\nwant %+v", got, before)
	}
	if len(tr.OpenIssues()) != 1 {
		t.Errorf("open issues = %d, want 1", len(tr.OpenIssues()))
	}
}

// TestMarkerIsCommentSafe: the marker is embedded in an issue body as an HTML
// comment, so no key — however hostile — may close it early or collide.
func TestMarkerIsCommentSafe(t *testing.T) {
	k := RegressionKey{
		CaseID: "case --> injected", Model: "m<1>", Backend: "b\nc", Mode: "e",
		Metric: "greedy-token-diff", FirstBad: "token:1",
	}
	m := k.Marker()
	if !strings.HasPrefix(m, "<!-- ") || !strings.HasSuffix(m, " -->") {
		t.Fatalf("marker is not a well-formed comment: %q", m)
	}
	if strings.Contains(strings.TrimSuffix(strings.TrimPrefix(m, "<!-- "), " -->"), "--") {
		t.Errorf("marker payload contains a comment terminator: %q", m)
	}
	for _, bad := range []string{"<", ">", "\n"} {
		if strings.Contains(strings.TrimSuffix(strings.TrimPrefix(m, "<!-- "), " -->"), bad) {
			t.Errorf("marker payload contains %q: %q", bad, m)
		}
	}
	// The marker must be INJECTIVE: a lossy one would fold two different defects
	// into one issue, which is the exact mistake deduplication exists to prevent.
	// The awkward cases are escaping that erases a distinction ("m<1>" vs "m_1_")
	// and a case id that itself contains the field separator — which the nightly
	// matrix mints for every cell ("nightly-matrix/<model>/<backend>/...").
	seen := map[string]RegressionKey{}
	for _, cand := range []RegressionKey{
		k,
		{CaseID: "case --> injected", Model: "m_1_", Backend: "b\nc", Mode: "e", Metric: "greedy-token-diff", FirstBad: "token:1"},
		{CaseID: "nightly-matrix/small/cuda/graph/throughput", Metric: "greedy-token-diff", FirstBad: "token:1"},
		{CaseID: "nightly-matrix", Model: "small", Backend: "cuda", Mode: "graph/throughput", Metric: "greedy-token-diff", FirstBad: "token:1"},
		{CaseID: "c", Metric: "m", FirstBad: "token:1"},
		{CaseID: "c", Model: "-", Metric: "m", FirstBad: "token:1"},
		{CaseID: "c%2F", Metric: "m", FirstBad: "token:1"},
		{CaseID: "c/", Metric: "m", FirstBad: "token:1"},
	} {
		mk := cand.Marker()
		if prior, dup := seen[mk]; dup {
			t.Errorf("keys %+v and %+v collided on marker %q", prior, cand, mk)
		}
		payload := strings.TrimSuffix(strings.TrimPrefix(mk, "<!-- "), " -->")
		if strings.ContainsAny(payload, "<>\n") || strings.Contains(payload, "--") {
			t.Errorf("marker payload is not comment-safe: %q", mk)
		}
		seen[mk] = cand
	}
}

// fencedJSON cuts the one ```json fenced block out of an issue body.
func fencedJSON(t *testing.T, body string) string {
	t.Helper()
	const open = "```json\n"
	i := strings.Index(body, open)
	if i < 0 {
		t.Fatalf("body carries no fenced json artifact:\n%s", body)
	}
	rest := body[i+len(open):]
	j := strings.Index(rest, "\n```")
	if j < 0 {
		t.Fatalf("fenced json artifact is unterminated:\n%s", body)
	}
	return rest[:j]
}
