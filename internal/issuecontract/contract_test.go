package issuecontract

import (
	"strings"
	"testing"
)

func completeCandidate() Candidate {
	return Candidate{
		Schema:          Schema,
		Key:             "task_push_next/strict-scope",
		Title:           "taskmgr: enforce strict handoff scope",
		ParentRef:       "task_push_next",
		CurrentState:    "Task handoff can already create stable follow-up issues.",
		WhyNow:          "Generated issues are the next weak point before dispatch.",
		WorkingSpine:    "A verified task completion creates one scoped follow-up issue.",
		PriorityContext: "Working path: clean Stop handoff -> scoped issue -> dispatch. Current blocker: vague follow-ups waste dispatch cycles. Unblocks: guard live handoff. Not polish: enforce the smallest leaf before optimization.",
		WorkUnit:        "leaf",
		ExpectedSteps:   3,
		Assumptions:     []string{"The handoff producer can derive the candidate before syncing."},
		ConfusionRisks:  []string{"A broad follow-up can be mistaken for an epic unless scoped."},
		Coordination:    []string{"Do not dispatch concurrently with taskmgr handoff body edits."},
		Trigger:         "A verified completion handoff proposes this next leaf.",
		BatchPolicy:     "At most two follow-up issues per handoff; update by marker key on rerun.",
		InScope:         "Review the next-step candidate and render scoped sections.",
		OutOfScope:      "Do not optimize issue routing or add new scorecards.",
		DoneCondition:   "Legacy handoffs pass by default; strict handoffs refuse vague next steps.",
		Witness:         "go test ./internal/taskmgr",
		AcceptanceGate:  "go test ./cmd/fak -run TestTaskHandoff",
		Lane:            "taskmgr",
		Paths:           []string{"internal/taskmgr/handoff.go"},
		BoundaryNotes:   []string{"Public issue only; no private lab evidence."},
		ClosureBinding:  "Resolving commit cites #N and carries a matching (fak <leaf>) trailer.",
	}
}

func TestReviewCandidateDispatchableScoresFull(t *testing.T) {
	review := ReviewCandidate(completeCandidate(), Options{})
	if !review.OK || review.Dispatchability != Dispatchable || review.Verdict != "ready" {
		t.Fatalf("review = %+v, want ready dispatchable", review)
	}
	if review.Score.Total != 100 {
		t.Fatalf("score = %+v, want total 100", review.Score)
	}
	if review.SpinePriority.Total != 100 {
		t.Fatalf("spine priority = %+v, want total 100", review.SpinePriority)
	}
	if review.AgentContext.Total != 100 {
		t.Fatalf("agent context = %+v, want total 100", review.AgentContext)
	}
	if review.WorkUnit != "leaf" || review.ExpectedSteps != 3 ||
		review.Trigger != "A verified completion handoff proposes this next leaf." ||
		review.BatchPolicy != "At most two follow-up issues per handoff; update by marker key on rerun." {
		t.Fatalf("review metadata = work_unit %q expected_steps %d trigger %q batch_policy %q",
			review.WorkUnit, review.ExpectedSteps, review.Trigger, review.BatchPolicy)
	}
	if len(review.Assumptions) != 1 || len(review.ConfusionRisks) != 1 || len(review.Coordination) != 1 ||
		review.Coordination[0] != "Do not dispatch concurrently with taskmgr handoff body edits." {
		t.Fatalf("review agent notes = assumptions=%v confusion=%v coordination=%v",
			review.Assumptions, review.ConfusionRisks, review.Coordination)
	}
	if len(review.Reasons) != 0 || len(review.MissingFields) != 0 {
		t.Fatalf("unexpected reasons/missing: %+v %+v", review.Reasons, review.MissingFields)
	}
}

func TestReviewCandidateScoresGenerationFit(t *testing.T) {
	c := completeCandidate()
	c.Generation = "gen/next"
	c.Title = "generation(next): add branchless feature gating"
	c.Labels = []string{"generation", "gen/next"}
	c.WhyNow = "Next gen near-term foundation work needs a gate, handoff, and operator visibility before it is agent-runnable."
	c.InScope = "Add the generation checklist with promotion evidence, demotion evidence, and runtime feature gate boundaries."
	c.OutOfScope = "Do not create a branch per generation; priority, shared trunk, and runtime feature gates remain orthogonal."
	c.DoneCondition = "The issue names promotion evidence, demotion/retirement evidence, and an invalidating assumption."
	c.Witness = "Captured command witness from fak issue contract."
	c.Assumptions = []string{"Invalidating assumption: generation labels stay available during issue grooming."}
	review := ReviewCandidate(c, Options{})
	if !review.OK {
		t.Fatalf("review = %+v, want scoped generation issue to remain dispatchable", review)
	}
	if review.GenerationFit.Stream != "gen/next" || review.GenerationFit.Total != 100 || len(review.GenerationFit.Flags) != 0 {
		t.Fatalf("generation fit = %+v, want clean gen/next score", review.GenerationFit)
	}
}

func TestReviewCandidateFlagsGenerationMismatch(t *testing.T) {
	c := completeCandidate()
	c.Generation = "gen/next"
	c.Title = "generation(next): add branchless feature gating"
	c.Labels = []string{"generation", "gen/future"}
	c.WhyNow = "Next gen near-term foundation work needs a gate, handoff, and operator visibility before it is agent-runnable."
	c.InScope = "Add the generation checklist with promotion evidence, demotion evidence, and runtime feature gate boundaries."
	c.OutOfScope = "Priority, shared trunk, and runtime feature gates remain orthogonal."
	c.DoneCondition = "The issue names promotion evidence, demotion/retirement evidence, and an invalidating assumption."
	c.Witness = "Captured command witness from fak issue contract."
	c.Assumptions = []string{"Invalidating assumption: generation labels stay available during issue grooming."}
	review := ReviewCandidate(c, Options{})
	if !review.OK {
		t.Fatalf("generation mismatch is advisory, got refused review: %+v", review)
	}
	if review.GenerationFit.Stream != "gen/future" || !has(review.GenerationFit.Flags, "generation_body_mismatch") {
		t.Fatalf("generation fit = %+v, want body mismatch on gen/future label", review.GenerationFit)
	}
	if review.GenerationFit.Total >= 100 {
		t.Fatalf("generation fit = %+v, want mismatch below full score", review.GenerationFit)
	}
}

func TestReviewIssueDraftRefusesUnexpandedTemplateTokens(t *testing.T) {
	body := strings.Join([]string{
		"## Generation stream",
		"- Generation: $(@{gen=next; title=x; labels=dispatch}.gen)",
		"- Milestone: $(System.Collections.Hashtable.title)",
		"",
		"## Current state",
		"Scoped body content exists.",
		"",
		"## Why this is next",
		"The issue should be repaired before dispatch.",
		"",
		"## Working spine",
		"Repair generated issue metadata before worker launch.",
		"",
		"## In scope",
		"Reject raw PowerShell template tokens.",
		"",
		"## Out of scope",
		"Do not launch a worker from this corrupt body.",
		"",
		"## Done condition",
		"The contract refuses the issue row.",
		"",
		"## Witness",
		"go test ./internal/issuecontract",
		"",
		"## Lane",
		"docs",
		"",
		"## Path hints",
		"- `docs/**`",
	}, "\n")
	review := ReviewIssueDraft(IssueDraft{
		Number: 1727,
		Title:  "generation(next): broken filer output",
		Body:   body,
	}, Options{})
	if review.OK || review.Dispatchability != Refused || review.Verdict != "refused" {
		t.Fatalf("review = %+v, want refused corrupt issue body", review)
	}
	if !has(review.Reasons, ReasonUnexpandedTemplate) {
		t.Fatalf("reasons = %+v, want %s", review.Reasons, ReasonUnexpandedTemplate)
	}
}

func TestReviewCandidateScoresGoldPlatingBelowSpineWork(t *testing.T) {
	c := completeCandidate()
	c.PriorityContext = "Nice later: polish helper names after the workflow already works."
	c.CurrentState = "The user-facing workflow already works and has a passing witness."
	c.WhyNow = "This would be cleanup someday."
	c.WorkingSpine = "No working path changes."
	c.OutOfScope = "No adjacent work is addressed."
	review := ReviewCandidate(c, Options{})
	if !review.OK {
		t.Fatalf("gold-plating candidate should still be dispatchable when scoped: %+v", review)
	}
	if review.SpinePriority.Total >= 50 {
		t.Fatalf("spine priority = %+v, want below 50 for scoped polish", review.SpinePriority)
	}
}

func TestReviewCandidateFlagsNonLeafWorkUnits(t *testing.T) {
	c := completeCandidate()
	c.WorkUnit = "epic"
	review := ReviewCandidate(c, Options{})
	if review.OK || review.Dispatchability != TriageOnly || review.Verdict != "needs_scope" {
		t.Fatalf("non-leaf review = %+v, want triage-only needs-scope", review)
	}
	if !has(review.Reasons, ReasonNotDispatchLeaf) {
		t.Fatalf("non-leaf reason missing: %+v", review.Reasons)
	}

	c = completeCandidate()
	c.WorkUnit = ""
	c.Labels = []string{"triage-only"}
	review = ReviewCandidate(c, Options{})
	if review.OK || !has(review.Reasons, ReasonNotDispatchLeaf) {
		t.Fatalf("triage-only label review = %+v, want non-dispatch leaf reason", review)
	}
}

func TestReviewCandidateFlagsOversizedExpectedSteps(t *testing.T) {
	c := completeCandidate()
	c.ExpectedSteps = MaxDispatchExpectedSteps + 1
	review := ReviewCandidate(c, Options{})
	if review.OK || review.Dispatchability != TriageOnly || review.Verdict != "needs_scope" {
		t.Fatalf("oversized review = %+v, want triage-only needs-scope", review)
	}
	if !has(review.Reasons, ReasonOversizedSteps) {
		t.Fatalf("oversized reason missing: %+v", review.Reasons)
	}
}

func TestReviewCandidateAllowsExistingColonKeyShape(t *testing.T) {
	c := completeCandidate()
	c.Key = "guard-rsi-route/guard-journal:blank_reason_on_deny"
	review := ReviewCandidate(c, Options{})
	if !review.OK {
		t.Fatalf("review = %+v, want colon-bearing stable marker key accepted", review)
	}
}

func TestReviewCandidateNeedsScopeForMissingSpineFields(t *testing.T) {
	c := completeCandidate()
	c.OutOfScope = ""
	c.DoneCondition = ""
	c.WorkingSpine = ""
	review := ReviewCandidate(c, Options{})
	if review.OK || review.Dispatchability != TriageOnly || review.Verdict != "needs_scope" {
		t.Fatalf("review = %+v, want needs_scope triage_only", review)
	}
	if !has(review.Reasons, ReasonScopeIncomplete) {
		t.Fatalf("scope reason missing: %+v", review.Reasons)
	}
	for _, want := range []string{"working_spine", "out_of_scope", "done_condition"} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent: %+v", want, review.MissingFields)
		}
	}
	if review.Score.Total >= 100 {
		t.Fatalf("partial candidate got full score: %+v", review.Score)
	}
}

func TestReviewCandidateNeedsRoute(t *testing.T) {
	c := completeCandidate()
	c.Lane = ""
	c.Paths = nil
	review := ReviewCandidate(c, Options{})
	if review.OK || !has(review.Reasons, ReasonUnrouted) {
		t.Fatalf("review = %+v, want unrouted refusal reason", review)
	}
	if review.Dispatchability != TriageOnly {
		t.Fatalf("dispatchability = %q, want triage_only", review.Dispatchability)
	}
}

func TestReviewCandidateRefusesPrivateBoundary(t *testing.T) {
	c := completeCandidate()
	c.BoundaryNotes = []string{"Requires fak-private Slack control transcript."}
	review := ReviewCandidate(c, Options{})
	if review.OK || review.Dispatchability != Refused || review.Verdict != "refused" {
		t.Fatalf("review = %+v, want refused", review)
	}
	if !has(review.Reasons, ReasonPrivateBoundary) {
		t.Fatalf("private-boundary reason missing: %+v", review.Reasons)
	}
}

func TestReviewCandidateLiveRequiresDedupeArmor(t *testing.T) {
	review := ReviewCandidate(completeCandidate(), Options{Live: true})
	if review.OK || review.Dispatchability != Refused {
		t.Fatalf("unarmored live review = %+v, want refused", review)
	}
	if !has(review.Reasons, ReasonLiveUnarmored) {
		t.Fatalf("live-unarmored reason missing: %+v", review.Reasons)
	}

	armed := ReviewCandidate(completeCandidate(), Options{Live: true, DedupeChecked: true, DedupeCap: 300})
	if !armed.OK {
		t.Fatalf("armed live review refused: %+v", armed)
	}
}

func TestReviewCandidateLiveRequiresNoiseControl(t *testing.T) {
	c := completeCandidate()
	c.Trigger = ""
	c.BatchPolicy = ""
	review := ReviewCandidate(c, Options{Live: true, DedupeChecked: true, DedupeCap: 300})
	if review.OK || review.Dispatchability != Refused || review.Verdict != "refused" {
		t.Fatalf("noise-uncontrolled live review = %+v, want refused", review)
	}
	if !has(review.Reasons, ReasonNoiseIncomplete) {
		t.Fatalf("noise-control reason missing: %+v", review.Reasons)
	}
	for _, want := range []string{"trigger", "batch_policy"} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent: %+v", want, review.MissingFields)
		}
	}

	c.Trigger = "A scored feeder crosses the issue threshold."
	c.BatchPolicy = "Handle repeated signals carefully."
	review = ReviewCandidate(c, Options{Live: true, DedupeChecked: true, DedupeCap: 300})
	if review.OK || !has(review.Reasons, ReasonNoiseIncomplete) {
		t.Fatalf("vague batch policy review = %+v, want noise-control refusal", review)
	}
	if !has(review.MissingFields, "batch_policy") {
		t.Fatalf("vague batch policy did not name batch_policy missing: %+v", review.MissingFields)
	}

	c.BatchPolicy = "One issue per marker key; reruns update existing issues."
	review = ReviewCandidate(c, Options{Live: true, DedupeChecked: true, DedupeCap: 300})
	if !review.OK {
		t.Fatalf("noise-controlled live review refused: %+v", review)
	}
}

func TestReviewCandidateLiveRequiresAgentContext(t *testing.T) {
	c := completeCandidate()
	c.WorkUnit = ""
	c.ExpectedSteps = 0
	c.Assumptions = nil
	c.ConfusionRisks = nil
	c.Coordination = nil
	review := ReviewCandidate(c, Options{Live: true, DedupeChecked: true, DedupeCap: 300})
	if review.OK || review.Dispatchability != Refused || review.Verdict != "refused" {
		t.Fatalf("agent-context-incomplete live review = %+v, want refused", review)
	}
	if !has(review.Reasons, ReasonAgentIncomplete) {
		t.Fatalf("agent-context reason missing: %+v", review.Reasons)
	}
	if has(review.Reasons, ReasonNoiseIncomplete) {
		t.Fatalf("noise-control reason should not fire when trigger/batch are present: %+v", review.Reasons)
	}
	for _, want := range []string{"work_unit", "expected_steps", "assumptions", "confusion_risks", "coordination"} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent: %+v", want, review.MissingFields)
		}
	}
}

func TestReviewIssueDraftParsesStandardSections(t *testing.T) {
	review := ReviewIssueDraft(IssueDraft{
		Number: 1440,
		Title:  "guardrsi: require a reason on every block",
		Labels: []IssueLabel{{Name: "guardrsi"}},
		Body: strings.Join([]string{
			"### Parent context",
			"guard-verdict-rsi route",
			"### Current state",
			"The guard journal can surface an unexplained block bucket.",
			"### Why this is next",
			"This is a load-bearing honesty hole before threshold tuning.",
			"### Working spine",
			"Every denied guard verdict carries one closed-vocabulary reason.",
			"### Priority context",
			"Working path: guard preflight -> reasoned denial -> worker repair.",
			"Current blocker: blank reasons hide the failing gate.",
			"Unblocks: guard-rsi tuning depends on reason buckets.",
			"Not polish: this fixes the smallest witnessed guard hole before threshold optimization.",
			"### Work unit",
			"leaf",
			"### Expected steps",
			"3",
			"### Assumptions",
			"- The guard journal fixture can reproduce the blank reason.",
			"### Confusion risks",
			"- Reason labels and threshold tuning are adjacent but separate.",
			"### Coordination notes",
			"- Avoid concurrent edits to the guard reason taxonomy.",
			"### Trigger",
			"Guard journal emits a denied verdict with no reason.",
			"### Batch policy",
			"One issue per repeated reason class; update existing marker on rerun.",
			"### In scope",
			"Add the missing reason mapping and one regression fixture.",
			"### Out of scope",
			"Do not retune model thresholds or rewrite unrelated guard code.",
			"### Done condition",
			"The regression fixture no longer reports a blank reason.",
			"### Witness",
			"go test ./internal/guardrsi ./internal/guardroute",
			"### Acceptance gate",
			"go test ./internal/guardrsi ./internal/guardroute",
			"### Lane",
			"guardrsi",
			"### Path hints",
			"- `internal/guardrsi/**`",
			"### Boundary notes",
			"- Public guard-journal defect only.",
			"### Closure binding",
			"Resolving commit cites #N and carries `(fak guardrsi)`.",
		}, "\n"),
	}, Options{})
	if !review.OK || review.Dispatchability != Dispatchable || review.Score.Total != 100 {
		t.Fatalf("review = %+v, want dispatchable full-score issue draft", review)
	}
	if review.SpinePriority.Total != 100 {
		t.Fatalf("spine priority = %+v, want full-score issue draft", review.SpinePriority)
	}
	if review.AgentContext.Total != 100 {
		t.Fatalf("agent context = %+v, want full-score issue draft", review.AgentContext)
	}
	if review.Key != "issue/1440" || review.Lane != "guardrsi" {
		t.Fatalf("identity = key %q lane %q", review.Key, review.Lane)
	}
	if len(review.Paths) != 1 || review.Paths[0] != "internal/guardrsi/**" {
		t.Fatalf("paths = %+v", review.Paths)
	}
}

func TestReviewIssueDraftHoldsMissingDoneConditionOrWitness(t *testing.T) {
	cases := []struct {
		name               string
		done               string
		witness            string
		includeLikelyFiles bool
		wantOK             bool
		wantMissing        string
		wantMissingSection string
		wantDispatch       string
	}{
		{
			name:               "complete issue passes",
			done:               "The lint reports no missing proof sections.",
			witness:            "go test ./internal/issuecontract",
			includeLikelyFiles: true,
			wantOK:             true,
			wantDispatch:       Dispatchable,
		},
		{
			name:               "missing done condition is held",
			witness:            "go test ./internal/issuecontract",
			includeLikelyFiles: true,
			wantMissing:        "done_condition",
			wantMissingSection: "done_condition",
			wantDispatch:       TriageOnly,
		},
		{
			name:               "missing witness is held",
			done:               "The lint reports missing witness sections.",
			includeLikelyFiles: true,
			wantMissing:        "witness",
			wantMissingSection: "witness",
			wantDispatch:       TriageOnly,
		},
		{
			name:               "missing likely files is held",
			done:               "The lint reports missing likely file sections.",
			witness:            "go test ./internal/issuecontract",
			wantMissing:        "likely_files",
			wantMissingSection: "likely_files",
			wantDispatch:       TriageOnly,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			review := ReviewIssueDraft(IssueDraft{
				Number: 1815,
				Title:  "dispatch: require issue proof sections",
				Body:   issueProofSectionBodyWithLikelyFiles(tc.done, tc.witness, tc.includeLikelyFiles),
			}, Options{})
			if review.OK != tc.wantOK || review.Dispatchability != tc.wantDispatch {
				t.Fatalf("review = %+v, want ok=%v dispatchability=%s", review, tc.wantOK, tc.wantDispatch)
			}
			if tc.wantOK {
				if len(review.Reasons) != 0 || len(review.MissingFields) != 0 || len(review.MissingSections) != 0 || review.Score.Total != 100 {
					t.Fatalf("complete issue review = %+v, want no findings and full score", review)
				}
				return
			}
			if !has(review.Reasons, ReasonScopeIncomplete) {
				t.Fatalf("reasons = %+v, want %s", review.Reasons, ReasonScopeIncomplete)
			}
			if !has(review.MissingFields, tc.wantMissing) {
				t.Fatalf("missing fields = %+v, want %s", review.MissingFields, tc.wantMissing)
			}
			if !has(review.MissingSections, tc.wantMissingSection) {
				t.Fatalf("missing sections = %+v, want %s", review.MissingSections, tc.wantMissingSection)
			}
		})
	}
}

func TestReviewIssueDraftUsesGeneratedMarkerKey(t *testing.T) {
	body := "<!-- fak-task-handoff-key: task_push_next/issue-sync -->\n" + strings.Join([]string{
		"### Parent context",
		"task_push_next",
		"### Current state",
		"The handoff created a scoped follow-up.",
		"### Why this is next",
		"The follow-up is the next dispatchable leaf.",
		"### Working spine",
		"verified handoff -> issue -> worker",
		"### Work unit",
		"leaf",
		"### Expected steps",
		"3",
		"### Assumptions",
		"- Existing marker dedupe is available.",
		"### Confusion risks",
		"- Do not treat this as the parent epic.",
		"### Coordination notes",
		"- Serialize with taskmgr issue-body edits.",
		"### Trigger",
		"Verified task handoff proposed this next step.",
		"### Batch policy",
		"At most two follow-up issues per handoff; reruns update by marker.",
		"### In scope",
		"Update this one follow-up.",
		"### Out of scope",
		"Do not change the parent task.",
		"### Done condition",
		"The issue review uses the stable marker key.",
		"### Witness",
		"go test ./internal/issuecontract",
		"### Acceptance gate",
		"go test ./internal/issuecontract",
		"### Lane",
		"taskmgr",
		"### Closure binding",
		"Resolving commit cites #N.",
	}, "\n")
	review := ReviewIssueDraft(IssueDraft{Number: 1444, Title: "taskmgr: follow up", Body: body}, Options{})
	if review.Key != "task_push_next/issue-sync" || review.IssueNumber != 1444 {
		t.Fatalf("identity = key %q issue_number %d, want marker key and issue number", review.Key, review.IssueNumber)
	}
}

func TestReviewIssueDraftFlagsOversizedExpectedSteps(t *testing.T) {
	review := ReviewIssueDraft(IssueDraft{
		Number: 42,
		Title:  "taskmgr: too large",
		Body: strings.Join([]string{
			"### Parent context",
			"task handoff",
			"### Current state",
			"Generated issues can already carry worker metadata.",
			"### Why this is next",
			"The producer would otherwise sync a bundled task.",
			"### Working spine",
			"Keep one issue to a dispatchable worker leaf.",
			"### Work unit",
			"leaf",
			"### Expected steps",
			"12",
			"### In scope",
			"Split the oversized leaf.",
			"### Out of scope",
			"Do not implement every child issue.",
			"### Done condition",
			"The candidate is refused before live sync.",
			"### Witness",
			"go test ./internal/issuecontract",
			"### Acceptance gate",
			"go test ./internal/issuecontract",
			"### Lane",
			"issuecontract",
			"### Path hints",
			"- `internal/issuecontract/contract.go`",
			"### Boundary notes",
			"Public issue only.",
			"### Closure binding",
			"Resolving commit cites #N and carries a matching trailer.",
		}, "\n"),
	}, Options{})
	if review.OK || review.Dispatchability != TriageOnly || !has(review.Reasons, ReasonOversizedSteps) {
		t.Fatalf("issue draft review = %+v, want oversized expected steps", review)
	}
}

func TestCandidateFromIssueDraftParsesAgentContext(t *testing.T) {
	body := strings.Join([]string{
		"### Parent context",
		"issue-catalog",
		"### Current state",
		"The feeder has a stable source row.",
		"### Why this is next",
		"The generated issue needs agent context before dispatch.",
		"### Working spine",
		"source row -> scoped issue -> worker prompt",
		"### Work unit",
		"step",
		"### Expected steps",
		"Expected: 4 steps.",
		"### Assumptions",
		"- Existing marker dedupe is available.",
		"### Confusion risks",
		"- Do not split this into an epic.",
		"### Coordination",
		"- Serialize with issuecontract body parser edits.",
		"### Trigger",
		"Catalog row crosses the default-on threshold.",
		"### Noise control",
		"Batch at most 20 creates per live wave.",
		"### In scope",
		"Render the fields.",
		"### Out of scope",
		"Do not sync live.",
		"### Done condition",
		"The parser returns the agent fields.",
		"### Witness",
		"go test ./internal/issuecontract",
		"### Acceptance gate",
		"go test ./internal/issuecontract",
		"### Lane",
		"issuecontract",
		"### Closure binding",
		"Commit cites #N.",
	}, "\n")
	c := CandidateFromIssueDraft(IssueDraft{Number: 1443, Title: "issuecontract: parse agent context", Body: body})
	if c.WorkUnit != "step" || c.ExpectedSteps != 4 || c.BatchPolicy != "Batch at most 20 creates per live wave." {
		t.Fatalf("candidate agent context = %+v", c)
	}
	if len(c.Assumptions) != 1 || len(c.ConfusionRisks) != 1 || len(c.Coordination) != 1 {
		t.Fatalf("candidate lists = assumptions=%v confusion=%v coordination=%v", c.Assumptions, c.ConfusionRisks, c.Coordination)
	}
}

func TestReviewIssueDraftParsesDependencyMarkers(t *testing.T) {
	body := issueProofSectionBody(
		"Dependency markers are parsed for dispatch holds.",
		"go test ./internal/issuecontract",
	) + "\n" + strings.Join([]string{
		"### Dependencies",
		"- after: #1756 must be witnessed before this issue runs.",
		"- related-only: #1706 is context, not a dispatch hold.",
		"- blocks: #1772 waits on this issue's witnessed result.",
	}, "\n")
	review := ReviewIssueDraft(IssueDraft{
		Number: 1755,
		Title:  "issuecontract: parse dependency markers",
		Body:   body,
	}, Options{})
	if !review.OK {
		t.Fatalf("review = %+v, want dependency markers to preserve dispatchability", review)
	}
	if len(review.Dependencies) != 3 {
		t.Fatalf("dependencies = %+v, want after, related, blocks", review.Dependencies)
	}
	if got := review.Dependencies[0]; got.Relation != "after" || got.Issue != 1756 || !got.Blocking {
		t.Fatalf("first dependency = %+v, want blocking after #1756", got)
	}
	if got := review.Dependencies[1]; got.Relation != "related" || got.Issue != 1706 || got.Blocking {
		t.Fatalf("second dependency = %+v, want non-blocking related #1706", got)
	}
	if got := review.Dependencies[2]; got.Relation != "blocks" || got.Issue != 1772 || !got.Blocking {
		t.Fatalf("third dependency = %+v, want blocking blocks #1772", got)
	}
}

func TestReviewIssueDraftLiveRejectsPlaceholderAgentContext(t *testing.T) {
	body := strings.Join([]string{
		"### Parent context",
		"issue-catalog",
		"### Current state",
		"The feeder can create an issue row.",
		"### Why this is next",
		"Live sync would otherwise create an ambiguous worker task.",
		"### Working spine",
		"live source -> scoped issue -> worker prompt",
		"### Work unit",
		"leaf",
		"### Expected steps",
		"3",
		"### Assumptions",
		"None named.",
		"### Confusion risks",
		"None named.",
		"### Coordination notes",
		"No special coordination beyond the lane lease.",
		"### Trigger",
		"A live feeder crossed the issue threshold.",
		"### Batch policy",
		"One issue per marker key; reruns update in place.",
		"### In scope",
		"Reject placeholder context.",
		"### Out of scope",
		"Do not sync live.",
		"### Done condition",
		"The review names missing agent context.",
		"### Witness",
		"go test ./internal/issuecontract",
		"### Acceptance gate",
		"go test ./internal/issuecontract",
		"### Lane",
		"issuecontract",
		"### Closure binding",
		"Commit cites #N.",
	}, "\n")
	review := ReviewIssueDraft(IssueDraft{Number: 1444, Title: "issuecontract: reject placeholders", Body: body}, Options{Live: true, DedupeChecked: true, DedupeCap: 300})
	if review.OK || !has(review.Reasons, ReasonAgentIncomplete) {
		t.Fatalf("live issue draft review = %+v, want agent-context refusal", review)
	}
	for _, want := range []string{"assumptions", "confusion_risks", "coordination"} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent: %+v", want, review.MissingFields)
		}
	}
}

func TestReviewIssueDraftFlagsVagueManualIssue(t *testing.T) {
	review := ReviewIssueDraft(IssueDraft{
		Number: 1441,
		Title:  "make it better",
		Body: strings.Join([]string{
			"### Current state",
			"The feature exists.",
			"### In scope",
			"Improve things.",
		}, "\n"),
	}, Options{})
	if review.OK || review.Dispatchability != TriageOnly {
		t.Fatalf("review = %+v, want triage-only incomplete issue", review)
	}
	for _, want := range []string{"parent_ref", "why_now", "working_spine", "out_of_scope", "done_condition", "witness", "acceptance_gate", "closure_binding"} {
		if !has(review.MissingFields, want) {
			t.Fatalf("missing field %q absent from %+v", want, review.MissingFields)
		}
	}
	if !has(review.Reasons, ReasonUnrouted) {
		t.Fatalf("unrouted reason absent: %+v", review.Reasons)
	}
}

func TestReviewIssueDraftParsesCombinedDoneWitnessSection(t *testing.T) {
	body := strings.Join([]string{
		"### Parent context",
		"task handoff",
		"### Current state",
		"A verified handoff created this follow-up.",
		"### Why this is next",
		"It unblocks the next dispatch cycle before polish.",
		"### Working spine",
		"One handoff issue carries scope and proof.",
		"### Priority context",
		"Working path: handoff -> issue -> worker. Current blocker: old body shape. Unblocks: prompt briefing. Not polish: parser compatibility.",
		"### In scope",
		"Parse the existing combined section.",
		"### Out of scope",
		"Do not redesign handoff sync.",
		"### Done condition / witness",
		"Done condition: The parser extracts a done condition.",
		"Witness: `go test ./internal/issuecontract`",
		"### Acceptance gate",
		"go test ./internal/issuecontract",
		"### Lane",
		"issuecontract",
		"### Path hints",
		"- `internal/issuecontract/contract.go`",
		"### Closure binding",
		"Resolving commit cites #N.",
	}, "\n")
	review := ReviewIssueDraft(IssueDraft{Number: 1442, Title: "taskmgr: parse combined proof", Body: body}, Options{})
	if !review.OK || review.Score.Witness != 25 {
		t.Fatalf("review = %+v, want combined done/witness parsed as ready", review)
	}
}

func issueProofSectionBody(done, witness string) string {
	return issueProofSectionBodyWithLikelyFiles(done, witness, true)
}

func issueProofSectionBodyWithLikelyFiles(done, witness string, includeLikelyFiles bool) string {
	parts := []string{
		"### Parent context",
		"fleet-400iph issue contract",
		"### Current state",
		"Generated issues can enter the dispatch queue.",
		"### Why this is next",
		"The queue needs proof sections before high-throughput workers pick items.",
		"### Working spine",
		"issue draft -> proof-section lint -> safe dispatch candidate",
		"### Priority context",
		"Working path: issue draft -> linter -> dispatch. Current blocker: missing proof sections. Unblocks: high-throughput worker launch. Not polish: this is the smallest proof gate.",
		"### In scope",
		"Lint the issue body for proof sections.",
		"### Out of scope",
		"Do not launch workers or mutate live issues.",
	}
	if strings.TrimSpace(done) != "" {
		parts = append(parts, "### Done condition", done)
	}
	if strings.TrimSpace(witness) != "" {
		parts = append(parts, "### Witness", witness)
	}
	parts = append(parts,
		"### Acceptance gate",
		"go test ./internal/issuecontract",
		"### Lane",
		"issuecontract",
	)
	if includeLikelyFiles {
		parts = append(parts,
			"### Likely files",
			"- `internal/issuecontract/contract.go`",
		)
	}
	parts = append(parts,
		"### Closure binding",
		"Resolving commit cites #1815.",
	)
	return strings.Join(parts, "\n")
}

func has(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}
