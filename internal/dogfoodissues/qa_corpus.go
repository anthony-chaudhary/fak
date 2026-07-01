package dogfoodissues

// QADogfoodCorpusRow is one synthetic "found late" defect that should now map
// to an origin control instead of becoming another after-the-fact cleanup item.
type QADogfoodCorpusRow struct {
	Key            string
	LateFailure    string
	OriginControl  string
	WitnessCommand string
	Item           ActionItem
}

// QADogfoodCorpus returns the standing fixture set for at-origin QA dogfooding.
// Each row is deliberately dispatchable: it names a root control, a focused
// witness, and enough issue-contract context for an agent to fix the origin
// rather than only the symptom.
func QADogfoodCorpus() []QADogfoodCorpusRow {
	rows := []QADogfoodCorpusRow{
		{
			Key:            "qa-dogfood-corpus/task-output-shape",
			LateFailure:    "A task was live and heartbeating while emitting degenerate repeated output.",
			OriginControl:  "taskmgr.BeatTaskWithEvidence + ShapeWitness",
			WitnessCommand: "go test ./internal/taskmgr -run Beat.*ShapeWitness",
			Item: qaCorpusItem(ActionItem{
				Key:          "qa-dogfood-corpus/task-output-shape",
				Title:        "qa dogfood: catch degenerate task output at beat origin",
				SourceProbe:  "qa-dogfood-corpus",
				ScoreName:    "origin_control",
				Score:        "missing",
				Grade:        "ACTION",
				DebtName:     "late_found",
				DebtCount:    1,
				EvidencePath: "internal/taskmgr/shapewitness_test.go",
				NextAction:   "Move output-shape grading to BeatTaskWithEvidence/BeatStepWithEvidence at the producer beat.",
				Finding:      "task_output_shape",
				Lane:         "taskmgr",
				Paths:        []string{"internal/taskmgr/**"},
			}, "A task was live and heartbeating while emitting degenerate repeated output.", "taskmgr.BeatTaskWithEvidence + ShapeWitness", "go test ./internal/taskmgr -run Beat.*ShapeWitness"),
		},
		{
			Key:            "qa-dogfood-corpus/task-origin-path-evidence",
			LateFailure:    "A task handoff claimed an artifact existed, but the missing path was found only during review.",
			OriginControl:  "taskmgr.WithDefaultOriginWitnesses + PathWitness",
			WitnessCommand: "go test ./internal/taskmgr -run OriginWitness.*Registry",
			Item: qaCorpusItem(ActionItem{
				Key:          "qa-dogfood-corpus/task-origin-path-evidence",
				Title:        "qa dogfood: read artifact paths at task origin",
				SourceProbe:  "qa-dogfood-corpus",
				ScoreName:    "origin_control",
				Score:        "missing",
				Grade:        "ACTION",
				DebtName:     "late_found",
				DebtCount:    1,
				EvidencePath: "internal/taskmgr/evidence_test.go",
				NextAction:   "Attach path EvidenceRefs to TaskSpec/StepSpec and run the default origin witness registry immediately.",
				Finding:      "missing_artifact_path",
				Lane:         "taskmgr",
				Paths:        []string{"internal/taskmgr/**"},
			}, "A task handoff claimed an artifact existed, but the missing path was found only during review.", "taskmgr.WithDefaultOriginWitnesses + PathWitness", "go test ./internal/taskmgr -run OriginWitness.*Registry"),
		},
		{
			Key:            "qa-dogfood-corpus/handoff-evidence-refs",
			LateFailure:    "A follow-up issue had to be edited by hand because changed paths and targeted tests were absent from the handoff.",
			OriginControl:  "taskmgr.DraftHandoffFromTask + DeriveHandoffEvidenceRefs",
			WitnessCommand: "go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence",
			Item: qaCorpusItem(ActionItem{
				Key:          "qa-dogfood-corpus/handoff-evidence-refs",
				Title:        "qa dogfood: derive handoff path and test evidence at origin",
				SourceProbe:  "qa-dogfood-corpus",
				ScoreName:    "origin_control",
				Score:        "missing",
				Grade:        "ACTION",
				DebtName:     "late_found",
				DebtCount:    1,
				EvidencePath: "internal/taskmgr/handoff_test.go",
				NextAction:   "Derive changed-path and targeted-test evidence before the handoff reaches issue sync.",
				Finding:      "handoff_missing_evidence_refs",
				Lane:         "taskmgr",
				Paths:        []string{"internal/taskmgr/**", "cmd/fak/task_handoff_test.go"},
			}, "A follow-up issue had to be edited by hand because changed paths and targeted tests were absent from the handoff.", "taskmgr.DraftHandoffFromTask + DeriveHandoffEvidenceRefs", "go test ./internal/taskmgr ./cmd/fak -run Handoff.*Evidence"),
		},
		{
			Key:            "qa-dogfood-corpus/dogfood-issue-scope",
			LateFailure:    "A scorecard ACTION row produced a vague GitHub issue that could not be dispatched safely.",
			OriginControl:  "dogfoodissues.BuildPlanWithOptions + issuecontract.ReviewCandidate",
			WitnessCommand: "go test ./internal/dogfoodissues -run ReviewedPlan",
			Item: qaCorpusItem(ActionItem{
				Key:          "qa-dogfood-corpus/dogfood-issue-scope",
				Title:        "qa dogfood: reject vague dogfood issues before creation",
				SourceProbe:  "qa-dogfood-corpus",
				ScoreName:    "origin_control",
				Score:        "missing",
				Grade:        "ACTION",
				DebtName:     "late_found",
				DebtCount:    1,
				EvidencePath: "internal/dogfoodissues/dogfoodissues_test.go",
				NextAction:   "Review every machine-created dogfood issue against the issue contract before live gh create.",
				Finding:      "vague_dogfood_issue",
				Lane:         "dogfoodissues",
				Paths:        []string{"internal/dogfoodissues/**", "cmd/fak/dogfoodissues.go"},
			}, "A scorecard ACTION row produced a vague GitHub issue that could not be dispatched safely.", "dogfoodissues.BuildPlanWithOptions + issuecontract.ReviewCandidate", "go test ./internal/dogfoodissues -run ReviewedPlan"),
		},
	}
	return rows
}

func qaCorpusItem(item ActionItem, lateFailure, originControl, witness string) ActionItem {
	item.ParentRef = "qa-dogfood-origin-corpus"
	item.CurrentState = "Late-found failure: " + lateFailure
	item.WhyNow = "This corpus row names a QA failure that should be caught by " + originControl + " before after-the-fact review."
	item.WorkingSpine = "Move the failure to the at-origin control: " + originControl + "."
	item.WorkUnit = "leaf"
	item.ExpectedSteps = 3
	item.Assumptions = []string{"The late-found failure can be represented as a deterministic fixture."}
	item.ConfusionRisks = []string{"Do not fix only the generated issue text; keep the origin control named in the row."}
	item.Coordination = []string{"Use the stable corpus key so reruns update the same issue."}
	item.Trigger = "QA dogfood corpus row mapped a late-found failure to an origin control."
	item.BatchPolicy = "One issue per corpus key; reruns update the existing marker."
	item.InScope = "Add or repair the named at-origin control and prove it with " + witness + "."
	item.OutOfScope = "Do not broaden the corpus into unrelated scorecard rewrites."
	item.DoneCondition = "The row's late-found failure is refused or surfaced by " + originControl + " before the operator edits issue text."
	item.Witness = witness
	item.AcceptanceGate = witness
	item.Labels = []string{"qa-dogfood", "origin-control"}
	item.BoundaryNotes = []string{"Synthetic public fixture only; no restricted transcript or local-only evidence."}
	item.ClosureBinding = "Resolving commit cites the issue and explains which origin control moved earlier."
	return item
}
