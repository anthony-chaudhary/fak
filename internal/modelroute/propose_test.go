package modelroute

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// baseManifestForPropose creates a test manifest with two rules and a default plan.
func baseManifestForPropose() Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{Members: []Member{{Model: "default-model", Role: "primary"}}},
		Rules: []Rule{
			{
				Name:  "tool-writes",
				Match: Match{Aspect: AspectToolCall, Tool: "write_file"},
				Plan:  Plan{Members: []Member{{Model: "small", Role: "primary"}}},
			},
			{
				Name:  "hard-reasoning",
				Match: Match{MinComplexity: ComplexityHigh},
				Plan:  Plan{Members: []Member{{Model: "large", Role: "primary"}}},
			},
		},
	}
}

// TestProposeEvaluatesOnHeldOutSplitAndImproves witnesses the core Issue #600 deliverable:
// 1. Candidate rule changes are evaluated on a held-out split.
// 2. The proposed manifest achieves a strictly better measured objective on the held-out split.
// 3. The reviewable manifest diff is generated and valid JSON.
// 4. No policy is silently auto-applied (keep=false default, Apply fails closed).
func TestProposeEvaluatesOnHeldOutSplitAndImproves(t *testing.T) {
	base := baseManifestForPropose()

	var records []OutcomeRecord

	// Generate 10 records for "tool-writes" on "small": moderate quality (0.65), low cost ($0.01)
	for i := 0; i < 10; i++ {
		records = append(records, OutcomeRecord{
			Key:   AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model: "small",
			Subject: Subject{
				Aspect: AspectToolCall,
				Tool:   "write_file",
			},
			Outcome: Outcome{
				Cost:    0.01,
				Latency: 100 * time.Millisecond,
				Quality: 0.65,
			},
		})
	}

	// Generate 10 records for "tool-writes" on "large": high quality (0.95), slightly higher cost ($0.05)
	for i := 0; i < 10; i++ {
		records = append(records, OutcomeRecord{
			Key:   AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model: "large",
			Subject: Subject{
				Aspect: AspectToolCall,
				Tool:   "write_file",
			},
			Outcome: Outcome{
				Cost:    0.05,
				Latency: 250 * time.Millisecond,
				Quality: 0.95,
			},
		})
	}

	// Generate 10 records for "hard-reasoning" on "large": high quality (0.90), cost ($0.10)
	for i := 0; i < 10; i++ {
		records = append(records, OutcomeRecord{
			Key:   AspectRuleKey{Aspect: AspectStep, Rule: "hard-reasoning"},
			Model: "large",
			Subject: Subject{
				Aspect:     AspectStep,
				Complexity: ComplexityHigh,
			},
			Outcome: Outcome{
				Cost:    0.10,
				Latency: 500 * time.Millisecond,
				Quality: 0.90,
			},
		})
	}

	opts := ProposeOptions{
		BaseManifest: base,
		Records:      records,
		TrainRatio:   0.70,
		Objective: Objective{
			QualityWeight: 1.0,
			CostWeight:    0.0, // pure quality maximization
		},
	}

	proposal, err := Propose(opts)
	if err != nil {
		t.Fatalf("Propose unexpected error: %v", err)
	}
	if proposal == nil {
		t.Fatal("expected proposal to be produced, got nil")
	}

	// 1. Verify candidate rule change was proposed for "tool-writes"
	if len(proposal.RuleProposals) != 1 {
		t.Fatalf("expected 1 rule proposal, got %d", len(proposal.RuleProposals))
	}
	rp := proposal.RuleProposals[0]
	if rp.RuleName != "tool-writes" {
		t.Errorf("proposed rule name: got %q, want %q", rp.RuleName, "tool-writes")
	}
	if rp.CurrentModel != "small" || rp.ProposedModel != "large" {
		t.Errorf("proposed model change: got %s -> %s, want small -> large", rp.CurrentModel, rp.ProposedModel)
	}

	// 2. Assert that the proposed manifest achieves a strictly better measured objective on the held-out split
	if !proposal.HeldOutScore.Improved {
		t.Errorf("expected proposal.HeldOutScore.Improved to be true")
	}
	if proposal.HeldOutScore.Delta <= 0 {
		t.Errorf("expected held-out score delta > 0, got %v", proposal.HeldOutScore.Delta)
	}
	if proposal.HeldOutScore.After.Score <= proposal.HeldOutScore.Before.Score {
		t.Errorf("expected After.Score (%v) > Before.Score (%v)",
			proposal.HeldOutScore.After.Score, proposal.HeldOutScore.Before.Score)
	}
	if !proposal.Improved() {
		t.Errorf("expected proposal.Improved() to be true")
	}

	// 3. Assert that the reviewable manifest diff is generated and valid JSON
	if proposal.Diff == "" {
		t.Fatal("manifest diff is empty")
	}
	var parsedDiff map[string]any
	if err := json.Unmarshal([]byte(proposal.Diff), &parsedDiff); err != nil {
		t.Fatalf("diff is not valid JSON: %v\ndiff content:\n%s", err, proposal.Diff)
	}
	if parsedDiff["schema"] != "fak-modelroute-diff/v1" {
		t.Errorf("diff schema: got %v, want fak-modelroute-diff/v1", parsedDiff["schema"])
	}
	if proposal.DiffStruct == nil || len(proposal.DiffStruct.ChangedRules) != 1 {
		t.Errorf("diff struct missing or changed rules count != 1")
	}

	// 4. Assert that no policy is silently auto-applied (keep=false)
	if proposal.ReviewAudit.Keep != false {
		t.Errorf("expected ReviewAudit.Keep to be false, got %v", proposal.ReviewAudit.Keep)
	}
	if !proposal.ReviewAudit.RequiresHumanReview {
		t.Errorf("expected ReviewAudit.RequiresHumanReview to be true")
	}
	if proposal.ReviewAudit.Status != "PENDING_REVIEW" {
		t.Errorf("expected ReviewAudit.Status to be PENDING_REVIEW, got %q", proposal.ReviewAudit.Status)
	}

	// Attempting to apply without explicit approval MUST fail closed
	_, err = proposal.Apply()
	if err == nil {
		t.Fatal("Apply() succeeded on unapproved proposal; want fail-closed error")
	}
	if !strings.Contains(err.Error(), "keep=false") {
		t.Errorf("Apply() error should mention keep=false, got %v", err)
	}

	// Once explicitly approved by operator, Apply() succeeds
	proposal.ReviewAudit.Keep = true
	applied, err := proposal.Apply()
	if err != nil {
		t.Fatalf("Apply() failed after Keep=true: %v", err)
	}
	if applied.Rules[0].Plan.Primary() != "large" {
		t.Errorf("applied rule plan primary model: got %q, want large", applied.Rules[0].Plan.Primary())
	}
}

// TestProposeRejectsWhenCandidateDegradesHeldOut witnesses the held-out generalization guard:
// a candidate that improves training data but degrades on the held-out split must be rejected.
func TestProposeRejectsWhenCandidateDegradesHeldOut(t *testing.T) {
	base := baseManifestForPropose()

	var trainRecords []OutcomeRecord
	var heldOutRecords []OutcomeRecord

	// In train: "large" looks better than "small"
	// "small": quality 0.50
	// "large": quality 0.90
	trainRecords = append(trainRecords,
		OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "small",
			Outcome: Outcome{Cost: 0.01, Quality: 0.50},
		},
		OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "large",
			Outcome: Outcome{Cost: 0.05, Quality: 0.90},
		},
	)

	// In heldOut: "large" degrades and is worse than "small"
	// "small": quality 0.80
	// "large": quality 0.40 (overfitting or distribution shift)
	heldOutRecords = append(heldOutRecords,
		OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "small",
			Outcome: Outcome{Cost: 0.01, Quality: 0.80},
		},
		OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "large",
			Outcome: Outcome{Cost: 0.05, Quality: 0.40},
		},
	)

	opts := ProposeOptions{
		BaseManifest:   base,
		TrainRecords:   trainRecords,
		HeldOutRecords: heldOutRecords,
		Objective: Objective{
			QualityWeight: 1.0,
		},
	}

	proposal, err := Propose(opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if proposal != nil {
		t.Fatalf("expected candidate to be rejected due to held-out degradation, but got proposal: %+v", proposal)
	}
}

// TestProposeRespectsOperationalConstraints witnesses that candidate rule changes
// violating declared operational constraints (e.g. MaxMeanCost) are rejected.
func TestProposeRespectsOperationalConstraints(t *testing.T) {
	base := baseManifestForPropose()

	var records []OutcomeRecord
	for i := 0; i < 10; i++ {
		// "small": cost $0.01, quality 0.60
		records = append(records, OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "small",
			Outcome: Outcome{Cost: 0.01, Quality: 0.60},
		})
		// "large": cost $0.10, quality 0.95
		records = append(records, OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "large",
			Outcome: Outcome{Cost: 0.10, Quality: 0.95},
		})
	}

	// Constraint: MaxMeanCost is 0.05. "large" costs 0.10, so it must be rejected as unfeasible.
	optsConstrained := ProposeOptions{
		BaseManifest: base,
		Records:      records,
		TrainRatio:   0.70,
		Objective: Objective{
			QualityWeight: 1.0,
			MaxMeanCost:   0.05,
		},
	}

	prop1, err := Propose(optsConstrained)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if prop1 != nil {
		t.Fatalf("expected proposal to be rejected due to MaxMeanCost constraint, got %+v", prop1)
	}

	// Relax constraint to 0.15: "large" is now feasible and should be proposed.
	optsRelaxed := ProposeOptions{
		BaseManifest: base,
		Records:      records,
		TrainRatio:   0.70,
		Objective: Objective{
			QualityWeight: 1.0,
			MaxMeanCost:   0.15,
		},
	}

	prop2, err := Propose(optsRelaxed)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if prop2 == nil {
		t.Fatal("expected proposal under relaxed constraint, got nil")
	}
	if prop2.RuleProposals[0].ProposedModel != "large" {
		t.Errorf("proposed model: got %q, want large", prop2.RuleProposals[0].ProposedModel)
	}
}

// TestProposeWeightedObjective witnesses multi-objective trade-offs:
// weighting cost heavily penalizes higher-cost models even if quality is higher.
func TestProposeWeightedObjective(t *testing.T) {
	base := baseManifestForPropose()

	var records []OutcomeRecord
	for i := 0; i < 10; i++ {
		// "small": cost $0.01, quality 0.70
		records = append(records, OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "small",
			Outcome: Outcome{Cost: 0.01, Quality: 0.70},
		})
		// "large": cost $0.05, quality 0.80
		records = append(records, OutcomeRecord{
			Key:     AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"},
			Model:   "large",
			Outcome: Outcome{Cost: 0.05, Quality: 0.80},
		})
	}

	// With heavy cost penalty (CostWeight = 10.0):
	// small score: 1.0 * 0.70 - 10.0 * 0.01 = 0.60
	// large score: 1.0 * 0.80 - 10.0 * 0.05 = 0.30
	// "small" has a higher objective score than "large", so "large" must NOT be proposed.
	optsCostSensitive := ProposeOptions{
		BaseManifest: base,
		Records:      records,
		TrainRatio:   0.70,
		Objective: Objective{
			QualityWeight: 1.0,
			CostWeight:    10.0,
		},
	}

	prop, err := Propose(optsCostSensitive)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if prop != nil {
		t.Fatalf("expected cost-sensitive objective to reject large, got proposal: %+v", prop)
	}
}

// TestProposeWithOutcomeJournal witnesses that Propose accepts an OutcomeJournal directly.
func TestProposeWithOutcomeJournal(t *testing.T) {
	base := baseManifestForPropose()
	var j OutcomeJournal

	dSmall := Decision{
		Subject:  Subject{Aspect: AspectToolCall, Tool: "write_file"},
		RuleName: "tool-writes",
		Plan:     Plan{Members: []Member{{Model: "small"}}},
	}
	dLarge := Decision{
		Subject:  Subject{Aspect: AspectToolCall, Tool: "write_file"},
		RuleName: "tool-writes",
		Plan:     Plan{Members: []Member{{Model: "large"}}},
	}

	for i := 0; i < 10; i++ {
		j.Record(base.Version, dSmall, Outcome{Cost: 0.01, Quality: 0.60})
		j.Record(base.Version, dLarge, Outcome{Cost: 0.02, Quality: 0.95})
	}

	prop, err := Propose(ProposeOptions{
		BaseManifest: base,
		Journal:      &j,
		TrainRatio:   0.70,
		Objective:    Objective{QualityWeight: 1.0},
	})
	if err != nil {
		t.Fatalf("Propose with journal failed: %v", err)
	}
	if prop == nil {
		t.Fatal("expected proposal from journal, got nil")
	}
	if prop.RuleProposals[0].ProposedModel != "large" {
		t.Errorf("proposed model: got %q, want large", prop.RuleProposals[0].ProposedModel)
	}
}

// TestSplitRecordsStratified witnesses deterministic stratified partitioning.
func TestSplitRecordsStratified(t *testing.T) {
	var records []OutcomeRecord
	for i := 0; i < 10; i++ {
		records = append(records, OutcomeRecord{
			Key:   AspectRuleKey{Aspect: AspectToolCall, Rule: "rule1"},
			Model: "model-a",
		})
		records = append(records, OutcomeRecord{
			Key:   AspectRuleKey{Aspect: AspectToolCall, Rule: "rule1"},
			Model: "model-b",
		})
	}

	train, heldOut := SplitRecords(records, 0.70)
	if len(train) != 14 {
		t.Errorf("train count: got %d, want 14 (7 per model)", len(train))
	}
	if len(heldOut) != 6 {
		t.Errorf("heldOut count: got %d, want 6 (3 per model)", len(heldOut))
	}

	// Calling SplitRecords again on the exact same input yields the exact same split
	train2, heldOut2 := SplitRecords(records, 0.70)
	if len(train) != len(train2) || len(heldOut) != len(heldOut2) {
		t.Fatal("SplitRecords is not deterministic")
	}
}
