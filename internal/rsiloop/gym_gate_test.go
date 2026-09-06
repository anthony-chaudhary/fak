package rsiloop

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gym"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

func TestPolicyGatewayCandidateRequiresGymReceipt(t *testing.T) {
	h := Harness{
		MetricName:      "latency_p50",
		LowerBetter:     true,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 100.0, "sha-base", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{
					Label:   "gateway_compaction_v2",
					Payload: []string{"internal/gateway/responses_elide.go"},
				},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			// Candidate passes unit tests and improves KPI, but provides NO Gym receipt
			return Measurement{
				Metric:      50.0, // Strict gain: 50 < 100
				SuiteGreen:  true,
				TruthClean:  true,
				GymReceipt:  nil,
				GymVerified: false,
			}, nil
		},
	}

	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if len(res.Rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(res.Rows))
	}

	row := res.Rows[0]
	if row.Kept {
		t.Errorf("candidate must NOT be kept without Gym multi-turn receipt")
	}
	if row.Decision != "REVERT" {
		t.Errorf("decision = %q, want REVERT", row.Decision)
	}
	if row.GymVerified {
		t.Errorf("expected GymVerified = false")
	}
	if row.SuiteGreen {
		t.Errorf("expected SuiteGreen to be flipped to false when Gym proof missing")
	}
}

func TestPolicyGatewayCandidateRejectsFailingGymReceipt(t *testing.T) {
	failingReceipt := &gym.GymReceipt{
		Schema:           gym.GymReceiptSchema,
		ScenarioID:       "gateway_subturn_valve",
		Timestamp:        time.Now().UTC(),
		TurnsExecuted:    4,
		LivelockDetected: true, // Induced client livelock!
		MultiTurnPass:    false,
		Outcome:          gym.OutcomeFail,
		FailureReason:    "client livelock runaway repetition observed",
	}

	h := Harness{
		MetricName:      "token_shed",
		LowerBetter:     false,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 50.0, "sha-base", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{
					Label:   "gateway_subturn_valve",
					Payload: []string{"internal/gateway/responses.go"},
				},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			// Unit tests passed, but Gym trajectory destabilized harness
			return Measurement{
				Metric:      80.0, // 80 > 50 gain
				SuiteGreen:  true,
				TruthClean:  true,
				GymReceipt:  failingReceipt,
				GymVerified: false,
			}, nil
		},
	}

	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	row := res.Rows[0]
	if row.Kept {
		t.Errorf("candidate must NOT be kept with failing Gym receipt")
	}
	if row.Decision != "REVERT" {
		t.Errorf("decision = %q, want REVERT", row.Decision)
	}
	if row.GymVerified {
		t.Errorf("expected GymVerified = false")
	}
}

func TestPolicyGatewayCandidateAcceptsPassingGymReceipt(t *testing.T) {
	passingReceipt := &gym.GymReceipt{
		Schema:           gym.GymReceiptSchema,
		ScenarioID:       "gateway_safe_elision",
		Timestamp:        time.Now().UTC(),
		TurnsExecuted:    5,
		TotalToolCalls:   8,
		ElisionsObserved: 3,
		RestoresObserved: 2,
		LivelockDetected: false,
		MultiTurnPass:    true,
		Outcome:          gym.OutcomePass,
		TranscriptDigest: "abc123sha",
	}

	h := Harness{
		MetricName:      "p99_latency",
		LowerBetter:     true,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 200.0, "sha-base", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{
					Label:   "gateway_safe_elision",
					Payload: []string{"internal/gateway/tool_dialect.go"},
				},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			return Measurement{
				Metric:     120.0, // 120 < 200
				SuiteGreen: true,
				TruthClean: true,
				GymReceipt: passingReceipt,
			}, nil
		},
	}

	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	row := res.Rows[0]
	if !row.Kept {
		t.Errorf("candidate with verified Gym receipt should be KEPT")
	}
	if row.Decision != "KEEP" {
		t.Errorf("decision = %q, want KEEP", row.Decision)
	}
	if !row.GymVerified {
		t.Errorf("expected GymVerified = true")
	}
	if row.GymReceiptRef != "gateway_safe_elision@abc123sha" {
		t.Errorf("gym_receipt_ref = %q, want gateway_safe_elision@abc123sha", row.GymReceiptRef)
	}
}

func TestNonPolicyCandidateBypassesGymProof(t *testing.T) {
	h := Harness{
		MetricName:      "doc_completeness",
		LowerBetter:     false,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 10.0, "sha-base", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{
					Label:   "docs_update",
					Payload: []string{"docs/fak/README.md"},
				},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			return Measurement{
				Metric:     20.0, // 20 > 10
				SuiteGreen: true,
				TruthClean: true,
				GymReceipt: nil, // Not required for docs
			}, nil
		},
	}

	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	row := res.Rows[0]
	if !row.Kept {
		t.Errorf("non-policy candidate should be kept without gym receipt")
	}
	if row.Decision != "KEEP" {
		t.Errorf("decision = %q, want KEEP", row.Decision)
	}
}

func TestEvaluateProposalWithGymProof(t *testing.T) {
	before := []Row{
		{Mode: "improve", Kept: false, TruthClean: true},
	}
	after := []Row{
		{Mode: "improve", Kept: true, TruthClean: true, SuiteGreen: true},
	}

	// 1. Without passing Gym receipt -> REVERT
	dec, _ := EvaluateProposalWithGymProof(before, after, nil)
	if dec != shipgate.REVERT {
		t.Errorf("expected REVERT with nil receipt, got %s", dec)
	}

	// 2. With passing Gym receipt -> KEEP
	validReceipt := &gym.GymReceipt{
		Schema:        gym.GymReceiptSchema,
		MultiTurnPass: true,
		Outcome:       gym.OutcomePass,
	}
	dec, _ = EvaluateProposalWithGymProof(before, after, validReceipt)
	if dec != shipgate.KEEP {
		t.Errorf("expected KEEP with valid receipt, got %s", dec)
	}
}

func TestIsPolicyOrGatewayCandidateConsistency(t *testing.T) {
	candidates := []Candidate{
		{Label: "gateway_test", Payload: "gateway"},
		{Label: "policy_candidate", Payload: "policy"},
		{Label: "compaction_rule", Payload: "compaction"},
		{Label: "subturn_valve", Payload: "subturn"},
		{Label: "unrelated_worker", Payload: []string{"docs/fak/README.md"}},
		{Label: "gateway_code", Payload: []string{"internal/gateway/tool_dialect.go"}},
		{Label: "policy_code", Payload: []string{"internal/policy/engine.go"}},
		{Label: "rsiloop_policy", Payload: []string{"internal/rsiloop/policy.go"}},
	}
	for _, c := range candidates {
		want := needsGymExecution(c)
		got := isPolicyOrGatewayCandidate(c)
		if got != want {
			t.Errorf("candidate %q: isPolicyOrGatewayCandidate = %v, want %v (needsGymExecution)", c.Label, got, want)
		}
	}
}
