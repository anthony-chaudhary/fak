package ultracodebench

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestSafeAccelerationCampaign(t *testing.T) {
	tests := []struct {
		name           string
		mutate         func(*SafeAccelerationCampaign)
		verify         func(*SafeAccelerationVerification, SafeAccelerationVerificationRequest)
		noVerifier     bool
		verifierError  bool
		want           SafeAccelerationVerdict
		reason         SafeAccelerationReasonCode
		wantRoundedTen bool
	}{
		{name: "gain 10x", want: SafeAccelerationGain10X},
		{name: "complete but below claim", mutate: func(c *SafeAccelerationCampaign) {
			for i := range c.Pairs {
				setSafeAccelerationWall(&c.Pairs[i].Candidate, 250)
			}
		}, want: SafeAccelerationNoGain, reason: "paired_rate_ratio_lower_95_below_10"},
		{name: "raw lower bound controls verdict", mutate: func(c *SafeAccelerationCampaign) {
			for i := range c.Pairs {
				setSafeAccelerationWall(&c.Pairs[i].Baseline, 1_000_000_000)
				setSafeAccelerationWall(&c.Pairs[i].Candidate, 100_000_001)
			}
		}, want: SafeAccelerationNoGain, reason: "paired_rate_ratio_lower_95_below_10", wantRoundedTen: true},
		{name: "schema", mutate: func(c *SafeAccelerationCampaign) { c.Schema = "old" }, want: SafeAccelerationAbstain, reason: "schema_unsupported"},
		{name: "twenty tasks", mutate: func(c *SafeAccelerationCampaign) { c.Tasks = c.Tasks[:19] }, want: SafeAccelerationAbstain, reason: "task_count_below_20"},
		{name: "atomic unique tasks", mutate: func(c *SafeAccelerationCampaign) { c.Tasks[0].Scope = "S2" }, want: SafeAccelerationAbstain, reason: "tasks_not_unique_s0_s1"},
		{name: "five pairs", mutate: func(c *SafeAccelerationCampaign) { c.Pairs = c.Pairs[:4] }, want: SafeAccelerationAbstain, reason: "matched_pair_count_below_5"},
		{name: "public revision", mutate: func(c *SafeAccelerationCampaign) { c.PublicRevision = "main" }, want: SafeAccelerationAbstain, reason: "public_revision_not_exact"},
		{name: "private revision", mutate: func(c *SafeAccelerationCampaign) { c.PrivateRevision = "latest" }, want: SafeAccelerationAbstain, reason: "private_revision_not_exact"},
		{name: "baseline identity", mutate: func(c *SafeAccelerationCampaign) { c.Baseline.Source = "" }, want: SafeAccelerationAbstain, reason: "baseline_identity_incomplete"},
		{name: "candidate identity", mutate: func(c *SafeAccelerationCampaign) { c.Candidate.ConfigurationDigest = "" }, want: SafeAccelerationAbstain, reason: "candidate_identity_incomplete"},
		{name: "distinct candidate", mutate: func(c *SafeAccelerationCampaign) { c.Candidate = c.Baseline }, want: SafeAccelerationAbstain, reason: "candidate_identity_not_distinct"},
		{name: "matched workload", mutate: func(c *SafeAccelerationCampaign) { c.Candidate.WorkloadDigest = testSafeAccelerationDigest(99) }, want: SafeAccelerationAbstain, reason: "workload_digest_mismatch"},
		{name: "identical task ids", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.TaskIDs[0] = "other" }, want: SafeAccelerationAbstain, reason: "candidate_task_ids_not_identical"},
		{name: "full wall", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.FullWallMS = 0 }, want: SafeAccelerationAbstain, reason: "candidate_full_wall_not_positive"},
		{name: "complete phases", mutate: func(c *SafeAccelerationCampaign) { delete(c.Pairs[0].Candidate.PhaseMS, "recovery") }, want: SafeAccelerationAbstain, reason: "candidate_phase_accounting_incomplete"},
		{name: "accepted units", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.AcceptedUnits = 21 }, want: SafeAccelerationAbstain, reason: "candidate_accepted_units_invalid"},
		{name: "positive accepted rate", mutate: func(c *SafeAccelerationCampaign) {
			c.Pairs[0].Baseline.AcceptedUnits, c.Pairs[0].Candidate.AcceptedUnits = 0, 0
		}, want: SafeAccelerationAbstain, reason: "accepted_unit_rate_not_positive"},
		{name: "latencies", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.TaskLatencyMS = nil }, want: SafeAccelerationAbstain, reason: "candidate_task_latency_incomplete"},
		{name: "observed evidence", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.EvidenceKind = "fixture" }, want: SafeAccelerationAbstain, reason: "candidate_evidence_not_observed"},
		{name: "receipt digest", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.ReceiptDigest = "" }, want: SafeAccelerationAbstain, reason: "candidate_receipt_digest_missing"},
		{name: "independent acceptance", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.AcceptanceWitnessIndependent = false }, want: SafeAccelerationAbstain, reason: "candidate_acceptance_witness_not_independent"},
		{name: "engine identity", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Engine = "" }, want: SafeAccelerationAbstain, reason: "candidate_engine_identity_missing"},
		{name: "backend identity", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Backend = "" }, want: SafeAccelerationAbstain, reason: "candidate_backend_identity_missing"},
		{name: "device identity", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Device = "" }, want: SafeAccelerationAbstain, reason: "candidate_device_identity_missing"},
		{name: "native fallback", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.FallbackCount = 1 }, want: SafeAccelerationAbstain, reason: "candidate_native_engine_or_fallback_invalid"},
		{name: "native opt out", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Native = false }, want: SafeAccelerationAbstain, reason: "candidate_native_engine_or_fallback_invalid"},
		{name: "policy violation", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Violations.Policy = 1 }, want: SafeAccelerationAbstain, reason: "candidate_violation_policy"},
		{name: "secret violation", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Violations.Secret = 1 }, want: SafeAccelerationAbstain, reason: "candidate_violation_secret"},
		{name: "boundary violation", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Violations.Boundary = 1 }, want: SafeAccelerationAbstain, reason: "candidate_violation_boundary"},
		{name: "lease violation", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Violations.Lease = 1 }, want: SafeAccelerationAbstain, reason: "candidate_violation_lease"},
		{name: "scope violation", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Violations.Scope = 1 }, want: SafeAccelerationAbstain, reason: "candidate_violation_scope"},
		{name: "provenance violation", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Violations.Provenance = 1 }, want: SafeAccelerationAbstain, reason: "candidate_violation_provenance"},
		{name: "resource completeness", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.ResourceMetricsComplete = false }, want: SafeAccelerationAbstain, reason: "candidate_resource_metrics_incomplete"},
		{name: "finite resources", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.CostUSD = math.NaN() }, want: SafeAccelerationAbstain, reason: "candidate_resource_metrics_incomplete"},
		{name: "acceptance below baseline", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.AcceptedUnits = 19 }, want: SafeAccelerationAbstain, reason: "candidate_acceptance_below_baseline"},
		{name: "acceptance below eighty", mutate: func(c *SafeAccelerationCampaign) {
			for i := range c.Pairs {
				c.Pairs[i].Baseline.AcceptedUnits, c.Pairs[i].Candidate.AcceptedUnits = 15, 15
			}
		}, want: SafeAccelerationAbstain, reason: "candidate_acceptance_below_80_percent"},
		{name: "p90 regression", mutate: func(c *SafeAccelerationCampaign) {
			for i := range c.Pairs {
				for j := range c.Pairs[i].Candidate.TaskLatencyMS {
					c.Pairs[i].Candidate.TaskLatencyMS[j] = 2000
				}
			}
		}, want: SafeAccelerationAbstain, reason: "candidate_p90_latency_regression"},
		{name: "cost regression", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.CostUSD = 2 }, want: SafeAccelerationAbstain, reason: "candidate_cost_regression"},
		{name: "joules regression", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.Joules = 200 }, want: SafeAccelerationAbstain, reason: "candidate_joules_regression"},
		{name: "operator regression", mutate: func(c *SafeAccelerationCampaign) { c.Pairs[0].Candidate.OperatorMinutes = 2 }, want: SafeAccelerationAbstain, reason: "candidate_operator_minutes_regression"},
		{name: "trusted verifier required", noVerifier: true, want: SafeAccelerationAbstain, reason: "trusted_verifier_missing"},
		{name: "verifier failure", verifierError: true, want: SafeAccelerationAbstain, reason: "trusted_verification_failed"},
		{name: "resolved revision mismatch", verify: func(v *SafeAccelerationVerification, _ SafeAccelerationVerificationRequest) {
			v.ResolvedPublicRevision = testSafeAccelerationRevision(9)
		}, want: SafeAccelerationAbstain, reason: "verified_revision_mismatch"},
		{name: "rehashed evidence mismatch", verify: func(v *SafeAccelerationVerification, _ SafeAccelerationVerificationRequest) {
			v.Arms[0].RehashedReceiptDigest = testSafeAccelerationDigest(90)
		}, want: SafeAccelerationAbstain, reason: "verified_evidence_hash_mismatch"},
		{name: "independent verified issuers", verify: func(v *SafeAccelerationVerification, _ SafeAccelerationVerificationRequest) {
			v.Arms[0].WitnessIssuer = v.Arms[0].ReceiptIssuer
		}, want: SafeAccelerationAbstain, reason: "verified_issuer_not_independent"},
		{name: "verified provenance", verify: func(v *SafeAccelerationVerification, _ SafeAccelerationVerificationRequest) {
			v.Arms[0].ProvenanceDigest = ""
		}, want: SafeAccelerationAbstain, reason: "verified_provenance_missing"},
		{name: "run binding", verify: func(v *SafeAccelerationVerification, _ SafeAccelerationVerificationRequest) {
			v.Arms[0].InputBindingDigest = ""
		}, want: SafeAccelerationAbstain, reason: "verified_run_binding_mismatch"},
		{name: "run id replay", verify: func(v *SafeAccelerationVerification, request SafeAccelerationVerificationRequest) {
			v.Arms[1].RunID = v.Arms[0].RunID
			v.Arms[1].InputBindingDigest = SafeAccelerationRunBinding(request, request.Evidence[1], v.Arms[1].RunID)
		}, want: SafeAccelerationAbstain, reason: "verified_identity_replay"},
		{name: "cross class identity collision", verify: func(v *SafeAccelerationVerification, request SafeAccelerationVerificationRequest) {
			v.Arms[0].RunID = request.Evidence[1].ReceiptDigest
			v.Arms[0].InputBindingDigest = SafeAccelerationRunBinding(request, request.Evidence[0], v.Arms[0].RunID)
		}, want: SafeAccelerationAbstain, reason: "verified_identity_replay"},
		{name: "evidence digest replay", mutate: func(c *SafeAccelerationCampaign) {
			c.Pairs[0].Candidate.ReceiptDigest = c.Pairs[0].Baseline.AcceptanceWitnessDigest
		}, want: SafeAccelerationAbstain, reason: "evidence_digest_replay"},
		{name: "verified arm set", verify: func(v *SafeAccelerationVerification, _ SafeAccelerationVerificationRequest) {
			v.Arms = v.Arms[:len(v.Arms)-1]
		}, want: SafeAccelerationAbstain, reason: "verified_arm_set_mismatch"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			campaign := validSafeAccelerationCampaign()
			if test.mutate != nil {
				test.mutate(&campaign)
			}
			var verifier SafeAccelerationVerifier
			if !test.noVerifier {
				verifier = deterministicSafeAccelerationVerifier{mutate: test.verify, fail: test.verifierError}
			}
			report := EvaluateSafeAcceleration(campaign, verifier)
			if report.Verdict != test.want {
				t.Fatalf("verdict=%s want=%s reasons=%v", report.Verdict, test.want, report.Reasons)
			}
			if test.reason != "" && !slices.ContainsFunc(report.Reasons, func(reason SafeAccelerationReason) bool { return reason.Code == test.reason }) {
				t.Fatalf("missing reason %q in %+v", test.reason, report.Reasons)
			}
			if report.Verdict == SafeAccelerationGain10X && report.RateRatioLower95 < 10 {
				t.Fatalf("GAIN_10X lower bound=%v", report.RateRatioLower95)
			}
			if test.wantRoundedTen && report.RateRatioLower95 != 10 {
				t.Fatalf("rounded lower bound=%v, want 10 while raw verdict remains NO_GAIN", report.RateRatioLower95)
			}
		})
	}
}

type deterministicSafeAccelerationVerifier struct {
	mutate func(*SafeAccelerationVerification, SafeAccelerationVerificationRequest)
	fail   bool
}

func (v deterministicSafeAccelerationVerifier) VerifySafeAcceleration(request SafeAccelerationVerificationRequest) (SafeAccelerationVerification, error) {
	if v.fail {
		return SafeAccelerationVerification{}, errors.New("deterministic verifier failure")
	}
	verification := SafeAccelerationVerification{
		ResolvedPublicRevision: request.PublicRevision, ResolvedPrivateRevision: request.PrivateRevision,
		Arms: make([]SafeAccelerationArmVerification, 0, len(request.Evidence)),
	}
	for i, evidence := range request.Evidence {
		runID := fmt.Sprintf("verified-run-%d-%s", evidence.Pair, evidence.Arm)
		verification.Arms = append(verification.Arms, SafeAccelerationArmVerification{
			Pair: evidence.Pair, Arm: evidence.Arm, RunID: runID,
			RehashedReceiptDigest: evidence.ReceiptDigest, RehashedWitnessDigest: evidence.AcceptanceWitnessDigest,
			ReceiptIssuer: "runner:" + evidence.Arm, WitnessIssuer: "witness:" + evidence.Arm,
			ProvenanceDigest:   testSafeAccelerationDigest(100 + i),
			InputBindingDigest: SafeAccelerationRunBinding(request, evidence, runID),
		})
	}
	if v.mutate != nil {
		v.mutate(&verification, request)
	}
	return verification, nil
}

func validSafeAccelerationCampaign() SafeAccelerationCampaign {
	c := SafeAccelerationCampaign{
		Schema:          SafeAccelerationSchema,
		PublicRevision:  "5cff6da95a830000000000000000000000000000",
		PrivateRevision: "d680eba198470000000000000000000000000000",
		Baseline: SafeAccelerationIdentity{
			Source: "frontier-single@2026-09-08", ArtifactDigest: testSafeAccelerationDigest(1),
			ConfigurationDigest: testSafeAccelerationDigest(2), WorkloadDigest: testSafeAccelerationDigest(3),
		},
		Candidate: SafeAccelerationIdentity{
			Source: "strix-fleet/glm53-flash@2026-09-08", ArtifactDigest: testSafeAccelerationDigest(4),
			ConfigurationDigest: testSafeAccelerationDigest(5), WorkloadDigest: testSafeAccelerationDigest(3),
		},
	}
	for i := 0; i < 20; i++ {
		c.Tasks = append(c.Tasks, SafeAccelerationTask{ID: "task-" + string(rune('a'+i)), Scope: "S1"})
	}
	ids := make([]string, len(c.Tasks))
	for i := range c.Tasks {
		ids[i] = c.Tasks[i].ID
	}
	for i := 0; i < 5; i++ {
		baseline := validSafeAccelerationArm(ids, 2000, testSafeAccelerationDigest(10+i), testSafeAccelerationDigest(20+i))
		candidate := validSafeAccelerationArm(ids, 100, testSafeAccelerationDigest(30+i), testSafeAccelerationDigest(40+i))
		c.Pairs = append(c.Pairs, SafeAccelerationPair{Baseline: baseline, Candidate: candidate})
	}
	return c
}

func testSafeAccelerationDigest(seed int) string {
	return fmt.Sprintf("sha256:%064x", seed)
}

func testSafeAccelerationRevision(seed int) string {
	return fmt.Sprintf("%040x", seed)
}

func validSafeAccelerationArm(ids []string, wall int64, receipt, witness string) SafeAccelerationArm {
	arm := SafeAccelerationArm{
		TaskIDs: append([]string(nil), ids...), FullWallMS: wall, AcceptedUnits: len(ids),
		EvidenceKind: "observed", ReceiptDigest: receipt,
		AcceptanceWitnessDigest: witness, AcceptanceWitnessIndependent: true,
		Engine: "fak-native", Backend: "vulkan", Device: "strix-halo-395",
		Native: true, ResourceMetricsComplete: true, CostUSD: 1, Joules: 100, OperatorMinutes: 1,
	}
	arm.TaskLatencyMS = make([]int64, len(ids))
	for i := range arm.TaskLatencyMS {
		arm.TaskLatencyMS[i] = wall / int64(len(ids))
	}
	setSafeAccelerationWall(&arm, wall)
	return arm
}

func setSafeAccelerationWall(arm *SafeAccelerationArm, wall int64) {
	arm.FullWallMS = wall
	arm.PhaseMS = map[string]int64{
		"admission": wall / 10, "queueing": wall / 10, "execution": wall * 4 / 10,
		"retries": 0, "verification": wall / 10, "fan_in": wall / 10,
		"stragglers": wall / 10, "recovery": 0,
		"terminal_failures": wall - (wall/10 + wall/10 + wall*4/10 + wall/10 + wall/10 + wall/10),
	}
}
