package ultracodebench

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestSafeAccelerationCampaign(t *testing.T) {
	t.Run("raw verified decision", func(t *testing.T) {
		if got := safeAccelerationVerifiedVerdict(10); got != SafeAccelerationGain10X {
			t.Fatalf("threshold verdict=%s", got)
		}
		if got := safeAccelerationVerifiedVerdict(math.Nextafter(10, 0)); got != SafeAccelerationNoGain {
			t.Fatalf("raw below-threshold verdict=%s", got)
		}
		_, lower := safeAccelerationPairedLower95([]float64{8, 8, 8, 8, 8})
		if got := safeAccelerationVerifiedVerdict(lower); got != SafeAccelerationNoGain {
			t.Fatalf("complete lower-bound verdict=%s lower=%v", got, lower)
		}
	})

	tests := []struct {
		name    string
		mutate  func(*SafeAccelerationCampaign, *safeAccelerationVerificationCapability)
		dropCap bool
		reason  SafeAccelerationReasonCode
	}{
		{name: "fixture cannot claim gain", reason: "baseline_evidence_not_observed"},
		{name: "authoritative capability required", dropCap: true, reason: "authoritative_verification_missing"},
		{name: "full wall mutation after verification", mutate: func(c *SafeAccelerationCampaign, _ *safeAccelerationVerificationCapability) {
			c.Pairs[0].Candidate.FullWallMS++
			c.Pairs[0].Candidate.PhaseMS["terminal_failures"]++
		}, reason: "authoritative_campaign_binding_mismatch"},
		{name: "safety mutation after verification", mutate: func(c *SafeAccelerationCampaign, _ *safeAccelerationVerificationCapability) {
			c.Pairs[0].Candidate.Violations.Policy++
		}, reason: "authoritative_campaign_binding_mismatch"},
		{name: "resource mutation after verification", mutate: func(c *SafeAccelerationCampaign, _ *safeAccelerationVerificationCapability) {
			c.Pairs[0].Candidate.CostUSD += 0.01
		}, reason: "authoritative_campaign_binding_mismatch"},
		{name: "capability seal", mutate: func(_ *SafeAccelerationCampaign, capability *safeAccelerationVerificationCapability) {
			capability.seal = ""
		}, reason: "authoritative_capability_invalid"},
		{name: "workload match", mutate: func(c *SafeAccelerationCampaign, _ *safeAccelerationVerificationCapability) {
			c.Candidate.WorkloadDigest = testSafeAccelerationDigest(999)
		}, reason: "workload_digest_mismatch"},
		{name: "native opt out", mutate: func(c *SafeAccelerationCampaign, _ *safeAccelerationVerificationCapability) {
			c.Pairs[0].Candidate.Native = false
		}, reason: "candidate_native_engine_or_fallback_invalid"},
		{name: "evidence replay", mutate: func(c *SafeAccelerationCampaign, _ *safeAccelerationVerificationCapability) {
			c.Pairs[0].Candidate.ReceiptDigest = c.Pairs[0].Baseline.AcceptanceWitnessDigest
		}, reason: "evidence_digest_replay"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			campaign, capability, _, err := testOnlyAuthoritativeFixture(validSafeAccelerationCampaign())
			if err != nil {
				t.Fatal(err)
			}
			if test.mutate != nil {
				test.mutate(&campaign, capability)
			}
			if test.dropCap {
				capability = nil
			}
			report := EvaluateSafeAcceleration(campaign, capability)
			if report.Verdict != SafeAccelerationAbstain {
				t.Fatalf("fixture verdict=%s reasons=%v", report.Verdict, report.Reasons)
			}
			if !slices.ContainsFunc(report.Reasons, func(reason SafeAccelerationReason) bool { return reason.Code == test.reason }) {
				t.Fatalf("missing reason %q in %+v", test.reason, report.Reasons)
			}
		})
	}

	t.Run("authoritative path rederives measurements", func(t *testing.T) {
		campaign, _, materials, err := testOnlyAuthoritativeFixture(validSafeAccelerationCampaign())
		if err != nil {
			t.Fatal(err)
		}
		var receipt safeAccelerationReceipt
		if err := json.Unmarshal(materials[0].Receipt, &receipt); err != nil {
			t.Fatal(err)
		}
		receipt.Measurements.FullWallMS++
		materials[0].Receipt, _ = json.Marshal(receipt)
		campaign.Pairs[0].Baseline.ReceiptDigest = safeAccelerationHash(materials[0].Receipt)
		var witness safeAccelerationWitness
		if err := json.Unmarshal(materials[0].Witness, &witness); err != nil {
			t.Fatal(err)
		}
		witness.ReceiptDigest = campaign.Pairs[0].Baseline.ReceiptDigest
		materials[0].Witness, _ = json.Marshal(witness)
		campaign.Pairs[0].Baseline.AcceptanceWitnessDigest = safeAccelerationHash(materials[0].Witness)
		if _, err := safeAccelerationAuthorizeResolved(campaign, materials); err == nil {
			t.Fatal("mutated resolved measurement was authorized")
		}
	})
}

// testOnlyAuthoritativeFixture is package-private and deliberately labels all
// synthesized material as fixture. It exercises capability construction but
// can never produce a GAIN_10X report.
func testOnlyAuthoritativeFixture(c SafeAccelerationCampaign) (SafeAccelerationCampaign, *safeAccelerationVerificationCapability, []safeAccelerationResolvedMaterial, error) {
	materials := make([]safeAccelerationResolvedMaterial, 0, len(c.Pairs)*2)
	for pairIndex := range c.Pairs {
		for _, armName := range []string{"baseline", "candidate"} {
			arm, identity := testSafeAccelerationArmPointer(&c, pairIndex, armName)
			arm.EvidenceKind = "fixture"
			runID := fmt.Sprintf("fixture-run-%d-%s", pairIndex+1, armName)
			receipt := safeAccelerationReceipt{
				Schema: "fak-safe-acceleration-receipt/1", Pair: pairIndex + 1, Arm: armName,
				RunID: runID, Issuer: "fixture-runner:" + armName, ProvenanceDigest: testSafeAccelerationDigest(100 + pairIndex*2),
				PublicRevision: c.PublicRevision, PrivateRevision: c.PrivateRevision, Identity: identity,
				Measurements: safeAccelerationMeasurements(*arm),
			}
			receiptBytes, err := json.Marshal(receipt)
			if err != nil {
				return c, nil, nil, err
			}
			arm.ReceiptDigest = safeAccelerationHash(receiptBytes)
			witness := safeAccelerationWitness{
				Schema: "fak-safe-acceleration-witness/1", Pair: pairIndex + 1, Arm: armName,
				RunID: runID, Issuer: "fixture-witness:" + armName, ProvenanceDigest: testSafeAccelerationDigest(200 + pairIndex*2),
				ReceiptDigest: arm.ReceiptDigest, AcceptedTaskIDs: append([]string(nil), arm.TaskIDs[:arm.AcceptedUnits]...), Violations: arm.Violations,
			}
			witnessBytes, err := json.Marshal(witness)
			if err != nil {
				return c, nil, nil, err
			}
			arm.AcceptanceWitnessDigest = safeAccelerationHash(witnessBytes)
			materials = append(materials, safeAccelerationResolvedMaterial{Receipt: receiptBytes, Witness: witnessBytes})
		}
	}
	capability, err := safeAccelerationAuthorizeResolved(c, materials)
	return c, capability, materials, err
}

func testSafeAccelerationArmPointer(c *SafeAccelerationCampaign, pairIndex int, arm string) (*SafeAccelerationArm, SafeAccelerationIdentity) {
	if arm == "baseline" {
		return &c.Pairs[pairIndex].Baseline, c.Baseline
	}
	return &c.Pairs[pairIndex].Candidate, c.Candidate
}

func validSafeAccelerationCampaign() SafeAccelerationCampaign {
	c := SafeAccelerationCampaign{
		Schema: SafeAccelerationSchema, PublicRevision: testSafeAccelerationRevision(1), PrivateRevision: testSafeAccelerationRevision(2),
		Baseline:  SafeAccelerationIdentity{Source: "fixture-baseline", ArtifactDigest: testSafeAccelerationDigest(1), ConfigurationDigest: testSafeAccelerationDigest(2), WorkloadDigest: testSafeAccelerationDigest(3)},
		Candidate: SafeAccelerationIdentity{Source: "fixture-candidate", ArtifactDigest: testSafeAccelerationDigest(4), ConfigurationDigest: testSafeAccelerationDigest(5), WorkloadDigest: testSafeAccelerationDigest(3)},
	}
	for i := 0; i < 20; i++ {
		c.Tasks = append(c.Tasks, SafeAccelerationTask{ID: fmt.Sprintf("fixture-task-%02d", i), Scope: "S1"})
	}
	ids := make([]string, len(c.Tasks))
	for i := range c.Tasks {
		ids[i] = c.Tasks[i].ID
	}
	for i := 0; i < 5; i++ {
		c.Pairs = append(c.Pairs, SafeAccelerationPair{Baseline: validSafeAccelerationArm(ids, 2000), Candidate: validSafeAccelerationArm(ids, 100)})
	}
	return c
}

func validSafeAccelerationArm(ids []string, wall int64) SafeAccelerationArm {
	arm := SafeAccelerationArm{
		TaskIDs: append([]string(nil), ids...), FullWallMS: wall, AcceptedUnits: len(ids),
		AcceptanceWitnessIndependent: true, Engine: "fak-native", Backend: "vulkan", Device: "strix-halo-fixture",
		Native: true, ResourceMetricsComplete: true, CostUSD: 1, Joules: 100, OperatorMinutes: 1,
		TaskLatencyMS: make([]int64, len(ids)),
	}
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

func testSafeAccelerationDigest(seed int) string   { return fmt.Sprintf("sha256:%064x", seed) }
func testSafeAccelerationRevision(seed int) string { return fmt.Sprintf("%040x", seed) }
