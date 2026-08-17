package harnesscreationstudy

import (
	"encoding/json"
	"strings"
	"testing"
)

func frozenStudy() Study {
	return Study{
		Schema: Schema,
		ID:     "study-1",
		Protocol: Protocol{TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Frozen: true, TenMinuteLimitSeconds: 600,
			AssistancePolicy: "task-card-and-help-only", FailuresInDenominator: true,
			Parity: MatchedStudySpec{Frozen: true, MinimumPairs: 2, MaxMedianElapsedRatio: 1.25, CounterbalancedOrder: true}},
		Baseline: Baseline{ID: "tuned-alt", Runnable: true, Tuned: true, Frozen: true, Evidence: "receipts/baseline.json"},
	}
}

func TestEvaluateKeepsFailuresAndExcludesCalibration(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "maintainer", ParticipantID: "maintainer-1", Track: "ten-minute", ParticipantClass: "maintainer-calibration", Outcome: "success", ElapsedSeconds: 10, Receipt: "receipts/m.json"},
		{ID: "builder-a", ParticipantID: "builder-a", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 540, Receipt: "receipts/a.json"},
		{ID: "builder-b", ParticipantID: "builder-b", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "timeout", ElapsedSeconds: 600, Receipt: "receipts/b.json"},
	}
	r := Evaluate(s)
	if r.Calibration != 1 || r.TenMinute.EligibleRuns != 2 || r.TenMinute.Successes != 1 || r.TenMinute.Failures != 1 || r.TenMinute.PassRate != .5 {
		t.Fatalf("unexpected fold: %+v", r)
	}
	if r.TenMinute.ClaimStatus != "not_yet" {
		t.Fatalf("one success must not unlock claim: %+v", r.TenMinute)
	}
}

func TestEvaluateSupportsOnlyCompleteIndependentEnvelopes(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "builder-a", ParticipantID: "builder-a", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 500, Receipt: "receipts/a.json"},
		{ID: "builder-b", ParticipantID: "builder-b", Track: "ten-minute", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 580, Receipt: "receipts/b.json"},
		{ID: "builder-c", ParticipantID: "builder-c", Track: "weekend", ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 7200, IndependentlyAuthored: true, ConformancePassed: true, Receipt: "receipts/c.json"},
	}
	r := Evaluate(s)
	if r.TenMinute.ClaimStatus != "supported" || r.Weekend.ClaimStatus != "supported" {
		t.Fatalf("complete envelopes rejected: %+v", r)
	}
	if r.TenMinute.MedianSuccessSeconds == nil || *r.TenMinute.MedianSuccessSeconds != 540 {
		t.Fatalf("median=%v", r.TenMinute.MedianSuccessSeconds)
	}
}

func TestParseRejectsTaskDigestOutsideProtocol(t *testing.T) {
	study := frozenStudy()
	study.Runs = []Run{{
		ID: "run-one", ParticipantID: "person-one", Track: "ten-minute", Arm: "fak", PairID: "pair-one",
		TaskDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		MachineID:  "machine-one", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder",
		Independent: true, OS: "linux", CPU: "x86_64", NetworkState: "online", CacheState: "empty",
		Outcome: "success", ElapsedSeconds: 100, Receipt: "receipt.json",
	}}
	raw, err := json.Marshal(study)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(raw); err == nil || !strings.Contains(err.Error(), "protocol.task_digest") {
		t.Fatalf("protocol digest drift accepted: %v", err)
	}
}

func TestParseFailsClosedOnPIIShapedIDsAndMutableProtocol(t *testing.T) {
	raw := `{"schema":"fak.harness-creation-study/v1alpha1","id":"study","protocol":{"frozen":false,"ten_minute_limit_seconds":600,"assistance_policy":"task-card-and-help-only","failures_in_denominator":true,"task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","parity":{"frozen":true,"minimum_pairs":2,"max_median_elapsed_ratio":1.25,"counterbalanced_order":true}},"baseline":{"id":"alt","runnable":true,"tuned":true,"frozen":true,"evidence":"x"},"runs":[]}`
	if _, err := Parse([]byte(raw)); err == nil || !strings.Contains(err.Error(), "protocol") {
		t.Fatalf("mutable protocol accepted: %v", err)
	}
	raw = strings.Replace(raw, `"frozen":false`, `"frozen":true`, 1)
	raw = strings.Replace(raw, `"runs":[]`, `"runs":[{"id":"r","participant_id":"person@example.com","track":"ten-minute","participant_class":"unfamiliar-builder","independent":true,"outcome":"failure","elapsed_seconds":600,"receipt":"x"}]`, 1)
	if _, err := Parse([]byte(raw)); err == nil || !strings.Contains(err.Error(), "privacy-safe") {
		t.Fatalf("PII-shaped id accepted: %v", err)
	}
}

func TestEvaluateReportsPairedParityWithoutCountingBaselineAsFak(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "a-fak", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a-fak.json"},
		{ID: "a-base", ParticipantID: "builder-a", Track: "ten-minute", Arm: "baseline", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 90, Receipt: "a-base.json"},
		{ID: "b-fak", ParticipantID: "builder-b", Track: "ten-minute", Arm: "fak", PairID: "pair-b", TaskDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", MachineID: "machine-b", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "baseline-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 120, Receipt: "b-fak.json"},
		{ID: "b-base", ParticipantID: "builder-b", Track: "ten-minute", Arm: "baseline", PairID: "pair-b", TaskDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", MachineID: "machine-b", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "baseline-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "b-base.json"},
	}
	r := Evaluate(s)
	if r.Parity.ClaimStatus != "supported" || r.Parity.CompletePairs != 2 || r.Parity.FakSuccesses != 2 || r.Parity.BaselineSuccesses != 2 {
		t.Fatalf("paired parity rejected: %+v", r.Parity)
	}
	if r.TenMinute.EligibleRuns != 2 || r.TenMinute.Successes != 2 {
		t.Fatalf("baseline arms contaminated fak claim: %+v", r.TenMinute)
	}
}

func TestEvaluateParityKeepsMissingAndFailedArmsVisible(t *testing.T) {
	s := frozenStudy()
	s.Protocol.Parity.MinimumPairs = 1
	s.Runs = []Run{
		{ID: "a-fak", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "failure", ElapsedSeconds: 600, Receipt: "a-fak.json"},
		{ID: "a-base", ParticipantID: "builder-a", Track: "ten-minute", Arm: "baseline", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a-base.json"},
		{ID: "b-fak", ParticipantID: "builder-b", Track: "ten-minute", Arm: "fak", PairID: "pair-b", TaskDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", MachineID: "machine-b", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "baseline-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "b-fak.json"},
	}
	r := Evaluate(s)
	if r.Parity.ClaimStatus != "refuted" || r.Parity.CompletePairs != 1 || r.Parity.IncompletePairs != 1 || r.Parity.FakSuccesses != 0 || r.Parity.BaselineSuccesses != 1 {
		t.Fatalf("failure or missing arm hidden: %+v", r.Parity)
	}
}

func TestEvaluateParityRefutesElapsedRatioOutsideFrozenBound(t *testing.T) {
	s := frozenStudy()
	s.Protocol.Parity.MinimumPairs = 1
	s.Runs = []Run{
		{ID: "a-fak", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 200, Receipt: "a-fak.json"},
		{ID: "a-base", ParticipantID: "builder-a", Track: "ten-minute", Arm: "baseline", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a-base.json"},
	}
	r := Evaluate(s)
	if r.Parity.ClaimStatus != "refuted" || r.Parity.MedianElapsedRatio == nil || *r.Parity.MedianElapsedRatio != 2 {
		t.Fatalf("slow fak arm not refuted: %+v", r.Parity)
	}
}

func TestParseRejectsDuplicateOrUnknownPairArms(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "a", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a.json"},
		{ID: "b", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "b.json"},
	}
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = Parse(raw); err == nil || !strings.Contains(err.Error(), "duplicate pair arm") {
		t.Fatalf("duplicate arm accepted: %v", err)
	}
	s.Runs[1].PairID, s.Runs[1].Arm = "pair-b", "other"
	raw, err = json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = Parse(raw); err == nil || !strings.Contains(err.Error(), "unknown arm") {
		t.Fatalf("unknown arm accepted: %v", err)
	}
}

func TestParseRejectsPairSpanningParticipants(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "a-fak", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a.json"},
		{ID: "b-base", ParticipantID: "builder-b", Track: "ten-minute", Arm: "baseline", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 90, Receipt: "b.json"},
	}
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = Parse(raw); err == nil || !strings.Contains(err.Error(), "spans participants") {
		t.Fatalf("cross-participant pair accepted: %v", err)
	}
}

func TestEvaluateParityRequiresCounterbalancedCompletePairs(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "a-fak", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a-fak.json"},
		{ID: "a-base", ParticipantID: "builder-a", Track: "ten-minute", Arm: "baseline", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a-base.json"},
		{ID: "b-fak", ParticipantID: "builder-b", Track: "ten-minute", Arm: "fak", PairID: "pair-b", TaskDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", MachineID: "machine-b", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "b-fak.json"},
		{ID: "b-base", ParticipantID: "builder-b", Track: "ten-minute", Arm: "baseline", PairID: "pair-b", TaskDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", MachineID: "machine-b", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 2, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "b-base.json"},
	}
	r := Evaluate(s)
	if r.Parity.ClaimStatus != "not_yet" || r.Parity.FakFirstPairs != 2 || r.Parity.BaselineFirstPairs != 0 {
		t.Fatalf("same-order evidence supported parity: %+v", r.Parity)
	}
}

func TestParseRejectsInconsistentPairOrderAndPosition(t *testing.T) {
	s := frozenStudy()
	s.Runs = []Run{
		{ID: "a-fak", ParticipantID: "builder-a", Track: "ten-minute", Arm: "fak", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "fak-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "a.json"},
		{ID: "a-base", ParticipantID: "builder-a", Track: "ten-minute", Arm: "baseline", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", PairOrder: "baseline-first", ArmPosition: 1, ParticipantClass: "unfamiliar-builder", Independent: true, Outcome: "success", ElapsedSeconds: 100, Receipt: "b.json"},
	}
	raw, _ := json.Marshal(s)
	if _, err := Parse(raw); err == nil || !strings.Contains(err.Error(), "conflicting order") {
		t.Fatalf("conflicting pair order accepted: %v", err)
	}
	s.Runs[1].PairOrder, s.Runs[1].ArmPosition = "fak-first", 1
	raw, _ = json.Marshal(s)
	if _, err := Parse(raw); err == nil || !strings.Contains(err.Error(), "arm_position") {
		t.Fatalf("wrong arm position accepted: %v", err)
	}
}

func TestParseRejectsPairedComparisonEnvelopeDrift(t *testing.T) {
	s := frozenStudy()
	base := Run{ParticipantID: "builder-a", Track: "ten-minute", PairID: "pair-a", TaskDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MachineID: "machine-a", PairOrder: "fak-first", ParticipantClass: "unfamiliar-builder", Independent: true, OS: "linux", CPU: "amd64", NetworkState: "online", CacheState: "empty", Outcome: "success", ElapsedSeconds: 100}
	fak, baseline := base, base
	fak.ID, fak.Arm, fak.ArmPosition, fak.Receipt = "fak", "fak", 1, "fak.json"
	baseline.ID, baseline.Arm, baseline.ArmPosition, baseline.Receipt = "baseline", "baseline", 2, "baseline.json"
	baseline.CacheState = "warm"
	s.Runs = []Run{fak, baseline}
	raw, _ := json.Marshal(s)
	if _, err := Parse(raw); err == nil || !strings.Contains(err.Error(), "envelope") {
		t.Fatalf("envelope drift accepted: %v", err)
	}
}
