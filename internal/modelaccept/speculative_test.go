package modelaccept

import (
	"strings"
	"testing"
)

func TestEvaluateSpeculativeFrontierPricesLoadabilityAndRejectsUnsafeArms(t *testing.T) {
	in := SpeculativeFrontier{RuntimeRevision: "sglang@1cf2b8c54d81802abc15dcf23a29b9cc687bc01e", Checkpoint: "Qwen/Qwen3.8-27B", HardwareTopology: "1xRTX5090", Arms: []SpeculativeArm{
		{Name: "none", Mode: "none", Loadable: true, FailureBounded: true, StartupSeconds: 10, RunSeconds: 100, Requests: 10, PeakMemoryBytes: 20, EnergyJoules: 100, TaskQuality: 0.90},
		{Name: "eagle", Mode: "eagle", Loadable: true, FailureBounded: true, StartupSeconds: 20, RunSeconds: 60, Requests: 10, PeakMemoryBytes: 24, EnergyJoules: 80, TaskQuality: 0.90, AcceptedTokensPerStep: 2.7},
		{Name: "mtp-lockup", Mode: "mtp", Loadable: true, FailureBounded: false, StartupSeconds: 12, RunSeconds: 50, Requests: 10, PeakMemoryBytes: 23, EnergyJoules: 70, TaskQuality: 0.91, AcceptedTokensPerStep: 2.9},
	}}
	got, err := EvaluateSpeculativeFrontier(in)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Pass || got.PromotedArm != "eagle" {
		t.Fatalf("got %+v", got)
	}
	if got.Arms[1].NetSpeedup != 1.375 || got.Arms[1].PeakMemoryDelta != 4 || got.Arms[1].EnergyPerRequest != 8 {
		t.Fatalf("startup was not priced: %+v", got.Arms[1])
	}
	if got.Arms[2].Disposition != SpeculativeReject || !strings.Contains(strings.Join(got.Arms[2].Reasons, " "), "unbounded_failure") {
		t.Fatalf("unsafe loadable arm promoted: %+v", got.Arms[2])
	}
}

func TestEvaluateSpeculativeFrontierFailsClosedOnQualityAndPins(t *testing.T) {
	base := SpeculativeFrontier{RuntimeRevision: "runtime@rev", Checkpoint: "Qwen/Qwen3.8-27B", HardwareTopology: "gpu", Arms: []SpeculativeArm{
		{Name: "none", Mode: "none", Loadable: true, FailureBounded: true, RunSeconds: 100, Requests: 10, PeakMemoryBytes: 20, EnergyJoules: 100, TaskQuality: .9},
		{Name: "fast-bad", Mode: "eagle", Loadable: true, FailureBounded: true, RunSeconds: 40, Requests: 10, PeakMemoryBytes: 24, EnergyJoules: 80, TaskQuality: .89},
		{Name: "not-loaded", Mode: "dspark", FailureBounded: true, RunSeconds: 30, Requests: 10, PeakMemoryBytes: 22, EnergyJoules: 70, TaskQuality: .9},
	}}
	got, err := EvaluateSpeculativeFrontier(base)
	if err != nil {
		t.Fatal(err)
	}
	if got.Pass || got.PromotedArm != "" {
		t.Fatalf("quality/load failure must hold: %+v", got)
	}
	if got.Arms[1].Disposition != SpeculativeReject || got.Arms[2].Disposition != SpeculativeReject {
		t.Fatalf("arms must reject: %+v", got.Arms)
	}

	base.Arms = base.Arms[:2]
	if _, err := EvaluateSpeculativeFrontier(base); err == nil || !strings.Contains(err.Error(), "at least three") {
		t.Fatalf("want bounded matrix error, got %v", err)
	}
	base.Arms = append(base.Arms, SpeculativeArm{Name: "other", Mode: "dspark", RuntimeRevision: "different", Loadable: true, FailureBounded: true, RunSeconds: 30, Requests: 10, PeakMemoryBytes: 22, EnergyJoules: 70, TaskQuality: .9})
	if _, err := EvaluateSpeculativeFrontier(base); err == nil || !strings.Contains(err.Error(), "runtime revision") {
		t.Fatalf("want pin mismatch, got %v", err)
	}
}
