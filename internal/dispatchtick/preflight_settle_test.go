package dispatchtick

import (
	"strings"
	"testing"
)

func settledPreflightInput(alive, target int) PreflightInput {
	return PreflightInput{MaxWorkers: 10, Host: HostCheck{Safe: true}, Account: AccountCheck{Available: true, Tag: "a", Tier: 1}, Kernel: KernelCheck{Alive: IntPtr(alive), Target: IntPtr(target)}, Seat: SeatCheck{Total: IntPtr(10), Free: IntPtr(8), Leased: IntPtr(alive)}, Resources: HostResources{}, OSWorkerProcs: alive}
}

func TestEvaluatePreflightObservesScaleUpInFlight(t *testing.T) {
	in := settledPreflightInput(2, 5)
	in.WorkerFloor = 7 // a second proposed scale action while target 5 is still converging
	got := EvaluatePreflight(in)
	if got.OK || got.Verdict != PreflightObserveSettling || !strings.Contains(got.Reason, "2 alive, target 5") {
		t.Fatalf("got=%+v", got)
	}
}
func TestEvaluatePreflightObservesScaleDownInFlight(t *testing.T) {
	in := settledPreflightInput(5, 2)
	in.MaxWorkers = 10
	got := EvaluatePreflight(in)
	if got.OK || got.Verdict != PreflightObserveSettling || !strings.Contains(got.Reason, "5 alive, target 2") {
		t.Fatalf("got=%+v", got)
	}
}
func TestEvaluatePreflightSettledTargetAllowsNormalClassification(t *testing.T) {
	in := settledPreflightInput(2, 2)
	got := EvaluatePreflight(in)
	if got.Verdict == PreflightObserveSettling {
		t.Fatalf("settled target was still observed: %+v", got)
	}
	if got.Verdict != PreflightRefuseAtCap {
		t.Fatalf("got=%+v, want ordinary at-cap classification", got)
	}
}
