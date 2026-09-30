package macbench

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestGpuEngagementGradeSpine(t *testing.T) {
	for _, tc := range []struct {
		name            string
		cpu, gpu, floor float64
		wantOK          bool
	}{
		{"CPU dominant device idle", 90, 0, 20, false},
		{"CPU dominant device below floor", 90, 19, 20, false},
		{"CPU dominant device exactly at floor", 90, 20, 20, true},
		{"CPU dominant device above floor", 90, 40, 20, true},
		{"device dominant below floor", 3, 4, 20, true},
		{"equal low utilization", 4, 4, 20, true},
		{"known zero pair", 0, 0, 20, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			verdict := GradeEngagement(tc.cpu, tc.gpu, tc.floor)
			if verdict == nil {
				t.Fatal("armed engagement floor requires structural verdict")
			}
			if verdict.OK != tc.wantOK || verdict.CPUDominant != (tc.cpu > tc.gpu) || verdict.DeviceOK != (tc.gpu >= tc.floor) {
				t.Fatalf("known samples CPU=%g GPU=%g floor=%g: got %+v, want OK=%v", tc.cpu, tc.gpu, tc.floor, verdict, tc.wantOK)
			}
		})
	}
	// Disabled gating must not add a verdict to the existing serialized report,
	// even when the available sample would fail an armed gate.
	report := Report{Schema: Schema, GPUUtilPct: 0}
	before, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	report.Engagement = GradeEngagement(100, 0, 0)
	if report.Engagement != nil {
		t.Fatal("zero floor must disable engagement verdict")
	}
	after, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("zero floor changed report bytes:\nbefore %s\nafter %s", before, after)
	}
}

func TestGpuEngagementObservedRequiresBoundPresentSamples(t *testing.T) {
	for _, tc := range []struct {
		name                                string
		cpuPresent, gpuPresent, engineBound bool
	}{
		{"missing CPU", false, true, true},
		{"missing device", true, false, true},
		{"both absent", false, false, true},
		{"cross engine pair", true, true, false},
		{"unbound absent pair", false, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A plausible, fully engaged numeric pair cannot substitute for
			// evidence that both samples exist and measure the benchmark engine.
			v := GradeEngagementObserved(5, 80, 20, tc.cpuPresent, tc.gpuPresent, tc.engineBound)
			if v == nil || v.Status != "unknown" || v.OK || v.Reason == "" {
				t.Fatalf("missing or unbound samples must be unknown and non-success: %+v", v)
			}
		})
	}
	for _, pair := range [][2]float64{{90, 0}, {90, 20}, {3, 4}, {0, 0}} {
		known := GradeEngagement(pair[0], pair[1], 20)
		observed := GradeEngagementObserved(pair[0], pair[1], 20, true, true, true)
		wantStatus := "pass"
		if !known.OK {
			wantStatus = "fail"
		}
		if observed == nil || observed.Status != wantStatus || observed.OK != known.OK || observed.CPUDominant != known.CPUDominant || observed.DeviceOK != known.DeviceOK {
			t.Fatalf("bound present pair %v must retain known-sample truth table: observed=%+v known=%+v", pair, observed, known)
		}
	}
	for _, bound := range []bool{false, true} {
		if v := GradeEngagementObserved(100, 0, 0, false, false, bound); v != nil {
			t.Fatalf("zero floor must disable gate even when telemetry unavailable: %+v", v)
		}
	}
}
