package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestDispatchHostSpaceAdmitAboveBoundary(t *testing.T) {
	const (
		reserve   = int64(2 << 30)
		perWorker = int64(1 << 30)
		requested = 3
	)
	free := reserve + requested*perWorker
	got := dispatchHostSpaceAdmit(t.TempDir(), requested, reserve, perWorker, "test thresholds", func(string) (int64, int64, bool) {
		return 100 << 30, free, true
	})
	if !got.OK || got.AdmittedWorkers != requested {
		t.Fatalf("admission = %+v, want all %d workers admitted", got, requested)
	}
	if got.PredictedDemandBytes != requested*perWorker || got.AdmittedDemandBytes != requested*perWorker {
		t.Fatalf("demand receipt = predicted %d admitted %d, want %d", got.PredictedDemandBytes, got.AdmittedDemandBytes, requested*perWorker)
	}
	if got.FreeBytes != free || got.ReserveBytes != reserve || got.ThresholdProvenance != "test thresholds" {
		t.Fatalf("receipt lost threshold provenance: %+v", got)
	}
}

func TestDispatchHostSpaceAdmitBelowBoundaryCapsWorkers(t *testing.T) {
	const (
		reserve   = int64(2 << 30)
		perWorker = int64(1 << 30)
		requested = 4
	)
	got := dispatchHostSpaceAdmit(t.TempDir(), requested, reserve, perWorker, "test thresholds", func(string) (int64, int64, bool) {
		return 100 << 30, reserve + 2*perWorker, true
	})
	if !got.OK || got.AdmittedWorkers != 2 {
		t.Fatalf("admission = %+v, want safe partial admission of 2", got)
	}
	if got.ReasonCode != "HOST_FREE_SPACE_PARTIAL" || got.AdmittedDemandBytes != 2*perWorker {
		t.Fatalf("partial receipt = %+v, want typed partial reason and bounded admitted demand", got)
	}
	if got.Remediation != dispatchHostSpaceRemediation {
		t.Fatalf("remediation = %q, want %q", got.Remediation, dispatchHostSpaceRemediation)
	}
}

func TestDispatchHostSpaceAdmitBelowReserveRefuses(t *testing.T) {
	const reserve = int64(2 << 30)
	got := dispatchHostSpaceAdmit(t.TempDir(), 1, reserve, 1<<30, "test thresholds", func(string) (int64, int64, bool) {
		return 100 << 30, reserve, true
	})
	if got.OK || got.AdmittedWorkers != 0 || got.ReasonCode != "HOST_FREE_SPACE_BUDGET" {
		t.Fatalf("admission = %+v, want fail-closed zero-worker refusal", got)
	}
}

func TestDispatchHostSpaceAdmitProbeErrorFailsClosed(t *testing.T) {
	target := filepath.Join(t.TempDir(), "not-created", "worker")
	got := dispatchHostSpaceAdmit(target, 2, 2<<30, 1<<30, "test thresholds", func(path string) (int64, int64, bool) {
		if path == target {
			t.Fatalf("probe path = nonexistent target %q; want existing filesystem ancestor", path)
		}
		return 0, 0, false
	})
	if got.OK || got.AdmittedWorkers != 0 || got.MeasurementKnown || got.ReasonCode != "HOST_FREE_SPACE_UNKNOWN" {
		t.Fatalf("admission = %+v, want unknown measurement to fail closed", got)
	}
	if got.Remediation != dispatchHostSpaceRemediation {
		t.Fatalf("remediation = %q, want %q", got.Remediation, dispatchHostSpaceRemediation)
	}
}

func TestDispatchWaveDryRunEmitsHostSpaceRefusalReceipt(t *testing.T) {
	oldProbe := dispatchHostSpaceProbeFn
	dispatchHostSpaceProbeFn = func(string) (int64, int64, bool) { return 0, 0, false }
	t.Cleanup(func() { dispatchHostSpaceProbeFn = oldProbe })

	var stdout, stderr bytes.Buffer
	code := runDispatchWave(&stdout, &stderr, []string{
		"--workspace", t.TempDir(),
		"--count", "3",
		"--host-disk-reserve-bytes", "4096",
		"--host-disk-per-worker-bytes", "1024",
		"--json",
	})
	if code == 0 {
		t.Fatalf("runDispatchWave code = 0, want refusal; stdout=%s stderr=%s", stdout.String(), stderr.String())
	}
	var rec struct {
		Live      bool                       `json:"live"`
		Requested int                        `json:"requested"`
		Admission dispatchHostSpaceAdmission `json:"host_space_admission"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &rec); err != nil {
		t.Fatalf("decode dry-run receipt: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	if rec.Live || rec.Requested != 3 {
		t.Fatalf("dry-run header = live %v requested %d, want false/3", rec.Live, rec.Requested)
	}
	if rec.Admission.OK || rec.Admission.ReasonCode != "HOST_FREE_SPACE_UNKNOWN" || rec.Admission.RequestedWorkers != 3 {
		t.Fatalf("host admission receipt = %+v", rec.Admission)
	}
	if rec.Admission.ReserveBytes != 4096 || rec.Admission.PerWorkerBytes != 1024 || rec.Admission.PredictedDemandBytes != 3072 {
		t.Fatalf("host admission arithmetic = %+v", rec.Admission)
	}
	if rec.Admission.ThresholdProvenance != "explicit flags" || rec.Admission.Remediation != dispatchHostSpaceRemediation {
		t.Fatalf("host admission provenance/remediation = %+v", rec.Admission)
	}
}
