package hostfault

import (
	"bytes"
	"strings"
	"testing"
)

// TestClassifyHostFault pins the closed classifier against the real event-log
// shapes the #2170 host audit surfaced on this Windows host, and confirms the
// fail-closed drops: an unrelated app crash and a NON-GPU live-kernel event both
// return ok=false rather than being fabricated into a host-fault row.
func TestClassifyHostFault(t *testing.T) {
	cases := []struct {
		name string
		rec  WinFaultRecord
		want HostFaultClass
		ok   bool
	}{
		{
			name: "wu install failure",
			rec: WinFaultRecord{
				Provider: "Microsoft-Windows-WindowsUpdateClient", ID: 20,
				Message: "Installation Failure: Windows failed to install the following update with error 0x80073D02: 9MSSGKG348SP-MicrosoftWindows.Client.WebExperience.",
			},
			want: HostWUInstallFailure, ok: true,
		},
		{
			name: "wu orchestrator fault via app identity",
			rec: WinFaultRecord{
				Provider: "Windows Error Reporting", ID: 1001,
				App:     "Update;MoUpdateOrchestratorDeviceScan-MoUsoCoreWorker-SearchForAllUpdatesWithUpdateOptionsAsync",
				Message: "Fault bucket , type 0\nEvent Name: AppCrash",
			},
			want: HostWUOrchestratorFault, ok: true,
		},
		{
			name: "gpu live-kernel (AMD watchdog path)",
			rec: WinFaultRecord{
				Provider: "Windows Error Reporting", ID: 1001,
				Message: "Fault bucket , type 0\nEvent Name: LiveKernelEvent\nAttached files:\n\\\\?\\C:\\WINDOWS\\LiveKernelReports\\AMD_WATCHDOG\\AMD_WATCHDOG-20260624-1411.dmp",
			},
			want: HostGPULiveKernel, ok: true,
		},
		{
			name: "app-termination hang dump",
			rec: WinFaultRecord{
				Provider: "Windows Error Reporting", ID: 1001,
				Message: "Fault bucket , type 0\nEvent Name: AppTermFailureEvent\nResponse: Not available",
			},
			want: HostAppTermFailure, ok: true,
		},
		{
			name: "unrelated app crash dropped fail-closed",
			rec: WinFaultRecord{
				Provider: "Windows Error Reporting", ID: 1001,
				App:     "msedge.exe",
				Message: "Fault bucket 12345, type 4\nEvent Name: AppCrash\nP1: msedge.exe",
			},
			ok: false,
		},
		{
			name: "non-GPU live-kernel dropped fail-closed",
			rec: WinFaultRecord{
				Provider: "Windows Error Reporting", ID: 1001,
				Message: "Fault bucket , type 0\nEvent Name: LiveKernelEvent\nP1: 1a1", // no GPU signal
			},
			ok: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ClassifyHostFault(tc.rec)
			if ok != tc.ok {
				t.Fatalf("ok = %v, want %v (class=%q)", ok, tc.ok, got)
			}
			if ok && got != tc.want {
				t.Fatalf("class = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestHostFaultEventRoundTrip confirms an emitted row survives
// AppendHostFaultEvent -> ParseHostFaultEvents unchanged, and that parsing is
// fail-closed on an unknown class and an unknown field.
func TestHostFaultEventRoundTrip(t *testing.T) {
	events := []HostFaultEvent{
		{Class: HostWUInstallFailure, AtMS: 1_700_000_000_000, Source: "WindowsUpdateClient/20", App: "WebExperience", Code: "0x80073D02"},
		{Class: HostGPULiveKernel, AtMS: 1_700_000_000_001, Source: "Windows Error Reporting/1001", Code: "141"},
	}
	var buf bytes.Buffer
	for _, ev := range events {
		if err := AppendHostFaultEvent(&buf, ev); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	got, err := ParseHostFaultEvents(&buf)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != len(events) {
		t.Fatalf("round-trip rows = %d, want %d", len(got), len(events))
	}
	if got[0].Class != HostWUInstallFailure || got[0].Code != "0x80073D02" || got[0].Schema != HostFaultEventSchema {
		t.Errorf("row0 = %+v, want WU install failure with schema stamped", got[0])
	}

	// Unknown class refuses the whole report.
	if _, err := ParseHostFaultEvents(strings.NewReader(`{"class":"NOPE","at_unix_ms":1}`)); err == nil {
		t.Error("parse accepted an unknown class")
	}
	// Unknown field refuses the whole report (drift guard).
	if _, err := ParseHostFaultEvents(strings.NewReader(`{"class":"GPU_LIVEKERNEL","at_unix_ms":1,"mystery":true}`)); err == nil {
		t.Error("parse accepted an unknown field")
	}
}

// TestValidateHostFaultEvent pins the required-field guards.
func TestValidateHostFaultEvent(t *testing.T) {
	if err := ValidateHostFaultEvent(HostFaultEvent{Class: HostAppTermFailure, AtMS: 1}); err != nil {
		t.Errorf("valid row rejected: %v", err)
	}
	if err := ValidateHostFaultEvent(HostFaultEvent{Class: "BOGUS", AtMS: 1}); err == nil {
		t.Error("unknown class accepted")
	}
	if err := ValidateHostFaultEvent(HostFaultEvent{Class: HostAppTermFailure, AtMS: 0}); err == nil {
		t.Error("non-positive at_unix_ms accepted")
	}
	if err := ValidateHostFaultEvent(HostFaultEvent{Class: HostAppTermFailure, AtMS: 1, Detail: strings.Repeat("x", hostFaultDetailLimit+1)}); err == nil {
		t.Error("over-long detail accepted")
	}
}

// TestHostFaultReportFromEvents folds counts by class/source/app.
func TestHostFaultReportFromEvents(t *testing.T) {
	events := []HostFaultEvent{
		{Class: HostGPULiveKernel, AtMS: 1, Source: "Windows Error Reporting/1001"},
		{Class: HostGPULiveKernel, AtMS: 2, Source: "Windows Error Reporting/1001"},
		{Class: HostWUInstallFailure, AtMS: 3, Source: "WindowsUpdateClient/20", App: "WhatsAppDesktop"},
	}
	rep := HostFaultReportFromEvents(events)
	if rep.Rows != 3 {
		t.Fatalf("rows = %d, want 3", rep.Rows)
	}
	if rep.Counts.ByClass["GPU_LIVEKERNEL"] != 2 {
		t.Errorf("GPU_LIVEKERNEL count = %d, want 2", rep.Counts.ByClass["GPU_LIVEKERNEL"])
	}
	if rep.Counts.ByClass["WU_INSTALL_FAILURE"] != 1 {
		t.Errorf("WU_INSTALL_FAILURE count = %d, want 1", rep.Counts.ByClass["WU_INSTALL_FAILURE"])
	}
	if rep.Counts.ByApp["WhatsAppDesktop"] != 1 {
		t.Errorf("app count = %d, want 1", rep.Counts.ByApp["WhatsAppDesktop"])
	}
}
