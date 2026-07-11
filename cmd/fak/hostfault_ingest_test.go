package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

// TestHostFaultEventsFromWinRecords pins the live-producer mapping against the
// real event shapes this host emits: a WindowsUpdateClient Event 20 install
// failure (code + package derived), a GPU live-kernel WER 1001 (category
// derived), a MoUso orchestrator fault, and an unrelated crash dropped
// fail-closed (never fabricated into a host-fault row).
func TestHostFaultEventsFromWinRecords(t *testing.T) {
	recs := []hostfault.WinFaultRecord{
		{
			Provider: "Microsoft-Windows-WindowsUpdateClient", ID: 20, TimeMS: 1_700_000_000_000,
			Message: "Installation Failure: Windows failed to install the following update with error 0x80073D02: 9NKSQGP7F2NH-5319275A.WhatsAppDesktop.",
		},
		{
			Provider: "Windows Error Reporting", ID: 1001, TimeMS: 0,
			Message: "Fault bucket , type 0\nEvent Name: LiveKernelEvent\nAttached files:\n\\\\?\\C:\\WINDOWS\\LiveKernelReports\\AMD_WATCHDOG\\AMD_WATCHDOG-20260624-1411.dmp",
		},
		{
			Provider: "Windows Error Reporting", ID: 1001, TimeMS: 1_700_000_000_002,
			App:     "Update;MoUpdateOrchestratorDeviceScan-MoUsoCoreWorker-SearchForAllUpdatesWithUpdateOptionsAsync",
			Message: "Fault bucket , type 0\nEvent Name: AppCrash",
		},
		{
			Provider: "Windows Error Reporting", ID: 1001, TimeMS: 1_700_000_000_003,
			App: "msedge.exe", Message: "Fault bucket 12345, type 4\nEvent Name: AppCrash",
		},
	}

	got := hostFaultEventsFromWinRecords(recs, 42)
	if len(got) != 3 {
		t.Fatalf("want 3 classified rows (unrelated dropped fail-closed), got %d", len(got))
	}

	// Row 0: WU install failure — code + package derived, source labelled.
	if got[0].Class != hostfault.HostWUInstallFailure {
		t.Errorf("row0 class = %q, want WU_INSTALL_FAILURE", got[0].Class)
	}
	if got[0].Code != "0x80073D02" {
		t.Errorf("row0 code = %q, want 0x80073D02", got[0].Code)
	}
	if got[0].App != "9NKSQGP7F2NH-5319275A.WhatsAppDesktop" {
		t.Errorf("row0 app = %q, want the package identity", got[0].App)
	}
	if got[0].Source != "WindowsUpdateClient/20" {
		t.Errorf("row0 source = %q, want WindowsUpdateClient/20", got[0].Source)
	}

	// Row 1: GPU live-kernel — category derived, fallback time applied (was 0).
	if got[1].Class != hostfault.HostGPULiveKernel {
		t.Errorf("row1 class = %q, want GPU_LIVEKERNEL", got[1].Class)
	}
	if got[1].Code != "AMD_WATCHDOG" {
		t.Errorf("row1 code = %q, want AMD_WATCHDOG", got[1].Code)
	}
	if got[1].AtMS != 42 {
		t.Errorf("row1 at_ms = %d, want the fallback (record time was 0)", got[1].AtMS)
	}
	if got[1].Source != "Windows Error Reporting/1001" {
		t.Errorf("row1 source = %q, want Windows Error Reporting/1001", got[1].Source)
	}

	// Row 2: orchestrator fault — app identity carried through.
	if got[2].Class != hostfault.HostWUOrchestratorFault {
		t.Errorf("row2 class = %q, want WU_ORCHESTRATOR_FAULT", got[2].Class)
	}

	// Every emitted row must satisfy the durable schema so it round-trips.
	for i, ev := range got {
		if err := hostfault.ValidateHostFaultEvent(ev); err != nil {
			t.Errorf("row%d invalid: %v", i, err)
		}
	}
}

// TestHostFaultExtractionHelpers pins the pure field extractors.
func TestHostFaultExtractionHelpers(t *testing.T) {
	if got := extractHexCode("failed with error 0x80073D02: pkg"); got != "0x80073D02" {
		t.Errorf("extractHexCode = %q, want 0x80073D02", got)
	}
	if got := extractHexCode("no code here"); got != "" {
		t.Errorf("extractHexCode = %q, want empty", got)
	}
	if got := extractLiveKernelCategory(`\\?\C:\WINDOWS\LiveKernelReports\AMD_REPORT_UM\x.dmp`); got != "AMD_REPORT_UM" {
		t.Errorf("extractLiveKernelCategory = %q, want AMD_REPORT_UM", got)
	}
	if got := extractWUPackage("...error 0x80073D02: 9MSSGKG348SP-MicrosoftWindows.Client.WebExperience."); got != "9MSSGKG348SP-MicrosoftWindows.Client.WebExperience" {
		t.Errorf("extractWUPackage = %q, want the package (trailing period trimmed)", got)
	}
}
