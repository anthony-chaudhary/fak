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
			name: "gpu xid fault with pci-qualified code",
			rec: WinFaultRecord{
				Provider: "nvlddmkm", ID: 14,
				Message: "Xid (PCI:0000:01:00.0): 79, pid=1234, GPU has fallen off the bus",
			},
			want: HostGpuXID, ok: true,
		},
		{
			name: "gpu xid fault from bare code token",
			rec: WinFaultRecord{
				Provider: "Windows Error Reporting", ID: 1001,
				App:     "nvlddmkm.sys",
				Message: "NVIDIA reported XID 48 on device 0",
			},
			want: HostGpuXID, ok: true,
		},
		{
			name: "xid-shaped word without a standalone token dropped fail-closed",
			rec: WinFaultRecord{
				Provider: "Microsoft-Windows-Kernel-PnP", ID: 219,
				Message: "Driver oxidation warning: rapid power state transition",
			},
			ok: false,
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

	// Pinned XID vocabulary (DIRECT-PORT of the dcgm-exporter table at
	// 4e663a526c46). These are the VALUE assertions: an operator receipt's fault
	// text is the closed contract, so a reworded or re-sorted row must fail here
	// rather than silently change what an operator reads.
	pinnedXIDText := map[int]string{
		0:   "No Error",
		48:  "Double Bit ECC Error",
		63:  "ECC page retirement or row remapping recording event",
		64:  "ECC page retirement or row remapper recording failure",
		74:  "NVLINK Error",
		79:  "GPU has fallen off the bus",
		119: "GSP RPC Timeout",
		120: "GSP Error",
		144: "NVLINK: Saw Error",
		145: "NVLINK: RLW Error",
		146: "NVLINK: TLW Error",
		147: "NVLINK: TREX Error",
		148: "NVLINK: NVLPW_CTRL Error",
		149: "NVLINK: NETIR Error",
		150: "NVLINK: MSE Error",
		172: "Uncorrectable SRAM Error",
	}
	for code, want := range pinnedXIDText {
		got, ok := XIDOperatorText(code)
		if !ok {
			t.Errorf("XIDOperatorText(%d) not ok, want %q", code, want)
			continue
		}
		if got != want {
			t.Errorf("XIDOperatorText(%d) = %q, want %q", code, got, want)
		}
	}
	// Fail-closed outside the pinned catalog: no text, and no ok.
	for _, code := range []int{-1, -79, 173, 99999} {
		if txt, ok := XIDOperatorText(code); ok {
			t.Errorf("XIDOperatorText(%d) = (%q, true), want fail-closed", code, txt)
		}
	}
	// Extraction: the code comes out of the same shapes the classifier accepts.
	for _, tc := range []struct {
		name string
		rec  WinFaultRecord
		want int
		ok   bool
	}{
		{"pci-qualified", WinFaultRecord{Message: "Xid (PCI:0000:01:00.0): 79"}, 79, true},
		{"bare token", WinFaultRecord{Message: "NVIDIA reported XID 48"}, 48, true},
		{"app identity", WinFaultRecord{App: "Xid: 120"}, 120, true},
		{"standalone word without code", WinFaultRecord{Message: "xid encountered"}, 0, false},
		{"no xid at all", WinFaultRecord{Message: "LiveKernelEvent bucket 141"}, 0, false},
	} {
		got, ok := ExtractXID(tc.rec)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Errorf("ExtractXID(%s) = (%d, %v), want (%d, %v)", tc.name, got, ok, tc.want, tc.ok)
		}
	}
	// A code beyond the pinned catalog is still a witnessed XID fault: the event
	// is classified, its operator text stays unknown.
	if got, ok := ClassifyHostFault(WinFaultRecord{Provider: "nvlddmkm", Message: "Xid 99999"}); !ok || got != HostGpuXID {
		t.Errorf("uncatalogued XID classified (%q, %v), want (GPU_XID, true)", got, ok)
	} else if _, textOK := XIDOperatorText(99999); textOK {
		t.Error("uncatalogued XID produced operator text; want fail-closed")
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
