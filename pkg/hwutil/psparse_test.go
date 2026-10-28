package hwutil

// Fake-probe tests for the PowerShell output contract: pinned sectioned
// command output, asserting the normalized rows the Windows collector must
// emit, including the null-on-error edges for absent WMI fields.

import (
	"testing"
)

// fakePSOutput mirrors what the Windows collector's inline powershell.exe
// expression prints: section markers plus ConvertTo-Json payloads, where WMI
// device paths arrive JSON-escaped (each backslash doubled).
const fakePSOutput = "DRIVE\r\n[{\"DeviceID\": \"\\\\\\\\.\\\\PHYSICALDRIVE0\", \"BusType\": \"NVMe\", \"Size\": 1099511627776}]\r\nPHYS\r\n[{\"Name\": \"\\\\\\\\.\\\\PhysicalDrive0\", \"Index\": 0}]\r\nADAPTER\r\n[{\"InterfaceIndex\": 12, \"Name\": \"Ethernet\", \"Speed\": 10000000000}]\r\nPERF\r\n[{\"AdapterDisplayName\": \"Ethernet\", \"ReceivedBytesPersec\": 1000000, \"SentBytesPersec\": 500000, \"PacketsReceivedPersec\": 800, \"PacketsSentPersec\": 400}]\r\n"

func TestPSSectionsSplit(t *testing.T) {
	sec := psSections([]byte(fakePSOutput))
	for _, want := range []string{"DRIVE", "PHYS", "ADAPTER", "PERF"} {
		if sec[want] == "" {
			t.Fatalf("section %s missing: %v", want, sec)
		}
	}
	if len(sec) != 4 {
		t.Fatalf("unexpected sections: %v", sec)
	}
}

func TestDecodePSRowsArrayAndObject(t *testing.T) {
	rows := decodePSRows(`[{"a": 1}, {"a": 2}]`)
	if len(rows) != 2 || rows[0]["a"] != float64(1) {
		t.Fatalf("array payload = %v", rows)
	}
	one := decodePSRows(`{"a": 3}`)
	if len(one) != 1 || one[0]["a"] != float64(3) {
		t.Fatalf("object payload = %v", one)
	}
	if got := decodePSRows(""); got != nil {
		t.Fatalf("empty payload = %v", got)
	}
}

func TestNumberFieldNullOnErrorEdges(t *testing.T) {
	m := map[string]any{
		"float":   float64(12.5),
		"strnum":  "300",
		"nullval": nil,
		"junk":    "abc",
	}
	if v, ok := numberField(m, "float"); !ok || v != 12.5 {
		t.Fatalf("float = %v ok=%v", v, ok)
	}
	if v, ok := numberField(m, "strnum"); !ok || v != 300 {
		t.Fatalf("string number = %v ok=%v", v, ok)
	}
	if _, ok := numberField(m, "nullval"); ok {
		t.Fatal("null field reported present")
	}
	if _, ok := numberField(m, "missing"); ok {
		t.Fatal("absent field reported present")
	}
	if _, ok := numberField(m, "junk"); ok {
		t.Fatal("non-numeric string accepted")
	}
}

func TestWindowsDiskKindMapping(t *testing.T) {
	cases := map[string]DiskKind{
		"NVMe": DiskNVMe, "nvme": DiskNVMe, "USB": DiskUSB, "IDE": DiskSATA,
		"SAS": DiskUnknown, "": DiskUnknown,
	}
	for in, want := range cases {
		if got := windowsDiskKind(in); got != want {
			t.Errorf("windowsDiskKind(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestDriveByNumberCaseInsensitive(t *testing.T) {
	drives := []map[string]any{
		{"DeviceID": `\\.\PHYSICALDRIVE0`, "BusType": "NVMe"},
		{"DeviceID": `\\.\PHYSICALDRIVE1`, "BusType": "USB"},
	}
	if d, ok := driveByNumber(drives, 1); !ok || stringField(d, "BusType") != "USB" {
		t.Fatalf("drive 1 = %v ok=%v", d, ok)
	}
	if _, ok := driveByNumber(drives, 2); ok {
		t.Fatal("absent drive matched")
	}
}

func TestSplitCounterPath(t *testing.T) {
	obj, inst, counter, ok := splitCounterPath(`\\host\physicaldisk(0 c:)\Disk Read Bytes/sec`)
	if !ok || obj != "physicaldisk" || inst != "0 c:" || counter != "disk read bytes/sec" {
		t.Fatalf("got (%q,%q,%q) ok=%v", obj, inst, counter, ok)
	}
	_, inst2, _, ok2 := splitCounterPath(`\\h\network interface(Abort Adapter _2)\bytes sent/sec`)
	if !ok2 || inst2 != "Abort Adapter _2" {
		t.Fatalf("net instance = %q ok=%v", inst2, ok2)
	}
	for _, bad := range []string{"", "no-backslashes", `\\h\plainobject\counter`, "x/y/z/w"} {
		if _, _, _, ok := splitCounterPath(bad); ok {
			t.Fatalf("malformed path accepted: %q", bad)
		}
	}
}

func TestCounterInstanceNumber(t *testing.T) {
	if n, ok := counterInstanceNumber("0 c:"); !ok || n != 0 {
		t.Fatalf("got %d ok=%v", n, ok)
	}
	if n, ok := counterInstanceNumber("12 D:"); !ok || n != 12 {
		t.Fatalf("got %d ok=%v", n, ok)
	}
	for _, bad := range []string{"", "_total", "c:"} {
		if _, ok := counterInstanceNumber(bad); ok {
			t.Fatalf("non-drive instance accepted: %q", bad)
		}
	}
}

// Two sample sets in order; only the LAST set survives per (instance,
// counter), and null CookedValues stay unread.
func TestLastCounterValuesKeepsLastSample(t *testing.T) {
	rows := decodePSRows(`[
		{"Path":"\\\\h\\physicaldisk(0 c:)\\disk reads/sec","CookedValue":11},
		{"Path":"\\\\h\\physicaldisk(0 c:)\\current disk queue length","CookedValue":null},
		{"Path":"\\\\h\\physicaldisk(1 d:)\\disk writes/sec","CookedValue":22},
		{"Path":"\\\\h\\physicaldisk(0 c:)\\disk reads/sec","CookedValue":33},
		{"Path":"\\\\h\\physicaldisk(0 c:)\\current disk queue length","CookedValue":4},
		{"Path":"\\\\h\\physicaldisk(1 d:)\\disk writes/sec","CookedValue":55}
	]`)
	vals := lastCounterValues(rows)
	m0 := vals["0 c:"]
	if m0["disk reads/sec"] != 33 {
		t.Fatalf("first sample leaked: %v", m0)
	}
	if m0["current disk queue length"] != 4 {
		t.Fatalf("null then value = %v", m0)
	}
	if got := vals["1 d:"]["disk writes/sec"]; got != 55 {
		t.Fatalf("drive 1 = %v", got)
	}
	if len(vals) != 2 {
		t.Fatalf("instances = %v", vals)
	}
}

func TestMatchCounterInstanceClaimsOnce(t *testing.T) {
	rates := map[string]map[string]float64{
		"Ethernet":         {"bytes received/sec": 1},
		"Tailscale Tunnel": {"bytes received/sec": 2},
	}
	matched := map[string]bool{}
	if _, m, ok := matchCounterInstance([]string{"ethernet"}, rates, matched); !ok || m["bytes received/sec"] != 1 {
		t.Fatalf("case-insensitive match failed: %v ok=%v", m, ok)
	}
	if _, _, ok := matchCounterInstance([]string{"ethernet"}, rates, matched); ok {
		t.Fatal("instance claimed twice")
	}
	// Connection-id alias: an instance named after the conn id still matches.
	rates2 := map[string]map[string]float64{"Tailscale": {"bytes sent/sec": 9}}
	if _, m, ok := matchCounterInstance([]string{"tailscale tunnel", "tailscale"}, rates2, matched); !ok || m["bytes sent/sec"] != 9 {
		t.Fatalf("conn-id alias unmatched: %v ok=%v", m, ok)
	}
}
