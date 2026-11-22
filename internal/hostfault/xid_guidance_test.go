package hostfault

import "testing"

func TestXIDGuidance(t *testing.T) {
	xidCases := []struct {
		code      uint32
		preAmpere bool
		want      XIDGuidance
	}{
		{63, false, XIDGuidanceNoUserAction},
		{106, false, XIDGuidanceNoUserAction},
		{107, false, XIDGuidanceNoUserAction},
		{8, false, XIDGuidanceRestartApplication},
		{13, false, XIDGuidanceRestartApplication},
		{105, false, XIDGuidanceRestartApplication},
		{139, false, XIDGuidanceRestartApplication},
		{46, false, XIDGuidanceResetGPU},
		{119, false, XIDGuidanceResetGPU},
		{158, false, XIDGuidanceResetGPU},
		{48, false, XIDGuidanceCheckDocumentation},
		{79, false, XIDGuidanceCheckDocumentation},
		{48, true, XIDGuidancePreAmpere},
		{63, true, XIDGuidancePreAmpere},
	}
	for _, c := range xidCases {
		if got := ClassifyXIDGuidance(c.code, c.preAmpere); got != c.want {
			t.Errorf("ClassifyXIDGuidance(%d, %v) = %v, want %v", c.code, c.preAmpere, got, c.want)
		}
	}
	// Each code must carry exactly one action, so the XID sets must be disjoint.
	seen := map[uint32]string{}
	for name, set := range map[string][]uint32{
		"no_user_action": xidNoUserAction, "restart_application": xidRestartApplication, "reset_gpu": xidResetGPU,
	} {
		for i, code := range set {
			if i > 0 && set[i-1] >= code {
				t.Errorf("%s set not strictly ascending at %d", name, code)
			}
			if prev, dup := seen[code]; dup {
				t.Errorf("xid %d in both %s and %s", code, prev, name)
			}
			seen[code] = name
		}
	}

	sxidCases := []struct {
		code  uint32
		class SXIDClass
		want  XIDGuidance
	}{
		{11012, SXIDClassNonFatal, XIDGuidanceNoUserAction},
		{24003, SXIDClassNonFatal, XIDGuidanceNoUserAction},
		{11012, SXIDClassFatal, XIDGuidanceCheckDocumentation},
		{11012, SXIDClassUnknown, XIDGuidanceCheckDocumentation},
		{99999, SXIDClassNonFatal, XIDGuidanceCheckDocumentation},
	}
	for _, c := range sxidCases {
		if got := ClassifySXIDGuidance(c.code, c.class); got != c.want {
			t.Errorf("ClassifySXIDGuidance(%d, %v) = %v, want %v", c.code, c.class, got, c.want)
		}
	}

	const (
		catalog = "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
		archive = "https://docs.nvidia.com/deploy/xid-errors/archive/index.html"
		fabric  = "https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html"
	)
	if got := XIDGuidanceResetGPU.DocURL(XIDEventXID); got != catalog {
		t.Errorf("xid doc = %q", got)
	}
	if got := XIDGuidancePreAmpere.DocURL(XIDEventXID); got != archive {
		t.Errorf("pre-ampere doc = %q", got)
	}
	if got := XIDGuidanceNoUserAction.DocURL(XIDEventSXID); got != fabric {
		t.Errorf("sxid doc = %q", got)
	}

	parsed := []struct {
		line string
		want XIDEvent
	}{
		{"NVRM: Xid (PCI:0000:01:00): 79, pid=1", XIDEvent{Type: XIDEventXID, Code: 79, Link: -1, BDF: 1 << 12, BDFValid: true}},
		{"NVRM: Xid 13, Graphics Exception", XIDEvent{Type: XIDEventXID, Code: 13, Link: -1}},
		{"NVRM: Xid (gpu-x): 48, pid=1", XIDEvent{Type: XIDEventXID, Code: 48, Link: -1}},
		{"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11012, Non-fatal, Link 32",
			XIDEvent{Type: XIDEventSXID, Code: 11012, SXIDClass: SXIDClassNonFatal, Link: 32, BDF: 5 << 12, BDFValid: true}},
		{"NVRM: Xid (PCI:0x0000: 3b:00000000001f.7): 31", XIDEvent{Type: XIDEventXID, Code: 31, Link: -1, BDF: 0x3b<<12 | 0x1f<<4 | 7, BDFValid: true}},
		{"NVRM: Xid (PCI:0000:3b:20): -0", XIDEvent{Type: XIDEventXID, Code: 0, Link: -1}},
		{"nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12028, fatal , Link\t7 (0x1)",
			XIDEvent{Type: XIDEventSXID, Code: 12028, SXIDClass: SXIDClassFatal, Link: 7, BDF: 0xc1 << 12, BDFValid: true}},
	}
	for _, c := range parsed {
		got, ok := ParseXIDLine(c.line)
		if !ok || got != c.want {
			t.Errorf("ParseXIDLine(%q) = %+v, %v; want %+v", c.line, got, ok, c.want)
		}
	}
	if !SameXIDEvent(parsed[0].want, parsed[0].want) || SameXIDEvent(parsed[0].want, parsed[1].want) {
		t.Errorf("SameXIDEvent identity broken")
	}

	rejected := []string{
		"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11012, Severity 0 Engine instance 32",
		"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11012, Data {0x0, 0x1}",
		"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11012, Warning, Link 3",
		"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11012, Non-fatal, Link 40000",
		"nvidia-nvswitch0: SXid PCI:0000:05:00.0: 11012, Non-fatal, Link 1",
		"oxidation of the heatsink",
		"NVRM: xid 79",
		"NVRM: Xid (PCI:0000:01:00): 4294967296",
		"NVRM: Xid (PCI:0000:01:00): -5",
		"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11012, Non-fatal, Link -1",
	}
	for _, line := range rejected {
		if got, ok := ParseXIDLine(line); ok {
			t.Errorf("ParseXIDLine(%q) = %+v, want rejection", line, got)
		}
	}

	ev, _ := ParseXIDLine("nvidia-nvswitch0: SXid (PCI:0000:05:00.0): 11012, Non-fatal, Link 32")
	if got := ClassifyXIDEventGuidance(ev, true); got != XIDGuidanceNoUserAction {
		t.Errorf("preAmpere must not override SXid guidance, got %v", got)
	}
}
