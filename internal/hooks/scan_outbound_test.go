package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// All leak fixtures are constructed at runtime (concatenation) so this file never carries
// a literal needle or shape — the same discipline the gate_publicleak/gate_secretshape
// tests follow, keeping the repo's own PUBLIC_LEAK/SECRET_SHAPE gates quiet about it.

func TestScanOutboundTextCleanPayloadPasses(t *testing.T) {
	if got := ScanOutboundText("run 42 finished\nverdict SHIPPED sha=abc123", ""); len(got) != 0 {
		t.Fatalf("clean payload flagged: %+v", got)
	}
}

func TestScanOutboundTextFlagsNeedleWithLineNumber(t *testing.T) {
	needle := "node-" + "windows-a" // a base audit needle, assembled at runtime
	payload := "line one is fine\nhost " + needle + " did the work"
	got := ScanOutboundText(payload, "")
	if len(got) != 1 {
		t.Fatalf("want 1 finding, got %+v", got)
	}
	if got[0].Gate != "PUBLIC_LEAK" || got[0].Line != 2 || !strings.Contains(got[0].Detail, needle) {
		t.Fatalf("finding wrong: %+v", got[0])
	}
}

func TestScanOutboundTextFlagsInternalHostShape(t *testing.T) {
	host := "build1" + "." + "lab" // *.lab internal-host shape, assembled at runtime
	got := ScanOutboundText("deployed to "+host+" ok", "")
	if len(got) != 1 || got[0].Gate != "SECRET_SHAPE" || !strings.Contains(got[0].Detail, host) {
		t.Fatalf("want one SECRET_SHAPE finding for %q, got %+v", host, got)
	}
}

func TestScanOutboundTextScansEveryLineUnlikeMessageScan(t *testing.T) {
	// A commit-message scan skips '#' comment lines; an outbound payload has no such
	// convention — a leak on a heading-style line must still be caught.
	needle := "node-" + "windows-a"
	payload := "# status for " + needle
	if got := ScanOutboundText(payload, ""); len(got) != 1 {
		t.Fatalf("comment-style line was not scanned: %+v", got)
	}
	if got := ScanMessageNeedles(payload, ""); len(got) != 0 {
		t.Fatalf("message scan contract changed — this test's premise is stale: %+v", got)
	}
}

func TestScanOutboundTextUnionsPrivateSidecarNeedles(t *testing.T) {
	root := t.TempDir()
	side := filepath.Join(root, filepath.FromSlash(privateNeedlesRel))
	if err := os.MkdirAll(filepath.Dir(side), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(side, []byte(`{"audit_needles":["dgx-secret-host"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	got := ScanOutboundText("posted from dgx-secret-host today", root)
	if len(got) != 1 || got[0].Gate != "PUBLIC_LEAK" {
		t.Fatalf("sidecar needle not applied: %+v", got)
	}
	if got := ScanOutboundText("posted from dgx-secret-host today", ""); len(got) != 0 {
		t.Fatalf("empty root must skip the sidecar: %+v", got)
	}
}

func TestScanOutboundTextPlaceholderUserPathIsNotALeak(t *testing.T) {
	payload := `log at C:\Users\USER\logs\run.txt` // placeholder username — documentation shape
	if got := ScanOutboundText(payload, ""); len(got) != 0 {
		t.Fatalf("placeholder user path flagged: %+v", got)
	}
}
