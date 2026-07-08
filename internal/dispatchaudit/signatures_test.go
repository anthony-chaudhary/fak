package dispatchaudit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNormalizeSigMessageCollapse mirrors the Python NormalizeTest: two messages
// differing only in an index map to one signature, and a digit becomes `#`.
func TestNormalizeSigMessageCollapse(t *testing.T) {
	a := normalizeSigMessage("index out of range [7]")
	b := normalizeSigMessage("index out of range [12]")
	if a != b {
		t.Fatalf("indices must collapse to one signature: %q vs %q", a, b)
	}
	if !strings.Contains(a, "#") {
		t.Fatalf("normalized message must carry the `#` digit placeholder: %q", a)
	}
}

// TestSignatureKeyShape locks the exact cross-tool key shape the Python tool
// stamps (`test_signature_key_shape`): a hex address collapses to `#`.
func TestSignatureKeyShape(t *testing.T) {
	got := signatureKey(SigPanicTraceback, BackendClaude, "panic: boom 0x4a")
	want := "panic-traceback::claude::panic: boom #"
	if got != want {
		t.Fatalf("signatureKey = %q, want %q", got, want)
	}
}

// TestSignatureDetectorsTable is the per-signature table test the issue witness
// requires: one fixture per class, asserting ScanLogText emits that class with
// the expected count.
func TestSignatureDetectorsTable(t *testing.T) {
	th := DefaultSignatureThresholds()
	cases := []struct {
		name      string
		text      string
		wantClass SignatureClass
		wantCount int
		msgSub    string
	}{
		{
			name:      "go panic at line start",
			text:      "doing work\npanic: runtime error: index out of range [5]\n  goroutine 1\n",
			wantClass: SigPanicTraceback, wantCount: 1, msgSub: "panic:",
		},
		{
			name:      "python traceback headline",
			text:      "Traceback (most recent call last):\n  File \"x.py\", line 3, in <module>\nKeyError: 'missing'\n",
			wantClass: SigPanicTraceback, wantCount: 1, msgSub: "KeyError",
		},
		{
			name:      "hook failure storm",
			text:      "hook: PreToolUse Failed\nhook: PreToolUse Failed\nhook: PreToolUse Failed\n",
			wantClass: SigHookFailureStorm, wantCount: 3, msgSub: "hook handler failures",
		},
		{
			name:      "off-trunk refusal",
			text:      "guard: OFF_TRUNK refused push to feature/x (not on main)\n",
			wantClass: SigOffTrunkStorm, wantCount: 1, msgSub: "OFF_TRUNK guard refusal",
		},
		{
			name:      "auth wall not logged in",
			text:      "Error: Not logged in. Please authenticate.\n",
			wantClass: SigAuthWall, wantCount: 1, msgSub: "not logged in",
		},
		{
			name:      "banner-only no-op",
			text:      "fak guard: turn 1 starting\ngateway: connected upstream\n",
			wantClass: SigBannerOnlyNoOp, wantCount: 1, msgSub: "banner-only",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ScanLogText("resolve-1-x.log", BackendClaude, tc.text, th)
			if len(got) != 1 {
				t.Fatalf("want exactly 1 finding, got %d: %+v", len(got), got)
			}
			f := got[0]
			if f.Class != tc.wantClass {
				t.Fatalf("class = %s, want %s", f.Class, tc.wantClass)
			}
			if f.Count != tc.wantCount {
				t.Fatalf("count = %d, want %d", f.Count, tc.wantCount)
			}
			if !strings.Contains(f.Message, tc.msgSub) {
				t.Fatalf("message %q does not contain %q", f.Message, tc.msgSub)
			}
		})
	}
}

// TestMatchPanicCollapsesAndSkipsQuotes covers the panic-detector nuances the
// Python PanicDetectorTest pins.
func TestMatchPanicCollapsesAndSkipsQuotes(t *testing.T) {
	// identical panics collapse to one finding with a summed count.
	got := matchPanic("panic: boom [1]\npanic: boom [2]\npanic: boom [3]\n")
	if len(got) != 1 || got[0].Count != 3 {
		t.Fatalf("repeated identical panic must collapse to count 3, got %+v", got)
	}
	// a worker echoing grep/JSON output that MENTIONS Traceback mid-line is not a panic.
	quoted := `      "detail": "Traceback (most recent call last):\n  File ..."` + "\n"
	if hits := matchPanic(quoted); len(hits) != 0 {
		t.Fatalf("quoted/mid-line traceback must not be a panic, got %+v", hits)
	}
}

// TestMatchOffTrunkSkips covers the OFF_TRUNK false-positive guards.
func TestMatchOffTrunkSkips(t *testing.T) {
	// a ripgrep echo of a repo file mentioning OFF_TRUNK is not a refusal.
	quote := ".\\tools\\githooks\\reference-transaction:10:# the OFF_TRUNK reason\n"
	if hits := matchOffTrunk(quote); len(hits) != 0 {
		t.Fatalf("quoted repo line must be skipped, got %+v", hits)
	}
	// a bare mention with no refuse hint is not a refusal.
	if hits := matchOffTrunk("we discussed OFF_TRUNK in passing\n"); len(hits) != 0 {
		t.Fatalf("bare mention must be skipped, got %+v", hits)
	}
}

// TestHookStormFloorPerLog mirrors Python `test_storm_floor_applies_per_log`:
// the per-session floor is the tunable HookMin.
func TestHookStormFloorPerLog(t *testing.T) {
	text := "hook: A Failed\nhook: B Failed\n" // 2 lines
	if got := ScanLogText("resolve-1-x.log", BackendClaude, text, SignatureThresholds{HookMin: 3}); len(got) != 0 {
		t.Fatalf("2 hook lines under a floor of 3 must not storm, got %+v", got)
	}
	got := ScanLogText("resolve-1-x.log", BackendClaude, text, SignatureThresholds{HookMin: 2})
	if len(got) != 1 || got[0].Class != SigHookFailureStorm {
		t.Fatalf("2 hook lines at a floor of 2 must storm, got %+v", got)
	}
}

// TestAggregateSignatures mirrors the Python AggregateTest: same signature across
// logs collapses (summed count, unioned logs); distinct backends stay separate.
func TestAggregateSignatures(t *testing.T) {
	findings := []SignatureFinding{
		{Class: SigHookFailureStorm, Severity: 80, Backend: BackendCodex, Message: "hook handler failures", Count: 3, Logs: []string{"resolve-1-a.log"}, minTotal: 3},
		{Class: SigHookFailureStorm, Severity: 80, Backend: BackendCodex, Message: "hook handler failures", Count: 5, Logs: []string{"resolve-2-b.log"}, minTotal: 3},
	}
	agg := AggregateSignatures(findings)
	if len(agg) != 1 {
		t.Fatalf("same signature across logs must collapse to 1, got %d", len(agg))
	}
	if agg[0].Count != 8 {
		t.Fatalf("counts must sum to 8, got %d", agg[0].Count)
	}
	if len(agg[0].Logs) != 2 {
		t.Fatalf("logs must union to 2, got %v", agg[0].Logs)
	}

	byBackend := []SignatureFinding{
		{Class: SigAuthWall, Severity: 50, Backend: BackendClaude, Message: "auth wall: not logged in", Count: 1, Logs: []string{"resolve-c.log"}, minTotal: 3},
		{Class: SigAuthWall, Severity: 50, Backend: BackendOpencode, Message: "auth wall: not logged in", Count: 1, Logs: []string{"resolve-o.log"}, minTotal: 3},
	}
	if got := AggregateSignatures(byBackend); len(got) != 2 {
		t.Fatalf("distinct backends must stay separate, got %d", len(got))
	}
}

// TestFoldSignaturesMinTotalAndOrder proves the min-total floor filters
// under-threshold candidates and the survivors sort worst-first.
func TestFoldSignaturesMinTotalAndOrder(t *testing.T) {
	logs := []SigLog{
		// panic (sev 100, min_total 1) — kept from one hit.
		{Name: "resolve-1-a.log", Backend: BackendClaude, Text: "panic: boom\n"},
		// off-trunk needs 2 aggregate hits: two logs each with one refusal.
		{Name: "resolve-2-b.log", Backend: BackendClaude, Text: "guard: OFF_TRUNK refused push\n"},
		{Name: "resolve-3-c.log", Backend: BackendClaude, Text: "guard: OFF_TRUNK refused push\n"},
		// a lone auth line (count 1 < min_total 3) must be filtered out.
		{Name: "resolve-4-d.log", Backend: BackendClaude, Text: "credit balance is too low\n"},
	}
	got := FoldSignatures(logs, DefaultSignatureThresholds())
	if len(got) != 2 {
		t.Fatalf("want 2 surviving candidates (panic + off-trunk), got %d: %+v", len(got), got)
	}
	if got[0].Class != SigPanicTraceback {
		t.Fatalf("worst-first: panic (sev 100) must lead, got %s", got[0].Class)
	}
	if got[1].Class != SigOffTrunkStorm || got[1].Count != 2 {
		t.Fatalf("off-trunk storm must aggregate to count 2, got %+v", got[1])
	}
	for _, c := range got {
		if c.Class == SigAuthWall {
			t.Fatalf("under-min-total auth candidate must be filtered out")
		}
	}
}

// TestSignatureAsFindingIsStructured proves the bridge into the fileable Finding
// carries only STRUCTURED evidence (the repo's no-raw-worker-text discipline) and
// a namespaced fingerprint distinct from any outcome finding.
func TestSignatureAsFindingIsStructured(t *testing.T) {
	sf := SignatureFinding{
		Class: SigPanicTraceback, Severity: 100, Backend: BackendClaude,
		Message: "panic: secret token abc123 leaked", Count: 2,
		Logs:   []string{"resolve-1-a.log", "resolve-2-b.log"},
		Sample: []string{"panic: secret token abc123 leaked"},
	}
	f := sf.AsFinding()
	if f.SignatureClass != SigPanicTraceback {
		t.Fatalf("SignatureClass must round-trip, got %q", f.SignatureClass)
	}
	if len(f.Fingerprint) != 16 {
		t.Fatalf("fingerprint must be 16 hex, got %q", f.Fingerprint)
	}
	if f.Fingerprint != sf.Fingerprint() {
		t.Fatalf("AsFinding fingerprint must match SignatureFinding.Fingerprint()")
	}
	// The raw sample line must never leak into the fileable evidence/detail.
	for _, raw := range []string{"secret token abc123", "abc123 leaked"} {
		if strings.Contains(f.Detail, raw) || strings.Contains(f.Evidence, raw) {
			t.Fatalf("raw worker sample text leaked into a fileable finding: %q / %q", f.Detail, f.Evidence)
		}
	}
	if !strings.Contains(f.Evidence, "class=panic-traceback") {
		t.Fatalf("evidence must carry the structured class token, got %q", f.Evidence)
	}
}

// TestScanDirSignaturesFixture proves the I/O shell reads real .dispatch-runs
// logs, defaults a sidecar-less log to the claude backend (Python parity), and
// classifies a panic across two sessions.
func TestScanDirSignaturesFixture(t *testing.T) {
	dir := t.TempDir()
	// Two claude sessions with the same panic — no .backend sidecar (legacy →
	// claude), so they aggregate to one candidate spanning both logs.
	if err := os.WriteFile(filepath.Join(dir, "resolve-1-20260708-000001.log"),
		[]byte("# fak-spawn 20260708-000001 issue=1 lane=cmd backend=claude\npanic: runtime error: index out of range [3]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "resolve-2-20260708-000002.log"),
		[]byte("# fak-spawn 20260708-000002 issue=2 lane=cmd backend=claude\npanic: runtime error: index out of range [9]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ScanDirSignatures(dir, DefaultSignatureThresholds())
	if err != nil {
		t.Fatalf("ScanDirSignatures: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("two same-panic sessions must aggregate to 1 candidate, got %d: %+v", len(got), got)
	}
	if got[0].Class != SigPanicTraceback || got[0].Backend != BackendClaude {
		t.Fatalf("want panic on claude (sidecar-less default), got %s on %s", got[0].Class, got[0].Backend)
	}
	if got[0].Count != 2 || len(got[0].Logs) != 2 {
		t.Fatalf("panic must aggregate across both sessions, got count=%d logs=%v", got[0].Count, got[0].Logs)
	}
}
