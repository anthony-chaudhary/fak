package auditpane

import (
	"os"
	"path/filepath"
	"testing"
)

// Classify is the heart of the common verdict contract. These cases pin the
// precedence ladder documented in the package doc: spawn-error > timeout >
// envelope (skip-token > ok-bool > verdict-token) > exit-code fallback.
func TestClassifyContractLadder(t *testing.T) {
	cases := []struct {
		name        string
		out         RunOutcome
		wantVerdict string
		wantOK      bool
	}{
		{"spawn failure trips ERROR", RunOutcome{SpawnErr: "exec: \"python\": not found", ExitCode: -1}, VerdictError, false},
		{"timeout degrades to SKIP", RunOutcome{TimedOut: true}, VerdictSkip, true},
		{"ok:true is PASS", RunOutcome{Stdout: []byte(`{"ok":true}`), ExitCode: 0}, VerdictPass, true},
		{"ok:false is FAIL", RunOutcome{Stdout: []byte(`{"ok":false}`), ExitCode: 1}, VerdictFail, false},
		{"verdict GREEN is PASS without ok", RunOutcome{Stdout: []byte(`{"verdict":"GREEN"}`), ExitCode: 0}, VerdictPass, true},
		{"unknown verdict is FAIL", RunOutcome{Stdout: []byte(`{"verdict":"BROKEN"}`), ExitCode: 1}, VerdictFail, false},
		{"skip-token beats ok:false", RunOutcome{Stdout: []byte(`{"ok":false,"verdict":"HOST_GATED"}`), ExitCode: 3}, VerdictSkip, true},
		{"no envelope exit 0 is PASS", RunOutcome{ExitCode: 0}, VerdictPass, true},
		{"no envelope exit non-zero is FAIL", RunOutcome{Stderr: "boom", ExitCode: 2}, VerdictFail, false},
		{"human prefix before JSON still parses", RunOutcome{Stdout: []byte("running audit...\n{\"ok\":true}"), ExitCode: 0}, VerdictPass, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Classify("x_audit.py", c.out)
			if got.Verdict != c.wantVerdict {
				t.Errorf("verdict = %q, want %q (reason: %s)", got.Verdict, c.wantVerdict, got.Reason)
			}
			if got.OK != c.wantOK {
				t.Errorf("OK = %v, want %v", got.OK, c.wantOK)
			}
		})
	}
}

// A spawn error must report ExitCode -1, never a real exit code, so an operator
// can tell "never ran" apart from "ran and exited -1-ish".
func TestClassifySpawnErrorExitCode(t *testing.T) {
	got := Classify("x_audit.py", RunOutcome{SpawnErr: "no python"})
	if got.ExitCode != -1 {
		t.Fatalf("spawn-error ExitCode = %d, want -1", got.ExitCode)
	}
}

// Fold's verdict ladder: any FAIL or ERROR trips ACTION; zero auditors trips
// ACTION; an all-pass-or-skip set is OK.
func TestFoldVerdictLadder(t *testing.T) {
	t.Run("zero auditors is ACTION", func(t *testing.T) {
		p := Fold(nil, "ws", "abc")
		if p.OK || p.Verdict != "ACTION" || p.Finding != "no_auditors" {
			t.Fatalf("empty fold = ok:%v verdict:%s finding:%s, want ACTION/no_auditors", p.OK, p.Verdict, p.Finding)
		}
	})
	t.Run("all pass is OK", func(t *testing.T) {
		p := Fold([]Result{{Verdict: VerdictPass}, {Verdict: VerdictPass}}, "", "")
		if !p.OK || p.Verdict != "OK" || p.Passed != 2 {
			t.Fatalf("all-pass fold = ok:%v verdict:%s passed:%d, want OK/2", p.OK, p.Verdict, p.Passed)
		}
	})
	t.Run("pass plus skip stays OK", func(t *testing.T) {
		p := Fold([]Result{{Verdict: VerdictPass}, {Verdict: VerdictSkip}}, "", "")
		if !p.OK || p.Skipped != 1 {
			t.Fatalf("pass+skip fold = ok:%v skipped:%d, want OK with 1 skip", p.OK, p.Skipped)
		}
	})
	t.Run("one fail trips ACTION and names it", func(t *testing.T) {
		p := Fold([]Result{{Name: "sec_audit.py", Verdict: VerdictFail}, {Verdict: VerdictPass}}, "", "")
		if p.OK || p.Verdict != "ACTION" || p.Failed != 1 {
			t.Fatalf("one-fail fold = ok:%v verdict:%s failed:%d, want ACTION/1", p.OK, p.Verdict, p.Failed)
		}
	})
	t.Run("error trips ACTION", func(t *testing.T) {
		p := Fold([]Result{{Verdict: VerdictError}}, "", "")
		if p.OK || p.Errored != 1 {
			t.Fatalf("error fold = ok:%v errored:%d, want ACTION with 1 error", p.OK, p.Errored)
		}
	})
}

// CheckGate maps a folded Payload to the documented exit codes: 0 green, 1
// failing, 2 empty.
func TestCheckGateExitCodes(t *testing.T) {
	if code, _ := CheckGate(Fold(nil, "", "")); code != 2 {
		t.Errorf("empty rollup gate = %d, want 2", code)
	}
	if code, _ := CheckGate(Fold([]Result{{Verdict: VerdictFail}}, "", "")); code != 1 {
		t.Errorf("failing rollup gate = %d, want 1", code)
	}
	if code, _ := CheckGate(Fold([]Result{{Verdict: VerdictPass}}, "", "")); code != 0 {
		t.Errorf("green rollup gate = %d, want 0", code)
	}
}

// Discover globs tools/*_audit.py and must exclude the paired *_audit_test.py.
func TestDiscoverExcludesTestFiles(t *testing.T) {
	root := t.TempDir()
	toolsDir := filepath.Join(root, "tools")
	if err := os.MkdirAll(toolsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"sec_audit.py", "sec_audit_test.py", "crash_audit.py", "helper.py"} {
		if err := os.WriteFile(filepath.Join(toolsDir, f), []byte("# stub\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := Discover(root)
	if err != nil {
		t.Fatal(err)
	}
	// Sorted, only the two real auditors, no _test.py and no non-_audit helper.
	want := []string{
		filepath.Join(toolsDir, "crash_audit.py"),
		filepath.Join(toolsDir, "sec_audit.py"),
	}
	if len(got) != len(want) {
		t.Fatalf("Discover = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Discover[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}
