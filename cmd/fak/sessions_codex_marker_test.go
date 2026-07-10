package main

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func TestCodexLoopHookGuardMarkerShortCircuitsSilent(t *testing.T) {
	t.Setenv(guardActiveEnv, "1")
	old := codexLoopHookDiagnose
	called := false
	codexLoopHookDiagnose = func(_ io.Reader, _ string) (codexLoopDiagnosis, error) {
		called = true
		panic("diagnose must not run for guarded child")
	}
	t.Cleanup(func() { codexLoopHookDiagnose = old })
	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader("not even json"), nil); code != 0 {
		t.Fatalf("code=%d, want allow", code)
	}
	if called || stdout.Len() != 0 || stderr.Len() != 0 {
		t.Fatalf("called=%v stdout=%q stderr=%q, want O(1) byte-silent allow", called, stdout.String(), stderr.String())
	}
}

func TestGuardChildInjectsContinuationMarker(t *testing.T) {
	_, env := guardChildCommandEnv([]string{"codex"}, nil, false)
	want := guardActiveEnv + "=1"
	for _, item := range env {
		if item == want {
			return
		}
	}
	t.Fatalf("guard child env missing %q", want)
}
