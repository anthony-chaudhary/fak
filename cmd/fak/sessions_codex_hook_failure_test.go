package main

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

type failingCodexHookWriter struct{}

func (failingCodexHookWriter) Write([]byte) (int, error) {
	return 0, errors.New("injected encode failure")
}

func TestCodexLoopHookFailureRecoveryMessages(t *testing.T) {
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	for _, tc := range []struct {
		name, payload string
		argv          []string
		want          string
	}{
		{name: "UnreadablePayload", payload: "{", argv: []string{"codex-loop-hook"}, want: "recovery: relaunch with `fak codex`"},
		{name: "AbsentSessionId", payload: "{}", argv: []string{"codex-loop-hook"}, want: "set CODEX_THREAD_ID"},
		{name: "UnresolvedTranscript", payload: `{"session_id":"missing-session"}`, argv: []string{"codex-loop-hook", "--codex-home", t.TempDir()}, want: "verify --codex-home/CODEX_HOME"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := sessionsCodexLoopHookUnbounded(&stdout, &stderr, strings.NewReader(tc.payload), tc.argv[1:], probeCodexLoopProvider); code != 0 {
				t.Fatalf("exit=%d", code)
			}
			if stdout.Len() != 0 || !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("stdout=%q stderr=%q want=%q", stdout.String(), stderr.String(), tc.want)
			}
		})
	}
}

func TestCodexLoopHookDiagnoseFailureAllowsWithRecovery(t *testing.T) {
	// Both ambient opt-outs must be neutralized: either one returns 0 with an EMPTY
	// stderr before diagnose is ever called, so the recovery-string assertion below
	// would fail on any box running under `fak guard`.
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "openai")
	var stdout, stderr bytes.Buffer
	code := sessionsCodexLoopHookUnbounded(&stdout, &stderr, strings.NewReader(`{"session_id":"`+sessionID+`"}`), []string{"--codex-home", home}, func(io.Reader, string) (codexLoopDiagnosis, error) {
		return codexLoopDiagnosis{}, errors.New("injected diagnose failure")
	})
	if code != 0 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "inspect the rollout JSONL") {
		t.Fatalf("exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func TestCodexLoopHookEncodeErrorNamesRecovery(t *testing.T) {
	// Same neutralizers: an ambient opt-out returns 0 without ever reaching the
	// encoder, so the injected write failure this test needs would never fire.
	t.Setenv(codexLoopHookOverrideEnv, "")
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "openai")
	var stderr bytes.Buffer
	code := sessionsCodexLoopHookUnbounded(failingCodexHookWriter{}, &stderr, strings.NewReader(`{"session_id":"`+sessionID+`"}`), []string{"--codex-home", home}, probeCodexLoopProvider)
	if code != 1 || !strings.Contains(stderr.String(), "turn not blocked") || !strings.Contains(stderr.String(), "relaunch with `fak codex`") {
		t.Fatalf("exit=%d stderr=%q", code, stderr.String())
	}
}
