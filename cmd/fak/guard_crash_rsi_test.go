package main

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestGuardCrashRSIFirstEligibleCrashLaunchesBoundedTaggedInvestigation(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	var got []guardCrashRSIRequest
	guardCrashRSILaunch = func(req guardCrashRSIRequest) error {
		got = append(got, req)
		return nil
	}
	var stderr bytes.Buffer
	if !guardMaybeLaunchCrashRSI(&stderr, "RID-secret-looking-source", "codex", "NONZERO_EXIT", 17, 0) {
		t.Fatal("first eligible crash did not launch")
	}
	if len(got) != 1 {
		t.Fatalf("launch count=%d, want 1", len(got))
	}
	req := got[0]
	if !strings.HasPrefix(req.Tag, "guard-crash-rsi/") || req.Source == "" || !strings.HasSuffix(req.Tag, req.Source) {
		t.Fatalf("tag/source = %q/%q", req.Tag, req.Source)
	}
	for _, want := range []string{"ORIGINAL fak guard child crash", req.Tag, req.Source, "NONZERO_EXIT", "exit_code: 17", req.Workspace} {
		if !strings.Contains(req.Prompt, want) {
			t.Fatalf("prompt missing %q: %s", want, req.Prompt)
		}
	}
	if strings.Contains(req.Prompt, "RID-secret-looking-source") {
		t.Fatalf("prompt leaked raw guard identity: %s", req.Prompt)
	}
	if !strings.Contains(stderr.String(), req.Tag) {
		t.Fatalf("status missing tag: %s", stderr.String())
	}
}

func TestGuardCrashRSIOnlyFirstCrashAndNeverRecurses(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { launches++; return nil }
	guardMaybeLaunchCrashRSI(nil, "trace", "claude", "SIGNAL", -1, 0)
	guardMaybeLaunchCrashRSI(nil, "trace", "claude", "SIGNAL", -1, 1)
	if launches != 1 {
		t.Fatalf("launches=%d, want exactly 1", launches)
	}
	t.Setenv(guardCrashRSIMarkerEnv, "guard-crash-rsi/already")
	guardMaybeLaunchCrashRSI(nil, "other", "claude", "OOM", 137, 0)
	if launches != 1 {
		t.Fatalf("recursive session launched: %d", launches)
	}
}

func TestGuardCrashRSIUnsafeAndNonCrashCasesSkip(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	launches := 0
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { launches++; return nil }
	cases := []struct {
		trace, agent, class string
		code                int
	}{
		{"", "codex", "NONZERO_EXIT", 1},
		{"trace", "unknown", "NONZERO_EXIT", 1},
		{"trace", "codex", "", 1},
		{"trace", "codex", "CLEAN_EXIT", 0},
	}
	for _, tc := range cases {
		guardMaybeLaunchCrashRSI(nil, tc.trace, tc.agent, tc.class, tc.code, 0)
	}
	if launches != 0 {
		t.Fatalf("unsafe/non-crash launches=%d", launches)
	}
}

func TestGuardCrashRSILaunchFailureIsFailOpen(t *testing.T) {
	t.Setenv(guardCrashRSIMarkerEnv, "")
	old := guardCrashRSILaunch
	t.Cleanup(func() { guardCrashRSILaunch = old })
	guardCrashRSILaunch = func(guardCrashRSIRequest) error { return errors.New("synthetic launch failure") }
	var stderr bytes.Buffer
	if guardMaybeLaunchCrashRSI(&stderr, "trace", "codex", "NONZERO_EXIT", 9, 0) {
		t.Fatal("failed launch reported success")
	}
	if !strings.Contains(stderr.String(), "synthetic launch failure") {
		t.Fatalf("missing fail-open diagnostic: %s", stderr.String())
	}
}

func TestGuardCrashRSIEnvironmentExcludesSecretsAndCarriesOnlyMarker(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "do-not-forward")
	t.Setenv("ANTHROPIC_API_KEY", "do-not-forward")
	t.Setenv("FAK_SECRET_ORIGINAL_ARG", "--token=do-not-forward")
	env := guardCrashRSIEnvironment("guard-crash-rsi/source")
	joined := strings.Join(env, "\n")
	for _, forbidden := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FAK_SECRET_ORIGINAL_ARG", "do-not-forward"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("environment leaked %q: %s", forbidden, joined)
		}
	}
	if !strings.Contains(joined, guardCrashRSIMarkerEnv+"=guard-crash-rsi/source") {
		t.Fatalf("environment missing recursion marker: %v", env)
	}
	_ = os.Getenv("PATH") // documents that bootstrap paths may be retained without asserting host shape.
}
