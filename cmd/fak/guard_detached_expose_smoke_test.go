package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const (
	guardDetachedExposeHelperEnv = "FAK_TEST_GUARD_DETACHED_EXPOSE_HELPER"
	guardDetachedPromptMarker    = "fak-6346-prompt-reached"
)

// TestGuardHeadlessProfileDetachedLaunchReachesPrompt is #6346's captured launch
// witness. It crosses the real detached dispatch seam, initializes a real gateway
// with the guard's headless expose profile, and emits the prompt marker only after
// that argument-validation boundary succeeds. Before the fix gateway.New rejected
// the stale fak_index_work name, so the marker was never emitted and the log ended
// with "matches no known tool".
func TestGuardHeadlessProfileDetachedLaunchReachesPrompt(t *testing.T) {
	if os.Getenv(guardDetachedExposeHelperEnv) == "1" {
		if _, err := gateway.New(gateway.Config{
			EngineID:     "inkernel",
			Model:        "guard-detached-expose-smoke",
			Invalidation: "global",
			ExposeTools:  resolveGuardExposeTools("headless"),
			Logf:         func(string, ...any) {},
		}); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stdout, guardDetachedPromptMarker)
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	runsDir := t.TempDir()
	env := envMap(os.Environ())
	env[guardDetachedExposeHelperEnv] = "1"
	delete(env, "FAK_GUARD_EXPOSE_PROFILE")

	spawned, err := spawnDispatchIssueWorker(
		[]string{exe, "-test.run=^TestGuardHeadlessProfileDetachedLaunchReachesPrompt$"},
		env,
		t.TempDir(),
		runsDir,
		6346,
		"cmd",
		"claude",
		"issue-6346",
		[]string{"cmd/fak/**"},
		dispatchtick.Account{},
		nil,
		"",
		"",
		20,
	)
	if err != nil {
		t.Fatalf("spawn detached guard smoke: %v", err)
	}
	t.Cleanup(func() {
		if dispatchPIDAlive(spawned.PID) {
			if p, findErr := os.FindProcess(spawned.PID); findErr == nil {
				_ = p.Kill()
			}
		}
	})

	deadline := time.Now().Add(30 * time.Second)
	var transcript string
	for time.Now().Before(deadline) {
		b, _ := os.ReadFile(spawned.Log)
		transcript = string(b)
		if strings.Contains(transcript, guardDetachedPromptMarker) || !dispatchPIDAlive(spawned.PID) {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if strings.TrimSpace(transcript) == "" {
		t.Fatal("detached worker transcript is empty; startup produced no captured output")
	}
	if strings.Contains(transcript, "matches no known tool") || strings.Contains(transcript, "unknown tool: fak_index_work") {
		t.Fatalf("headless gateway rejected its expose profile before the prompt:\n%s", transcript)
	}
	if !strings.Contains(transcript, guardDetachedPromptMarker) {
		t.Fatalf("detached guard never reached the wrapped prompt marker %q:\n%s", guardDetachedPromptMarker, transcript)
	}
	if early := spawned.EarlyExit; dispatchMapBool(early, "silent") {
		t.Fatalf("detached launch was classified silent despite captured prompt output: %#v", early)
	}
	if got := filepath.Base(spawned.Log); !strings.HasPrefix(got, "resolve-6346-") {
		t.Fatalf("captured transcript %q is not the issue-scoped detached worker log", got)
	}
}
