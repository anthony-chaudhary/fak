package main

import (
	"bytes"
	"strings"
	"testing"
)

// claude_router_model_test.go — the default-backend model preference for
// `fak claude`.

// init keeps the whole test binary hermetic with respect to the new
// preference seam: without it, every runClaude test would inherit the real
// claudeRouterPreferredModelFn, which dials the host router and reads host
// config. The default stub returns "" (preference unavailable), so the
// pre-existing direct-default and first-advertised-id tests keep observing
// their old behavior; a test that means to witness the preference opts back
// in with stubClaudeRouterPreferredModel.
func init() {
	claudeRouterPreferredModelFn = func(string) string { return "" }
}

// stubClaudeRouterPreferredModel overrides the default-backend preference so
// tests are hermetic and never dial a router. It returns a restore func
// registered with t.Cleanup.
func stubClaudeRouterPreferredModel(t *testing.T, model string) {
	t.Helper()
	orig := claudeRouterPreferredModelFn
	claudeRouterPreferredModelFn = func(string) string { return model }
	t.Cleanup(func() { claudeRouterPreferredModelFn = orig })
}

// TestClaudeRouterPreferredModelWinsOverFirstAdvertised drives the real flag
// path: with a live router and no pin, the default-backend preference beats
// the first advertised id.
func TestClaudeRouterPreferredModelWinsOverFirstAdvertised(t *testing.T) {
	stubClaudeRouterLive(t, "testprov.first", true)
	stubClaudeRouterModel(t, "testprov.first")
	stubClaudeRouterPreferredModel(t, "testprov.large")
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_MODEL", "")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:18101")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	out := stderr.String()
	if !strings.Contains(out, "model       = testprov.large") {
		t.Errorf("dry-run did not adopt the preferred default-backend model:\n%s", out)
	}
	if strings.Contains(out, "model       = testprov.first") {
		t.Errorf("dry-run kept the first advertised id over the preference:\n%s", out)
	}
}

// TestClaudeRouterPreferredModelEmptyKeepsFirstAdvertised pins the previous
// behavior: when the preference yields nothing, the first advertised id is
// still used.
func TestClaudeRouterPreferredModelEmptyKeepsFirstAdvertised(t *testing.T) {
	stubClaudeRouterLive(t, "testprov.first", true)
	stubClaudeRouterModel(t, "testprov.first")
	stubClaudeRouterPreferredModel(t, "")
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_MODEL", "")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:18101")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "model       = testprov.first") {
		t.Errorf("dry-run lost the first advertised id:\n%s", stderr.String())
	}
}

// TestClaudeRouterExplicitModelBeatsPreferred pins that an explicit --model
// still wins over the default-backend preference.
func TestClaudeRouterExplicitModelBeatsPreferred(t *testing.T) {
	stubClaudeRouterLive(t, "testprov.first", true)
	stubClaudeRouterModel(t, "testprov.first")
	stubClaudeRouterPreferredModel(t, "testprov.large")
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_MODEL", "")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:18101")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe", "--model", "operator-model"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "model       = operator-model") {
		t.Errorf("explicit --model did not win over the preference:\n%s", stderr.String())
	}
}

// TestClaudeRouterModelEnvBeatsPreferred pins that a FAK_ROUTER_MODEL pin
// still wins over the default-backend preference. The router-model stub
// returns the pinned value, as the real probe does when the env var is set.
func TestClaudeRouterModelEnvBeatsPreferred(t *testing.T) {
	stubClaudeRouterLive(t, "testprov.first", true)
	stubClaudeRouterModel(t, "pinned-model")
	stubClaudeRouterPreferredModel(t, "testprov.large")
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_MODEL", "pinned-model")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:18101")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "model       = pinned-model") {
		t.Errorf("FAK_ROUTER_MODEL pin did not win over the preference:\n%s", stderr.String())
	}
}
