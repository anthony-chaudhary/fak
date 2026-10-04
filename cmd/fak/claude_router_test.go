package main

import (
	"bytes"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// claude_router_test.go — the live-router default for `fak claude`.

// init makes the whole test binary hermetic with respect to the live-router
// default: without it, every runClaude test in claude_launcher_test.go would
// inherit this host's real router (if one is live on 18101) and change which
// backend it picks, coupling the package's assertions to a running service.
// The default stubs the router DOWN, so the pre-existing direct-default tests
// keep observing the 8080 placeholder; a test that means to witness the router
// path opts back in with stubClaudeRouterLive.
func init() {
	claudeRouterLive = func(string, time.Duration) (string, bool) { return "", false }
	claudeRouterModel = func(*http.Client, string) string { return "" }
}

// stubClaudeRouterLive overrides the live-router probe so tests are hermetic and
// never depend on a router that happens to be running on the test host. It
// returns a restore func registered with t.Cleanup.
func stubClaudeRouterLive(t *testing.T, model string, live bool) {
	t.Helper()
	orig := claudeRouterLive
	claudeRouterLive = func(string, time.Duration) (string, bool) { return model, live }
	t.Cleanup(func() { claudeRouterLive = orig })
}

// stubClaudeRouterModel overrides /v1/models discovery so tests are hermetic and
// never dial a host.
func stubClaudeRouterModel(t *testing.T, model string) {
	t.Helper()
	orig := claudeRouterModel
	claudeRouterModel = func(*http.Client, string) string { return model }
	t.Cleanup(func() { claudeRouterModel = orig })
}

func TestClaudeRouterOriginPrecedence(t *testing.T) {
	t.Run("explicit wins", func(t *testing.T) {
		t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:9999")
		if got := claudeRouterOrigin("http://10.0.0.5:1234/"); got != "http://10.0.0.5:1234" {
			t.Fatalf("claudeRouterOrigin explicit = %q", got)
		}
	})
	t.Run("env wins over default", func(t *testing.T) {
		t.Setenv("FAK_ROUTER_URL", "http://router.lan:18101/v1")
		if got := claudeRouterOrigin(""); got != "http://router.lan:18101" {
			t.Fatalf("claudeRouterOrigin env = %q", got)
		}
	})
	t.Run("default is the supervised router", func(t *testing.T) {
		t.Setenv("FAK_ROUTER_URL", "")
		if got := claudeRouterOrigin(""); got != claudeRouterDefaultOrigin {
			t.Fatalf("claudeRouterOrigin default = %q, want %q", got, claudeRouterDefaultOrigin)
		}
	})
}

func TestClaudeFrontDoorArgvTranslatesAnthropicToOpenAI(t *testing.T) {
	argv := claudeFrontDoorArgv("127.0.0.1:18123", "http://127.0.0.1:18101", "deepseek-ai/DeepSeek-V4.1-Flash")
	joined := strings.Join(argv, " ")
	want := []string{
		"serve",
		"--provider openai",
		"--base-url http://127.0.0.1:18101/v1",
		"--api-key-env " + claudeRouterKeyEnv,
		"--model deepseek-ai/DeepSeek-V4.1-Flash",
		"--addr 127.0.0.1:18123",
		"--require-key-env " + claudeRouterFrontDoorKeyEnv,
		"--session-state off",
		"--claude",
	}
	for _, w := range want {
		if !strings.Contains(joined, w) {
			t.Errorf("front-door argv missing %q:\n%s", w, joined)
		}
	}
}

func TestClaudeRouterCredentialPlaceholder(t *testing.T) {
	t.Setenv(claudeRouterKeyEnv, "")
	if got := claudeFrontDoorRouterCredential(""); got != claudeRouterKeylessValue {
		t.Errorf("keyless credential = %q, want %q", got, claudeRouterKeylessValue)
	}
	if got := claudeFrontDoorRouterCredential("real-key"); got != "real-key" {
		t.Errorf("explicit credential = %q", got)
	}
	t.Setenv(claudeRouterKeyEnv, "env-key")
	if got := claudeFrontDoorRouterCredential(""); got != "env-key" {
		t.Errorf("env credential = %q", got)
	}
}

// TestClaudeLiveRouterDefault drives the real flag path: with a live router and
// no explicit backend, the launcher prefers the router, serves its model, and
// (dry-run) names the loopback front door.
func TestClaudeLiveRouterDefault(t *testing.T) {
	stubClaudeRouterLive(t, "router-served-model", true)
	stubClaudeRouterModel(t, "router-served-model")
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:18101")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	out := stderr.String()
	if !strings.Contains(out, "live fak router via loopback front door") {
		t.Errorf("dry-run did not name the live-router backend:\n%s", out)
	}
	if !strings.Contains(out, "model       = router-served-model") {
		t.Errorf("dry-run did not adopt the router model:\n%s", out)
	}
	if strings.Contains(out, "gateway     = http://127.0.0.1:8080") {
		t.Errorf("dry-run kept the stale 8080 default:\n%s", out)
	}
	if !strings.Contains(out, "ANTHROPIC_MODEL=router-served-model") {
		t.Errorf("child ANTHROPIC_MODEL was not the router model:\n%s", out)
	}
}

// TestClaudeNoRouterFlagOptsOut pins --no-router: the direct default is kept even
// when a router is live.
func TestClaudeNoRouterFlagOptsOut(t *testing.T) {
	stubClaudeRouterLive(t, "router-served-model", true)
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_URL", "")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe", "--no-router"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	out := stderr.String()
	if !strings.Contains(out, "gateway     = http://127.0.0.1:8080") {
		t.Errorf("--no-router did not keep the direct default:\n%s", out)
	}
	if strings.Contains(out, "live fak router") {
		t.Errorf("--no-router still used the router:\n%s", out)
	}
}

// TestClaudeExplicitGatewayBypassesRouter pins that an explicit --gateway-url
// always wins over the router default.
func TestClaudeExplicitGatewayBypassesRouter(t *testing.T) {
	stubClaudeRouterLive(t, "router-served-model", true)
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe", "--gateway-url", "http://127.0.0.1:65530", "--model", "fixture"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	out := stderr.String()
	if !strings.Contains(out, "gateway     = http://127.0.0.1:65530") {
		t.Errorf("explicit --gateway-url did not win:\n%s", out)
	}
	if strings.Contains(out, "live fak router") {
		t.Errorf("explicit --gateway-url still used the router:\n%s", out)
	}
}

// TestClaudeExplicitModelWinsOverRouter pins that an explicit --model is kept
// even when the router is live.
func TestClaudeExplicitModelWinsOverRouter(t *testing.T) {
	stubClaudeRouterLive(t, "router-served-model", true)
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_URL", "")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe", "--model", "operator-model"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	out := stderr.String()
	if !strings.Contains(out, "model       = operator-model") {
		t.Errorf("explicit --model did not win over the router:\n%s", out)
	}
}

// TestClaudeRouterLiveIsBoundedAndNonFatal: a down router must not flip the
// launcher off its direct default and must not error.
func TestClaudeRouterLiveIsBoundedAndNonFatal(t *testing.T) {
	stubClaudeRouterLive(t, "", false)
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:1")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "gateway     = http://127.0.0.1:8080") {
		t.Errorf("down router did not fall back to the direct default:\n%s", stderr.String())
	}
}

// TestClaudeFrontDoorWaitReadyStopsOnChildExit: a front door that exits during
// startup must be reported, not waited on until the deadline.
func TestClaudeFrontDoorWaitReadyStopsOnChildExit(t *testing.T) {
	exited := make(chan struct{})
	close(exited)
	client := &http.Client{Timeout: time.Second}
	err := claudeFrontDoorWaitReady(t.Context(), client, "http://127.0.0.1:1", "tok", exited, time.Second)
	if err == nil || !strings.Contains(err.Error(), "exited during startup") {
		t.Fatalf("wait ready error = %v, want child-exit error", err)
	}
}

// TestClaudeFrontDoorLogPathIsTempScoped pins the log lands under the temp dir.
func TestClaudeFrontDoorLogPathIsTempScoped(t *testing.T) {
	p := claudeFrontDoorLogPath("127.0.0.1:18123")
	if filepath.Dir(p) != os.TempDir() {
		t.Errorf("log path %q not under temp dir %q", p, os.TempDir())
	}
	if !strings.Contains(filepath.Base(p), "127.0.0.1-18123") {
		t.Errorf("log path %q does not encode the address", p)
	}
}
