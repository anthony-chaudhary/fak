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
		"--model-passthrough",
	}
	for _, w := range want {
		if !strings.Contains(joined, w) {
			t.Errorf("front-door argv missing %q:\n%s", w, joined)
		}
	}
}

// isolateClaudeHome points os.UserHomeDir at an empty temp dir so credential
// discovery never sees this host's real ~/.fak/router/gateway.key.
func isolateClaudeHome(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("USERPROFILE", dir)
	t.Setenv("HOME", dir)
	return dir
}

func TestClaudeRouterCredentialPlaceholder(t *testing.T) {
	isolateClaudeHome(t)
	t.Setenv(claudeRouterKeyEnv, "")
	t.Setenv(claudeRouterGatewayKeyEnv, "")
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

// TestClaudeRouterGatewayKeyDiscovery pins the discovery order: the operator
// env var wins over the supervised gateway key file; the file is read and
// trimmed when the env var is unset; nothing found leaves discovery empty so
// the caller falls back to the keyless placeholder.
func TestClaudeRouterGatewayKeyDiscovery(t *testing.T) {
	home := isolateClaudeHome(t)
	keyPath := filepath.Join(home, ".fak", "router", "gateway.key")
	if err := os.MkdirAll(filepath.Dir(keyPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyPath, []byte("file-gateway-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(claudeRouterGatewayKeyEnv, "")
	if got := claudeRouterDiscoveredKey(); got != "file-gateway-key" {
		t.Errorf("discovered key = %q, want the trimmed file value", got)
	}
	t.Setenv(claudeRouterGatewayKeyEnv, "env-gateway-key")
	if got := claudeRouterDiscoveredKey(); got != "env-gateway-key" {
		t.Errorf("discovered key = %q, want the env value to win", got)
	}
	t.Setenv(claudeRouterGatewayKeyEnv, "")
	if err := os.Remove(keyPath); err != nil {
		t.Fatal(err)
	}
	if got := claudeRouterDiscoveredKey(); got != "" {
		t.Errorf("discovered key = %q after removing the file, want empty", got)
	}
}

// TestClaudeRouterCredentialOrderWithDiscovery pins that the front-door
// upstream credential prefers the explicit value, then FAK_CLAUDE_ROUTER_KEY,
// then the discovered gateway key, then the keyless placeholder.
func TestClaudeRouterCredentialOrderWithDiscovery(t *testing.T) {
	home := isolateClaudeHome(t)
	keyPath := filepath.Join(home, ".fak", "router", "gateway.key")
	if err := os.MkdirAll(filepath.Dir(keyPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyPath, []byte("file-gateway-key"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(claudeRouterKeyEnv, "")
	t.Setenv(claudeRouterGatewayKeyEnv, "")
	if got := claudeFrontDoorRouterCredential(""); got != "file-gateway-key" {
		t.Errorf("credential = %q, want the discovered gateway key", got)
	}
	if got := claudeFrontDoorRouterCredential("explicit"); got != "explicit" {
		t.Errorf("credential = %q, want the explicit value", got)
	}
	t.Setenv(claudeRouterKeyEnv, "frontdoor-env-key")
	if got := claudeFrontDoorRouterCredential(""); got != "frontdoor-env-key" {
		t.Errorf("credential = %q, want FAK_CLAUDE_ROUTER_KEY over the file", got)
	}
}

// TestClaudeRouterAuthHeader pins that probes send Bearer only for real keys:
// the keyless placeholder sends no Authorization header at all.
func TestClaudeRouterAuthHeader(t *testing.T) {
	isolateClaudeHome(t)
	t.Setenv(claudeRouterKeyEnv, "")
	t.Setenv(claudeRouterGatewayKeyEnv, "")
	if got := claudeRouterAuthHeader(); got != "" {
		t.Errorf("auth header with only the placeholder = %q, want empty", got)
	}
	t.Setenv(claudeRouterKeyEnv, "some-key")
	if got := claudeRouterAuthHeader(); got != "Bearer some-key" {
		t.Errorf("auth header = %q, want Bearer some-key", got)
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

// TestClaudeGuardJoinsLiveRouterFrontDoor pins the guarded end-to-end path: with
// --guard and a live router and no explicit backend, the launcher starts the same
// loopback front door (no dead-8080 fallback) and the guard command names
// ANTHROPIC_API_KEY as the upstream key env, so the guard's gateway forwards the
// per-launch front-door bearer instead of the Claude subscription OAuth token.
func TestClaudeGuardJoinsLiveRouterFrontDoor(t *testing.T) {
	stubClaudeRouterLive(t, "router-served-model", true)
	stubClaudeRouterModel(t, "router-served-model")
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:18101")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--guard", "--dry-run", "--no-probe"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	out := stderr.String()
	if !strings.Contains(out, "harness     = claude (fak guard enabled)") {
		t.Errorf("guarded dry-run did not report the guard harness:\n%s", out)
	}
	if !strings.Contains(out, "backend     = live fak router via loopback front door") {
		t.Errorf("guarded dry-run did not name the live-router front door:\n%s", out)
	}
	if strings.Contains(out, "gateway     = http://127.0.0.1:8080") {
		t.Errorf("guarded dry-run fell back to the stale 8080 default:\n%s", out)
	}
	cmd := stdout.String()
	for _, want := range []string{
		"guard --provider anthropic",
		"--base-url http://127.0.0.1:<free>",
		"--model router-served-model",
		"--api-key-env ANTHROPIC_API_KEY",
	} {
		if !strings.Contains(cmd, want) {
			t.Errorf("guarded front-door command missing %q:\n%s", want, cmd)
		}
	}
}

// TestClaudeGuardExplicitGatewayKeepsSubscriptionDefault pins the non-router
// guard path: an explicit --gateway-url keeps guard's historical upstream
// contract (subscription OAuth default; no api-key-env injected).
func TestClaudeGuardExplicitGatewayKeepsSubscriptionDefault(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--guard", "--dry-run", "--no-probe", "--gateway-url", "http://127.0.0.1:65530", "--model", "fixture"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	cmd := stdout.String()
	if strings.Contains(cmd, "--api-key-env") {
		t.Errorf("explicit-backend guard injected an api-key-env; subscription default must stay:\n%s", cmd)
	}
	if !strings.Contains(cmd, "--base-url http://127.0.0.1:65530") {
		t.Errorf("guard command lost the explicit gateway:\n%s", cmd)
	}
}

// TestClaudeFrontDoorShadowsOperatorKeyEnvNotice pins the operator notice: when
// an operator names --api-key-env on a live-router front-door launch, stderr
// says the front-door bearer shadows it, and --quiet suppresses the notice.
func TestClaudeFrontDoorShadowsOperatorKeyEnvNotice(t *testing.T) {
	stubClaudeRouterLive(t, "router-served-model", true)
	stubClaudeRouterModel(t, "router-served-model")
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_ROUTER_URL", "http://127.0.0.1:18101")

	var stdout, stderr bytes.Buffer
	code := runClaude(&stdout, &stderr, []string{"--dry-run", "--no-probe", "--api-key-env", "OPERATOR_KEY"})
	if code != 0 {
		t.Fatalf("runClaude = %d; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "--api-key-env OPERATOR_KEY is shadowed on the live-router front door") {
		t.Errorf("front-door launch did not name the shadowed key env:\n%s", stderr.String())
	}

	var qOut, qErr bytes.Buffer
	if code := runClaude(&qOut, &qErr, []string{"--dry-run", "--no-probe", "--quiet", "--api-key-env", "OPERATOR_KEY"}); code != 0 {
		t.Fatalf("runClaude --quiet = %d; stderr=%s", code, qErr.String())
	}
	if strings.Contains(qErr.String(), "shadowed") {
		t.Errorf("--quiet did not suppress the shadow notice:\n%s", qErr.String())
	}
}

// TestClaudeGuardLaunchArgvFrontDoorPrecedence pins the argv switch directly:
// the front door's bearer wins over an operator-named key env, while a
// non-front-door launch keeps the operator's env and the bare default.
func TestClaudeGuardLaunchArgvFrontDoorPrecedence(t *testing.T) {
	child := []string{"claude", "--dangerously-skip-permissions"}

	frontDoor := buildClaudeGuardLaunchArgv("fak", claudeLaunchOptions{serverURL: "http://127.0.0.1:<free>", model: "m", apiKeyEnv: "OPERATOR_KEY", frontDoor: true}, child)
	if !strings.Contains(strings.Join(frontDoor, " "), "--api-key-env ANTHROPIC_API_KEY") {
		t.Errorf("front-door guard argv did not pin ANTHROPIC_API_KEY: %v", frontDoor)
	}
	if strings.Contains(strings.Join(frontDoor, " "), "OPERATOR_KEY") {
		t.Errorf("front-door guard argv leaked the operator key env: %v", frontDoor)
	}

	explicit := buildClaudeGuardLaunchArgv("fak", claudeLaunchOptions{serverURL: "http://127.0.0.1:8080", apiKeyEnv: "OPERATOR_KEY"}, child)
	if !strings.Contains(strings.Join(explicit, " "), "--api-key-env OPERATOR_KEY") {
		t.Errorf("explicit key env lost on the direct-backend guard: %v", explicit)
	}

	bare := buildClaudeGuardLaunchArgv("fak", claudeLaunchOptions{serverURL: "http://127.0.0.1:8080"}, child)
	if strings.Contains(strings.Join(bare, " "), "--api-key-env") {
		t.Errorf("bare guard argv must keep the subscription OAuth default: %v", bare)
	}
}
