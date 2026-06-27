package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// hermeticTargets points FAK_TARGETS_FILE at a missing path so a test only ever sees
// the four built-in compute targets, never a developer's ~/.fak/targets.json.
func hermeticTargets(t *testing.T) {
	t.Helper()
	t.Setenv("FAK_TARGETS_FILE", filepath.Join(t.TempDir(), "no-such-targets.json"))
	// Force the documented defaults for the env-sourced built-ins (an inherited
	// FAK_MAC_GATEWAY etc. on the runner would skew the expected URLs).
	for _, name := range []string{"FAK_MAC_GATEWAY", "FAK_MAC_MODEL", "FAK_GLM_GCP_BASE_URL", "FAK_GLM_GCP_MODEL", "FAK_LOCAL_GATEWAY", "ANTHROPIC_BASE_URL"} {
		t.Setenv(name, "")
	}
}

// TestResolveLeadingTarget is the core back-compat/footgun unit: a leading token that
// NAMES a target is stripped and returned; an unknown token (or a flag) is left in place
// so it still forwards to claude verbatim.
func TestResolveLeadingTarget(t *testing.T) {
	hermeticTargets(t)
	reg, err := loadComputeTargets(defaultComputeTargetsFile())
	if err != nil {
		t.Fatalf("loadComputeTargets: %v", err)
	}
	var sink bytes.Buffer

	// Known leading token -> stripped, rest preserved.
	name, rest := resolveLeadingTarget([]string{"mac", "--json", "extra"}, reg, &sink)
	if name != "mac" || strings.Join(rest, " ") != "--json extra" {
		t.Fatalf("known token: name=%q rest=%v, want mac + [--json extra]", name, rest)
	}
	// Unknown token -> left in place (forwards to claude), name empty.
	name, rest = resolveLeadingTarget([]string{"zzznope", "--foo"}, reg, &sink)
	if name != "" || strings.Join(rest, " ") != "zzznope --foo" {
		t.Fatalf("unknown token: name=%q rest=%v, want passthrough", name, rest)
	}
	// A leading flag is never a target token.
	name, rest = resolveLeadingTarget([]string{"--json", "mac"}, reg, &sink)
	if name != "" || strings.Join(rest, " ") != "--json mac" {
		t.Fatalf("leading flag: name=%q rest=%v, want untouched", name, rest)
	}
	// Empty argv is tolerated.
	if name, rest := resolveLeadingTarget(nil, reg, &sink); name != "" || len(rest) != 0 {
		t.Fatalf("empty argv: name=%q rest=%v", name, rest)
	}
	// A close-but-unknown token earns a "did you mean" hint (still passes through).
	sink.Reset()
	if name, _ := resolveLeadingTarget([]string{"anthropi"}, reg, &sink); name != "" {
		t.Fatalf("close token should not resolve as a target: %q", name)
	}
	if !strings.Contains(sink.String(), "did you mean \"anthropic\"") {
		t.Fatalf("expected a did-you-mean hint for a close token, got: %q", sink.String())
	}
}

// TestTUIAgentTargetResolvesBuiltins is the table test over the four built-in targets:
// `fak c <name> --json` resolves each to the right launch plan (gateway/model/provider)
// and records the named target in the report.
func TestTUIAgentTargetResolvesBuiltins(t *testing.T) {
	cases := []struct {
		target     string
		provider   string
		gatewayURL string // "" => default guard path (no gateway env)
		model      string
		wantGuard  bool
	}{
		{"mac", "existing-fak-gateway", "http://node-macos-a.local:8080", defaultClaudeMacModel, false},
		{"gcp", "existing-fak-gateway", "http://127.0.0.1:8200", "glm-5.2", false},
		{"local", "existing-fak-gateway", "http://127.0.0.1:8080", "", false},
		{"anthropic", "anthropic", "", "claude-opus-4-8", true},
	}
	for _, c := range cases {
		t.Run(c.target, func(t *testing.T) {
			hermeticTargets(t)
			t.Setenv("FAK_GATEWAY_KEY", "test-bearer") // mac is remote and needs a bearer
			var stdout, stderr bytes.Buffer
			code := runTUI(&stdout, &stderr, []string{"agent", c.target, "--json"})
			if code != 0 {
				t.Fatalf("runTUI agent %s --json code=%d stderr=%s", c.target, code, stderr.String())
			}
			var report tuiAgentReport
			if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
				t.Fatalf("unmarshal report: %v\n%s", err, stdout.String())
			}
			if report.Target != c.target {
				t.Errorf("report.Target = %q, want %q", report.Target, c.target)
			}
			if report.Provider != c.provider {
				t.Errorf("provider = %q, want %q", report.Provider, c.provider)
			}
			if report.GatewayURL != c.gatewayURL {
				t.Errorf("gateway = %q, want %q", report.GatewayURL, c.gatewayURL)
			}
			if c.model != "" && report.Model != c.model {
				t.Errorf("model = %q, want %q", report.Model, c.model)
			}
			if c.wantGuard {
				if !hasTUIString(report.Launch, "guard") || !hasTUIString(report.Launch, "--provider") || !hasTUIString(report.Launch, "anthropic") {
					t.Errorf("anthropic target should route through fak guard: %v", report.Launch)
				}
			} else if hasTUIString(report.Launch, "guard") {
				t.Errorf("gateway target should not start a local guard: %v", report.Launch)
			}
		})
	}
}

// TestTUIAgentTargetFlagMatchesPositional proves `--target NAME` resolves identically to
// the leading `fak c NAME` form.
func TestTUIAgentTargetFlagMatchesPositional(t *testing.T) {
	hermeticTargets(t)
	t.Setenv("FAK_GATEWAY_KEY", "test-bearer")

	reportFor := func(args []string) tuiAgentReport {
		t.Helper()
		var stdout, stderr bytes.Buffer
		if code := runTUI(&stdout, &stderr, args); code != 0 {
			t.Fatalf("runTUI %v code=%d stderr=%s", args, code, stderr.String())
		}
		var r tuiAgentReport
		if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
			t.Fatalf("unmarshal %v: %v\n%s", args, err, stdout.String())
		}
		return r
	}
	pos := reportFor([]string{"agent", "mac", "--json"})
	flagForm := reportFor([]string{"agent", "--target", "mac", "--json"})
	if pos.GatewayURL != flagForm.GatewayURL || pos.Model != flagForm.Model || pos.Target != flagForm.Target {
		t.Fatalf("positional %+v != --target %+v", pos, flagForm)
	}
	if flagForm.Target != "mac" {
		t.Fatalf("--target form target = %q, want mac", flagForm.Target)
	}
}

// TestTUIAgentExplicitFlagOverridesTarget proves a user-set flag (here --model) is NOT
// clobbered by the resolved target's value.
func TestTUIAgentExplicitFlagOverridesTarget(t *testing.T) {
	hermeticTargets(t)
	t.Setenv("FAK_GATEWAY_KEY", "test-bearer")
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"agent", "mac", "--model", "my-override", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report tuiAgentReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Model != "my-override" {
		t.Fatalf("explicit --model lost to target default: model=%q", report.Model)
	}
}

// TestTUIAgentUnknownTokenForwardsToClaude is the back-compat guard: an unknown token is
// forwarded to claude unchanged and selects NO target (Target empty, default path).
func TestTUIAgentUnknownTokenForwardsToClaude(t *testing.T) {
	hermeticTargets(t)
	var stdout, stderr bytes.Buffer
	// Flags first so --json is parsed before the unknown positional reaches claude args.
	code := runTUI(&stdout, &stderr, []string{"agent", "--json", "zzznope"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report tuiAgentReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Target != "" {
		t.Errorf("unknown token selected a target: %q", report.Target)
	}
	if report.Provider != "anthropic" {
		t.Errorf("unknown token should fall through to the default anthropic path, got provider %q", report.Provider)
	}
	if !hasTUIString(report.Command, "zzznope") {
		t.Errorf("unknown token should be forwarded to claude, command=%v", report.Command)
	}
}

// TestTUIAgentTargetConflict proves a positional target and a differing --target flag
// are rejected rather than silently picking one.
func TestTUIAgentTargetConflict(t *testing.T) {
	hermeticTargets(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"agent", "mac", "--target", "gcp", "--json"})
	if code != 2 {
		t.Fatalf("conflicting targets code=%d (want 2) stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "conflicting target") {
		t.Fatalf("stderr = %q, want a conflicting-target error", stderr.String())
	}
}

// TestTUIAgentUnknownTargetFlagErrors proves an explicit --target with an unknown name is
// an error (unlike an unknown positional, which passes through), with a hint.
func TestTUIAgentUnknownTargetFlagErrors(t *testing.T) {
	hermeticTargets(t)
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"agent", "--target", "anthropi", "--json"})
	if code != 2 {
		t.Fatalf("unknown --target code=%d (want 2) stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "unknown --target") || !strings.Contains(stderr.String(), "did you mean") {
		t.Fatalf("stderr = %q, want unknown+hint", stderr.String())
	}
}

// TestTUIAgentLocalTargetToleratesNoBearer proves the loopback `local` target launches
// without demanding a bogus gateway key — the loopback bearer relaxation.
func TestTUIAgentLocalTargetToleratesNoBearer(t *testing.T) {
	hermeticTargets(t)
	t.Setenv("FAK_GATEWAY_KEY", "") // no key at all
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"agent", "local", "--json"})
	if code != 0 {
		t.Fatalf("local target without a key should still resolve, code=%d stderr=%s", code, stderr.String())
	}
	var report tuiAgentReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if report.Auth != "none" {
		t.Errorf("loopback no-bearer auth = %q, want none", report.Auth)
	}
	for _, e := range report.Env {
		if e.Name == "ANTHROPIC_API_KEY" {
			t.Errorf("loopback no-bearer launch must not inject an empty ANTHROPIC_API_KEY: %+v", report.Env)
		}
	}
}

// TestPreflightComputeTargetGatesDeadGateway proves the interactive-launch preflight: an
// unreachable resolved gateway is gated (no launch), a live one proceeds, and a target
// with no /healthz (the real Anthropic API) never blocks.
func TestPreflightComputeTargetGatesDeadGateway(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer up.Close()
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer down.Close()

	var so, se bytes.Buffer
	upTgt := computeTarget{Name: "up", Kind: targetGatewayURL, GatewayURL: up.URL, Locality: localityLocal, HealthzPath: "/healthz"}
	if code, gated := preflightComputeTarget(&so, &se, upTgt); gated || code != 0 {
		t.Fatalf("live gateway gated=%v code=%d, want proceed", gated, code)
	}

	so.Reset()
	se.Reset()
	downTgt := computeTarget{Name: "down", Kind: targetGatewayURL, GatewayURL: down.URL, Locality: localityLocal, HealthzPath: "/healthz"}
	if code, gated := preflightComputeTarget(&so, &se, downTgt); !gated || code != 1 {
		t.Fatalf("dead gateway gated=%v code=%d, want gated with exit 1", gated, code)
	}
	if !strings.Contains(se.String(), "unreachable") {
		t.Fatalf("dead-gateway stderr = %q, want an unreachable message", se.String())
	}

	so.Reset()
	se.Reset()
	naTgt := computeTarget{Name: "anthropic", Kind: targetProviderProxy, GatewayURL: "https://api.anthropic.com", Locality: localityRemote}
	if code, gated := preflightComputeTarget(&so, &se, naTgt); gated || code != 0 {
		t.Fatalf("no-healthz target gated=%v code=%d, want proceed (n/a)", gated, code)
	}
}

// TestTUIAgentTargetDryRunShowsResolvedPlan proves --dry-run renders the resolved target
// → gateway/model plan (not a network launch).
func TestTUIAgentTargetDryRunShowsResolvedPlan(t *testing.T) {
	hermeticTargets(t)
	t.Setenv("FAK_GATEWAY_KEY", "test-bearer")
	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"agent", "gcp", "--dry-run", "--width", "1000"})
	if code != 0 {
		t.Fatalf("gcp dry-run code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"existing-fak-gateway", "http://127.0.0.1:8200", "glm-5.2"} {
		if !strings.Contains(out, want) {
			t.Fatalf("gcp dry-run missing %q:\n%s", want, out)
		}
	}
}
