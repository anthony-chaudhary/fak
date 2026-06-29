// guard_test.go — parity witness for the Go dogfood-guard port, mirroring the
// guard test table in tools/dispatch_worker_test.py. Hermetic: no process is
// spawned; a real temp file stands in for the resolved `fak` binary.
package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGuardEnabledDefaultOnAndOptOut(t *testing.T) {
	if !guardEnabled(map[string]string{}) { // unset -> ON (dogfood default)
		t.Error("absent FLEET_DOGFOOD_GUARD must be ON")
	}
	for _, on := range []string{"1", "on"} {
		if !guardEnabled(map[string]string{"FLEET_DOGFOOD_GUARD": on}) {
			t.Errorf("%q must be ON", on)
		}
	}
	for _, off := range []string{"0", "off", "false", "no", "", "disable", "DISABLED", " Off "} {
		if guardEnabled(map[string]string{"FLEET_DOGFOOD_GUARD": off}) {
			t.Errorf("%q must be OFF", off)
		}
	}
}

func TestResolveFakBinPrefersEnvThenIntreeThenPathElseEmpty(t *testing.T) {
	// An explicit FAK_BIN that exists wins.
	existing := filepath.Join(t.TempDir(), "fak-stand-in")
	writeFile(t, existing, "x")
	if got := resolveFakBin("C:/nope", map[string]string{"FAK_BIN": existing}); got != existing {
		t.Errorf("explicit FAK_BIN must win: got %q", got)
	}
	// A non-existent FAK_BIN is ignored; with a bogus workspace and a PATH holding no
	// `fak`, the result is "" (the fail-open signal).
	emptyDir := t.TempDir()
	got := resolveFakBin("C:/definitely/not/a/repo/xyz", map[string]string{
		"FAK_BIN": "C:/no/such/fak", "PATH": emptyDir})
	if got != "" {
		t.Errorf("unresolvable fak must be \"\": got %q", got)
	}
}

func TestGuardProviderMapsClaudeToAnthropicElseOpenai(t *testing.T) {
	if guardProvider("claude") != "anthropic" {
		t.Error("claude -> anthropic")
	}
	if guardProvider("opencode") != "openai" {
		t.Error("opencode -> openai")
	}
}

func TestGuardAuditPathIsPerSessionUnderDispatchRuns(t *testing.T) {
	p := guardAuditPath("C:/work/fak", "gate way/1", "claude")
	if filepath.Base(filepath.Dir(p)) != "guard-audit" {
		t.Errorf("parent dir = %q, want guard-audit", filepath.Base(filepath.Dir(p)))
	}
	if filepath.Base(filepath.Dir(filepath.Dir(p))) != ".dispatch-runs" {
		t.Errorf("grandparent = %q, want .dispatch-runs", filepath.Base(filepath.Dir(filepath.Dir(p))))
	}
	name := filepath.Base(p)
	if !strings.HasSuffix(name, ".jsonl") {
		t.Errorf("name %q must end .jsonl", name)
	}
	if strings.ContainsAny(name, "/ ") {
		t.Errorf("name %q must have lane separators/spaces sanitized out", name)
	}
	if !strings.HasPrefix(name, "gate_way_1-claude-") {
		t.Errorf("name %q must keep the sanitized lane-backend prefix for globbing", name)
	}
}

func TestGuardAuditPathUniquePerCall(t *testing.T) {
	// Two workers on the SAME lane must NOT resolve to the same journal file, or their
	// independent hash chains would braid into one unverifiable file.
	a := guardAuditPath("C:/work/fak", "gateway", "claude")
	b := guardAuditPath("C:/work/fak", "gateway", "claude")
	if a == b {
		t.Errorf("per-session journal paths must differ: %q == %q", a, b)
	}
}

func TestGuardWrapClaudeFrontsWithFakGuardAnthropic(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	wrapped := guardWrap(raw, "/usr/bin/fak", "gateway", "claude", "C:/work/fak", map[string]string{})
	if wrapped[0] != "/usr/bin/fak" || wrapped[1] != "guard" {
		t.Errorf("must front with `fak guard`: %v", wrapped[:2])
	}
	if wrapped[indexOf(wrapped, "--provider")+1] != "anthropic" {
		t.Error("claude provider must be anthropic")
	}
	if !contains(wrapped, "--audit") {
		t.Error("must pass --audit")
	}
	// The raw worker argv is preserved verbatim AFTER the `--` separator.
	sep := indexOf(wrapped, "--")
	if sep < 0 || !sliceEqual(wrapped[sep+1:], raw) {
		t.Errorf("raw argv must follow `--` verbatim: sep=%d wrapped=%v", sep, wrapped)
	}
}

func TestGuardWrapNoopWithoutFakBin(t *testing.T) {
	raw, _ := buildCommand("docs", "claude")
	if got := guardWrap(raw, "", "docs", "claude", ".", map[string]string{}); !sliceEqual(got, raw) {
		t.Errorf("no fak bin -> command unchanged: %v", got)
	}
}

func TestGuardWrapOpencodeSkipsWithoutBaseURLButWrapsWithOne(t *testing.T) {
	raw, _ := buildCommand("recall", "opencode")
	// No FLEET_DOGFOOD_GUARD_BASEURL -> refuse to misroute a local-upstream worker.
	if got := guardWrap(raw, "/usr/bin/fak", "recall", "opencode", ".", map[string]string{}); !sliceEqual(got, raw) {
		t.Errorf("opencode without base url must stay unwrapped: %v", got)
	}
	// With a base URL the operator names the local upstream and we DO front it.
	wrapped := guardWrap(raw, "/usr/bin/fak", "recall", "opencode", ".",
		map[string]string{"FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:8131/v1"})
	if wrapped[0] != "/usr/bin/fak" {
		t.Errorf("opencode with base url must front with fak: %v", wrapped)
	}
	if wrapped[indexOf(wrapped, "--provider")+1] != "openai" {
		t.Error("opencode provider must be openai")
	}
	if wrapped[indexOf(wrapped, "--base-url")+1] != "http://127.0.0.1:8131/v1" {
		t.Error("base url must be forwarded")
	}
}

func TestGuardedLaunchCommandOptsOutWhenDisabled(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	fak := filepath.Join(t.TempDir(), "fak")
	writeFile(t, fak, "x")
	cmd, guarded := guardedLaunchCommand(raw, "gateway", "claude", "C:/work/fak",
		map[string]string{"FLEET_DOGFOOD_GUARD": "0", "FAK_BIN": fak})
	if guarded || !sliceEqual(cmd, raw) {
		t.Errorf("disabled -> unguarded raw command: guarded=%v cmd=%v", guarded, cmd)
	}
}

func TestGuardedLaunchCommandWrapsWhenEnabledAndBinPresent(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	fak := filepath.Join(t.TempDir(), "fak")
	writeFile(t, fak, "x")
	cmd, guarded := guardedLaunchCommand(raw, "gateway", "claude", "C:/work/fak",
		map[string]string{"FAK_BIN": fak})
	if !guarded || cmd[0] != fak || cmd[1] != "guard" {
		t.Errorf("enabled + bin -> guarded fak-fronted command: guarded=%v cmd=%v", guarded, cmd)
	}
}

func TestGuardEnvAugmentSetsTimeoutFloorsWithoutClobbering(t *testing.T) {
	env := map[string]string{"FAK_PLANNER_TIMEOUT_S": "1200"}
	guardEnvAugment(env)
	if env["FAK_PLANNER_TIMEOUT_S"] != "1200" { // explicit value kept
		t.Error("explicit planner timeout must be kept")
	}
	if env["FAK_HTTP_WRITE_TIMEOUT_S"] != "600" {
		t.Errorf("write timeout floor must be set: %q", env["FAK_HTTP_WRITE_TIMEOUT_S"])
	}
}

func TestBuildPayloadCarriesGuardedAndExplicitCommand(t *testing.T) {
	p := buildPayload("gateway", "claude", "C:/work/fak", true, nil, "",
		[]string{"fak", "guard", "--", "claude"}, true)
	if !p.Guarded {
		t.Error("payload must carry guarded=true")
	}
	if p.Command[0] != "fak" {
		t.Errorf("payload must carry the explicit fronted command: %v", p.Command)
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
