package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func useGuardLaunchToolGrant(t *testing.T, names ...string) {
	t.Helper()
	previous := launchToolGrant()
	setLaunchToolGrant(names)
	t.Cleanup(func() { setLaunchToolGrant(previous.Allow) })
}

func isolateGuardLaunchGrantOverlays(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv(guardAllowOverlayEnv, filepath.Join(dir, "allow.json"))
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
}

func preserveGuardDefaultPolicy(t *testing.T) {
	t.Helper()
	previous := adjudicator.Default.PolicySnapshot()
	t.Cleanup(func() { adjudicator.Default.SetPolicy(previous) })
}

func guardLaunchGrantVerdict(t *testing.T, rt policy.Runtime, tool, args string) abi.Verdict {
	t.Helper()
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("no active ref resolver")
	}
	ref, err := res.Put(context.Background(), []byte(args))
	if err != nil {
		t.Fatalf("store %s args: %v", tool, err)
	}
	return adjudicator.New(rt.Adjudicator).Adjudicate(context.Background(), &abi.ToolCall{Tool: tool, Args: ref})
}

func TestGuardLaunchToolFlagCollectsExactNames(t *testing.T) {
	var flags launchToolFlag
	if err := flags.Set(" deploy_preview "); err != nil {
		t.Fatal(err)
	}
	if err := flags.Set("publish_docs"); err != nil {
		t.Fatal(err)
	}
	if got := flags.String(); got != "deploy_preview,publish_docs" {
		t.Fatalf("flag value = %q, want both exact names", got)
	}
	if err := flags.Set("  "); err == nil {
		t.Fatal("blank --allow-tool value was accepted")
	}
}

func TestGuardLaunchGrantAllowsExactToolWithoutSofteningHardRules(t *testing.T) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	grant := guardAllowOverlay{Allow: []string{"deploy_preview", "opencode.bash", "delete_file", "write_file"}}
	if added := guardApplyAllowOverlay(&rt, grant); added < 1 {
		t.Fatalf("launch grant added %d tools, want the previously unknown exact capability", added)
	}

	if got := guardLaunchGrantVerdict(t, rt, "deploy_preview", `{}`); got.Kind != abi.VerdictAllow {
		t.Fatalf("exact launch grant verdict = %v/%s, want ALLOW", got.Kind, abi.ReasonName(got.Reason))
	}
	hardCases := []struct {
		name   string
		tool   string
		args   string
		reason abi.ReasonCode
	}{
		{"shell danger", "opencode.bash", `{"command":"rm -rf /"}`, abi.ReasonPolicyBlock},
		{"explicit deny", "delete_file", `{"path":"notes.txt"}`, abi.ReasonPolicyBlock},
		{"self modification", "write_file", `{"path":".hermes/config.yaml","content":"x"}`, abi.ReasonSelfModify},
	}
	for _, tc := range hardCases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardLaunchGrantVerdict(t, rt, tc.tool, tc.args)
			if got.Kind != abi.VerdictDeny || got.Reason != tc.reason {
				t.Fatalf("%s grant verdict = %v/%s, want DENY/%s", tc.tool, got.Kind, abi.ReasonName(got.Reason), abi.ReasonName(tc.reason))
			}
		})
	}
}

func TestGuardLaunchGrantIsAttestedAndPreservedByReloads(t *testing.T) {
	isolateGuardLaunchGrantOverlays(t)
	useGuardLaunchToolGrant(t, "deploy_preview")
	preserveGuardDefaultPolicy(t)

	rt, source, launchDigest, _ := loadGuardCapabilityFloor("")
	if !rt.Adjudicator.Allow["deploy_preview"] {
		t.Fatal("initial floor did not install the launch grant")
	}
	if !strings.Contains(source, "launch-scoped allow grant") || !strings.Contains(source, "expires with this process") {
		t.Fatalf("floor provenance does not disclose launch grant lifetime: %s", source)
	}
	wantDigest := guardEffectivePolicyDigest(guardDefaultPolicyJSON, launchToolGrant(), guardDenyOverlay{})
	if launchDigest != wantDigest || launchDigest == guardPolicyDigest(guardDefaultPolicyJSON) {
		t.Fatalf("launch digest = %q, want grant-aware %q distinct from base", launchDigest, wantDigest)
	}

	defaultResp, err := guardPolicyReloader("")(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if defaultResp.EffectiveDigest != wantDigest {
		t.Fatalf("default reload digest = %q, want %q", defaultResp.EffectiveDigest, wantDigest)
	}
	if got := adjudicator.Default.Adjudicate(context.Background(), guardToolCall(t, "deploy_preview", map[string]any{})); got.Kind != abi.VerdictAllow {
		t.Fatalf("default reload lost launch grant: %v/%s", got.Kind, abi.ReasonName(got.Reason))
	}

	policyPath := filepath.Join(t.TempDir(), "floor.json")
	if err := os.WriteFile(policyPath, guardDefaultPolicyJSON, 0o600); err != nil {
		t.Fatal(err)
	}
	explicitResp, err := guardPolicyReloader(policyPath)(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if explicitResp.EffectiveDigest != wantDigest {
		t.Fatalf("explicit reload digest = %q, want %q", explicitResp.EffectiveDigest, wantDigest)
	}
	if got := adjudicator.Default.Adjudicate(context.Background(), guardToolCall(t, "deploy_preview", map[string]any{})); got.Kind != abi.VerdictAllow {
		t.Fatalf("explicit reload lost launch grant: %v/%s", got.Kind, abi.ReasonName(got.Reason))
	}
}

func TestGuardLaunchGrantCLIHelperProcess(t *testing.T) {
	if os.Getenv("GUARD_LAUNCH_GRANT_HELPER") != "1" {
		return
	}
	cmdGuard([]string{
		"--allow-tool", "deploy_preview",
		"--allow-tool", "opencode.bash",
		"--replay-trace", os.Getenv("GUARD_LAUNCH_GRANT_FIXTURE"),
		"--audit", "off",
	})
}

func TestGuardLaunchGrantCLIReplayCapturesGrantAndHardDeny(t *testing.T) {
	t.Cleanup(journal.ResetActiveForTest)

	fixturePath := filepath.Join(t.TempDir(), "launch-grant-trace.json")
	fixture := `{
  "slice_id": "guard-launch-grant",
  "turns": [{
    "usage": {"input_tokens": 20, "output_tokens": 5},
    "calls": [
      {"id": "granted", "tool": "deploy_preview", "args": {}, "class": "allow"},
      {"id": "hard-deny", "tool": "opencode.bash", "args": {"command": "rm -rf /"}, "class": "deny", "reason": "POLICY_BLOCK"}
    ]
  }]
}`
	if err := os.WriteFile(fixturePath, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestGuardLaunchGrantCLIHelperProcess$")
	cmd.Env = append(os.Environ(),
		"GUARD_LAUNCH_GRANT_HELPER=1",
		"GUARD_LAUNCH_GRANT_FIXTURE="+fixturePath,
		guardAllowOverlayEnv+"="+filepath.Join(t.TempDir(), "allow.json"),
		guardDenyOverlayEnv+"="+filepath.Join(t.TempDir(), "deny.json"),
	)
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		t.Fatalf("launch-grant CLI replay failed: %v\n%s", err, out.String())
	}
	for _, want := range []string{
		"launch-scoped allow grant (2 exact tool(s))",
		"deploy_preview",
		"ALLOW",
		"opencode.bash",
		"DENY[POLICY_BLOCK]",
		"every call landed on its expected disposition",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("captured replay missing %q:\n%s", want, out.String())
		}
	}
}

func TestGuardNoLaunchGrantKeepsHistoricalDigest(t *testing.T) {
	isolateGuardLaunchGrantOverlays(t)
	useGuardLaunchToolGrant(t)
	preserveGuardDefaultPolicy(t)
	_, source, digest, _ := loadGuardCapabilityFloor("")
	if strings.Contains(source, "launch-scoped allow grant") {
		t.Fatalf("empty grant changed provenance: %s", source)
	}
	if want := guardPolicyDigest(guardDefaultPolicyJSON); digest != want {
		t.Fatalf("empty grant digest = %q, want historical %q", digest, want)
	}
	resp, err := guardPolicyReloader("")(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if want := "built-in guard floor + operator allow overlay"; resp.Source != want {
		t.Fatalf("empty grant reload source = %q, want historical %q", resp.Source, want)
	}
}
