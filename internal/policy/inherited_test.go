package policy

import (
	"encoding/json"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

const (
	inheritedProbeEnv  = "FAK_INHERITED_CHILD_PROBE"
	inheritedKeepEnv   = "FAK_KEEP_FOR_CHILD"
	inheritedSecretEnv = "FAK_CANARY_SECRET"
	inheritedRefEnv    = "FAK_CANARY_SECRET_REF"
	inheritedCanary    = "sk-proj-fakcanarynotforchild123456"
	inheritedRef       = "secret://synthetic/canary"
)

func TestInheritedCapabilitiesDefaultDenyStripsEnvSecretAndScopes(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"version":"fak-policy/v1","allow":["Agent"]}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.InheritedCapabilities != nil {
		t.Fatalf("Runtime.InheritedCapabilities = %+v, want nil for absent block", rt.InheritedCapabilities)
	}
	env := rt.InheritedCapabilities.ResolveLaunch("Agent", inheritedParentFixture())
	if len(env.Env) != 0 || len(env.SecretRefs) != 0 || env.CWD != "" ||
		len(env.WritablePaths) != 0 || len(env.PersistencePaths) != 0 || len(env.EgressRefs) != 0 {
		t.Fatalf("default-deny inherited envelope granted scope: %+v", env)
	}
	if rows := rt.InheritedCapabilities.Rules(); rows != nil {
		t.Fatalf("nil inherited table Rules() = %+v, want nil", rows)
	}
	if got := env.Environ(); got != nil {
		t.Fatalf("default-deny inherited Environ() = %v, want nil", got)
	}
	if s := SummaryRuntime(rt); !strings.Contains(s, "inherited launch   : (none") {
		t.Fatalf("SummaryRuntime should flag inherited default deny:\n%s", s)
	}
}

func TestInheritedCapabilitiesAllowsNonSecretEnvAndSecretRefButNotRawSecret(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["Agent"],
		"inherited_capabilities": [{
			"tool": "Agent",
			"env": ["FAK_KEEP_FOR_CHILD", "FAK_CANARY_SECRET"],
			"secret_refs": [{"env": "FAK_CANARY_SECRET_REF", "ref": "secret://synthetic/canary"}],
			"cwd": "workspace",
			"writable_paths": ["workspace/out/**"],
			"persistence_paths": ["workspace/.fak/**"],
			"egress_refs": ["research-web"]
		}]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	env := rt.InheritedCapabilities.ResolveLaunch("Agent", inheritedParentFixture())
	if got := env.Env[inheritedKeepEnv]; got != "ok" {
		t.Fatalf("allowed non-secret env %s = %q, want ok", inheritedKeepEnv, got)
	}
	if _, ok := env.Env[inheritedSecretEnv]; ok {
		t.Fatalf("secret-shaped env %s was inherited despite being marked secret", inheritedSecretEnv)
	}
	if got, want := env.SecretRefs, []SecretRefGrant{{Env: inheritedRefEnv, Ref: inheritedRef}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("secret refs = %+v, want %+v", got, want)
	}
	if env.CWD != "workspace" {
		t.Fatalf("cwd = %q, want workspace", env.CWD)
	}
	if got, want := env.WritablePaths, []string{"workspace/out/**"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("writable paths = %v, want %v", got, want)
	}
	if got, want := env.PersistencePaths, []string{"workspace/.fak/**"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("persistence paths = %v, want %v", got, want)
	}
	if got, want := env.EgressRefs, []string{"research-web"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("egress refs = %v, want %v", got, want)
	}

	seen := inheritedLaunchProbe(t, env.Environ())
	if seen[inheritedKeepEnv] != "ok" {
		t.Fatalf("child did not see allowed non-secret env %s", inheritedKeepEnv)
	}
	if seen[inheritedSecretEnv] != "" {
		t.Fatalf("child saw stripped canary secret env %s", inheritedSecretEnv)
	}
	if seen[inheritedRefEnv] != inheritedRef {
		t.Fatalf("child secret ref = %q, want the ref handoff", seen[inheritedRefEnv])
	}

	audit, err := json.Marshal(env.Audit)
	if err != nil {
		t.Fatalf("marshal audit: %v", err)
	}
	auditText := string(audit)
	for _, forbidden := range []string{inheritedCanary, inheritedRef, "workspace/out", "workspace/.fak"} {
		if strings.Contains(auditText, forbidden) {
			t.Fatalf("audit exposed raw inherited material %q", forbidden)
		}
	}
	for _, want := range []string{inheritedKeepEnv, inheritedRefEnv, "sha256:"} {
		if !strings.Contains(auditText, want) {
			t.Fatalf("audit missing bounded metadata %q in %s", want, auditText)
		}
	}
}

func TestInheritedCredentialValueShapeIsStrippedEvenWhenNameLooksBenign(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"inherited_capabilities": [{
			"tool": "Agent",
			"env": ["FAK_KEEP_FOR_CHILD", "FAK_BENIGN_NAME"]
		}]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	env := rt.InheritedCapabilities.ResolveLaunch("Agent", InheritedParent{
		Env: map[string]string{
			inheritedKeepEnv:  "ok",
			"FAK_BENIGN_NAME": inheritedCanary,
		},
	})
	if env.Env[inheritedKeepEnv] != "ok" {
		t.Fatalf("allowed non-secret env was stripped")
	}
	if _, ok := env.Env["FAK_BENIGN_NAME"]; ok {
		t.Fatalf("secret-shaped value crossed through a benign env name")
	}
}

func TestInheritedCapabilitiesValidationFailsLoud(t *testing.T) {
	cases := []struct {
		name, manifest, want string
	}{
		{"missing tool", `{"inherited_capabilities":[{"env":["SAFE"]}]}`, "tool is required"},
		{"bad env", `{"inherited_capabilities":[{"tool":"Agent","env":["BAD-NAME"]}]}`, "invalid env"},
		{"duplicate tool", `{"inherited_capabilities":[{"tool":"Agent","env":["A"]},{"tool":"Agent","env":["B"]}]}`, "duplicate row"},
		{"secret ref env clash", `{"inherited_capabilities":[{"tool":"Agent","env":["A"],"secret_refs":[{"env":"A","ref":"secret://a"}]}]}`, "already used"},
		{"empty ref", `{"inherited_capabilities":[{"tool":"Agent","secret_refs":[{"env":"A","ref":" "}]}]}`, "ref is required"},
		{"empty writable", `{"inherited_capabilities":[{"tool":"Agent","writable_paths":[" "]}]}`, "scope is required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseRuntime([]byte(tc.manifest))
			if err == nil {
				t.Fatalf("manifest should fail to load")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestInheritedCapabilitiesCatchAllAndSummary(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"inherited_capabilities": [
			{"tool": "*", "env": ["FAK_KEEP_FOR_CHILD"]},
			{"tool": "Task", "env": ["FAK_KEEP_FOR_CHILD"], "egress_refs": ["task-egress"]}
		]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	parent := inheritedParentFixture()
	if got := rt.InheritedCapabilities.ResolveLaunch("Agent", parent).Env[inheritedKeepEnv]; got != "ok" {
		t.Fatalf("catch-all env = %q, want ok", got)
	}
	if got := rt.InheritedCapabilities.ResolveLaunch("Task", parent).EgressRefs; len(got) != 0 {
		t.Fatalf("exact Task row should not inherit ungranted parent egress, got %v", got)
	}
	sum := SummaryRuntime(rt)
	if !strings.Contains(sum, "inherited launch   : 2 envelope(s)") ||
		!strings.Contains(sum, "* -> env=1 secret_refs=0 cwd=false writable=0 persistence=0 egress=0") ||
		!strings.Contains(sum, "Task -> env=1 secret_refs=0 cwd=false writable=0 persistence=0 egress=1") {
		t.Fatalf("SummaryRuntime missing inherited grants:\n%s", sum)
	}
}

func TestInheritedChildEnvProbe(t *testing.T) {
	if os.Getenv(inheritedProbeEnv) != "1" {
		return
	}
	payload := map[string]string{
		inheritedKeepEnv:   os.Getenv(inheritedKeepEnv),
		inheritedSecretEnv: os.Getenv(inheritedSecretEnv),
		inheritedRefEnv:    os.Getenv(inheritedRefEnv),
	}
	_ = json.NewEncoder(os.Stdout).Encode(payload)
	os.Exit(0)
}

func inheritedParentFixture() InheritedParent {
	return InheritedParent{
		Env: map[string]string{
			inheritedKeepEnv:   "ok",
			inheritedSecretEnv: inheritedCanary,
		},
		SecretEnv:        map[string]bool{inheritedSecretEnv: true},
		CWD:              "workspace",
		WritablePaths:    []string{"workspace/out/**", "workspace/private/**"},
		PersistencePaths: []string{"workspace/.fak/**", "workspace/.ssh/**"},
		EgressRefs:       []string{"research-web", "metadata"},
	}
}

func inheritedLaunchProbe(t *testing.T, env []string) map[string]string {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestInheritedChildEnvProbe", "--")
	cmd.Env = append(append([]string(nil), env...), inheritedProbeEnv+"=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("child env probe failed: %v", err)
	}
	var seen map[string]string
	if err := json.Unmarshal(out, &seen); err != nil {
		t.Fatalf("child env probe returned invalid JSON (%d bytes): %v", len(out), err)
	}
	return seen
}
