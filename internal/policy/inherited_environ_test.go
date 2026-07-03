package policy

import (
	"encoding/json"
	"strings"
	"testing"
)

// canaries for the live-environ helper tests. The *_VALUE constants are the raw
// secret material that must NEVER reach a child or a marshaled audit.
const (
	environAllowedName  = "FAK_KEEP_FOR_CHILD"
	environAllowedValue = "ok"

	environAPIKeyName   = "ANTHROPIC_API_KEY"
	environAPIKeyValue  = "sk-ant-fakcanarynotforchild-0001"
	environAWSName      = "AWS_SECRET_ACCESS_KEY"
	environAWSValue     = "fakcanaryawssecretnotforchild-0002"
	environSessionToken = "SESSION_TOKEN"
	environTokenValue   = "fakcanarytokennotforchild-0003"
)

func environFixture() []string {
	return []string{
		environAllowedName + "=" + environAllowedValue,
		environAPIKeyName + "=" + environAPIKeyValue,
		environAWSName + "=" + environAWSValue,
		environSessionToken + "=" + environTokenValue,
		"MALFORMED_NO_EQUALS",
		"=leadingequalsisdropped",
	}
}

// TestNewInheritedParentFromEnvironFlagsSecretNames: the environ parser marks
// KEY/SECRET/TOKEN-named variables secret and leaves benign ones unflagged, and
// drops malformed entries.
func TestNewInheritedParentFromEnvironFlagsSecretNames(t *testing.T) {
	p := NewInheritedParentFromEnviron(environFixture())
	if p.Env[environAllowedName] != environAllowedValue {
		t.Fatalf("benign env %s = %q, want %q", environAllowedName, p.Env[environAllowedName], environAllowedValue)
	}
	for _, secretName := range []string{environAPIKeyName, environAWSName, environSessionToken} {
		if !p.SecretEnv[secretName] {
			t.Fatalf("secret-named env %s was not flagged secret", secretName)
		}
	}
	if p.SecretEnv[environAllowedName] {
		t.Fatalf("benign env %s was wrongly flagged secret", environAllowedName)
	}
	if _, ok := p.Env["MALFORMED_NO_EQUALS"]; ok {
		t.Fatalf("malformed environ entry without '=' was parsed")
	}
	if _, ok := p.Env[""]; ok {
		t.Fatalf("leading-'=' environ entry produced an empty-name env var")
	}
}

// TestResolveFromEnvironStripsSecretNamedEnvEvenWhenAllowlisted: a rule with a
// broad env allow-list that names the secret variables still does NOT hand their
// values to the child, because the environ parser flagged them secret; the benign
// allowlisted variable IS passed; and the audit carries no raw secret value.
func TestResolveFromEnvironStripsSecretNamedEnvEvenWhenAllowlisted(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["Agent"],
		"inherited_capabilities": [{
			"tool": "Agent",
			"env": ["FAK_KEEP_FOR_CHILD", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "SESSION_TOKEN"]
		}]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	env := rt.InheritedCapabilities.ResolveFromEnviron("Agent", environFixture(), InheritedParent{})

	if got := env.Env[environAllowedName]; got != environAllowedValue {
		t.Fatalf("benign allowlisted env %s = %q, want %q", environAllowedName, got, environAllowedValue)
	}
	for _, secretName := range []string{environAPIKeyName, environAWSName, environSessionToken} {
		if _, ok := env.Env[secretName]; ok {
			t.Fatalf("secret-named env %s was granted to child despite broad allow-list", secretName)
		}
	}

	audit, err := json.Marshal(env.Audit)
	if err != nil {
		t.Fatalf("marshal audit: %v", err)
	}
	auditText := string(audit)
	for _, canary := range []string{environAPIKeyValue, environAWSValue, environTokenValue} {
		if strings.Contains(auditText, canary) {
			t.Fatalf("audit exposed a raw secret value %q", canary)
		}
	}
	if !strings.Contains(auditText, environAllowedName) {
		t.Fatalf("audit missing the benign env name %s in %s", environAllowedName, auditText)
	}

	// The rendered child environ must carry the benign value and none of the secrets.
	childEnviron := strings.Join(env.Environ(), "\n")
	if !strings.Contains(childEnviron, environAllowedName+"="+environAllowedValue) {
		t.Fatalf("child environ missing benign %s: %q", environAllowedName, childEnviron)
	}
	for _, canary := range []string{environAPIKeyValue, environAWSValue, environTokenValue} {
		if strings.Contains(childEnviron, canary) {
			t.Fatalf("child environ leaked a raw secret value %q", canary)
		}
	}
}

// TestResolveFromEnvironDefaultDenyWithNilTable: with NO inherited_capabilities
// block (nil table), the live-environ helper returns an empty envelope end to end.
func TestResolveFromEnvironDefaultDenyWithNilTable(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"version":"fak-policy/v1","allow":["Agent"]}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	if rt.InheritedCapabilities != nil {
		t.Fatalf("Runtime.InheritedCapabilities = %+v, want nil for absent block", rt.InheritedCapabilities)
	}
	env := rt.InheritedCapabilities.ResolveFromEnviron("Agent", environFixture(), InheritedParent{})
	if len(env.Env) != 0 || len(env.SecretRefs) != 0 || env.CWD != "" ||
		len(env.WritablePaths) != 0 || len(env.PersistencePaths) != 0 || len(env.EgressRefs) != 0 {
		t.Fatalf("nil-table live-environ resolve granted scope: %+v", env)
	}
	if got := env.Environ(); got != nil {
		t.Fatalf("nil-table live-environ Environ() = %v, want nil", got)
	}
}

// TestResolveFromEnvironOverlaysCallerScope: the caller-supplied non-environ
// scope (cwd/paths/egress) and extra secret flags overlay the environ-derived
// parent and pass through Resolve's intersection unchanged.
func TestResolveFromEnvironOverlaysCallerScope(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{
		"version": "fak-policy/v1",
		"allow": ["Agent"],
		"inherited_capabilities": [{
			"tool": "Agent",
			"env": ["FAK_KEEP_FOR_CHILD", "PLAIN_NAME"],
			"cwd": "workspace",
			"writable_paths": ["workspace/out/**"],
			"egress_refs": ["research-web"]
		}]
	}`))
	if err != nil {
		t.Fatalf("ParseRuntime: %v", err)
	}
	environ := []string{
		environAllowedName + "=" + environAllowedValue,
		"PLAIN_NAME=plainvalue",
	}
	extra := InheritedParent{
		SecretEnv:     map[string]bool{"PLAIN_NAME": true}, // caller re-flags a benign name as secret
		CWD:           "workspace",
		WritablePaths: []string{"workspace/out/**", "workspace/other/**"},
		EgressRefs:    []string{"research-web", "metadata"},
	}
	env := rt.InheritedCapabilities.ResolveFromEnviron("Agent", environ, extra)

	if env.Env[environAllowedName] != environAllowedValue {
		t.Fatalf("benign env dropped: %+v", env.Env)
	}
	if _, ok := env.Env["PLAIN_NAME"]; ok {
		t.Fatalf("caller-reflagged secret PLAIN_NAME crossed to child")
	}
	if env.CWD != "workspace" {
		t.Fatalf("cwd = %q, want workspace", env.CWD)
	}
	if len(env.WritablePaths) != 1 || env.WritablePaths[0] != "workspace/out/**" {
		t.Fatalf("writable paths = %v, want the rule intersection only", env.WritablePaths)
	}
	if len(env.EgressRefs) != 1 || env.EgressRefs[0] != "research-web" {
		t.Fatalf("egress refs = %v, want the rule intersection only", env.EgressRefs)
	}
}
