package secretload

import (
	"strings"
	"testing"
)

func TestSandboxEnvMasksInheritedByDefaultAndKeepsExplicit(t *testing.T) {
	env := []string{
		"PATH=/bin",
		"HOME=/home/fak",
		"SECRET_TOKEN=do-not-cross",
		"FAK_SWEBENCH_INSTANCE_ID=parent-value",
	}
	got := SandboxEnvWithLoader(New(), env, "FAK_SWEBENCH_INSTANCE_ID=explicit", "EXPLICIT_SECRET=ok")

	if !envHas(got, "PATH", "/bin") || !envHas(got, "HOME", "/home/fak") {
		t.Fatalf("default platform env missing from %v", got)
	}
	if envHasKey(got, "SECRET_TOKEN") {
		t.Fatalf("inherited secret crossed sandbox env: %v", got)
	}
	if !envHas(got, "FAK_SWEBENCH_INSTANCE_ID", "explicit") || !envHas(got, "EXPLICIT_SECRET", "ok") {
		t.Fatalf("explicit vars did not cross sandbox env: %v", got)
	}
	if envHas(got, "FAK_SWEBENCH_INSTANCE_ID", "parent-value") {
		t.Fatalf("explicit var did not override parent value: %v", got)
	}
}

// TestSandboxEnvKeepsEnterpriseTrustAndRouteSelectors pins the #8172 widening: a
// child inherits the trust POINTERS and cloud-route SELECTORS its runtimes need,
// and still inherits no credential. Before this, a Claude Code invocation that
// worked in the operator's shell stopped working the moment fak wrapped it —
// NODE_EXTRA_CA_CERTS is the sharpest case, since Node is the wrapped harness
// itself and without it the agent cannot validate its own upstream.
func TestSandboxEnvKeepsEnterpriseTrustAndRouteSelectors(t *testing.T) {
	env := []string{
		"NODE_EXTRA_CA_CERTS=/etc/corp/roots.pem",
		"AWS_CA_BUNDLE=/etc/corp/roots.pem",
		"CURL_CA_BUNDLE=/etc/corp/roots.pem",
		"GIT_SSL_CAINFO=/etc/corp/roots.pem",
		"CLAUDE_CODE_USE_BEDROCK=1",
		"AWS_PROFILE=corp-sso",
		"AWS_REGION=us-east-1",
		"CLAUDE_CODE_USE_VERTEX=1",
		"GOOGLE_CLOUD_PROJECT=corp-ml",
		// The credential-shaped members of the same chains. They must NOT ride a
		// static allow-list: a launch that needs one declares it per-launch through
		// policy.StripInheritedSecretsExcept.
		"AWS_SECRET_ACCESS_KEY=nope",
		"AWS_SESSION_TOKEN=nope",
		"GOOGLE_APPLICATION_CREDENTIALS=/home/u/adc.json",
	}
	got := SandboxEnvWithLoader(New(), env)
	for _, want := range []string{
		"NODE_EXTRA_CA_CERTS", "AWS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO",
		"CLAUDE_CODE_USE_BEDROCK", "AWS_PROFILE", "AWS_REGION",
		"CLAUDE_CODE_USE_VERTEX", "GOOGLE_CLOUD_PROJECT",
	} {
		if !envHasKey(got, want) {
			t.Fatalf("%s was stripped; a child that loses it loses its trust store or its provider: %v", want, got)
		}
	}
	for _, deny := range []string{"AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS"} {
		if envHasKey(got, deny) {
			t.Fatalf("%s crossed the boundary on the STATIC allow-list; deny-by-default must still hold for credentials: %v", deny, got)
		}
	}
}

func TestSandboxEnvAllowListComesFromLoader(t *testing.T) {
	l := New(mapSource{name: "policy", m: map[string]string{SandboxEnvAllowKey: "CUSTOM_KEEP; ALSO_KEEP"}})
	env := []string{"CUSTOM_KEEP=1", "ALSO_KEEP=2", "DROP_ME=3"}

	got := SandboxEnvWithLoader(l, env)
	if !envHas(got, "CUSTOM_KEEP", "1") || !envHas(got, "ALSO_KEEP", "2") {
		t.Fatalf("loader allow-list not honored: %v", got)
	}
	if envHasKey(got, "DROP_ME") {
		t.Fatalf("unlisted var crossed sandbox env: %v", got)
	}
}

func TestSandboxEnvInheritEscape(t *testing.T) {
	l := New(mapSource{name: "policy", m: map[string]string{SandboxEnvInheritKey: "all"}})
	env := []string{"SECRET_TOKEN=kept-for-legacy", "PATH=/bin", "K=old"}

	got := SandboxEnvWithLoader(l, env, "K=new")
	if !envHas(got, "SECRET_TOKEN", "kept-for-legacy") {
		t.Fatalf("inherit escape did not preserve parent env: %v", got)
	}
	if !envHas(got, "K", "new") || envHas(got, "K", "old") {
		t.Fatalf("explicit vars should still override under inherit escape: %v", got)
	}
}

func TestEnvMapKeyKeepsUnixCaseSensitivity(t *testing.T) {
	if got := envMapKeyForGOOS("Path", "linux"); got != "Path" {
		t.Fatalf("linux env key = %q, want Path", got)
	}
	if got := envMapKeyForGOOS("Path", "windows"); got != "PATH" {
		t.Fatalf("windows env key = %q, want PATH", got)
	}
}

func envHasKey(env []string, key string) bool {
	prefix := key + "="
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return true
		}
	}
	return false
}

func envHas(env []string, key, value string) bool {
	want := key + "=" + value
	for _, kv := range env {
		if kv == want {
			return true
		}
	}
	return false
}
