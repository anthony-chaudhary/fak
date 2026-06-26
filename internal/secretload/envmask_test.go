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
