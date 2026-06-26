package guard

import (
	"strings"
	"testing"
)

func TestLandlockChildEnvMasksInheritedSecrets(t *testing.T) {
	t.Setenv("FAK_PARENT_SECRET", "do-not-cross")
	t.Setenv("FAK_KEEP_FOR_CHILD", "ok")
	t.Setenv("FAK_SANDBOX_ENV_ALLOW", "FAK_KEEP_FOR_CHILD")

	env := landlockChildEnv("FAK_EXPLICIT=present")
	if envContainsKey(env, "FAK_PARENT_SECRET") {
		t.Fatalf("landlock child env inherited parent secret: %v", env)
	}
	if !envContains(env, "FAK_KEEP_FOR_CHILD=ok") {
		t.Fatalf("landlock child env did not honor allow-list: %v", env)
	}
	if !envContains(env, "FAK_EXPLICIT=present") {
		t.Fatalf("landlock child env dropped explicit var: %v", env)
	}
}

func envContainsKey(env []string, key string) bool {
	prefix := key + "="
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return true
		}
	}
	return false
}

func envContains(env []string, want string) bool {
	for _, kv := range env {
		if kv == want {
			return true
		}
	}
	return false
}
