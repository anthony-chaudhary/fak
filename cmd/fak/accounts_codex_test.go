package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

func TestCodexLaunchPosturePinsHomeInArgvAndChildEnv(t *testing.T) {
	homes := []string{t.TempDir(), t.TempDir()}
	type result struct {
		argv []string
		env  []string
	}
	results := make(chan result, len(homes))
	for _, home := range homes {
		home := home
		go func() {
			argv := buildLaunchArgv("fak", launchOpts{command: "codex", useGuard: true, codexHome: home})
			env := launchCodexEnv([]string{"PATH=/bin", "CODEX_HOME=parent"}, home)
			results <- result{argv: argv, env: env}
		}()
	}
	seen := map[string]bool{}
	for range homes {
		got := <-results
		joined := strings.Join(got.argv, "\x00")
		var pinned string
		for _, home := range homes {
			if strings.Contains(joined, "--codex-home\x00"+home) {
				pinned = home
			}
		}
		if pinned == "" {
			t.Fatalf("guard argv lacks a named --codex-home pin: %#v", got.argv)
		}
		if strings.Count(strings.ToUpper(strings.Join(got.env, "\n")), "CODEX_HOME=") != 1 || !envHasValue(got.env, "CODEX_HOME", pinned) {
			t.Fatalf("child env is not pinned to %q: %#v", pinned, got.env)
		}
		seen[pinned] = true
	}
	if len(seen) != 2 {
		t.Fatalf("concurrent launches crossed homes: %#v", seen)
	}
}

func TestLaunchCodexEnvDoesNotMutateParentSlice(t *testing.T) {
	base := []string{"CODEX_HOME=parent", "CODEX_SESSION_ID=parent-session", "CODEX_THREAD_ID=parent-thread", "FAK_REGISTRATION_ID=parent-registration", "FAK_ATTEMPT_ID=parent-attempt", "FAK_PARENT_REGISTRATION_ID=grandparent", "FAK_PARENT_ATTEMPT_ID=grandparent-attempt", "FAK_ROOT_REGISTRATION_ID=root", "PATH=/bin"}
	got := launchCodexEnv(base, "child")
	if base[0] != "CODEX_HOME=parent" || !envHasValue(got, "CODEX_HOME", "child") || envHasKey(got, "CODEX_SESSION_ID") || envHasKey(got, "CODEX_THREAD_ID") || envHasKey(got, "FAK_REGISTRATION_ID") || envHasKey(got, "FAK_ATTEMPT_ID") || envHasKey(got, "FAK_PARENT_REGISTRATION_ID") || envHasKey(got, "FAK_PARENT_ATTEMPT_ID") || envHasKey(got, "FAK_ROOT_REGISTRATION_ID") {
		t.Fatalf("base=%#v child=%#v", base, got)
	}
}

func envHasValue(env []string, key, value string) bool {
	for _, kv := range env {
		if kv == key+"="+value {
			return true
		}
	}
	return false
}

func TestDiscoveredCodexHomesEnterUnifiedAccountView(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	ready := filepath.Join(home, ".codex-blue")
	if err := os.MkdirAll(ready, 0o755); err != nil {
		t.Fatal(err)
	}
	const token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJhY2N0LWJsdWUifQ.sig"
	auth := `{"auth_mode":"chatgpt","tokens":{"access_token":"` + token + `","account_id":"acct-blue"}}`
	if err := os.WriteFile(filepath.Join(ready, "auth.json"), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(root, "config"))
	t.Setenv("FLEET_REG_DIR", filepath.Join(root, "registry"))
	t.Setenv("FLEET_POLICY_DIR", filepath.Join(root, "policy"))
	homes := discoveredCodexHomes()
	if len(homes) != 1 || homes[0].Name != "blue" || homes[0].Dir != ready || !homes[0].CanServe() {
		t.Fatalf("discovered Codex homes = %+v", homes)
	}
}

func envHasKey(env []string, key string) bool {
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		if strings.EqualFold(name, key) {
			return true
		}
	}
	return false
}

func TestCodexExplicitNameGuidanceListsOnlyReadyHomes(t *testing.T) {
	disabled := false
	homes := []accounts.Home{
		{Name: "blue", Dir: t.TempDir(), Identity: accounts.Identity{Exists: true, HasCreds: true}},
		{Name: "red", Dir: t.TempDir(), Enabled: &disabled, Identity: accounts.Identity{Exists: true, HasCreds: true}},
	}
	got := codexExplicitNameGuidance(homes)
	for _, want := range []string{
		"Codex requires an explicit --name",
		`ready Codex seat "blue" (named child launch, not live-session rehome): fak accounts launch --name blue --command codex`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("guidance missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "red") {
		t.Fatalf("guidance advertised disabled home:\n%s", got)
	}
}

func TestRunAccountsStatusIncludesDiscoveredCodexHomes(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	ready := filepath.Join(home, ".codex-blue")
	if err := os.MkdirAll(ready, 0o755); err != nil {
		t.Fatal(err)
	}
	const token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJhY2N0LWJsdWUifQ.sig"
	if err := os.WriteFile(filepath.Join(ready, "auth.json"), []byte(`{"auth_mode":"chatgpt","tokens":{"access_token":"`+token+`","account_id":"acct-blue"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(root, "config"))
	t.Setenv("FLEET_REG_DIR", filepath.Join(root, "registry"))
	t.Setenv("FLEET_POLICY_DIR", filepath.Join(root, "policy"))
	var stdout, stderr bytes.Buffer
	if rc := runAccounts(&stdout, &stderr, []string{"status", "--json", "--registry", filepath.Join(root, "accounts.json"), "--home", filepath.Join(root, "claude-home")}); rc != 0 {
		t.Fatalf("status rc=%d stderr=%s", rc, stderr.String())
	}
	var report accounts.LoginReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode status: %v\n%s", err, stdout.String())
	}
	var found bool
	for _, seat := range report.Seats {
		if seat.Name == "blue" && seat.CanServe {
			found = true
		}
	}
	if !found {
		t.Fatalf("status omitted ready Codex home:\n%s", stdout.String())
	}
}

func TestCodexLaunchAlternativesExcludeRegistryNameCollisions(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	for _, name := range []string{"default", "blue"} {
		dir := filepath.Join(home, ".codex-"+name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "auth.json"), []byte(`{"auth_mode":"chatgpt","tokens":{"access_token":"eyJhbGciOiJub25lIn0.eyJzdWIiOiJhY2N0LQ.sig","account_id":"acct"}}`), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(root, "config"))
	t.Setenv("FLEET_REG_DIR", filepath.Join(root, "registry"))
	t.Setenv("FLEET_POLICY_DIR", filepath.Join(root, "policy"))
	got := codexLaunchAlternatives(accounts.Registry{Homes: []accounts.Home{{Name: "default"}}})
	if len(got) != 1 || got[0].Name != "blue" {
		t.Fatalf("alternatives = %+v, want only blue", got)
	}
}
