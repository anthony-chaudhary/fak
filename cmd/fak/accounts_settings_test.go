package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// regWithDefaults returns a registry.json string with two active seats plus a dos view carrying
// the defaults.settings block (bypass + skip-dialog) that `sync` projects into each seat's
// settings.json. dirA/dirB are the seats' config dirs.
func regWithDefaults(nameA, dirA, nameB, dirB string) string {
	return `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"` + nameA + `","dir":"` + jsonPath(dirA) + `"},` +
		`{"name":"` + nameB + `","dir":"` + jsonPath(dirB) + `"}` +
		`],"roles":{"active":"` + nameA + `","anchor":"` + nameA + `"},` +
		`"views":{"dos":{"blocks":{"defaults":{"settings":{` +
		`"skipDangerousModePermissionPrompt":true,` +
		`"permissions":{"defaultMode":"bypassPermissions"}}}}}}}`
}

// settingsBypass reads a seat's settings.json and returns permissions.defaultMode, or "" if the
// file/keys are absent.
func settingsBypass(t *testing.T, dir string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "settings.json"))
	if err != nil {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("settings.json not JSON: %v", err)
	}
	perms, _ := m["permissions"].(map[string]any)
	mode, _ := perms["defaultMode"].(string)
	return mode
}

// TestAccountsSyncProjectsSettings proves `fak accounts sync` deep-merges defaults.settings into
// every active seat's settings.json and is idempotent on a second run.
func TestAccountsSyncProjectsSettings(t *testing.T) {
	home := t.TempDir()
	a := mkHome(t, home, ".claude-a-seat", "a@example.test", true)
	b := mkHome(t, home, ".claude-b-seat", "b@example.test", true)
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(regWithDefaults("a-seat", a, "b-seat", b)), 0o644); err != nil {
		t.Fatal(err)
	}
	dosView := filepath.Join(home, ".claude", "accounts.yaml")

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"sync", "--registry", regPath, "--home", home, "--dos-view", dosView, "--job-view", ""}); rc != 0 {
		t.Fatalf("sync rc=%d stderr=%s", rc, errb.String())
	}
	if got := settingsBypass(t, a); got != "bypassPermissions" {
		t.Errorf("seat a missing bypass after sync (got %q)", got)
	}
	if got := settingsBypass(t, b); got != "bypassPermissions" {
		t.Errorf("seat b missing bypass after sync (got %q)", got)
	}
	if !strings.Contains(out.String(), "settings: 2 account(s) changed") {
		t.Errorf("sync should report 2 changed:\n%s", out.String())
	}

	// Second run is idempotent — no account changes.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"sync", "--registry", regPath, "--home", home, "--dos-view", dosView, "--job-view", ""}); rc != 0 {
		t.Fatalf("re-sync rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "settings: 0 account(s) changed") {
		t.Errorf("second sync should report 0 changed:\n%s", out.String())
	}
}

// TestAccountsAddSeedsSettings proves a brand-new account enrolled via `fak accounts add
// --no-login --token` gets its settings.json seeded with the bypass default on creation.
func TestAccountsAddSeedsSettings(t *testing.T) {
	home := t.TempDir()
	// A pre-existing seat so the registry already carries the defaults block the add reads.
	existing := mkHome(t, home, ".claude-anchor-seat", "anchor@example.test", true)
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(
		`{"version":"fak-config-homes/v1","homes":[`+
			`{"name":"anchor-seat","dir":"`+jsonPath(existing)+`"}`+
			`],"roles":{"active":"anchor-seat","anchor":"anchor-seat"},`+
			`"views":{"dos":{"blocks":{"defaults":{"settings":{`+
			`"permissions":{"defaultMode":"bypassPermissions"}}}}}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	dosView := filepath.Join(home, ".claude", "accounts.yaml")

	var out, errb bytes.Buffer
	// A syntactically valid setup-token via --no-login --token so the flow never hits the network.
	// --job-view "" keeps the add hermetic (never write the real job roster); a bare --name keeps
	// the dir basename predictable (.claude-day99-netra).
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "day99", "--no-login",
		"--token", "sk-ant-oat01-testtokentesttokentesttoken",
		"--registry", regPath, "--home", home, "--dos-view", dosView, "--job-view", "",
	})
	if rc != 0 {
		t.Fatalf("add rc=%d stderr=%s\nstdout=%s", rc, errb.String(), out.String())
	}
	newDir := filepath.Join(home, ".claude-day99-netra")
	if got := settingsBypass(t, newDir); got != "bypassPermissions" {
		t.Errorf("new seat %s missing bypass in settings.json (got %q)\nstdout=%s", newDir, got, out.String())
	}
}
