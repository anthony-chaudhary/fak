package fleetaccounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

// TestCodexHomeIsAvailableToAccountPicker is the end-to-end picker witness: a real
// CODEX_HOME marker + live auth.json enters the production roster, carries the managed
// gpt-5.6-sol/xhigh profile, and can be selected by the same tier-aware route path used by
// dispatch. A config-only home remains visible but is never offered until `codex login`.
func TestCodexHomeIsAvailableToAccountPicker(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	configHome := filepath.Join(root, "config")
	if err := os.MkdirAll(home, 0o755); err != nil {
		t.Fatal(err)
	}

	readyDir := filepath.Join(home, ".codex")
	if err := os.MkdirAll(readyDir, 0o755); err != nil {
		t.Fatal(err)
	}
	const token = "picker-witness-token-must-not-render"
	auth := `{"auth_mode":"chatgpt","tokens":{"access_token":"` + token + `","account_id":"acct-codex-ready"}}`
	if err := os.WriteFile(filepath.Join(readyDir, "auth.json"), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}

	needsLoginDir := filepath.Join(home, ".codex-needs-login")
	if err := os.MkdirAll(needsLoginDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(needsLoginDir, "config.toml"), []byte("# no auth yet\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// The legacy registry is Claude-only and commonly owns a seat named "default". That
	// short tag must never tombstone the unrelated `.codex` default home.
	registryDir := filepath.Join(home, ".claude-accounts")
	if err := os.MkdirAll(registryDir, 0o755); err != nil {
		t.Fatal(err)
	}
	registry := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"default","dir":"` + jsonPath(filepath.Join(home, ".claude")) + `","status":"tombstoned","rehome_to":"other","tombstone_reason":"Claude-only default retired"},` +
		`{"name":"other","dir":"` + jsonPath(filepath.Join(home, ".claude-other")) + `"}` +
		`]}`
	if err := os.WriteFile(filepath.Join(registryDir, "registry.json"), []byte(registry), 0o600); err != nil {
		t.Fatal(err)
	}

	rows := AnnotatedRoster(home, configHome, DefaultPolicy(), Registry{})
	ready := find(rows, ".codex")
	if ready == nil {
		t.Fatalf("Codex home was absent from picker roster: %+v", rows)
	}
	if ready.Product != "codex" || ready.Tag != "default" || ready.Kind != KindWorker {
		t.Fatalf("Codex row classification = product/tag/kind %q/%q/%q, want codex/default/worker",
			ready.Product, ready.Tag, ready.Kind)
	}
	if derefStr(ready.Agent) != "codex" || derefStr(ready.Model) != configaccounts.CodexDefaultModel ||
		derefStr(ready.ModelEffort) != configaccounts.CodexDefaultReasoningEffort || derefInt(ready.ModelTier) != TierFrontier {
		t.Fatalf("Codex picker profile = agent/model/effort/tier %q/%q/%q/%d, want codex/%s/%s/%d",
			derefStr(ready.Agent), derefStr(ready.Model), derefStr(ready.ModelEffort), derefInt(ready.ModelTier),
			configaccounts.CodexDefaultModel, configaccounts.CodexDefaultReasoningEffort, TierFrontier)
	}
	if derefStr(ready.LoginStatus) != string(configaccounts.LoginReady) || !derefBool(ready.CanServe) || !derefBool(ready.Available) {
		t.Fatalf("ready Codex picker row login/can_serve/available = %q/%v/%v",
			derefStr(ready.LoginStatus), derefBool(ready.CanServe), derefBool(ready.Available))
	}

	needs := find(rows, ".codex-needs-login")
	if needs == nil || needs.Kind != KindWorker {
		t.Fatalf("config-only Codex home should stay visible as a worker row: %+v", needs)
	}
	if derefStr(needs.LoginStatus) != string(configaccounts.LoginNeedsLogin) || derefBool(needs.CanServe) || derefBool(needs.Available) {
		t.Fatalf("config-only Codex row login/can_serve/available = %q/%v/%v, want needs_login/false/false",
			derefStr(needs.LoginStatus), derefBool(needs.CanServe), derefBool(needs.Available))
	}

	resolved := RouteAccount(rows, "audit the repository", "engineering", false, false, "codex", DefaultPolicy())
	if !resolved.OK || resolved.Account == nil || resolved.Account.Account != ".codex" {
		t.Fatalf("Codex-filtered account picker result = %+v, want authenticated .codex home", resolved)
	}

	encoded, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), token) {
		t.Fatal("picker roster leaked the Codex access token")
	}
}

func TestCodexAccountProductAndTag(t *testing.T) {
	cases := []struct {
		account string
		product string
		tag     string
	}{
		{account: ".codex", product: "codex", tag: "default"},
		{account: ".codex-work-acct", product: "codex", tag: "work"},
		{account: ".claude", product: "claude", tag: "default"},
		{account: "opencode-glm", product: "opencode", tag: "glm"},
	}
	for _, tc := range cases {
		if got := AccountProduct(tc.account); got != tc.product {
			t.Errorf("AccountProduct(%q) = %q, want %q", tc.account, got, tc.product)
		}
		if got := AccountTag(tc.account); got != tc.tag {
			t.Errorf("AccountTag(%q) = %q, want %q", tc.account, got, tc.tag)
		}
	}
}
