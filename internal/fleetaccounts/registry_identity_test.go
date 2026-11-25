package fleetaccounts

import (
	"os"
	"path/filepath"
	"testing"
)

// A tombstone excludes by the identity it recorded, not by whoever has since logged into its
// dir: reusing a retired dir for a new login must not retire every seat sharing that login.
func TestTombstoneExcludesByRecordedIdentity(t *testing.T) {
	home := t.TempDir()
	cfg := t.TempDir()
	write := func(p, body string) {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	login := func(uuid string) string {
		return `{"oauthAccount":{"accountUuid":"` + uuid + `","emailAddress":"` + uuid + `@example.com"}}`
	}
	write(filepath.Join(home, ".claude", ".claude.json"), login("UUID-NEW"))
	write(filepath.Join(home, ".claude-audit", ".claude.json"), login("uuid-new"))
	write(filepath.Join(home, ".claude-twin", ".claude.json"), login("uuid-old"))
	write(filepath.Join(home, ".claude-accounts", "registry.json"),
		`{"version":"fak-config-homes/v1","homes":[`+
			`{"name":"default","dir":"`+jsonPath(filepath.Join(home, ".claude"))+`","status":"tombstoned","rehome_to":"audit",`+
			`"identity":{"account_uuid":"UUID-OLD","has_creds":true,"exists":true}}]}`)
	t.Setenv("FAK_ACCOUNTS_REGISTRY", "")

	wc := NewWorkerCheck(home, cfg, DefaultPolicy())
	if wc.IsWorker(".claude") {
		t.Error(".claude is tombstoned by name, want not a worker")
	}
	if !wc.IsWorker(".claude-audit") {
		t.Error(".claude-audit shares only the tombstoned dir's current login, want a worker")
	}
	if wc.IsWorker(".claude-twin") {
		t.Error(".claude-twin carries the tombstone's recorded identity, want not a worker")
	}
}
