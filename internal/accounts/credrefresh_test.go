package accounts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// credrefresh_test.go — witnesses for TriggerRefresh: the on-disk expiry, not the spawn's exit
// code, decides whether a refresh happened. A spawn that advances expiresAt is a refresh; one
// that no-ops, errors, or CLEARS the file is not — and the env scrub that keeps `claude -p` from
// billing an API key (and clearing the credential) is asserted directly, since it is the single
// highest-risk line in the change.

func writeCredExpiry(t *testing.T, path string, expiresAtMs int64) {
	t.Helper()
	body := fmt.Sprintf(`{"claudeAiOauth":{"accessToken":"sk-ant-oat01-x","expiresAt":%d}}`, expiresAtMs)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func fixedNow(ms int64) func() time.Time {
	return func() time.Time { return time.UnixMilli(ms) }
}

// TestTriggerRefresh_RotatesForward: a spawn that rewrites the credential with a later expiry is
// reported as refreshed=true.
func TestTriggerRefresh_RotatesForward(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	writeCredExpiry(t, credPath, nowMs-1) // expired

	spawn := func(ctx context.Context, cfgDir string) error {
		writeCredExpiry(t, filepath.Join(cfgDir, ".credentials.json"), nowMs+3_600_000)
		return nil
	}
	refreshed, err := TriggerRefresh(context.Background(), dir, spawn, fixedNow(nowMs))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !refreshed {
		t.Fatal("a spawn that advanced expiresAt must report refreshed=true")
	}
}

// TestTriggerRefresh_NoOpDoesNotClaimRefresh: a spawn that leaves the (still-expired) credential
// unchanged reports refreshed=false with no error — the caller then routes to its human backstop.
func TestTriggerRefresh_NoOpDoesNotClaimRefresh(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	writeCredExpiry(t, credPath, nowMs-1)

	spawn := func(ctx context.Context, cfgDir string) error { return nil } // rewrites nothing
	refreshed, err := TriggerRefresh(context.Background(), dir, spawn, fixedNow(nowMs))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if refreshed {
		t.Fatal("a no-op spawn must NOT report refreshed=true")
	}
}

// TestTriggerRefresh_SpawnErrorSurfaces: a spawn that errors surfaces the error and does not
// claim a refresh.
func TestTriggerRefresh_SpawnErrorSurfaces(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	writeCredExpiry(t, credPath, nowMs-1)

	wantErr := fmt.Errorf("boom")
	spawn := func(ctx context.Context, cfgDir string) error { return wantErr }
	refreshed, err := TriggerRefresh(context.Background(), dir, spawn, fixedNow(nowMs))
	if refreshed {
		t.Fatal("a spawn error must not report refreshed=true")
	}
	if err == nil {
		t.Fatal("a spawn error must be surfaced, not swallowed")
	}
}

// TestTriggerRefresh_ClearedFileIsNotARefresh: the API-key failure mode — the spawn deletes/blanks
// the credential — must be refreshed=false, never a crash.
func TestTriggerRefresh_ClearedFileIsNotARefresh(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	writeCredExpiry(t, credPath, nowMs-1)

	spawn := func(ctx context.Context, cfgDir string) error {
		return os.Remove(filepath.Join(cfgDir, ".credentials.json"))
	}
	refreshed, err := TriggerRefresh(context.Background(), dir, spawn, fixedNow(nowMs))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if refreshed {
		t.Fatal("a cleared credential must NOT report refreshed=true")
	}
}

// TestTriggerRefresh_EmptyDir: an empty config dir is a no-op, not a spawn.
func TestTriggerRefresh_EmptyDir(t *testing.T) {
	spawned := false
	spawn := func(ctx context.Context, cfgDir string) error { spawned = true; return nil }
	refreshed, err := TriggerRefresh(context.Background(), "", spawn, nil)
	if refreshed || err != nil {
		t.Fatalf("empty dir: got refreshed=%v err=%v, want false/nil", refreshed, err)
	}
	if spawned {
		t.Fatal("empty dir must not spawn")
	}
}

// TestRefreshEnv_ScrubsBlockers is the highest-risk assertion: the constructed refresh env must
// have ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, and CLAUDE_CODE_OAUTH_TOKEN removed (else `claude -p`
// bills the API key and clears the seat's credential), and CLAUDE_CONFIG_DIR set to the seat.
func TestRefreshEnv_ScrubsBlockers(t *testing.T) {
	base := []string{
		"PATH=/usr/bin",
		"ANTHROPIC_API_KEY=sk-secret",
		"ANTHROPIC_BASE_URL=https://example.test",
		"CLAUDE_CODE_OAUTH_TOKEN=oat-secret",
		"HOME=/home/x",
	}
	got := refreshEnv(base, "/cfg/dir")

	for _, kv := range got {
		for _, blocked := range refreshEnvBlockers {
			if len(kv) >= len(blocked)+1 && kv[:len(blocked)+1] == blocked+"=" {
				t.Fatalf("blocked env var %q must be scrubbed; found %q", blocked, kv)
			}
		}
	}
	var haveCfg, havePath bool
	for _, kv := range got {
		if kv == "CLAUDE_CONFIG_DIR=/cfg/dir" {
			haveCfg = true
		}
		if kv == "PATH=/usr/bin" {
			havePath = true
		}
	}
	if !haveCfg {
		t.Fatal("CLAUDE_CONFIG_DIR must be set to the seat's config dir")
	}
	if !havePath {
		t.Fatal("non-blocked env vars (PATH) must be preserved")
	}
}
