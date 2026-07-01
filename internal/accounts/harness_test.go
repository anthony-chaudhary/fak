package accounts

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

func codexProfile(t *testing.T) harnessprofile.HarnessProfile {
	t.Helper()
	p, ok := harnessprofile.Lookup("codex")
	if !ok {
		t.Fatal("harnessprofile.Lookup(codex) missed — registry regression")
	}
	return p
}

// writeCodexHome creates <root>/<dirName> as a codex config home with an auth.json carrying
// the given account id (as tokens.account_id) and an access token, and returns the home dir.
func writeCodexHome(t *testing.T, root, dirName, accountID string) string {
	t.Helper()
	dir := filepath.Join(root, dirName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	auth := `{"auth_mode":"chatgpt","tokens":{"access_token":"tok-live","account_id":"` + accountID + `"}}`
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestDiscoverProfileCodexHome is the C4 acceptance witness that a ~/.codex* home with a
// valid auth.json is discovered with a real identity: the ChatGPT account id becomes the
// AccountKey bucket and the live token makes it LoginReady — exactly the shape RotationPlan
// needs to admit it.
func TestDiscoverProfileCodexHome(t *testing.T) {
	root := t.TempDir()
	writeCodexHome(t, root, ".codex", "acct-alpha")
	writeCodexHome(t, root, ".codex-beta", "acct-beta")

	homes, err := DiscoverProfile(root, codexProfile(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(homes) != 2 {
		t.Fatalf("DiscoverProfile found %d codex homes, want 2: %+v", len(homes), homes)
	}
	// Sorted by name: "beta" then "default" (".codex" → default).
	byName := map[string]Home{}
	for _, h := range homes {
		byName[h.Name] = h
	}
	def, ok := byName["default"]
	if !ok {
		t.Fatalf("expected a 'default' seat for .codex, got names %v", homes)
	}
	if def.Identity.AccountUUID != "acct-alpha" {
		t.Errorf("default seat account = %q, want acct-alpha", def.Identity.AccountUUID)
	}
	if def.Identity.AccountKey() != "uuid:acct-alpha" {
		t.Errorf("default seat bucket = %q, want uuid:acct-alpha", def.Identity.AccountKey())
	}
	if !def.CanServe() {
		t.Errorf("default codex seat should be LoginReady (has auth.json token): %+v", def.Identity)
	}
}

// TestCodexHomesEnterRotationPool proves codex homes flow through the EXISTING RotationPlan:
// two distinct ChatGPT accounts are two pool buckets, and two homes on the SAME account id
// collapse to one bucket (the Claude uuid-dedup, unchanged, now applied to codex).
func TestCodexHomesEnterRotationPool(t *testing.T) {
	mk := func(name, acct string) Home {
		return Home{Name: name, Dir: "/x/" + name, Identity: Identity{Exists: true, HasCreds: true, AccountUUID: acct}}
	}

	t.Run("distinct accounts -> two buckets", func(t *testing.T) {
		reg := Registry{Homes: []Home{mk("a", "acct-1"), mk("b", "acct-2")}}
		pool := reg.RotationPlan().Pool
		if len(pool) != 2 {
			t.Fatalf("pool = %d buckets, want 2: %+v", len(pool), pool)
		}
	})

	t.Run("same account -> one bucket, other is a duplicate", func(t *testing.T) {
		reg := Registry{Homes: []Home{mk("a", "acct-1"), mk("b", "acct-1")}}
		res := reg.RotationPlan()
		if len(res.Pool) != 1 {
			t.Fatalf("pool = %d buckets, want 1 (same account collapses): %+v", len(res.Pool), res.Pool)
		}
		if res.Pool[0].Account != "uuid:acct-1" {
			t.Errorf("pool bucket = %q, want uuid:acct-1", res.Pool[0].Account)
		}
		foundDup := false
		for _, e := range res.Excluded {
			if e.Status == RotationDuplicate {
				foundDup = true
			}
		}
		if !foundDup {
			t.Errorf("second codex home on the same account should be a RotationDuplicate: %+v", res.Excluded)
		}
	})
}

// TestNextInRotationCodex proves codex→codex rotation off a walled bucket rides the existing
// NextInRotation contract: two accounts rotate to each other, and a pool of one bucket
// refuses (ok=false) rather than re-handing the walled account.
func TestNextInRotationCodex(t *testing.T) {
	mk := func(name, acct string) Home {
		return Home{Name: name, Dir: "/x/" + name, Identity: Identity{Exists: true, HasCreds: true, AccountUUID: acct}}
	}
	reg := Registry{Homes: []Home{mk("a", "acct-1"), mk("b", "acct-2")}}

	// `after` is the SEAT NAME the caller is leaving; NextInRotation resolves it to its
	// account bucket and returns a DIFFERENT one.
	next, ok := reg.NextInRotation("a")
	if !ok {
		t.Fatal("NextInRotation off seat a (acct-1) should find the other codex bucket")
	}
	if next.Account != "uuid:acct-2" {
		t.Errorf("rotated onto %q, want uuid:acct-2 (never re-hand the walled bucket)", next.Account)
	}

	// A single-bucket pool cannot rotate off itself — the walled account is the only one.
	one := Registry{Homes: []Home{mk("a", "acct-1")}}
	if _, ok := one.NextInRotation("a"); ok {
		t.Error("NextInRotation with one bucket must be ok=false (never re-hand the walled account)")
	}
}

// TestHasIdentityReaderCoverage records the honest #1955 coverage: claude and codex have a
// real per-home identity reader; the openai-generic (env-key) profile does not.
func TestHasIdentityReaderCoverage(t *testing.T) {
	cases := map[string]bool{"claude": true, "codex": true, "opencode": false}
	for agent, want := range cases {
		p, ok := harnessprofile.Lookup(agent)
		if !ok {
			t.Fatalf("Lookup(%q) missed", agent)
		}
		if got := HasIdentityReader(p); got != want {
			t.Errorf("HasIdentityReader(%s) = %v, want %v", agent, got, want)
		}
	}
}

// TestDeriveCodexIdentityFromJWT proves the account id is recovered from the id_token JWT
// claim when it is not at tokens.account_id / account_id — the codex-rs-version-tolerant path.
func TestDeriveCodexIdentityFromJWT(t *testing.T) {
	payload := `{"https://api.openai.com/auth":{"chatgpt_account_id":"acct-from-jwt"}}`
	jwt := "h." + base64.RawURLEncoding.EncodeToString([]byte(payload)) + ".sig"
	dir := t.TempDir()
	auth := `{"tokens":{"access_token":"tok","id_token":"` + jwt + `"}}`
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}
	id := DeriveIdentityForProfile(dir, codexProfile(t))
	if id.AccountUUID != "acct-from-jwt" {
		t.Errorf("account from JWT = %q, want acct-from-jwt", id.AccountUUID)
	}
	if !id.HasCreds {
		t.Error("home with an access token should read HasCreds")
	}
}

// TestDeriveIdentityForProfileClaudeUnchanged proves the claude path through the new
// dispatcher is byte-identical to DeriveIdentity (the byte-identical-Claude fence).
func TestDeriveIdentityForProfileClaudeUnchanged(t *testing.T) {
	claude, ok := harnessprofile.Lookup("claude")
	if !ok {
		t.Fatal("Lookup(claude) missed")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".claude.json"),
		[]byte(`{"oauthAccount":{"emailAddress":"x@y.z","accountUuid":"uuid-1"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got := DeriveIdentityForProfile(dir, claude)
	want := DeriveIdentity(dir)
	if got != want {
		t.Errorf("DeriveIdentityForProfile(claude) = %+v, want DeriveIdentity = %+v", got, want)
	}
}
