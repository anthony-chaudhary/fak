package accounts

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

func TestReadCodexHomeIdentityPrecedence(t *testing.T) {
	jwtAccount := "acct-jwt-secret"
	payload := `{"https://api.openai.com/auth":{"chatgpt_account_id":"` + jwtAccount + `"}}`
	jwt := "h." + base64.RawURLEncoding.EncodeToString([]byte(payload)) + ".sig"

	cases := []struct {
		name string
		auth string
		want string
	}{
		{
			name: "tokens account wins",
			auth: `{"account_id":"acct-top-secret","tokens":{"account_id":"acct-tokens-secret","id_token":"` + jwt + `"}}`,
			want: "acct-tokens-secret",
		},
		{
			name: "top level account wins over jwt",
			auth: `{"account_id":"acct-top-secret","tokens":{"id_token":"` + jwt + `"}}`,
			want: "acct-top-secret",
		},
		{
			name: "jwt fallback",
			auth: `{"tokens":{"id_token":"` + jwt + `"}}`,
			want: jwtAccount,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, codexAuthFile), []byte(tc.auth), 0o600); err != nil {
				t.Fatal(err)
			}
			got, err := ReadCodexHomeIdentity(dir)
			if err != nil {
				t.Fatal(err)
			}
			if !got.AuthPresent {
				t.Fatal("AuthPresent = false, want true")
			}
			if want := testCodexAccountDigest(tc.want); got.AccountDigest != want {
				t.Errorf("AccountDigest = %q, want %q", got.AccountDigest, want)
			}
		})
	}
}

func TestReadCodexHomeIdentityDigestStableAndDomainSeparated(t *testing.T) {
	const accountID = "acct-sensitive-value"
	dir := t.TempDir()
	auth := `{"tokens":{"account_id":"` + accountID + `"}}`
	if err := os.WriteFile(filepath.Join(dir, codexAuthFile), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}

	first, err := ReadCodexHomeIdentity(dir)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ReadCodexHomeIdentity(dir)
	if err != nil {
		t.Fatal(err)
	}
	if first.AccountDigest != second.AccountDigest {
		t.Fatalf("digest changed across reads: %q != %q", first.AccountDigest, second.AccountDigest)
	}
	if first.AccountDigest != testCodexAccountDigest(accountID) {
		t.Fatalf("AccountDigest = %q, want domain-separated digest", first.AccountDigest)
	}
	plain := sha256.Sum256([]byte(accountID))
	if first.AccountDigest == "sha256:"+hex.EncodeToString(plain[:]) {
		t.Fatal("AccountDigest equals an undomained SHA-256 digest")
	}
}

func TestReadCodexHomeIdentityMissingAuthAndIdentity(t *testing.T) {
	t.Run("missing auth", func(t *testing.T) {
		got, err := ReadCodexHomeIdentity(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		if got.AuthPresent || got.AccountDigest != "" {
			t.Fatalf("identity = %+v, want zero value", got)
		}
	})

	t.Run("auth without identity", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, codexAuthFile), []byte(`{"tokens":{"access_token":"token-secret"}}`), 0o600); err != nil {
			t.Fatal(err)
		}
		got, err := ReadCodexHomeIdentity(dir)
		if err != nil {
			t.Fatal(err)
		}
		if !got.AuthPresent || got.AccountDigest != "" {
			t.Fatalf("identity = %+v, want auth present with no digest", got)
		}
	})
}

func TestReadCodexHomeIdentityDoesNotExposeRawIdentity(t *testing.T) {
	const accountID = "acct-must-not-escape"
	dir := t.TempDir()
	auth := `{"account_id":"` + accountID + `","tokens":{"access_token":"token-must-not-escape"}}`
	if err := os.WriteFile(filepath.Join(dir, codexAuthFile), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := ReadCodexHomeIdentity(dir)
	if err != nil {
		t.Fatal(err)
	}
	rendered := fmt.Sprintf("%+v", got)
	for _, secret := range []string{accountID, "token-must-not-escape"} {
		if strings.Contains(rendered, secret) {
			t.Fatalf("credential-safe identity leaked %q in %q", secret, rendered)
		}
	}
	if strings.Contains(got.AccountDigest, accountID) {
		t.Fatalf("digest leaked raw account identity: %q", got.AccountDigest)
	}
}

func testCodexAccountDigest(accountID string) string {
	sum := sha256.Sum256([]byte("fak/codex-account-identity/v1\x00" + accountID))
	return "sha256:" + hex.EncodeToString(sum[:])
}
