package accounts

import (
	"os"
	"path/filepath"
	"testing"
)

// mkSeat creates a config-home dir with an optional login (.claude.json oauthAccount),
// an optional setup token (.oauth-token), and an optional .credentials.json, then returns
// the derived Identity for it. uuid/email/token empty -> that file is omitted.
func mkSeat(t *testing.T, root, dir, email, uuid, token string, creds bool) Home {
	t.Helper()
	full := filepath.Join(root, dir)
	if err := os.MkdirAll(filepath.Join(full, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if email != "" || uuid != "" {
		body := `{"oauthAccount":{"emailAddress":"` + email + `","accountUuid":"` + uuid + `"}}`
		if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if token != "" {
		if err := os.WriteFile(filepath.Join(full, ".oauth-token"), []byte(token+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if creds {
		if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	name := "default"
	if dir != ".claude" {
		name = dir[len(".claude-"):]
	}
	return Home{Name: name, Dir: full, Identity: DeriveIdentity(full)}
}

func TestTokenFingerprintNonSecretAndStable(t *testing.T) {
	dir := t.TempDir()
	a := mkSeat(t, dir, ".claude-a", "a@x.test", "u-a", "SECRET-TOKEN-VALUE", true)
	b := mkSeat(t, dir, ".claude-b", "b@x.test", "u-b", "SECRET-TOKEN-VALUE", true)
	c := mkSeat(t, dir, ".claude-c", "c@x.test", "u-c", "DIFFERENT-TOKEN", true)

	if a.Identity.TokenFP == "" {
		t.Fatal("a should have a token fingerprint")
	}
	// One-way: the fingerprint must never be (or contain) the raw secret.
	if a.Identity.TokenFP == "SECRET-TOKEN-VALUE" ||
		contains(a.Identity.TokenFP, "SECRET") {
		t.Fatalf("fingerprint leaks the token: %q", a.Identity.TokenFP)
	}
	// Same token (whitespace-trimmed) -> same fingerprint; different token -> different.
	if a.Identity.TokenFP != b.Identity.TokenFP {
		t.Fatalf("same token must fingerprint identically: %q vs %q", a.Identity.TokenFP, b.Identity.TokenFP)
	}
	if a.Identity.TokenFP == c.Identity.TokenFP {
		t.Fatal("different tokens must fingerprint differently")
	}
	// A dir with no .oauth-token has an empty fingerprint.
	noTok := mkSeat(t, dir, ".claude-d", "d@x.test", "u-d", "", true)
	if noTok.Identity.TokenFP != "" {
		t.Fatalf("absent token must yield empty fingerprint, got %q", noTok.Identity.TokenFP)
	}
}

func TestSetupTokenLoaderRedactsResolvedToken(t *testing.T) {
	root := t.TempDir()
	seat := mkSeat(t, root, ".claude-redact", "r@x.test", "u-r", "plain-account-token-value", true)

	loader := setupTokenLoader(seat.Dir)
	tok, src, ok := loader.LookupSource(setupTokenSecretKey)
	if !ok || tok != "plain-account-token-value" || src != filepath.Join(seat.Dir, ".oauth-token") {
		t.Fatalf("LookupSource = (%q,%q,%v), want setup token file source", tok, src, ok)
	}
	if out := loader.Redact("accounts loaded " + tok); contains(out, tok) {
		t.Fatalf("resolved setup token was not redacted: %q", out)
	}
}

func TestAccountKeyPrefersUUIDThenToken(t *testing.T) {
	withUUID := Identity{AccountUUID: "u-1", TokenFP: "ff00"}
	if got := withUUID.AccountKey(); got != "uuid:u-1" {
		t.Fatalf("AccountKey with uuid = %q, want uuid:u-1", got)
	}
	tokenOnly := Identity{TokenFP: "ff00"}
	if got := tokenOnly.AccountKey(); got != "tok:ff00" {
		t.Fatalf("AccountKey token-only = %q, want tok:ff00", got)
	}
	if got := (Identity{}).AccountKey(); got != "" {
		t.Fatalf("empty identity AccountKey = %q, want \"\"", got)
	}
}

// TestReconcileCollapsesSameAccount is the gem8/q/day24 incident in miniature: three
// dirs logged into one account plus a fourth whose setup token is that account's.
func TestReconcileCollapsesSameAccount(t *testing.T) {
	root := t.TempDir()
	tok := "gem8-setup-token"
	// gem8 account, present under three dirs sharing one uuid AND one token.
	gem8 := mkSeat(t, root, ".claude-gem8-netra", "gem8@netra.test", "u-gem8", tok, true)
	dflt := mkSeat(t, root, ".claude", "gem8@netra.test", "u-gem8", tok, true)
	qseat := mkSeat(t, root, ".claude-q-netra", "gem8@netra.test", "u-gem8", tok, true)
	// day24: its OWN login, but its setup token is gem8's (a leaked/copied token).
	day24 := mkSeat(t, root, ".claude-day24-netra", "day24@netra.test", "u-day24", tok, true)
	// gem7: a genuinely independent account.
	gem7 := mkSeat(t, root, ".claude-gem7-netra", "gem7@netra.test", "u-gem7", "gem7-token", true)

	reg := Registry{Homes: []Home{gem8, dflt, qseat, day24, gem7}}
	rec := reg.Reconcile()

	// gem8/default/q-netra collapse to ONE account; gem8-netra is the canonical (its name
	// matches the login and it is not the generic "default").
	for _, name := range []string{"gem8-netra", "default", "q-netra"} {
		si, ok := rec[name]
		if !ok {
			t.Fatalf("seat %q missing from reconcile", name)
		}
		if si.Canonical != "gem8-netra" {
			t.Fatalf("%q should collapse onto gem8-netra, got %q", name, si.Canonical)
		}
	}
	if rec["gem8-netra"].Role != RoleCanonical {
		t.Fatalf("gem8-netra role = %q, want canonical", rec["gem8-netra"].Role)
	}
	if rec["default"].Role != RoleDuplicate || rec["q-netra"].Role != RoleDuplicate {
		t.Fatalf("default/q-netra should be duplicates: %q / %q", rec["default"].Role, rec["q-netra"].Role)
	}
	// gem7 is alone on its account.
	if rec["gem7-netra"].Role != RoleUnique {
		t.Fatalf("gem7-netra role = %q, want unique", rec["gem7-netra"].Role)
	}
	// day24 has its OWN login (different uuid) -> not a duplicate of gem8 by account, but
	// its setup token is gem8's -> flagged as a token-twin of the gem8 dirs.
	d := rec["day24-netra"]
	if d.Role == RoleDuplicate {
		t.Fatalf("day24-netra has its own login; must NOT be an account duplicate")
	}
	if len(d.TokenTwin) == 0 {
		t.Fatalf("day24-netra shares gem8's setup token; expected a token-twin warning")
	}
	if !containsAll(d.TokenTwin, "gem8-netra", "default", "q-netra") {
		t.Fatalf("day24 token-twins = %v, want the three gem8 dirs", d.TokenTwin)
	}
	// Conversely, the gem8 dirs see day24 as a token-twin (different login, same token).
	if !contains2(rec["gem8-netra"].TokenTwin, "day24-netra") {
		t.Fatalf("gem8-netra token-twins = %v, want day24-netra", rec["gem8-netra"].TokenTwin)
	}
	// Same-login peers are NOT token-twins (same uuid), so gem8's twin list excludes them.
	if contains2(rec["gem8-netra"].TokenTwin, "default") {
		t.Fatalf("same-login peers must not be token-twins: %v", rec["gem8-netra"].TokenTwin)
	}
}

func TestReconcileTombstonedExcludedAndNoLogin(t *testing.T) {
	root := t.TempDir()
	live := mkSeat(t, root, ".claude-live", "x@y.test", "u-x", "tok-x", true)
	// A dir with neither a login nor a token -> no-login (ungroupable).
	blank := mkSeat(t, root, ".claude-blank", "", "", "", false)
	tomb := Home{Name: "dead", Status: StatusTombstoned, RehomeTo: "live"}

	reg := Registry{Homes: []Home{live, blank, tomb}}
	rec := reg.Reconcile()

	if _, ok := rec["dead"]; ok {
		t.Fatal("tombstoned seat must be excluded from reconcile")
	}
	if rec["blank"].Role != RoleNoLogin {
		t.Fatalf("blank seat role = %q, want no-login", rec["blank"].Role)
	}
	if rec["live"].Role != RoleUnique {
		t.Fatalf("live seat role = %q, want unique", rec["live"].Role)
	}
}

// TestNameLieIgnoresOrgSuffix covers the partial-match case the older TestNameLie did
// not: a name that shares SOME tokens with the login (the org-suffix pattern) is
// truthful, while a name that shares NONE is the lie.
func TestNameLieIgnoresOrgSuffix(t *testing.T) {
	cases := []struct {
		name, email string
		lie         bool
	}{
		{"gem8-netra", "gem8@example.test", false},                      // truthful: "gem8" matches, "-netra" org suffix ignored
		{"day24-netra", "day24@example.test", false},                    // truthful despite org suffix
		{"jack-barker-claude-netra", "jack.barker@example.test", false}, // "jack" matches
		{"q-netra", "gem8@example.test", true},                          // q is really gem8 — no token matches
		{"c10-netra", "anthony.agent@example.test", true},               // c10 dir is really anthony's
		{"default", "gem8@example.test", false},                         // role name, never a lie
	}
	for _, c := range cases {
		h := Home{Name: c.name, Identity: Identity{Email: c.email}}
		if got := h.NameLie(); got != c.lie {
			t.Errorf("NameLie(%q logged into %q) = %v, want %v", c.name, c.email, got, c.lie)
		}
	}
}

// --- tiny local helpers (avoid pulling in strings/slices just for the asserts) ---

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func contains2(list []string, want string) bool {
	for _, v := range list {
		if v == want {
			return true
		}
	}
	return false
}

func containsAll(list []string, want ...string) bool {
	for _, w := range want {
		if !contains2(list, w) {
			return false
		}
	}
	return true
}
