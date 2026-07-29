package fleetaccounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

// discover_apikey_test.go pins fleet discovery's credential-KIND awareness (#5331 "Gap B").
//
// internal/accounts already ships the api-key credential kind: a seat may declare
// cred_kind:"api_key" plus the NAME of the env var holding its Anthropic API key, and
// accounts.Validate deliberately exempts such a seat from the "an active home needs a dir"
// rule. Fleet discovery was blind to all of it — it rebuilt a synthetic Home from (dir, tag)
// alone and probed the OAuth disk credential — so an api-key seat reported HasCreds=false,
// needs_login, can_serve=false, and a DIR-LESS one never appeared on the roster at all. Either
// way the seat could never be routed to a dispatch worker.
//
// These tests are the witness for both halves plus the two things that must NOT change: an
// api-key seat with its key unset must report needs_login rather than falsely serving, and an
// OAuth seat's row must be bit-for-bit what it always was.
//
// SECRET HYGIENE: the fixtures reference an env var by NAME and set it to an obviously-fake
// placeholder. No assertion, reason string, or emitted row may ever carry the value — the last
// test in this file proves it does not.
const (
	// apiKeySeatEnv is the env-var NAME the fixture seats reference (a REFERENCE, never a key).
	apiKeySeatEnv = "FAK_TEST_FLEETACCOUNTS_APIKEY_ENV"
	// apiKeySeatFakeValue is a deliberately non-credential-shaped placeholder value.
	apiKeySeatFakeValue = "test-not-a-real-key"
)

// writeAPIKeySeatFixture builds a config-home tree under a fresh temp home, writes the
// config-home registry the homes closure renders (it receives the home path so a seat can name
// a real dir), and points discovery at that registry via FAK_ACCOUNTS_REGISTRY so the
// operator's live registry can never leak into the verdict. Every dir listed gets a projects/
// subdir, which is what makes it a Claude account dir to the glob.
func writeAPIKeySeatFixture(t *testing.T, dirs []string, homes func(home string) string) (home, cfg string) {
	t.Helper()
	home, cfg = t.TempDir(), t.TempDir()
	for _, d := range dirs {
		if err := os.MkdirAll(filepath.Join(home, d, "projects"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	regPath := filepath.Join(home, ".claude-accounts", "registry.json")
	if err := os.MkdirAll(filepath.Dir(regPath), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"version":"fak-config-homes/v1","homes":[` + homes(home) + `]}`
	if err := os.WriteFile(regPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_ACCOUNTS_REGISTRY", regPath)
	return home, cfg
}

// apiKeySeatJSON renders one api-key registry seat. dir may be "" — the dir-less shape
// accounts.Validate permits precisely because an api-key seat's existence is its key.
func apiKeySeatJSON(name, dir string) string {
	out := `{"name":"` + name + `","cred_kind":"api_key","api_key_env":"` + apiKeySeatEnv + `"`
	if dir != "" {
		out += `,"dir":"` + jsonPath(dir) + `"`
	}
	return out + `}`
}

// writeOAuthSeatDir lays down a normal subscription-OAuth config home: an oauthAccount
// identity plus the minimal .credentials.json that reads as a live login.
func writeOAuthSeatDir(t *testing.T, home, dir, email, uuid string) {
	t.Helper()
	body := `{"oauthAccount":{"accountUuid":"` + uuid + `","emailAddress":"` + email +
		`","organizationUuid":"org-` + uuid + `","organizationType":"claude_max"}}`
	if err := os.WriteFile(filepath.Join(home, dir, ".claude.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, dir, ".credentials.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestDiscoverAPIKeySeatCredentialKind is the table over the four shapes discovery must now
// get right — dir-backed and dir-less api-key seats, each with the key present and absent —
// plus the OAuth regression guard that proves the change is purely additive.
func TestDiscoverAPIKeySeatCredentialKind(t *testing.T) {
	const apiDir = ".claude-apiseat"

	cases := []struct {
		name string
		dirs []string
		// homes renders the registry's homes array; it receives the fixture home path.
		homes func(home string) string
		// setKey controls whether the referenced env var holds a (fake) key.
		setKey bool
		// account is the roster row asserted on (the dir basename, or the seat name for a
		// dir-less seat, which owns no basename to be found by).
		account    string
		wantTag    string
		wantKind   Kind
		wantCred   configaccounts.CredKind
		wantEnv    string
		wantStatus configaccounts.LoginStatus
		wantServe  bool
	}{
		{
			name:       "dir-backed api-key seat with its key present serves",
			dirs:       []string{apiDir},
			homes:      func(home string) string { return apiKeySeatJSON("apiseat", filepath.Join(home, apiDir)) },
			setKey:     true,
			account:    apiDir,
			wantTag:    "apiseat",
			wantKind:   KindWorker,
			wantCred:   configaccounts.CredKindAPIKey,
			wantEnv:    apiKeySeatEnv,
			wantStatus: configaccounts.LoginReady,
			wantServe:  true,
		},
		{
			name:       "dir-backed api-key seat with its key unset needs login, never serves",
			dirs:       []string{apiDir},
			homes:      func(home string) string { return apiKeySeatJSON("apiseat", filepath.Join(home, apiDir)) },
			setKey:     false,
			account:    apiDir,
			wantTag:    "apiseat",
			wantKind:   KindWorker,
			wantCred:   configaccounts.CredKindAPIKey,
			wantEnv:    apiKeySeatEnv,
			wantStatus: configaccounts.LoginNeedsLogin,
			wantServe:  false,
		},
		{
			name:       "dir-less api-key seat is discovered from the registry and serves",
			dirs:       nil,
			homes:      func(string) string { return apiKeySeatJSON("dirless", "") },
			setKey:     true,
			account:    "dirless",
			wantTag:    "dirless",
			wantKind:   KindWorker,
			wantCred:   configaccounts.CredKindAPIKey,
			wantEnv:    apiKeySeatEnv,
			wantStatus: configaccounts.LoginReady,
			wantServe:  true,
		},
		{
			name:       "dir-less api-key seat with its key unset is discovered but needs login",
			dirs:       nil,
			homes:      func(string) string { return apiKeySeatJSON("dirless", "") },
			setKey:     false,
			account:    "dirless",
			wantTag:    "dirless",
			wantKind:   KindWorker,
			wantCred:   configaccounts.CredKindAPIKey,
			wantEnv:    apiKeySeatEnv,
			wantStatus: configaccounts.LoginNeedsLogin,
			wantServe:  false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			home, cfg := writeAPIKeySeatFixture(t, tc.dirs, tc.homes)
			if tc.setKey {
				t.Setenv(apiKeySeatEnv, apiKeySeatFakeValue)
			} else {
				os.Unsetenv(apiKeySeatEnv)
			}

			row := find(Discover(home, cfg, DefaultPolicy()), tc.account)
			if row == nil {
				t.Fatalf("account %q was not discovered at all", tc.account)
			}
			if row.Kind != tc.wantKind {
				t.Errorf("kind = %q, want %q (reason=%q)", row.Kind, tc.wantKind, row.Reason)
			}
			if row.Tag != tc.wantTag {
				t.Errorf("tag = %q, want %q", row.Tag, tc.wantTag)
			}
			if row.CredKind != tc.wantCred {
				t.Errorf("cred_kind = %q, want %q — the seat's KIND must survive discovery", row.CredKind, tc.wantCred)
			}
			if row.APIKeyEnv != tc.wantEnv {
				t.Errorf("api_key_env = %q, want %q", row.APIKeyEnv, tc.wantEnv)
			}
			// Both roster paths must describe an api-key seat identically, and by env-var NAME.
			if want := apiKeySeatReason(apiKeySeatEnv); row.Reason != want {
				t.Errorf("reason = %q, want %q", row.Reason, want)
			}
			if got := derefStr(row.LoginStatus); got != string(tc.wantStatus) {
				t.Errorf("login_status = %q, want %q", got, tc.wantStatus)
			}
			if got := derefBool(row.CanServe); got != tc.wantServe {
				t.Errorf("can_serve = %v, want %v", got, tc.wantServe)
			}
			// A servable api-key seat must also be OFFERABLE: a worker that is not a duplicate
			// of another dir's identity is what the switcher/dispatch layer will route to.
			if tc.wantServe && !RoutableWorker(*row) {
				t.Errorf("seat can serve but is not a routable worker: %+v", row)
			}
		})
	}
}

// TestDiscoverOAuthSeatUnchangedByCredentialKind is the regression guard: the historical
// subscription-OAuth seat must be untouched by the kind-aware fold — still a worker, still
// ready, and still carrying NO cred_kind/api_key_env, which is what keeps its published JSON
// byte-identical to the legacy picker's row.
func TestDiscoverOAuthSeatUnchangedByCredentialKind(t *testing.T) {
	const oauthDir = ".claude-gem8-acct"
	home, cfg := writeAPIKeySeatFixture(t, []string{oauthDir}, func(home string) string {
		return `{"name":"gem8-acct","dir":"` + jsonPath(filepath.Join(home, oauthDir)) + `"},` +
			apiKeySeatJSON("dirless", "")
	})
	writeOAuthSeatDir(t, home, oauthDir, "gem8@example.test", "uuid-gem8")
	t.Setenv(apiKeySeatEnv, apiKeySeatFakeValue)

	rows := Discover(home, cfg, DefaultPolicy())
	row := find(rows, oauthDir)
	if row == nil {
		t.Fatalf("%s was not discovered", oauthDir)
	}
	if row.Kind != KindWorker {
		t.Errorf("kind = %q, want %q (reason=%q)", row.Kind, KindWorker, row.Reason)
	}
	if row.CredKind != "" || row.APIKeyEnv != "" {
		t.Errorf("oauth seat carries cred_kind=%q api_key_env=%q, want both empty", row.CredKind, row.APIKeyEnv)
	}
	if row.Reason != "real offered account" {
		t.Errorf("reason = %q, want the unchanged legacy text %q", row.Reason, "real offered account")
	}
	if got := derefStr(row.LoginStatus); got != string(configaccounts.LoginReady) {
		t.Errorf("login_status = %q, want %q", got, configaccounts.LoginReady)
	}
	if !derefBool(row.CanServe) {
		t.Errorf("oauth seat with credentials must still serve: %+v", row)
	}
	if derefStr(row.LoginEmail) != "gem8@example.test" {
		t.Errorf("login_email = %q, want the disk-derived address", derefStr(row.LoginEmail))
	}
	// The api-key seat alongside it must not have displaced or duplicated the OAuth row.
	if n := len(rowsWithAccount(rows, oauthDir)); n != 1 {
		t.Errorf("oauth seat appears %d times, want exactly 1", n)
	}
	if find(rows, "dirless") == nil {
		t.Errorf("the dir-less api-key seat must still surface beside the oauth seat")
	}
}

// TestAPIKeySeatRowPublishesReferenceNeverSecret pins the wire contract on both sides: an
// api-key row publishes its KIND and the env var's NAME (so a consumer can tell an API seat
// from an OAuth one), an OAuth row publishes neither key at all, and the key VALUE appears
// nowhere in the emitted bytes.
func TestAPIKeySeatRowPublishesReferenceNeverSecret(t *testing.T) {
	const oauthDir = ".claude-gem8-acct"
	home, cfg := writeAPIKeySeatFixture(t, []string{oauthDir}, func(home string) string {
		return `{"name":"gem8-acct","dir":"` + jsonPath(filepath.Join(home, oauthDir)) + `"},` +
			apiKeySeatJSON("dirless", "")
	})
	writeOAuthSeatDir(t, home, oauthDir, "gem8@example.test", "uuid-gem8")
	t.Setenv(apiKeySeatEnv, apiKeySeatFakeValue)

	rows := Discover(home, cfg, DefaultPolicy())
	encode := func(account string) map[string]any {
		t.Helper()
		row := find(rows, account)
		if row == nil {
			t.Fatalf("account %q was not discovered", account)
		}
		b, err := json.Marshal(*row)
		if err != nil {
			t.Fatalf("marshal %s: %v", account, err)
		}
		if strings.Contains(string(b), apiKeySeatFakeValue) {
			t.Fatalf("row %s leaked the credential VALUE into its published JSON", account)
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatalf("unmarshal %s: %v", account, err)
		}
		return m
	}

	api := encode("dirless")
	if api["cred_kind"] != string(configaccounts.CredKindAPIKey) {
		t.Errorf("api-key row cred_kind = %v, want %q", api["cred_kind"], configaccounts.CredKindAPIKey)
	}
	if api["api_key_env"] != apiKeySeatEnv {
		t.Errorf("api-key row api_key_env = %v, want the env var NAME %q", api["api_key_env"], apiKeySeatEnv)
	}

	oauth := encode(oauthDir)
	for _, key := range []string{"cred_kind", "api_key_env"} {
		if _, present := oauth[key]; present {
			t.Errorf("oauth row emits %q; it must stay absent so the legacy key set is unchanged", key)
		}
	}
}

// rowsWithAccount counts the rows carrying one account id, so a fold that double-emits a seat
// is caught rather than masked by find() returning the first hit.
func rowsWithAccount(rows []Account, account string) []Account {
	var out []Account
	for _, r := range rows {
		if r.Account == account {
			out = append(out, r)
		}
	}
	return out
}
