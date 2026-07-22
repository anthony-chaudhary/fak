package accounts

import (
	"strings"
	"testing"
)

// TestValidAPIKeyEnvName pins the reference-not-secret gate: only a syntactically valid
// env-var NAME is accepted, so a pasted Anthropic key (which carries `-`) can never be
// persisted as if it were a reference.
func TestValidAPIKeyEnvName(t *testing.T) {
	valid := []string{"ANTHROPIC_API_KEY", "MY_CORP_KEY_2", "_x", "  ANTHROPIC_API_KEY  "}
	for _, s := range valid {
		if !ValidAPIKeyEnvName(s) {
			t.Errorf("ValidAPIKeyEnvName(%q) = false, want true", s)
		}
	}
	invalid := []string{
		"", "   ",
		"sk-ant-api03-abcdef", // a pasted SECRET, the exact mistake this gate exists for
		"sk-ant-oat01-xyz",
		"2KEY",      // leading digit
		"MY KEY",    // interior space
		"MY.KEY",    // dot
		"KEY=value", // an assignment, not a name
	}
	for _, s := range invalid {
		if ValidAPIKeyEnvName(s) {
			t.Errorf("ValidAPIKeyEnvName(%q) = true, want false", s)
		}
	}
}

// TestCredentialKindZeroValueIsOAuth pins backward compatibility: every pre-#5331 registry
// row (no cred_kind field) must read as the subscription-OAuth seat it always was.
func TestCredentialKindZeroValueIsOAuth(t *testing.T) {
	if got := (Home{}).CredentialKind(); got != CredKindOAuth {
		t.Fatalf("zero-value CredentialKind() = %q, want %q", got, CredKindOAuth)
	}
	if got := (Home{CredKind: CredKindOAuth}).CredentialKind(); got != CredKindOAuth {
		t.Fatalf("explicit oauth CredentialKind() = %q, want %q", got, CredKindOAuth)
	}
	if got := (Home{CredKind: CredKindAPIKey}).CredentialKind(); got != CredKindAPIKey {
		t.Fatalf("api_key CredentialKind() = %q, want %q", got, CredKindAPIKey)
	}
}

// TestDeriveAPIKeyIdentity pins the offline identity derivation for an api-key seat: the
// env-var NAME is recorded as the credential reference, HasCreds reflects the key being
// PRESENT and non-empty in the environment, and the seat exists by carrying a reference
// even without a config dir.
func TestDeriveAPIKeyIdentity(t *testing.T) {
	lookupSet := func(name string) (string, bool) {
		if name == "CORP_KEY" {
			return "sk-ant-api03-secret", true
		}
		return "", false
	}
	lookupBlank := func(string) (string, bool) { return "   ", true }
	lookupUnset := func(string) (string, bool) { return "", false }

	id := DeriveAPIKeyIdentity("", "CORP_KEY", lookupSet)
	if !id.Exists || !id.HasCreds || id.APIKeyEnv != "CORP_KEY" {
		t.Fatalf("key set: got %+v, want Exists+HasCreds with APIKeyEnv=CORP_KEY", id)
	}
	if id.Email != "" || id.AccountUUID != "" {
		t.Fatalf("offline derivation must not invent an OAuth identity: %+v", id)
	}

	id = DeriveAPIKeyIdentity("", "CORP_KEY", lookupUnset)
	if !id.Exists || id.HasCreds {
		t.Fatalf("key unset: got %+v, want Exists=true HasCreds=false", id)
	}

	// A whitespace-only value is not a usable key.
	id = DeriveAPIKeyIdentity("", "CORP_KEY", lookupBlank)
	if id.HasCreds {
		t.Fatalf("blank key value must not read as credentials: %+v", id)
	}

	// No reference at all: falls back to the plain dir stat (a zero identity for no dir).
	id = DeriveAPIKeyIdentity("", "", lookupSet)
	if id.Exists || id.HasCreds || id.APIKeyEnv != "" {
		t.Fatalf("no reference: got %+v, want zero identity", id)
	}
}

// TestAccountKeyAPIKeyBucket pins the dedup bucket: an api-key seat with no probed org
// buckets on its env-var reference, and every OAuth field keeps strict precedence over it
// so existing seats' AccountKey never changes.
func TestAccountKeyAPIKeyBucket(t *testing.T) {
	if got := (Identity{APIKeyEnv: "CORP_KEY"}).AccountKey(); got != "apikey:CORP_KEY" {
		t.Fatalf("api-key AccountKey = %q, want %q", got, "apikey:CORP_KEY")
	}
	// TODO(#5331): once the live probe fills AccountUUID from the key's org, the UUID bucket wins.
	id := Identity{AccountUUID: "u-1", APIKeyEnv: "CORP_KEY"}
	if got := id.AccountKey(); got != "uuid:u-1" {
		t.Fatalf("uuid must outrank the api-key bucket, got %q", got)
	}
	id = Identity{TokenFP: "fp", APIKeyEnv: "CORP_KEY"}
	if got := id.AccountKey(); got != "tok:fp" {
		t.Fatalf("token fingerprint must outrank the api-key bucket, got %q", got)
	}
	// And the historical empty case stays empty.
	if got := (Identity{}).AccountKey(); got != "" {
		t.Fatalf("zero identity AccountKey = %q, want empty", got)
	}
}

// TestValidateAPIKeySeat pins the registry invariants for the new kind: a valid api-key
// seat passes without a dir; a pasted secret, a stray api_key_env on an OAuth seat, and an
// unknown cred_kind are all refused.
func TestValidateAPIKeySeat(t *testing.T) {
	ok := Registry{Version: RegistryVersion, Homes: []Home{
		{Name: "corp", CredKind: CredKindAPIKey, APIKeyEnv: "ANTHROPIC_API_KEY"},
	}}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid api-key seat (no dir) refused: %v", err)
	}

	secret := Registry{Version: RegistryVersion, Homes: []Home{
		{Name: "corp", CredKind: CredKindAPIKey, APIKeyEnv: "sk-ant-api03-oops"},
	}}
	if err := secret.Validate(); err == nil {
		t.Fatal("api-key seat with a pasted secret as its reference must be refused")
	}

	strayRef := Registry{Version: RegistryVersion, Homes: []Home{
		{Name: "gem8", Dir: "/tmp/x", APIKeyEnv: "ANTHROPIC_API_KEY"},
	}}
	if err := strayRef.Validate(); err == nil {
		t.Fatal("api_key_env on a non-api_key seat must be refused")
	}

	badKind := Registry{Version: RegistryVersion, Homes: []Home{
		{Name: "corp", Dir: "/tmp/x", CredKind: "sso"},
	}}
	if err := badKind.Validate(); err == nil {
		t.Fatal("unknown cred_kind must be refused")
	}

	// The historical invariant stands for OAuth seats: an active seat still needs a dir.
	noDir := Registry{Version: RegistryVersion, Homes: []Home{{Name: "gem8"}}}
	if err := noDir.Validate(); err == nil {
		t.Fatal("active OAuth seat without a dir must still be refused")
	}
}

// TestAPIKeySeatLoginStatus pins the login-status semantics: "ready" means the KEY is
// present in the environment, "needs_login" means the env var is unset — never OAuth
// login readiness — and Refresh derives that truthfully via the seat's credential kind.
func TestAPIKeySeatLoginStatus(t *testing.T) {
	const env = "FAK_TEST_5331_KEY"
	h := Home{Name: "corp", CredKind: CredKindAPIKey, APIKeyEnv: env}

	t.Setenv(env, "sk-ant-api03-live")
	reg := Registry{Version: RegistryVersion, Homes: []Home{h}}.Refresh()
	got := reg.Homes[0]
	if st := got.LoginStatus(); st != LoginReady {
		t.Fatalf("key present: LoginStatus = %q, want %q (identity %+v)", st, LoginReady, got.Identity)
	}
	if !got.CanServe() {
		t.Fatal("key present: CanServe = false, want true")
	}
	if got.Identity.AccountKey() != "apikey:"+env {
		t.Fatalf("refreshed AccountKey = %q, want apikey bucket", got.Identity.AccountKey())
	}
	reason, action := LoginReasonAction(LoginReady, got)
	if !strings.Contains(reason, env) || action != "" {
		t.Fatalf("ready reason/action = %q / %q, want the env-var named and no action", reason, action)
	}

	t.Setenv(env, "")
	reg = Registry{Version: RegistryVersion, Homes: []Home{h}}.Refresh()
	got = reg.Homes[0]
	if st := got.LoginStatus(); st != LoginNeedsLogin {
		t.Fatalf("key absent: LoginStatus = %q, want %q", st, LoginNeedsLogin)
	}
	reason, action = LoginReasonAction(LoginNeedsLogin, got)
	if !strings.Contains(reason, env) || !strings.Contains(action, env) {
		t.Fatalf("needs-login reason/action must name the env var, got %q / %q", reason, action)
	}
	if strings.Contains(action, "sk-ant") {
		t.Fatalf("action must never carry a secret: %q", action)
	}
}

// TestLoginReportCarriesAPIKeySeat pins the --json surface: an api-key seat appears with
// its cred_kind, its env-var reference, a truthful can_serve, and its apikey bucket —
// while an OAuth seat's observation carries neither new field.
func TestLoginReportCarriesAPIKeySeat(t *testing.T) {
	const env = "FAK_TEST_5331_REPORT_KEY"
	t.Setenv(env, "sk-ant-api03-live")
	reg := Registry{Version: RegistryVersion, Homes: []Home{
		{Name: "corp", CredKind: CredKindAPIKey, APIKeyEnv: env},
	}}.Refresh()
	rep := reg.LoginReport()
	if len(rep.Seats) != 1 {
		t.Fatalf("want 1 seat, got %d", len(rep.Seats))
	}
	obs := rep.Seats[0]
	if obs.CredKind != CredKindAPIKey || obs.APIKeyEnv != env {
		t.Fatalf("observation cred_kind/api_key_env = %q/%q, want api_key/%s", obs.CredKind, obs.APIKeyEnv, env)
	}
	if !obs.CanServe || obs.Status != LoginReady {
		t.Fatalf("key present: can_serve=%v status=%q, want servable ready", obs.CanServe, obs.Status)
	}
	if obs.Account != "apikey:"+env {
		t.Fatalf("observation account = %q, want the apikey bucket", obs.Account)
	}
	if rep.Summary.CanServe != 1 || rep.Summary.DistinctAccounts != 1 {
		t.Fatalf("summary = %+v, want can_serve=1 distinct=1", rep.Summary)
	}
}

// TestRefreshPreservesAPIKeySeatWithoutDir pins the rescan invariants: Refresh and
// MergeDiscovered must re-derive an api-key seat from its stored KIND (the env-var
// reference) rather than mis-reading it through the disk OAuth probe — even when the
// discovery scan cannot cover it (no dir).
func TestRefreshPreservesAPIKeySeatWithoutDir(t *testing.T) {
	const env = "FAK_TEST_5331_MERGE_KEY"
	t.Setenv(env, "sk-ant-api03-live")
	reg := Registry{Version: RegistryVersion, Homes: []Home{
		{Name: "corp", CredKind: CredKindAPIKey, APIKeyEnv: env},
	}}

	merged, err := reg.MergeDiscovered(t.TempDir()) // empty home: the scan covers nothing
	if err != nil {
		t.Fatalf("MergeDiscovered: %v", err)
	}
	got := merged.Homes[0]
	if got.CredKind != CredKindAPIKey || got.APIKeyEnv != env {
		t.Fatalf("merge dropped the authored credential kind: %+v", got)
	}
	if !got.Identity.HasCreds || got.Identity.APIKeyEnv != env {
		t.Fatalf("merge must re-derive the api-key identity, got %+v", got.Identity)
	}
}
