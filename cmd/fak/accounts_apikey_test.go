package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// TestRunAccountsAddAPIKeySeat pins the enroll verb for an API-key seat (#5331):
// `fak accounts add --name corp --api-key-env VAR` lands a kind=api_key registry row whose
// credential is the env-var REFERENCE — the secret itself must never reach the registry file.
func TestRunAccountsAddAPIKeySeat(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	const env = "FAK_TEST_5331_ADD_KEY"
	const secret = "sk-ant-api03-super-secret"
	t.Setenv(env, secret)
	home := t.TempDir()
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "corp", "--api-key-env", env, "--no-sync",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("add rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}

	h := findHome(t, regPath, "corp-netra")
	if h.CredentialKind() != accounts.CredKindAPIKey || h.APIKeyEnv != env {
		t.Fatalf("enrolled seat not api_key-kinded: %+v", h)
	}
	if !h.Active() {
		t.Fatalf("api-key seat should enroll active: %+v", h)
	}
	if !h.Identity.HasCreds || h.Identity.APIKeyEnv != env {
		t.Fatalf("identity must reflect the present key via its reference: %+v", h.Identity)
	}
	if got := h.Identity.AccountKey(); got != "apikey:"+env {
		t.Fatalf("AccountKey = %q, want the apikey bucket", got)
	}
	// The load-bearing invariant: the registry file carries the REFERENCE, never the secret.
	raw, err := os.ReadFile(regPath)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte(secret)) {
		t.Fatalf("registry file leaked the API-key secret:\n%s", raw)
	}
	if !bytes.Contains(raw, []byte(env)) {
		t.Fatalf("registry file should carry the env-var reference %q:\n%s", env, raw)
	}
	// An api-key seat must not seed a .claude.json OAuth identity (there is none).
	if _, err := os.Stat(filepath.Join(home, ".claude-corp-netra", ".claude.json")); err == nil {
		t.Fatalf("api-key seat should not seed an OAuth .claude.json")
	}
}

// TestRunAccountsAddAPIKeyRefusals pins the two front-door refusals: a pasted secret in
// place of an env-var NAME, and combining --api-key-env with an OAuth acquisition flag.
func TestRunAccountsAddAPIKeyRefusals(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home := t.TempDir()
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "corp", "--api-key-env", "sk-ant-api03-oops", "--no-sync",
		"--registry", regPath, "--home", home,
	})
	if rc != 2 || !strings.Contains(errb.String(), "not a valid env-var NAME") {
		t.Fatalf("pasted secret: rc=%d stderr=%s (want rc=2 + name refusal)", rc, errb.String())
	}

	errb.Reset()
	rc = runAccounts(&out, &errb, []string{
		"add", "--name", "corp", "--api-key-env", "ANTHROPIC_API_KEY", "--adopt", "--no-sync",
		"--registry", regPath, "--home", home,
	})
	if rc != 1 || !strings.Contains(errb.String(), "cannot be combined") {
		t.Fatalf("--adopt combo: rc=%d stderr=%s (want rc=1 + exclusivity refusal)", rc, errb.String())
	}
	if _, err := os.Stat(regPath); err == nil {
		t.Fatalf("a refused add must not write the registry")
	}
}

// TestRunAccountsListJSONShowsAPIKeySeat pins the list/status --json surface: the api-key
// seat appears with its cred_kind, env-var reference, and a truthful can_serve that tracks
// key presence.
func TestRunAccountsListJSONShowsAPIKeySeat(t *testing.T) {
	const env = "FAK_TEST_5331_LIST_KEY"
	t.Setenv(env, "sk-ant-api03-live")
	home := t.TempDir()
	reg := `{"version":"fak-config-homes/v1",` +
		`"homes":[{"name":"corp","cred_kind":"api_key","api_key_env":"` + env + `"}]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"list", "--json", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("list --json rc=%d stderr=%s", rc, errb.String())
	}
	var report accounts.LoginReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("list --json output not a LoginReport: %v\n%s", err, out.String())
	}
	var seat *accounts.LoginObservation
	for i := range report.Seats {
		if report.Seats[i].Name == "corp" {
			seat = &report.Seats[i]
		}
	}
	if seat == nil {
		t.Fatalf("api-key seat missing from list --json:\n%s", out.String())
	}
	if seat.CredKind != accounts.CredKindAPIKey || seat.APIKeyEnv != env {
		t.Fatalf("seat cred_kind/api_key_env = %q/%q, want api_key/%s", seat.CredKind, seat.APIKeyEnv, env)
	}
	if !seat.CanServe || seat.Account != "apikey:"+env {
		t.Fatalf("seat can_serve=%v account=%q, want servable apikey bucket", seat.CanServe, seat.Account)
	}
}

// TestRunAccountsLaunchAPIKeySeat pins the launch wiring (#5331): launching an api-key seat
// fronts `fak guard` with the seat's own --api-key-env reference plus the managed-cache
// posture, so guard bills the key and the managed cache resolves ACTIVE.
func TestRunAccountsLaunchAPIKeySeat(t *testing.T) {
	const env = "FAK_TEST_5331_LAUNCH_KEY"
	// The placeholder is deliberately NOT secret-SHAPED. The #2358 inherited-secret floor
	// strips an `sk-…` value held under any name outside providerAPIKeyNames, so the original
	// `sk-ant-api03-live` fixture pinned a launch whose child could never read the very
	// variable its argv referenced (#5503). That contradiction is now a refusal
	// (launchStrippedAPIKeyEnvRefusal), covered in accounts_launch_apikeyenv_test.go; this
	// test's subject is the argv SPLICING, so it uses a reference that survives the floor.
	const fakeKey = "placeholder-not-a-real-key"
	t.Setenv(env, fakeKey)
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "") // the seat's OWN reference must be spliced, not the fleet knob
	home := t.TempDir()
	seatDir := filepath.Join(home, ".claude-corp")
	if err := os.MkdirAll(seatDir, 0o755); err != nil {
		t.Fatal(err)
	}
	reg := `{"version":"fak-config-homes/v1",` +
		`"homes":[{"name":"corp","dir":"` + jsonPath(seatDir) + `","cred_kind":"api_key","api_key_env":"` + env + `"}]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var gotArgv []string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		gotArgv = argv
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "corp", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch rc=%d stderr=%s", rc, errb.String())
	}
	joined := strings.Join(gotArgv, " ")
	if !strings.Contains(joined, "guard --api-key-env "+env+" --managed-cache on --") {
		t.Fatalf("api-key seat launch must splice --api-key-env + managed-cache before --, got argv %q", joined)
	}
	// The plan summary names the seat's key reference (never the secret) as its identity.
	if !strings.Contains(errb.String(), "$"+env) {
		t.Fatalf("launch plan should name the env-var reference:\n%s", errb.String())
	}
	if strings.Contains(errb.String(), fakeKey) || strings.Contains(joined, fakeKey) {
		t.Fatalf("launch must never surface the secret itself")
	}
}
