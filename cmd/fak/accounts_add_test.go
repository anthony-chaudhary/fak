package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// mkLoggedInSeat creates a source seat with a realistic auto-refreshing credential
// (.credentials.json carrying a claudeAiOauth.accessToken) + an oauthAccount identity +
// the projects/ marker — the disk shape a live `claude` login leaves. When token is
// non-empty it also drops a static .oauth-token (a sk-ant-oat… setup-token).
func mkLoggedInSeat(t *testing.T, root, dir, email, uuid, token string) string {
	t.Helper()
	full := filepath.Join(root, dir)
	if err := os.MkdirAll(filepath.Join(full, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	claudeJSON := `{"oauthAccount":{"emailAddress":"` + email + `","accountUuid":"` + uuid + `"}}`
	if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(claudeJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	creds := `{"claudeAiOauth":{"accessToken":"live-access-` + uuid + `","refreshToken":"live-refresh","expiresAt":9999999999999}}`
	if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(creds), 0o600); err != nil {
		t.Fatal(err)
	}
	if token != "" {
		if err := os.WriteFile(filepath.Join(full, ".oauth-token"), []byte(token+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return full
}

// findHome returns the registry row for name, or a zero Home if absent (including when the
// registry file itself does not exist — e.g. a failed enrollment that never wrote it).
func findHome(t *testing.T, regPath, name string) accounts.Home {
	t.Helper()
	if _, statErr := os.Stat(regPath); os.IsNotExist(statErr) {
		return accounts.Home{}
	}
	reg, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("load registry %s: %v", regPath, err)
	}
	for _, h := range reg.Homes {
		if h.Name == name {
			return h
		}
	}
	return accounts.Home{}
}

func TestRunAccountsAddAdopt_CopiesBundle(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	// The live default seat we adopt FROM.
	mkLoggedInSeat(t, home, ".claude", "july6@example.test", "u-july6", "")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "july6", "--adopt",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("adopt rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}

	dir := filepath.Join(home, ".claude-july6-netra")
	// The auto-refreshing credential must be copied byte-for-byte.
	srcCreds, _ := os.ReadFile(filepath.Join(home, ".claude", ".credentials.json"))
	dstCreds, err := os.ReadFile(filepath.Join(dir, ".credentials.json"))
	if err != nil {
		t.Fatalf("adopted seat missing .credentials.json: %v", err)
	}
	if !bytes.Equal(srcCreds, dstCreds) {
		t.Fatalf("adopted .credentials.json not byte-equal to source")
	}
	if _, err := os.Stat(filepath.Join(dir, "projects")); err != nil {
		t.Fatalf("adopted seat missing projects/ marker: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, ".claude.json")); err != nil {
		t.Fatalf("adopted seat missing .claude.json identity: %v", err)
	}
	// The source had NO .oauth-token, so the adopt must not invent one.
	if _, err := os.Stat(filepath.Join(dir, ".oauth-token")); err == nil {
		t.Fatalf("adopt copied a .oauth-token the source did not have")
	}

	h := findHome(t, regPath, "july6-netra")
	if h.Name == "" || !h.Active() {
		t.Fatalf("july6-netra not enrolled active: %+v", h)
	}
	if h.Identity.Email != "july6@example.test" || h.Identity.AccountUUID != "u-july6" {
		t.Fatalf("adopted identity wrong: %+v", h.Identity)
	}
	if !h.Identity.HasCreds {
		t.Fatalf("adopted seat should have has_creds=true: %+v", h.Identity)
	}
	// The default ~/.claude seat must be left untouched.
	if _, err := os.Stat(filepath.Join(home, ".claude", ".credentials.json")); err != nil {
		t.Fatalf("source ~/.claude was disturbed: %v", err)
	}
}

func TestRunAccountsAddAdopt_CopiesBoth(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	// Adopt from a NAMED source that has BOTH a session cred and a static setup-token.
	mkLoggedInSeat(t, home, ".claude-src-netra", "gem@example.test", "u-gem", "sk-ant-oat01-BOTHTEST")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "twin", "--adopt", "--from", "src-netra",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("adopt-both rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	dir := filepath.Join(home, ".claude-twin-netra")
	for _, f := range []string{".credentials.json", ".oauth-token"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Fatalf("adopt-both must copy %s: %v", f, err)
		}
	}
	h := findHome(t, regPath, "twin-netra")
	if h.Identity.TokenFP == "" {
		t.Fatalf("copied .oauth-token should yield a TokenFP: %+v", h.Identity)
	}
}

func TestRunAccountsAddAdopt_NoCredsRefused(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home := t.TempDir()
	// A source dir with an identity but NO credential files.
	src := filepath.Join(home, ".claude-bare-netra")
	if err := os.MkdirAll(filepath.Join(src, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, ".claude.json"), []byte(`{"oauthAccount":{"emailAddress":"x@e.test","accountUuid":"u-x"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "nocreds", "--adopt", "--from", "bare-netra",
		"--registry", regPath, "--home", home,
	})
	if rc == 0 {
		t.Fatalf("adopt from a credential-less source must fail, got rc=0 stdout=%s", out.String())
	}
	if !strings.Contains(errb.String(), "no login") {
		t.Fatalf("expected a 'no login to adopt' error, got: %s", errb.String())
	}
	// The registry must not have gained a row for a failed adopt.
	if h := findHome(t, regPath, "nocreds-netra"); h.Name != "" {
		t.Fatalf("failed adopt must not enroll a row: %+v", h)
	}
}

func TestRunAccountsAddAdopt_ForceReconciles(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	mkLoggedInSeat(t, home, ".claude", "july6@example.test", "u-july6", "")
	regPath := filepath.Join(home, "registry.json")

	// First adopt: creates the seat.
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{
		"add", "--name", "july6", "--adopt",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("first adopt rc=%d stderr=%s", rc, errb.String())
	}

	// A bare re-run (no --force) must refuse the existing dir.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{
		"add", "--name", "july6", "--adopt",
		"--registry", regPath, "--home", home,
	}); rc == 0 {
		t.Fatalf("re-adopt without --force must refuse the existing dir")
	}

	// --adopt --force reconciles in place: refresh the source cred, then re-run.
	newCreds := `{"claudeAiOauth":{"accessToken":"rotated-token","refreshToken":"r2","expiresAt":9999999999999}}`
	if err := os.WriteFile(filepath.Join(home, ".claude", ".credentials.json"), []byte(newCreds), 0o600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{
		"add", "--name", "july6", "--adopt", "--force",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("force reconcile rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "reconciled") {
		t.Fatalf("expected a 'reconciled' summary, got:\n%s", out.String())
	}
	// The rotated credential must now be in the seat.
	got, _ := os.ReadFile(filepath.Join(home, ".claude-july6-netra", ".credentials.json"))
	if !strings.Contains(string(got), "rotated-token") {
		t.Fatalf("reconcile did not refresh the credential:\n%s", got)
	}
	// Exactly ONE row for july6-netra — reconcile must not duplicate.
	reg, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, h := range reg.Homes {
		if h.Name == "july6-netra" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("reconcile duplicated the registry row: got %d rows for july6-netra", n)
	}
}

func TestRunAccountsAddAdopt_SameSourceAndTargetRefused(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	// A seat whose dir IS its own adopt source — copying a login onto itself is a no-op the
	// flow refuses rather than silently doing nothing.
	src := mkLoggedInSeat(t, home, ".claude-loop-netra", "loop@example.test", "u-loop", "")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "loop", "--adopt", "--from", src, "--force",
		"--registry", regPath, "--home", home,
	})
	if rc == 0 {
		t.Fatalf("adopt must refuse when --from source == target dir")
	}
	if !strings.Contains(errb.String(), "same dir") {
		t.Fatalf("expected a same-dir refusal, got: %s", errb.String())
	}
}

func TestAccountDir(t *testing.T) {
	home := filepath.FromSlash("/h")
	cases := []struct {
		name, suffix, want string
	}{
		{"day26", "-netra", filepath.Join(home, ".claude-day26-netra")},
		{"day26-netra", "-netra", filepath.Join(home, ".claude-day26-netra")}, // already suffixed, not doubled
		{"plain", "", filepath.Join(home, ".claude-plain")},                   // no suffix
	}
	for _, c := range cases {
		if got := accountDir(home, c.name, c.suffix); got != c.want {
			t.Errorf("accountDir(%q,%q) = %q, want %q", c.name, c.suffix, got, c.want)
		}
	}
}

func TestExtractToken(t *testing.T) {
	cases := map[string]string{
		"sk-ant-oat01-abc":                          "sk-ant-oat01-abc",
		"Paste this token:\nsk-ant-oat01-xyz\nDone": "sk-ant-oat01-xyz",
		"  sk-ant-oat01-trimmed  ":                  "sk-ant-oat01-trimmed",
		"no token here":                             "no token here",
	}
	for in, want := range cases {
		if got := extractToken(in); got != want {
			t.Errorf("extractToken(%q) = %q, want %q", in, got, want)
		}
	}
}
