package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// enrollProfileServerFor stands in for Anthropic's oauth/profile endpoint, answering with the
// account keyed by the bearer token so a test can serve different identities per credential.
func enrollProfileServerFor(t *testing.T, byToken map[string]accounts.ProbedIdentity) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok := r.Header.Get("Authorization")
		const p = "Bearer "
		if len(tok) > len(p) {
			tok = tok[len(p):]
		}
		id, ok := byToken[tok]
		if !ok {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"account": map[string]any{
			"uuid":  id.AccountUUID,
			"email": id.Email,
		}})
	}))
	t.Cleanup(srv.Close)
	return srv
}

func writeFileString(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestEnrollCurrentIdentityFromCredentialWinsAndSkipsTwin is the #3215/#3216 acceptance: a source
// dir whose .claude.json names account A but whose .credentials.json serves account B enrolls as B
// (identity-from-credential wins, the seeded metadata is overwritten to B), and the cross-account
// twin .oauth-token is NOT copied into the seat.
func TestEnrollCurrentIdentityFromCredentialWinsAndSkipsTwin(t *testing.T) {
	home := t.TempDir()
	// Belt-and-suspenders: an ambient CLAUDE_CONFIG_DIR (this test may run under a guard session)
	// must not leak in as the source; --from points at the fixture explicitly.
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")

	source := filepath.Join(home, ".claude")
	// Source serves account B (accessToken at-b) but its metadata still names A (july9) — the
	// stale-.claude.json trap. It also carries a setup token that belongs to a THIRD account
	// (july7), present on a sibling seat: the cross-account twin that must be left behind.
	writeFileString(t, filepath.Join(source, ".credentials.json"), `{"claudeAiOauth":{"accessToken":"at-b","refreshToken":"rt-b"}}`)
	writeFileString(t, filepath.Join(source, ".claude.json"), `{"oauthAccount":{"emailAddress":"a-july9@example.test","accountUuid":"uuid-a-july9"}}`)
	writeFileString(t, filepath.Join(source, ".oauth-token"), "sk-ant-oat-TWIN-july7\n")
	sibling := filepath.Join(home, ".claude-july7-netra")
	writeFileString(t, filepath.Join(sibling, ".oauth-token"), "sk-ant-oat-TWIN-july7\n")
	writeFileString(t, filepath.Join(sibling, ".claude.json"), `{"oauthAccount":{"emailAddress":"c-july7@example.test","accountUuid":"uuid-c-july7"}}`)

	srv := enrollProfileServerFor(t, map[string]accounts.ProbedIdentity{
		"at-b": {Email: "b-july11@example.test", AccountUUID: "uuid-b-july11"},
	})
	t.Setenv("FAK_OAUTH_PROFILE_URL", srv.URL)

	regPath := filepath.Join(home, "reg.json")
	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"enroll-current", "--name", "seatb", "--from", source,
		"--home", home, "--registry", regPath, "--job-view", "",
	})
	if rc != 0 {
		t.Fatalf("enroll-current rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}

	seatDir := filepath.Join(home, ".claude-seatb-netra")

	// 1. The seed .claude.json (copied as stale A) was OVERWRITTEN to the credential's account B.
	claudeJSON, err := os.ReadFile(filepath.Join(seatDir, ".claude.json"))
	if err != nil {
		t.Fatalf("read seat .claude.json: %v", err)
	}
	var cj struct {
		OAuthAccount struct {
			EmailAddress string `json:"emailAddress"`
			AccountUUID  string `json:"accountUuid"`
		} `json:"oauthAccount"`
	}
	if err := json.Unmarshal(claudeJSON, &cj); err != nil {
		t.Fatalf("parse seat .claude.json: %v (%s)", err, claudeJSON)
	}
	if cj.OAuthAccount.EmailAddress != "b-july11@example.test" || cj.OAuthAccount.AccountUUID != "uuid-b-july11" {
		t.Errorf("seat .claude.json = %q/%q, want the credential identity b-july11 (uuid-b-july11) — identity-from-credential must win over stale july9",
			cj.OAuthAccount.EmailAddress, cj.OAuthAccount.AccountUUID)
	}

	// 2. The cross-account twin .oauth-token was NOT copied.
	if _, err := os.Stat(filepath.Join(seatDir, ".oauth-token")); err == nil {
		t.Errorf("seat carries a .oauth-token; the cross-account twin must be skipped")
	}
	// 3. The live session credential WAS copied.
	if _, err := os.Stat(filepath.Join(seatDir, ".credentials.json")); err != nil {
		t.Errorf("seat is missing .credentials.json (the live credential should have been copied): %v", err)
	}

	// 4. The registry records account B for the seat.
	reg, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("load registry: %v", err)
	}
	var found bool
	for _, h := range reg.Homes {
		if h.Name == "seatb-netra" {
			found = true
			if h.Identity.Email != "b-july11@example.test" || h.Identity.AccountUUID != "uuid-b-july11" {
				t.Errorf("registry identity = %q/%q, want b-july11 (uuid-b-july11)", h.Identity.Email, h.Identity.AccountUUID)
			}
		}
	}
	if !found {
		t.Errorf("registry has no seatb-netra row; homes=%v", reg.Homes)
	}
}

// TestEnrollCurrentResolvesSessionDirFromEnv pins the defining behavior of enroll-current that the
// #3215 identity acceptance (which passes an explicit --from) leaves uncovered: with NO --from, the
// verb resolves the CURRENT session's config dir from $CLAUDE_CONFIG_DIR — exactly what a launched
// `fak guard` seat exports — and enrolls THAT login, recording the credential-probed identity. This
// is the first bullet of the issue's proposed fix ("Resolves the current session's config dir
// (CLAUDE_CONFIG_DIR or ~/.claude)") turned into a regression witness, so "enroll the session I'm in"
// can't silently regress to ~/.claude or start requiring a hand-passed source.
func TestEnrollCurrentResolvesSessionDirFromEnv(t *testing.T) {
	home := t.TempDir()
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")

	// The live session runs from a NON-default dir the launcher exported via CLAUDE_CONFIG_DIR — not
	// ~/.claude, and never named on the command line. Its credential serves account B while its stale
	// .claude.json still names A, so a correct resolve-then-probe must land B, proving the source came
	// from the env dir (whose disk metadata said A) and the credential won.
	sessionDir := filepath.Join(home, "session-cfg")
	writeFileString(t, filepath.Join(sessionDir, ".credentials.json"), `{"claudeAiOauth":{"accessToken":"at-b","refreshToken":"rt-b"}}`)
	writeFileString(t, filepath.Join(sessionDir, ".claude.json"), `{"oauthAccount":{"emailAddress":"a-old@example.test","accountUuid":"uuid-a-old"}}`)
	t.Setenv("CLAUDE_CONFIG_DIR", sessionDir)

	srv := enrollProfileServerFor(t, map[string]accounts.ProbedIdentity{
		"at-b": {Email: "b-live@example.test", AccountUUID: "uuid-b-live"},
	})
	t.Setenv("FAK_OAUTH_PROFILE_URL", srv.URL)

	regPath := filepath.Join(home, "reg.json")
	var out, errb bytes.Buffer
	// NOTE: no --from — the source MUST be resolved from CLAUDE_CONFIG_DIR.
	rc := runAccounts(&out, &errb, []string{
		"enroll-current", "--name", "live",
		"--home", home, "--registry", regPath, "--dos-view", "", "--job-view", "",
	})
	if rc != 0 {
		t.Fatalf("enroll-current rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}

	seatDir := filepath.Join(home, ".claude-live-netra")
	// The env-named session credential was the source: it landed in the seat.
	if _, err := os.Stat(filepath.Join(seatDir, ".credentials.json")); err != nil {
		t.Errorf("seat missing .credentials.json copied from the CLAUDE_CONFIG_DIR session: %v", err)
	}
	// The registry records the credential-probed account B (resolved from the env dir), not stale A.
	reg, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("load registry: %v", err)
	}
	var found bool
	for _, h := range reg.Homes {
		if h.Name == "live-netra" {
			found = true
			if h.Identity.Email != "b-live@example.test" || h.Identity.AccountUUID != "uuid-b-live" {
				t.Errorf("registry identity = %q/%q, want the credential-probed b-live (uuid-b-live) resolved from CLAUDE_CONFIG_DIR",
					h.Identity.Email, h.Identity.AccountUUID)
			}
		}
	}
	if !found {
		t.Errorf("registry has no live-netra row; homes=%v", reg.Homes)
	}
	// The seat's .claude.json was reconciled to the probed account, confirming the source dir carried
	// the stale A metadata (only the CLAUDE_CONFIG_DIR fixture did) and the credential overwrote it.
	claudeJSON, err := os.ReadFile(filepath.Join(seatDir, ".claude.json"))
	if err != nil {
		t.Fatalf("read seat .claude.json: %v", err)
	}
	if !bytes.Contains(claudeJSON, []byte("uuid-b-live")) {
		t.Errorf("seat .claude.json = %s, want the probed account uuid-b-live written in", claudeJSON)
	}
}

// TestStatusProbeFlagsIdentityMetadataStale is the #3216 read-surface acceptance: `status --probe`
// probes each seat's live credential and, when its account disagrees with the on-disk .claude.json
// metadata, flags identity_metadata_stale and shows the credential's true account instead of
// silently trusting disk.
func TestStatusProbeFlagsIdentityMetadataStale(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", "")

	// A NAMED seat whose single .claude.json says A (july9) but whose credential serves B.
	seat := filepath.Join(home, ".claude-stale-netra")
	writeFileString(t, filepath.Join(seat, "projects", ".keep"), "")
	writeFileString(t, filepath.Join(seat, ".credentials.json"), `{"claudeAiOauth":{"accessToken":"at-b","refreshToken":"rt-b"}}`)
	writeFileString(t, filepath.Join(seat, ".claude.json"), `{"oauthAccount":{"emailAddress":"a-july9@example.test","accountUuid":"uuid-a-july9"}}`)

	srv := enrollProfileServerFor(t, map[string]accounts.ProbedIdentity{
		"at-b": {Email: "b-july11@example.test", AccountUUID: "uuid-b-july11"},
	})
	t.Setenv("FAK_OAUTH_PROFILE_URL", srv.URL)

	regPath := filepath.Join(home, "reg.json")
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"discover", "--write", "--home", home, "--registry", regPath, "--job-view", ""}); rc != 0 {
		t.Fatalf("discover --write rc=%d stderr=%s", rc, errb.String())
	}

	// Without --probe: disk-only, so the report names the stale A.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"status", "--json", "--home", home, "--registry", regPath}); rc != 0 {
		t.Fatalf("status rc=%d stderr=%s", rc, errb.String())
	}
	if seatObs := findSeat(t, out.Bytes(), "stale-netra"); seatObs.Email != "a-july9@example.test" {
		t.Fatalf("disk-only status email = %q, want the (stale) july9 the disk names", seatObs.Email)
	}

	// With --probe: the credential's true account B wins, and identity_metadata_stale is flagged.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"status", "--probe", "--json", "--home", home, "--registry", regPath}); rc != 0 {
		t.Fatalf("status --probe rc=%d stderr=%s", rc, errb.String())
	}
	seatObs := findSeat(t, out.Bytes(), "stale-netra")
	if seatObs.Email != "b-july11@example.test" || seatObs.Account != "uuid:uuid-b-july11" {
		t.Errorf("probed status = %q/%q, want the credential identity b-july11 (uuid:uuid-b-july11)", seatObs.Email, seatObs.Account)
	}
	if !hasWarning(seatObs.Warnings, accounts.LoginWarningIdentityStale) {
		t.Errorf("probed status warnings = %v, want identity_metadata_stale", seatObs.Warnings)
	}
}

// findSeat parses a LoginReport JSON blob and returns the observation for the named seat.
func findSeat(t *testing.T, blob []byte, name string) accounts.LoginObservation {
	t.Helper()
	var report accounts.LoginReport
	if err := json.Unmarshal(blob, &report); err != nil {
		t.Fatalf("parse status JSON: %v (%s)", err, blob)
	}
	for _, s := range report.Seats {
		if s.Name == name {
			return s
		}
	}
	t.Fatalf("seat %q not in report (%d seats)", name, len(report.Seats))
	return accounts.LoginObservation{}
}
