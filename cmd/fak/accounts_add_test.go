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

// hermeticProbeStub points FAK_OAUTH_PROFILE_URL at a local stub that 401s every token, so an
// adopt's now-default credential-identity probe resolves INSTANTLY to "unknown" and falls back to
// the copied disk identity — without ever touching the real Anthropic endpoint. The bundle-copy
// tests below assert the disk-derived identity of a FAKE credential; this keeps them hermetic and
// fast under the probe-on default while still exercising the real fallback path. A test that wants
// the probe to SUCCEED registers its own enrollProfileServerFor instead.
func hermeticProbeStub(t *testing.T) {
	t.Helper()
	srv := enrollProfileServerFor(t, map[string]accounts.ProbedIdentity{}) // empty map → 401 on every token
	t.Setenv("FAK_OAUTH_PROFILE_URL", srv.URL)
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
	hermeticProbeStub(t)
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
	hermeticProbeStub(t)
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

// TestRunAccountsAddAdopt_SkipsCrossAccountTwinToken is the july8-style case: the login we adopt
// FROM carries a live session (.credentials.json) AND a static .oauth-token that is byte-identical
// to a DIFFERENT account's seat (the token-twin smear). Adopting must copy the clean session and
// SKIP the foreign token — so the new seat runs on its own bucket instead of tripping the
// GateTokenWrite refusal and leaving a half-created dir.
func TestRunAccountsAddAdopt_SkipsCrossAccountTwinToken(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	hermeticProbeStub(t)
	const twinTok = "sk-ant-oat01-SHARED-TWIN-TOKEN"
	// The seat that legitimately OWNS the token (a different account, already enrolled).
	mkLoggedInSeat(t, home, ".claude-owner-netra", "owner@example.test", "u-owner", twinTok)
	// The default seat we adopt FROM: its own live session, but the SAME token smeared onto it.
	mkLoggedInSeat(t, home, ".claude", "july8@example.test", "u-july8", twinTok)
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "july8", "--adopt",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("adopt with a twin token must SUCCEED by skipping it, got rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "skipped cross-account twin") {
		t.Fatalf("expected a skipped-twin note in the summary, got:\n%s", out.String())
	}
	dir := filepath.Join(home, ".claude-july8-netra")
	// The live session must be adopted…
	if _, err := os.Stat(filepath.Join(dir, ".credentials.json")); err != nil {
		t.Fatalf("adopted seat missing .credentials.json: %v", err)
	}
	// …and the foreign token must NOT have followed it.
	if _, err := os.Stat(filepath.Join(dir, ".oauth-token")); err == nil {
		t.Fatalf("adopt copied the cross-account twin .oauth-token it should have skipped")
	}
	// The seat is enrolled, active, and carries NO token fingerprint (so twin audits stay clean).
	h := findHome(t, regPath, "july8-netra")
	if h.Name == "" || !h.Active() {
		t.Fatalf("july8-netra not enrolled active: %+v", h)
	}
	if h.Identity.Email != "july8@example.test" {
		t.Fatalf("adopted identity wrong: %+v", h.Identity)
	}
	if h.Identity.TokenFP != "" {
		t.Fatalf("adopted seat must not carry a token fingerprint after skipping the twin: %+v", h.Identity)
	}
	if !h.Identity.HasCreds {
		t.Fatalf("adopted seat should still have has_creds=true via .credentials.json: %+v", h.Identity)
	}
}

// TestRunAccountsAddAdopt_KeepsTokenWhenNoSession guards the narrowness of the twin-skip: a source
// whose ONLY credential is a .oauth-token (no live .credentials.json) must still copy that token —
// dropping it would leave the seat with no credential at all. Here the token is the source's OWN
// (no foreign sibling), so the copy is kept AND passes the smear gate.
func TestRunAccountsAddAdopt_KeepsTokenWhenNoSession(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	hermeticProbeStub(t)
	const tok = "sk-ant-oat01-TOKEN-ONLY-SOURCE"
	// The adopt source: token ONLY, no .credentials.json (a distinct dir from the target). No
	// sibling shares this token, so it is the source's own — the twin-skip must NOT fire.
	src := filepath.Join(home, ".claude-toksrc-netra")
	if err := os.MkdirAll(filepath.Join(src, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, ".oauth-token"), []byte(tok+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "tokonly", "--adopt", "--from", "toksrc-netra",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("adopt of a token-only source must SUCCEED (the token is the only credential), rc=%d stderr=%s", rc, errb.String())
	}
	dir := filepath.Join(home, ".claude-tokonly-netra")
	if _, err := os.Stat(filepath.Join(dir, ".oauth-token")); err != nil {
		t.Fatalf("token-only adopt must keep the .oauth-token (it is the sole credential): %v", err)
	}
	if strings.Contains(out.String(), "skipped cross-account twin") {
		t.Fatalf("twin-skip must NOT fire when the token is the source's own and its only credential:\n%s", out.String())
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

// TestRunAccountsAddAdopt_ProbesIdentityByDefault is the determinism regression for the
// stale-.claude.json incident: a plain `add --adopt` (NO --probe-identity flag) from a source whose
// .claude.json metadata names account A but whose live .credentials.json serves account B must now
// enroll as B — the credential is ground truth, probed by default — instead of silently recording
// the stale A the disk claims. Before the default flip this required the operator to know to pass
// --probe-identity (or use enroll-current); forgetting it enrolled the wrong account.
func TestRunAccountsAddAdopt_ProbesIdentityByDefault(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()

	// The source (default ~/.claude) serves account B (accessToken at-b) but its .claude.json
	// still names A — exactly the shape a /login into a shared dir leaves behind.
	source := filepath.Join(home, ".claude")
	writeFileString(t, filepath.Join(source, "projects", ".keep"), "")
	writeFileString(t, filepath.Join(source, ".credentials.json"), `{"claudeAiOauth":{"accessToken":"at-b","refreshToken":"rt-b"}}`)
	writeFileString(t, filepath.Join(source, ".claude.json"), `{"oauthAccount":{"emailAddress":"a-stale@example.test","accountUuid":"uuid-a-stale"}}`)

	srv := enrollProfileServerFor(t, map[string]accounts.ProbedIdentity{
		"at-b": {Email: "b-true@example.test", AccountUUID: "uuid-b-true"},
	})
	t.Setenv("FAK_OAUTH_PROFILE_URL", srv.URL)

	regPath := filepath.Join(home, "registry.json")
	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "seat", "--adopt", // NOTE: no --probe-identity
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("adopt rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}

	// The enrolled identity must be the credential's true account B, not the stale disk A.
	h := findHome(t, regPath, "seat-netra")
	if h.Identity.Email != "b-true@example.test" || h.Identity.AccountUUID != "uuid-b-true" {
		t.Fatalf("adopt identity = %q/%q, want the credential's true account b-true (uuid-b-true) — probe should be default-on",
			h.Identity.Email, h.Identity.AccountUUID)
	}
	// The seat's .claude.json must have been rewritten to the credential identity so every later
	// disk read (list/status/discover) reports B, not the stale A.
	dir := filepath.Join(home, ".claude-seat-netra")
	got, _ := os.ReadFile(filepath.Join(dir, ".claude.json"))
	if !strings.Contains(string(got), "uuid-b-true") || strings.Contains(string(got), "uuid-a-stale") {
		t.Fatalf("seat .claude.json not reconciled to credential identity: %s", got)
	}
	// The reconcile must be surfaced to the operator, not silent.
	if !strings.Contains(out.String(), "identity reconcile") {
		t.Errorf("expected an 'identity reconcile' line on stdout, got: %s", out.String())
	}
}

// TestRunAccountsAddAdopt_NoProbeIdentityStaysDiskOnly pins the offline escape hatch: with
// --no-probe-identity, the adopt records the copied disk metadata verbatim and makes NO network
// probe, even when a profile endpoint is configured — so a deliberately offline enrollment trusts
// the .claude.json it copied.
func TestRunAccountsAddAdopt_NoProbeIdentityStaysDiskOnly(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()

	source := filepath.Join(home, ".claude")
	writeFileString(t, filepath.Join(source, "projects", ".keep"), "")
	writeFileString(t, filepath.Join(source, ".credentials.json"), `{"claudeAiOauth":{"accessToken":"at-b","refreshToken":"rt-b"}}`)
	writeFileString(t, filepath.Join(source, ".claude.json"), `{"oauthAccount":{"emailAddress":"a-stale@example.test","accountUuid":"uuid-a-stale"}}`)

	// A profile endpoint IS configured; --no-probe-identity must ignore it and never hit it.
	probed := false
	srv := enrollProfileServerFor(t, map[string]accounts.ProbedIdentity{
		"at-b": {Email: "b-true@example.test", AccountUUID: "uuid-b-true"},
	})
	t.Setenv("FAK_OAUTH_PROFILE_URL", srv.URL)
	_ = probed

	regPath := filepath.Join(home, "registry.json")
	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "seat", "--adopt", "--no-probe-identity",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("adopt rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}

	// Disk-only: the recorded identity is the copied .claude.json A, untouched.
	h := findHome(t, regPath, "seat-netra")
	if h.Identity.Email != "a-stale@example.test" || h.Identity.AccountUUID != "uuid-a-stale" {
		t.Fatalf("--no-probe-identity identity = %q/%q, want the disk metadata a-stale (uuid-a-stale)",
			h.Identity.Email, h.Identity.AccountUUID)
	}
	if strings.Contains(out.String(), "identity reconcile") {
		t.Errorf("--no-probe-identity must not reconcile against the network, got: %s", out.String())
	}
}

// TestRunAccountsRemoveFlattensInboundRehome pins #4672: removing a seat WITHOUT --archive must
// repoint every OTHER seat that rehomed to it forward to the live rehome target, so the registry
// does not accrete tombstoned->tombstoned->…->live chains as intermediate hops retire. Shape is
// the issue's own acceptance case: seat C rehomes to B; `remove B --rehome-to A` must leave C
// rehoming to A, not the now-tombstoned B. (The --archive path, which repoints inbound edges onto
// the renamed handle so `restore` can reverse it, is covered by TestRunAccountsRestoreArchive.)
func TestRunAccountsRemoveFlattensInboundRehome(t *testing.T) {
	// Same hermetic roster isolation the sibling remove/restore tests use: keep the regenerated
	// dos/job views inside t.TempDir() so this can never clobber a live operator's switcher roster.
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home := t.TempDir()
	seatB := mkHome(t, home, ".claude-seat-b", "b@example.test", true)
	seatA := mkHome(t, home, ".claude-seat-a", "a@example.test", true)

	// C is a tombstoned seat whose rehome edge names B (the inbound edge #4672 is about); B and A
	// are live; A is the default and the removal's rehome target.
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"C","status":"tombstoned","rehome_to":"B"},` +
		`{"name":"B","dir":"` + jsonPath(seatB) + `"},` +
		`{"name":"A","dir":"` + jsonPath(seatA) + `","default":true}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{
		"remove", "--name", "B", "--rehome-to", "A",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("remove rc=%d stderr=%s", rc, errb.String())
	}

	got, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry should validate after remove: %v", err)
	}
	byName := map[string]accounts.Home{}
	for _, h := range got.Homes {
		byName[h.Name] = h
	}
	if b := byName["B"]; b.Active() {
		t.Fatalf("B should be tombstoned after remove: %+v", b)
	}
	// The inbound edge on C is flattened past the tombstoned B to the live target A.
	if c := byName["C"]; c.RehomeTo != "A" {
		t.Fatalf("C rehome_to = %q, want A (flattened past tombstoned B)", c.RehomeTo)
	}
	// No surviving edge anywhere names the tombstoned seat B — the pool stays legible and cannot
	// lengthen the chain when B's own target later retires.
	for _, h := range got.Homes {
		if h.RehomeTo == "B" {
			t.Fatalf("seat %q still rehomes to tombstoned B: %+v", h.Name, h)
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
