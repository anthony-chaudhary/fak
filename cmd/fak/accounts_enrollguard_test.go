package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// TestRunAccountsAdd_DryRunMutatesNothing pins the #3954 --dry-run contract for `add --adopt`: it
// prints the plan and touches nothing — no seat dir, no registry, no view — while still honoring the
// read-only refusals above the short-circuit.
func TestRunAccountsAdd_DryRunMutatesNothing(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	hermeticProbeStub(t)
	mkLoggedInSeat(t, home, ".claude", "july6@example.test", "u-july6", "")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "july6", "--adopt", "--dry-run",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("dry-run rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "DRY RUN") || !strings.Contains(out.String(), "would ADOPT") {
		t.Fatalf("dry-run should describe the plan, got:\n%s", out.String())
	}
	// Nothing durable may have been created.
	if _, err := os.Stat(filepath.Join(home, ".claude-july6-netra")); err == nil {
		t.Fatal("dry-run must not create the seat dir")
	}
	if _, err := os.Stat(regPath); err == nil {
		t.Fatal("dry-run must not write the registry")
	}
}

// TestRunAccountsEnrollCurrent_DryRun pins the same contract for enroll-current — the guided verb the
// ticket centers on — and that its plan advertises the live identity probe.
func TestRunAccountsEnrollCurrent_DryRun(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	home := t.TempDir()
	hermeticProbeStub(t)
	mkLoggedInSeat(t, home, ".claude", "day26@example.test", "u-day26", "")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"enroll-current", "--name", "day26", "--dry-run",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("enroll-current dry-run rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "would enroll") {
		t.Errorf("enroll-current dry-run should say 'would enroll', got:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "would PROBE") {
		t.Errorf("enroll-current dry-run plan should advertise the identity probe, got:\n%s", out.String())
	}
	if _, err := os.Stat(filepath.Join(home, ".claude-day26-netra")); err == nil {
		t.Fatal("enroll-current dry-run must not create the seat dir")
	}
	if h := findHome(t, regPath, "day26-netra"); h.Name != "" {
		t.Fatalf("enroll-current dry-run must not enroll a row: %+v", h)
	}
}

// TestRunAccountsAdd_RefusesIdentityHijack is the core #3954 collision guard: enrolling a login whose
// account is already held by a DIFFERENT active seat is refused (no --force), and the refusal leaves
// no half-created seat dir behind.
func TestRunAccountsAdd_RefusesIdentityHijack(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	hermeticProbeStub(t) // probe 401s → identity falls back to the copied disk account
	// Seat "alpha" already owns account u-shared.
	mkLoggedInSeat(t, home, ".claude-alphasrc-netra", "shared@example.test", "u-shared", "")
	// A DIFFERENT source dir logged into the SAME account u-shared.
	mkLoggedInSeat(t, home, ".claude-betasrc-netra", "shared@example.test", "u-shared", "")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{
		"add", "--name", "alpha", "--adopt", "--from", "alphasrc-netra",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("seed enroll of alpha rc=%d stderr=%s", rc, errb.String())
	}

	out.Reset()
	errb.Reset()
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "beta", "--adopt", "--from", "betasrc-netra",
		"--registry", regPath, "--home", home,
	})
	if rc == 0 {
		t.Fatalf("enrolling a login already held by seat alpha must be refused, got rc=0\nstdout=%s", out.String())
	}
	if !strings.Contains(errb.String(), "REFUSED (identity-hijack)") {
		t.Fatalf("expected an identity-hijack refusal, got stderr:\n%s", errb.String())
	}
	// No half-seat: the refused dir must be gone, and no registry row for beta.
	if _, err := os.Stat(filepath.Join(home, ".claude-beta-netra")); err == nil {
		t.Fatal("a refused hijack must not leave a half-created seat dir")
	}
	if h := findHome(t, regPath, "beta-netra"); h.Name != "" {
		t.Fatalf("a refused hijack must not enroll a registry row: %+v", h)
	}
	// alpha must be untouched.
	if h := findHome(t, regPath, "alpha-netra"); h.Name == "" || !h.Active() {
		t.Fatalf("the existing seat alpha must survive the refusal: %+v", h)
	}
}

// TestRunAccountsAdd_DuplicateRefusalOffersRemedies is the #3954 leg that makes the refusal
// ACTIONABLE rather than merely correct. Stopping the operator is only half the contract: the
// ticket asks the duplicate branch to point at the two safe exits by name — (a) log in again under
// a FRESH config dir when a different account was intended, and (b) the canonicalize/tombstone path
// when this login should become the seat that already holds the account. Without this the refusal's
// only signposted exits were `--force` (which commits the exact duplicate the guard exists to
// prevent) and a bare `remove` (which retires the other, working seat with no fall-forward).
func TestRunAccountsAdd_DuplicateRefusalOffersRemedies(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	hermeticProbeStub(t)
	// Seat "alpha" already owns account u-shared; a second dir logged into the SAME account.
	mkLoggedInSeat(t, home, ".claude-alphasrc-netra", "shared@example.test", "u-shared", "")
	mkLoggedInSeat(t, home, ".claude-betasrc-netra", "shared@example.test", "u-shared", "")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{
		"add", "--name", "alpha", "--adopt", "--from", "alphasrc-netra",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("seed enroll of alpha rc=%d stderr=%s", rc, errb.String())
	}

	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{
		"add", "--name", "beta", "--adopt", "--from", "betasrc-netra",
		"--registry", regPath, "--home", home,
	}); rc != 1 {
		t.Fatalf("a duplicate-identity enroll must refuse with rc=1, got rc=%d\nstdout=%s", rc, out.String())
	}
	got := errb.String()
	// The refusal names BOTH remedies the ticket asked for, as runnable commands against the
	// conflicting seat — not just the --force escape hatch.
	for _, want := range []string{
		"REFUSED (identity-hijack)",
		"FRESH config dir",            // criterion 2: the fresh-dir remedy
		"CLAUDE_CONFIG_DIR=<new dir>", // ...spelled as something runnable
		"canonicalize",                // criterion 3: make this login the canonical seat
		"fak accounts enroll-current --name alpha-netra --force",
		"tombstone", // ...or retire the other seat WITH fall-forward
		"fak accounts remove --name alpha-netra --rehome-to <seat>",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("duplicate refusal must offer %q; stderr:\n%s", want, got)
		}
	}
	// The refusal stays a refusal: no half-seat dir, no registry row, alpha untouched.
	if _, err := os.Stat(filepath.Join(home, ".claude-beta-netra")); err == nil {
		t.Error("an offered-remedy refusal must still leave no half-created seat dir")
	}
	if h := findHome(t, regPath, "beta-netra"); h.Name != "" {
		t.Errorf("an offered-remedy refusal must not enroll a registry row: %+v", h)
	}
	if h := findHome(t, regPath, "alpha-netra"); h.Name == "" || !h.Active() {
		t.Errorf("the existing seat alpha must survive the refusal: %+v", h)
	}
}

// TestRunAccountsAdd_ForceOverridesHijack documents the sanctioned escape hatch: --force enrolls the
// duplicate anyway, with a loud warning instead of a refusal.
func TestRunAccountsAdd_ForceOverridesHijack(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	hermeticProbeStub(t)
	mkLoggedInSeat(t, home, ".claude-alphasrc-netra", "shared@example.test", "u-shared", "")
	mkLoggedInSeat(t, home, ".claude-betasrc-netra", "shared@example.test", "u-shared", "")
	regPath := filepath.Join(home, "registry.json")

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{
		"add", "--name", "alpha", "--adopt", "--from", "alphasrc-netra",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("seed enroll of alpha rc=%d stderr=%s", rc, errb.String())
	}

	out.Reset()
	errb.Reset()
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "beta", "--adopt", "--from", "betasrc-netra", "--force",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("--force must override the hijack refusal, got rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "--force enrolling a duplicate") {
		t.Errorf("--force override should warn loudly, got stderr:\n%s", errb.String())
	}
	if h := findHome(t, regPath, "beta-netra"); h.Name == "" {
		t.Fatal("--force override should enroll the seat")
	}
}

// TestRunAccountsAdd_VerifiesServableInBothViews pins the post-sync servability witness: a clean
// enroll with both roster views configured reports the seat serveable in both.
func TestRunAccountsAdd_VerifiesServableInBothViews(t *testing.T) {
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	home := t.TempDir()
	hermeticProbeStub(t)
	mkLoggedInSeat(t, home, ".claude", "solo@example.test", "u-solo", "")
	regPath := filepath.Join(home, "registry.json")
	dosView := filepath.Join(home, "dos.yaml")
	jobView := filepath.Join(home, "job.yaml")

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"add", "--name", "solo", "--adopt",
		"--registry", regPath, "--home", home,
		"--dos-view", dosView, "--job-view", jobView,
	})
	if rc != 0 {
		t.Fatalf("add rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "servable: seat \"solo-netra\" is ready in both roster views") {
		t.Fatalf("expected a both-views servable confirmation, got:\n%s", out.String())
	}
	// The seat must genuinely appear in both rendered views.
	dosText, _ := os.ReadFile(dosView)
	jobText, _ := os.ReadFile(jobView)
	if !strings.Contains(string(dosText), "solo-netra") {
		t.Errorf("dos view missing the seat:\n%s", dosText)
	}
	if !strings.Contains(string(jobText), "solo-netra") {
		t.Errorf("job view missing the seat:\n%s", jobText)
	}
	// And the pure verifier agrees on the refreshed registry.
	reg, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	rep := accounts.VerifySeatServable(reg.Refresh(), "solo-netra", string(dosText), string(jobText))
	if !rep.Servable {
		t.Fatalf("VerifySeatServable disagrees with the CLI: %+v", rep)
	}
}
