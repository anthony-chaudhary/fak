package fleetaccounts

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
)

// writeAccountConfig writes a minimal .claude.json into a fresh account dir whose
// oauthAccount.accountUuid stamps WHO the dir is currently logged in as — the ground truth
// currentConfigIdentity reads — and returns the dir.
func writeAccountConfig(t *testing.T, accountUUID string) string {
	t.Helper()
	dir := t.TempDir()
	doc := `{"oauthAccount":{"accountUuid":"` + accountUUID + `"}}`
	if err := os.WriteFile(filepath.Join(dir, ".claude.json"), []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// futureResetStr renders a reset string resetTime parses ("Jan 2, 3:04pm") for an
// instant d from now. The dated form's 180-day rollover keeps a year-boundary +48h
// instant future, so these stay valid whenever the tests run.
func futureResetStr(d time.Duration) string {
	return time.Now().UTC().Add(d).Format("Jan 2, 3:04pm")
}

func TestLedgerOKClearsCarriedThrottle(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"reset": futureResetStr(48 * time.Hour)},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("fresh ledger OK should clear the carried throttle, got %+v", st)
	}
	if st.StatusSource != "probe-ledger" {
		t.Fatalf("status_source = %q, want probe-ledger", st.StatusSource)
	}
}

func TestLedgerOKHoldsActiveWeeklyCap(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{
			"reset":  futureResetStr(4 * time.Hour),
			"weekly": futureResetStr(72 * time.Hour),
		},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("fresh OK must not reopen an active weekly cap, got %+v", st)
	}
	if st.BlockKind != "usage" || st.Weekly == "" {
		t.Fatalf("weekly hold shape: kind=%q weekly=%q", st.BlockKind, st.Weekly)
	}
}

// TestLedgerOKClearsThrottleOnIdentityMismatch is the #2253 refinement: a fresh OK reopens
// a weekly-capped seat when the cap provably belonged to a DIFFERENT account (the dir was
// re-logged since the cap was stamped). This is the case TestLedgerOKHoldsActiveWeeklyCap
// deliberately does NOT flip — there the throttle carries no identity, so the hold stands.
func TestLedgerOKClearsThrottleOnIdentityMismatch(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	// The dir is logged in as BBBB now; the carried weekly cap was stamped for AAAA.
	dir := writeAccountConfig(t, "BBBB")
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{
			"reset":        futureResetStr(4 * time.Hour),
			"weekly":       futureResetStr(72 * time.Hour),
			"account_uuid": "AAAA",
		},
	}}
	st := computeRuntimeStatus(".claude-a", dir, reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("proven identity mismatch must clear the weekly hold, got %+v", st)
	}
	if st.StatusSource != "probe-ledger" {
		t.Fatalf("status_source = %q, want probe-ledger", st.StatusSource)
	}
}

// TestLedgerOKHoldsThrottleOnIdentityMatch is the other verdict direction: when the cap's
// stamped identity matches the seat's CURRENT login, the weekly window still binds it and a
// fresh OK does not reopen the seat.
func TestLedgerOKHoldsThrottleOnIdentityMatch(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	dir := writeAccountConfig(t, "AAAA")
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{
			"reset":        futureResetStr(4 * time.Hour),
			"weekly":       futureResetStr(72 * time.Hour),
			"account_uuid": "AAAA",
		},
	}}
	st := computeRuntimeStatus(".claude-a", dir, reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("matching identity must keep the active weekly cap, got %+v", st)
	}
	if st.BlockKind != "usage" || st.Weekly == "" {
		t.Fatalf("weekly hold shape: kind=%q weekly=%q", st.BlockKind, st.Weekly)
	}
}

func TestLedgerBlockOverridesRegistry(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "LIMIT", time.Now(), `"reset":"3pm"`))
	st := computeRuntimeStatus(".claude-a", "", Registry{})
	if st.Available || !st.Blocked || st.BlockKind != "usage" || !st.Throttled {
		t.Fatalf("fresh ledger LIMIT should block, got %+v", st)
	}
	if st.StatusSource != "probe-ledger" || st.Reset != "3pm" {
		t.Fatalf("source/reset = %q/%q, want probe-ledger/3pm", st.StatusSource, st.Reset)
	}
}

// TestLedgerIgnoredWithoutRegDir asserts what it always asserted — a ledger sitting in a dir
// this process does not resolve to is not consulted, so the passive fold keeps the carried
// throttle — but it can no longer say so by clearing FLEET_REG_DIR alone. Since #5439 the
// rung is gated on whether the RESOLVED registry can derive a block, and with the env var
// empty the resolver falls through to the discovered rungs; on a host that runs the prober
// one of those carries a real ledger, and the test would then be measuring the operator's
// laptop. pinRegistryRungs makes "nothing is derivable anywhere" a fact of the test.
func TestLedgerIgnoredWithoutRegDir(t *testing.T) {
	rd := t.TempDir()
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	pinRegistryRungs(t, "")
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"reset": futureResetStr(48 * time.Hour)},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Throttled {
		t.Fatalf("without FLEET_REG_DIR the passive fold should keep the throttle, got %+v", st)
	}
	if st.StatusSource != "registry" {
		t.Fatalf("status_source = %q, want registry", st.StatusSource)
	}
}

func TestSyntheticProbeRowWinsOverLedger(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Sessions: []Session{{
		Account:     ".claude-a",
		Project:     "_probe",
		ProbeStatus: "LIMIT",
		Reason:      "usage limit",
	}}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || st.BlockKind != "usage" || st.StatusSource != "probe" {
		t.Fatalf("synthetic _probe row should win over the ledger, got %+v", st)
	}
}

func TestFreshCodexAuthProbeMakesReadySeatUnavailable(t *testing.T) {
	regDir := t.TempDir()
	now := time.Now().UTC()
	if err := accountprobe.AppendLedger(regDir, accountprobe.LedgerEntry{
		TS: now.Format(time.RFC3339Nano), Account: ".codex-four", Status: "AUTH", BlockReason: "guarded subscription returned HTTP 401",
	}); err != nil {
		t.Fatal(err)
	}
	row := Account{
		Account: ".codex-four", Tag: "four", Product: "codex", Kind: KindWorker,
		LoginStatus: strp("ready"), CanServe: boolp(true),
	}
	got := AnnotateWithProbes([]Account{row}, Registry{}, regDir)[0]
	if derefBool(got.Available) || !derefBool(got.Blocked) || derefStr(got.BlockKind) != "auth" {
		t.Fatalf("fresh auth probe left seat offerable: %+v", got)
	}
	if derefStr(got.LoginStatus) != "needs_login" || derefBool(got.CanServe) {
		t.Fatalf("login gate did not match guarded auth failure: %+v", got)
	}
	if derefStr(got.StatusSource) != "fresh_probe" {
		t.Fatalf("status source = %q", derefStr(got.StatusSource))
	}
}

func TestStaleCodexAuthProbeDoesNotBlockRecoveredSeat(t *testing.T) {
	regDir := t.TempDir()
	if err := accountprobe.AppendLedger(regDir, accountprobe.LedgerEntry{
		TS:      time.Now().UTC().Add(-activeProbeFreshness - time.Minute).Format(time.RFC3339Nano),
		Account: ".codex-four", Status: "AUTH",
	}); err != nil {
		t.Fatal(err)
	}
	row := Account{Account: ".codex-four", Product: "codex", Kind: KindWorker, LoginStatus: strp("ready"), CanServe: boolp(true)}
	got := AnnotateWithProbes([]Account{row}, Registry{}, regDir)[0]
	if !derefBool(got.Available) || !derefBool(got.CanServe) {
		t.Fatalf("stale probe blocked recovered seat: %+v", got)
	}
}
