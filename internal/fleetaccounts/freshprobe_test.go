package fleetaccounts

import (
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
)

// writeProbeLedger writes probe ledger lines under regDir.
func writeProbeLedger(t *testing.T, regDir string, lines ...string) {
	t.Helper()
	if err := os.MkdirAll(regDir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := ""
	for _, l := range lines {
		body += l + "\n"
	}
	if err := os.WriteFile(accountprobe.ProbeLedgerPath(regDir), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func probeLine(t *testing.T, account, status string, when time.Time, extra string) string {
	t.Helper()
	ts := when.UTC().Format(time.RFC3339)
	line := `{"ts":"` + ts + `","account":"` + account + `","status":"` + status + `"`
	if extra != "" {
		line += "," + extra
	}
	return line + "}"
}

func TestFreshProbeFromLedgerNoEntry(t *testing.T) {
	rd := t.TempDir()
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	if got := FreshProbeFromLedger(".claude-missing", rd, time.Now().UTC(), 0); got != nil {
		t.Fatalf("FreshProbeFromLedger for missing account = %+v, want nil", got)
	}
}

func TestFreshProbeFromLedgerFreshOK(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", now.Add(-5*time.Minute), ""))
	got := FreshProbeFromLedger(".claude-a", rd, now, 0)
	if got == nil || !got.Available {
		t.Fatalf("fresh OK = %+v, want available", got)
	}
	if got.AgeMin < 4.9 || got.AgeMin > 5.1 {
		t.Fatalf("age_min = %v, want ~5", got.AgeMin)
	}
}

func TestFreshProbeFromLedgerFreshLimitWithReset(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "LIMIT", now.Add(-2*time.Minute), `"reset":"3pm","weekly":"Jul 5"`))
	got := FreshProbeFromLedger(".claude-a", rd, now, 0)
	if got == nil || got.Available {
		t.Fatalf("fresh LIMIT = %+v, want blocked", got)
	}
	if got.BlockKind != "usage" {
		t.Fatalf("block_kind = %q, want usage", got.BlockKind)
	}
	if got.BlockReason != "usage limit; resets 3pm" {
		t.Fatalf("block_reason = %q", got.BlockReason)
	}
	if got.Reset != "3pm" || got.Weekly != "Jul 5" {
		t.Fatalf("reset/weekly = %q/%q, want 3pm/Jul 5", got.Reset, got.Weekly)
	}
}

func TestFreshProbeFromLedgerAuthDefaultReason(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "AUTH", now.Add(-1*time.Minute), ""))
	got := FreshProbeFromLedger(".claude-a", rd, now, 0)
	if got == nil || got.Available {
		t.Fatalf("fresh AUTH = %+v, want blocked", got)
	}
	if got.BlockKind != "auth" || got.BlockReason != "auth block" {
		t.Fatalf("kind/reason = %q/%q, want auth/'auth block'", got.BlockKind, got.BlockReason)
	}
}

func TestFreshProbeFromLedgerStaleIsNil(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	// 25 min old, explicit fresh window 20 -> stale -> nil (independent of any
	// ambient FLEET_PROBE_FRESH_MIN).
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", now.Add(-25*time.Minute), ""))
	if got := FreshProbeFromLedger(".claude-a", rd, now, 20); got != nil {
		t.Fatalf("stale probe = %+v, want nil", got)
	}
}

func TestFreshProbeFromLedgerUnknownStatusIsNil(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "APIERR", now.Add(-1*time.Minute), ""))
	if got := FreshProbeFromLedger(".claude-a", rd, now, 0); got != nil {
		t.Fatalf("APIERR probe = %+v, want nil (not a clean signal)", got)
	}
}

func TestProbeLedgerFreshMinEnvOverride(t *testing.T) {
	t.Setenv("FLEET_PROBE_FRESH_MIN", "45")
	if got := ProbeLedgerFreshMin(); got != 45 {
		t.Fatalf("ProbeLedgerFreshMin = %v, want 45", got)
	}
}

func TestProbeLedgerEntitlementFreshMinDefaultAndEnvOverride(t *testing.T) {
	if got := ProbeLedgerEntitlementFreshMin(); got != 1440 {
		t.Fatalf("ProbeLedgerEntitlementFreshMin = %v, want 1440 (24h)", got)
	}
	t.Setenv("FLEET_PROBE_ENTITLEMENT_FRESH_MIN", "90")
	if got := ProbeLedgerEntitlementFreshMin(); got != 90 {
		t.Fatalf("ProbeLedgerEntitlementFreshMin = %v, want 90", got)
	}
}

// An entitlement verdict outlives the 20-minute window a usage wall gets. This is the
// live july2new-netra shape: an ACCESS probe ("Claude subscription access disabled")
// recorded ~19h ago, well past ProbeLedgerFreshMin. Before the kind-aware window it
// aged out to nil, the seat carried no block, and — because a 403 burns no quota — the
// headroom-weighted allocator ranked the dead seat among the emptiest and routed
// workers that died in ~0.5s.
func TestFreshProbeFromLedgerEntitlementOutlivesUsageWindow(t *testing.T) {
	now := time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		status string
		kind   string
	}{
		{"ACCESS", "access"},
		{"AUTH", "auth"},
		{"CREDIT", "credit"},
	} {
		t.Run(tc.status, func(t *testing.T) {
			rd := t.TempDir()
			writeProbeLedger(t, rd, probeLine(t, ".claude-a", tc.status, now.Add(-19*time.Hour), ""))
			got := FreshProbeFromLedger(".claude-a", rd, now, 0)
			if got == nil {
				t.Fatalf("19h-old %s = nil, want a blocked verdict (entitlement blocks do not heal by waiting)", tc.status)
			}
			if got.Available || got.BlockKind != tc.kind {
				t.Fatalf("19h-old %s = %+v, want blocked with kind %q", tc.status, got, tc.kind)
			}
		})
	}
}

// The widened window is entitlement-only: a usage wall genuinely expires when quota
// resets, and a stale OK must still never mask a real current limit. Both keep aging
// out at ProbeLedgerFreshMin.
func TestFreshProbeFromLedgerUsageAndOKStillAgeOut(t *testing.T) {
	now := time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)
	for _, status := range []string{"LIMIT", "OK"} {
		t.Run(status, func(t *testing.T) {
			rd := t.TempDir()
			writeProbeLedger(t, rd, probeLine(t, ".claude-a", status, now.Add(-19*time.Hour), ""))
			if got := FreshProbeFromLedger(".claude-a", rd, now, 0); got != nil {
				t.Fatalf("19h-old %s = %+v, want nil (only entitlement verdicts get the long window)", status, got)
			}
		})
	}
}

// Self-healing: honoring an old ACCESS never pins a seat that has recovered, because
// only the newest row per account is read. A later OK IS the entry, so the block is
// gone — no reset, no eviction, no operator step.
func TestFreshProbeFromLedgerNewerOKClearsOlderAccessBlock(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd,
		probeLine(t, ".claude-a", "ACCESS", now.Add(-19*time.Hour), `"block_reason":"Claude subscription access disabled"`),
		probeLine(t, ".claude-a", "OK", now.Add(-3*time.Minute), ""),
	)
	got := FreshProbeFromLedger(".claude-a", rd, now, 0)
	if got == nil || !got.Available {
		t.Fatalf("newer OK after ACCESS = %+v, want available", got)
	}
}

// The long window is finite on purpose. A prober that stops running must not strand a
// seat forever on its last bad verdict: past the window the fold returns nil and the
// registry's own status decides, exactly as it did before this window existed. This is
// the july17 trap — a transient org-disabled blip must not become permanent.
func TestFreshProbeFromLedgerEntitlementExpiresPastLongWindow(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "ACCESS", now.Add(-25*time.Hour), ""))
	if got := FreshProbeFromLedger(".claude-a", rd, now, 0); got != nil {
		t.Fatalf("25h-old ACCESS = %+v, want nil (a dead prober must not strand a seat)", got)
	}
}

// An explicit window still means what it says: it is not silently widened for an
// entitlement kind, so a caller that asks for 20 minutes gets 20 minutes.
func TestFreshProbeFromLedgerExplicitWindowNotWidened(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 26, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "ACCESS", now.Add(-25*time.Minute), ""))
	if got := FreshProbeFromLedger(".claude-a", rd, now, 20); got != nil {
		t.Fatalf("25min-old ACCESS with explicit window 20 = %+v, want nil", got)
	}
}
