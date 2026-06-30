package accounts

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

func TestCredFreshness(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	window := time.Hour
	if !CredFreshness(now.Add(-30*time.Minute), now, window) {
		t.Error("a token refreshed 30m ago within a 1h window must be fresh")
	}
	if CredFreshness(now.Add(-90*time.Minute), now, window) {
		t.Error("a token refreshed 90m ago past a 1h window must be stale")
	}
	if CredFreshness(time.Time{}, now, window) {
		t.Error("an unknown (zero) last-refresh must fail closed to stale")
	}
	if CredFreshness(now, now, 0) {
		t.Error("a non-positive window must fail closed to stale")
	}
}

func credCheck(fresh, refreshed bool) CredCheck {
	return func(context.Context) (bool, bool) { return fresh, refreshed }
}

func TestRehydrateCredRung_freshClears(t *testing.T) {
	adm := rehydrate.NewGate(NewRehydrateCredRung(credCheck(true, false))).Admit(context.Background(), dormancy.Cold)
	if !adm.Admitted {
		t.Fatalf("a fresh credential must clear; got refusedBy=%q", adm.RefusedBy)
	}
}

func TestRehydrateCredRung_staleButRefreshedClears(t *testing.T) {
	adm := rehydrate.NewGate(NewRehydrateCredRung(credCheck(false, true))).Admit(context.Background(), dormancy.Cold)
	if !adm.Admitted {
		t.Fatalf("a stale credential refreshed in place must clear; got refusedBy=%q", adm.RefusedBy)
	}
}

// #1183 acceptance: a credential past its refresh window that cannot refresh (login wall)
// refuses a cold re-entry with STALE_CRED, before any serve.
func TestRehydrateCredRung_staleUnrefreshableRefuses(t *testing.T) {
	adm := rehydrate.NewGate(NewRehydrateCredRung(credCheck(false, false))).Admit(context.Background(), dormancy.Cold)
	if adm.Admitted {
		t.Fatal("a stale, unrefreshable credential must refuse admission")
	}
	if adm.RefusedBy != rehydrate.StaleCred {
		t.Fatalf("refusal reason = %q, want STALE_CRED", adm.RefusedBy)
	}
}

func TestRehydrateCredRung_nilCheckFailsClosed(t *testing.T) {
	adm := rehydrate.NewGate(NewRehydrateCredRung(nil)).Admit(context.Background(), dormancy.Cold)
	if adm.Admitted || adm.RefusedBy != rehydrate.StaleCred {
		t.Fatalf("a nil check must fail closed with STALE_CRED; got %+v", adm)
	}
}

// #1183 acceptance: a warm re-entry does not run the credential rung (verbatim resume).
func TestRehydrateCredRung_warmDoesNotFire(t *testing.T) {
	adm := rehydrate.NewGate(NewRehydrateCredRung(credCheck(false, false))).Admit(context.Background(), dormancy.Warm)
	if !adm.Admitted {
		t.Fatalf("a warm wake must admit verbatim (no cred rung); got refusedBy=%q", adm.RefusedBy)
	}
	if len(adm.Ran) != 0 {
		t.Fatalf("a warm wake must run zero rungs; ran %v", adm.RanReasons())
	}
}

// End to end through the age check: a token stamped older than its window drives the rung to
// STALE_CRED when refresh is walled.
func TestRehydrateCredRung_oldTokenViaFreshnessRefuses(t *testing.T) {
	now := time.Unix(2_000_000, 0)
	lastRefresh := now.Add(-2 * time.Hour) // older than the 1h window
	check := func(context.Context) (bool, bool) {
		fresh := CredFreshness(lastRefresh, now, time.Hour)
		return fresh, false // refresh walled
	}
	adm := rehydrate.NewGate(NewRehydrateCredRung(check)).Admit(context.Background(), dormancy.Cold)
	if adm.Admitted || adm.RefusedBy != rehydrate.StaleCred {
		t.Fatalf("an over-window token with a walled refresh must refuse STALE_CRED; got %+v", adm)
	}
}
