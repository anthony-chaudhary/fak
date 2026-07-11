package main

import (
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// claudeRow builds a minimal annotated claude worker row for the signal-resolver tests: the
// account dir, its logged-in UUID (empty => no derivable bucket), and an optional usage_soon
// advisory string (nil => the seat is not flagged near a daily cap).
func claudeRow(dir, uuid string, usageSoon *string) fleetaccounts.Account {
	r := fleetaccounts.Account{Dir: dir, Product: "claude", UsageSoonReset: usageSoon}
	if uuid != "" {
		r.AccountUUID = &uuid
	}
	return r
}

// f64 returns a pointer to a headroom score for the nil-vs-value distinction proactiveWantsSwitch
// draws (nil => no runtime signal).
func f64(v float64) *float64 { return &v }

// TestProactiveWantsSwitch table-tests the pure predicate: it must act only when enabled AND the
// active seat is either headroom-walled or flagged usage_soon, preferring the walled reason, and
// hold with a closed reason otherwise.
func TestProactiveWantsSwitch(t *testing.T) {
	walled := f64(accounts.WalledBase + 0.5) // strictly < 0 => TierWalled
	offer := f64(accounts.OfferableBase)     // > 0 => TierOfferable
	tests := []struct {
		name      string
		enabled   bool
		headroom  *float64
		usageSoon bool
		want      proactiveReason
	}{
		{"disabled short-circuits even when walled", false, walled, true, ProactiveDisabled},
		{"walled active seat -> go_walled", true, walled, false, ProactiveGoWalled},
		{"walled wins over usage_soon", true, walled, true, ProactiveGoWalled},
		{"usage_soon with room -> go_usage_soon", true, offer, true, ProactiveGoUsageSoon},
		{"offerable, no soon flag -> hold", true, offer, false, ProactiveHoldRoom},
		{"no runtime signal, no soon flag -> hold", true, nil, false, ProactiveHoldRoom},
		{"no runtime signal but usage_soon -> go_usage_soon", true, nil, true, ProactiveGoUsageSoon},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := proactiveWantsSwitch(tc.enabled, tc.headroom, tc.usageSoon); got != tc.want {
				t.Fatalf("proactiveWantsSwitch(%v,%v,%v) = %q, want %q", tc.enabled, tc.headroom, tc.usageSoon, got, tc.want)
			}
		})
	}
}

// TestProactiveRehomeTick proves the dormant adapter: it holds (touching no roster) when the
// predicate declines, applies a real swap through forceRehome when it wants one and a healthy
// sibling exists, and surfaces the shared typed no-target reason when none does.
func TestProactiveRehomeTick(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	future := now.Add(time.Hour).UnixMilli()
	walled := f64(accounts.WalledBase + 0.5)
	offer := f64(accounts.OfferableBase)

	t.Run("holds and touches no roster when the predicate declines", func(t *testing.T) {
		root := t.TempDir()
		a := mkFailoverHome(t, root, ".claude-a", "a@x.test", "u-a", "sk-ant-oat01-a", future)
		af := newAccountFailover(root, a.Dir, func() time.Time { return now })
		out := af.proactiveRehomeTick(true, offer, false)
		if out.Acted || out.Reason != ProactiveHoldRoom {
			t.Fatalf("room seat should hold: got Acted=%v Reason=%q", out.Acted, out.Reason)
		}
		if got := af.currentConfigDir(); got != a.Dir {
			t.Fatalf("a hold must not advance the sticky dir: got %q want %q", got, a.Dir)
		}
	})

	t.Run("applies a real swap when walled and a live sibling exists", func(t *testing.T) {
		root := t.TempDir()
		a := mkFailoverHome(t, root, ".claude-a", "a@x.test", "u-a", "sk-ant-oat01-a", future)
		b := mkFailoverHome(t, root, ".claude-b", "b@x.test", "u-b", "sk-ant-oat01-b", future)
		_ = b
		af := newAccountFailover(root, a.Dir, func() time.Time { return now })
		out := af.proactiveRehomeTick(true, walled, false)
		if !out.Acted || out.Reason != ProactiveGoWalled {
			t.Fatalf("walled seat with a sibling should swap: got Acted=%v Reason=%q Err=%v", out.Acted, out.Reason, out.Err)
		}
		if got := af.currentConfigDir(); got != b.Dir {
			t.Fatalf("swap must advance the sticky dir onto the sibling: got %q want %q", got, b.Dir)
		}
	})

	t.Run("reports the shared no-target reason when the seat is walled but nowhere to go", func(t *testing.T) {
		root := t.TempDir()
		a := mkFailoverHome(t, root, ".claude-a", "a@x.test", "u-a", "sk-ant-oat01-a", future)
		af := newAccountFailover(root, a.Dir, func() time.Time { return now })
		out := af.proactiveRehomeTick(true, walled, false)
		if out.Acted {
			t.Fatal("a single-seat roster has nowhere to rehome; the tick must not claim it acted")
		}
		if out.Reason != ProactiveGoWalled {
			t.Fatalf("the decision reason should still be go_walled: got %q", out.Reason)
		}
		if out.NoTarget != FailoverNoSiblings || out.Err == nil {
			t.Fatalf("a refused swap must surface the typed no-target reason: got NoTarget=%q Err=%v", out.NoTarget, out.Err)
		}
		if got := af.currentConfigDir(); got != a.Dir {
			t.Fatalf("a refused swap must leave the sticky dir put: got %q want %q", got, a.Dir)
		}
	})
}

// TestProactiveSignalsForSeat proves the pure consumer that projects the fleet roster + rotation
// headroom map down to the active seat's (headroom, usage_soon): it reads the matched row's
// bucket score through hr (nil when the bucket carries no signal, distinct from a 0), reads the
// usage_soon advisory off the same row, and fails safe (nil,false) for a dir that names no row.
func TestProactiveSignalsForSeat(t *testing.T) {
	dir := "/homes/.claude-a"
	soon := "resets 11pm"
	walled := accounts.RotationHeadroom{accounts.UUIDBucketKey("u-a"): accounts.WalledBase}
	offer := accounts.RotationHeadroom{accounts.UUIDBucketKey("u-a"): accounts.OfferableBase}

	t.Run("walled bucket -> negative score, classifies walled", func(t *testing.T) {
		hs, us := proactiveSignalsForSeat(dir, []fleetaccounts.Account{claudeRow(dir, "u-a", nil)}, walled)
		if hs == nil || accounts.Classify(*hs) != accounts.TierWalled || us {
			t.Fatalf("want walled headroom + no usage_soon: got %v usageSoon=%v", hs, us)
		}
	})
	t.Run("offerable bucket -> positive score, classifies offerable", func(t *testing.T) {
		hs, us := proactiveSignalsForSeat(dir, []fleetaccounts.Account{claudeRow(dir, "u-a", nil)}, offer)
		if hs == nil || accounts.Classify(*hs) != accounts.TierOfferable || us {
			t.Fatalf("want offerable headroom + no usage_soon: got %v usageSoon=%v", hs, us)
		}
	})
	t.Run("bucket absent from hr -> nil headroom (no signal)", func(t *testing.T) {
		hs, us := proactiveSignalsForSeat(dir, []fleetaccounts.Account{claudeRow(dir, "u-a", nil)}, nil)
		if hs != nil || us {
			t.Fatalf("an unscored bucket must read as no signal: got %v usageSoon=%v", hs, us)
		}
	})
	t.Run("usage_soon advisory read off the row regardless of headroom", func(t *testing.T) {
		hs, us := proactiveSignalsForSeat(dir, []fleetaccounts.Account{claudeRow(dir, "u-a", &soon)}, offer)
		if hs == nil || accounts.Classify(*hs) != accounts.TierOfferable || !us {
			t.Fatalf("want offerable headroom + usage_soon: got %v usageSoon=%v", hs, us)
		}
	})
	t.Run("row without a UUID -> nil headroom but still reads usage_soon", func(t *testing.T) {
		hs, us := proactiveSignalsForSeat(dir, []fleetaccounts.Account{claudeRow(dir, "", &soon)}, walled)
		if hs != nil || !us {
			t.Fatalf("no bucket key => no headroom, but usage_soon still honored: got %v usageSoon=%v", hs, us)
		}
	})
	t.Run("normalization: a trailing separator still matches the row", func(t *testing.T) {
		hs, _ := proactiveSignalsForSeat(dir+string(os.PathSeparator), []fleetaccounts.Account{claudeRow(dir, "u-a", nil)}, walled)
		if hs == nil || accounts.Classify(*hs) != accounts.TierWalled {
			t.Fatalf("normalized dir match must find the seat: got %v", hs)
		}
	})
	t.Run("no matching row -> fail safe (nil,false)", func(t *testing.T) {
		hs, us := proactiveSignalsForSeat("/homes/.claude-ZZZ", []fleetaccounts.Account{claudeRow(dir, "u-a", nil)}, walled)
		if hs != nil || us {
			t.Fatalf("an unknown dir must yield no signal: got %v usageSoon=%v", hs, us)
		}
	})
	t.Run("non-claude row with the same dir is ignored", func(t *testing.T) {
		row := claudeRow(dir, "u-a", &soon)
		row.Product = "opencode"
		hs, us := proactiveSignalsForSeat(dir, []fleetaccounts.Account{row}, walled)
		if hs != nil || us {
			t.Fatalf("only claude worker seats carry the rotation signal: got %v usageSoon=%v", hs, us)
		}
	})
}

// TestProactiveTickFromRoster proves the dormant composition end to end: it resolves the active
// seat's signals off the roster + headroom and drives forceRehome, swapping onto a healthy sibling
// when the active seat is walled or usage_soon, and holding (touching no roster) when it has room.
func TestProactiveTickFromRoster(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	future := now.Add(time.Hour).UnixMilli()

	newFleet := func(t *testing.T) (root string, a, b accounts.Home) {
		root = t.TempDir()
		a = mkFailoverHome(t, root, ".claude-a", "a@x.test", "u-a", "sk-ant-oat01-a", future)
		b = mkFailoverHome(t, root, ".claude-b", "b@x.test", "u-b", "sk-ant-oat01-b", future)
		return root, a, b
	}

	t.Run("active seat walled -> swaps onto the healthy sibling", func(t *testing.T) {
		root, a, b := newFleet(t)
		af := newAccountFailover(root, a.Dir, func() time.Time { return now })
		roster := []fleetaccounts.Account{claudeRow(a.Dir, "u-a", nil), claudeRow(b.Dir, "u-b", nil)}
		hr := accounts.RotationHeadroom{accounts.UUIDBucketKey("u-a"): accounts.WalledBase}
		out := af.proactiveTickFromRoster(true, roster, hr)
		if !out.Acted || out.Reason != ProactiveGoWalled || af.currentConfigDir() != b.Dir {
			t.Fatalf("walled active seat should swap onto b: Acted=%v Reason=%q dir=%q Err=%v", out.Acted, out.Reason, af.currentConfigDir(), out.Err)
		}
	})

	t.Run("active seat usage_soon -> swaps with go_usage_soon", func(t *testing.T) {
		root, a, b := newFleet(t)
		af := newAccountFailover(root, a.Dir, func() time.Time { return now })
		soon := "resets 11pm"
		roster := []fleetaccounts.Account{claudeRow(a.Dir, "u-a", &soon), claudeRow(b.Dir, "u-b", nil)}
		hr := accounts.RotationHeadroom{accounts.UUIDBucketKey("u-a"): accounts.OfferableBase}
		out := af.proactiveTickFromRoster(true, roster, hr)
		if !out.Acted || out.Reason != ProactiveGoUsageSoon || af.currentConfigDir() != b.Dir {
			t.Fatalf("usage_soon active seat should swap onto b: Acted=%v Reason=%q dir=%q", out.Acted, out.Reason, af.currentConfigDir())
		}
	})

	t.Run("active seat has room -> holds, touches no roster", func(t *testing.T) {
		root, a, b := newFleet(t)
		_ = b
		af := newAccountFailover(root, a.Dir, func() time.Time { return now })
		roster := []fleetaccounts.Account{claudeRow(a.Dir, "u-a", nil), claudeRow(b.Dir, "u-b", nil)}
		hr := accounts.RotationHeadroom{accounts.UUIDBucketKey("u-a"): accounts.OfferableBase}
		out := af.proactiveTickFromRoster(true, roster, hr)
		if out.Acted || out.Reason != ProactiveHoldRoom || af.currentConfigDir() != a.Dir {
			t.Fatalf("a seat with room must hold on its dir: Acted=%v Reason=%q dir=%q", out.Acted, out.Reason, af.currentConfigDir())
		}
	})

	t.Run("disabled -> never resolves a swap even when walled", func(t *testing.T) {
		root, a, b := newFleet(t)
		_ = b
		af := newAccountFailover(root, a.Dir, func() time.Time { return now })
		roster := []fleetaccounts.Account{claudeRow(a.Dir, "u-a", nil), claudeRow(b.Dir, "u-b", nil)}
		hr := accounts.RotationHeadroom{accounts.UUIDBucketKey("u-a"): accounts.WalledBase}
		out := af.proactiveTickFromRoster(false, roster, hr)
		if out.Acted || out.Reason != ProactiveDisabled || af.currentConfigDir() != a.Dir {
			t.Fatalf("disabled tick must not move: Acted=%v Reason=%q dir=%q", out.Acted, out.Reason, af.currentConfigDir())
		}
	})
}
