package main

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// freshSeatFakePool builds a fake fleetaccounts seat pool: fresh accounts that can
// serve now plus walled accounts under an active usage block. Distinct uuids keep
// uniquePoolAccounts from deduping the rows into one seat.
func freshSeatFakePool(fresh, walled int) []fleetaccounts.Account {
	rows := make([]fleetaccounts.Account, 0, fresh+walled)
	for i := 0; i < fresh; i++ {
		rows = append(rows, fleetaccounts.Account{
			Account:     fmt.Sprintf(".claude-fresh-%d", i),
			Tag:         fmt.Sprintf("fresh-%d", i),
			Product:     "claude",
			Kind:        fleetaccounts.KindWorker,
			AccountUUID: strPtr(fmt.Sprintf("uuid-fresh-%d", i)),
			Available:   boolPtr(true),
		})
	}
	for i := 0; i < walled; i++ {
		rows = append(rows, fleetaccounts.Account{
			Account:     fmt.Sprintf(".claude-walled-%d", i),
			Tag:         fmt.Sprintf("walled-%d", i),
			Product:     "claude",
			Kind:        fleetaccounts.KindWorker,
			AccountUUID: strPtr(fmt.Sprintf("uuid-walled-%d", i)),
			Available:   boolPtr(false),
			Blocked:     boolPtr(true),
			BlockKind:   strPtr("usage"),
			BlockReason: strPtr("usage limit reached"),
		})
	}
	return rows
}

// freshSeatPreflightFixture evaluates a SPAWN_OK preflight whose session-LEASE pool
// shows six slots (the "slots" side of the seats-not-slots trap) with live workers
// already running.
func freshSeatPreflightFixture(t *testing.T, live int) dispatchtick.PreflightResult {
	t.Helper()
	res := dispatchtick.EvaluatePreflight(dispatchtick.PreflightInput{
		Workspace:  "ws",
		MaxWorkers: 6,
		Host:       dispatchtick.HostCheck{Safe: true},
		Account:    dispatchtick.AccountCheck{Available: true, Tag: "fresh-0", Tier: 1},
		Seat: dispatchtick.SeatCheck{
			Total:  dispatchtick.IntPtr(6),
			Free:   dispatchtick.IntPtr(6 - live),
			Leased: dispatchtick.IntPtr(live),
		},
		OSWorkerProcs: live,
	})
	if !res.OK || res.Cap != 6 || res.Live != live {
		t.Fatalf("fixture preflight = %+v, want SPAWN_OK at cap 6 with live %d", res, live)
	}
	return res
}

// The core #3579 acceptance: with the fresh ceiling below the session-slot cap, the
// effective launch cap equals the fresh ceiling (no over-admission), and the wave
// records the ceiling as the binding term. The ceiling itself is derived from a fake
// seat pool: one fresh account among two walled ones. The compatibility session-cap
// knob cannot widen the one-session Claude OAuth safety bound established by #6775.
func TestDispatchFreshSeatCeilingBindsWaveCap(t *testing.T) {
	t.Setenv(fleetaccounts.SessionsPerAccountEnv, "2")
	ceiling := dispatchFreshSeatCeilingFromRoster(freshSeatFakePool(1, 2), "claude")
	if ceiling != 1 {
		t.Fatalf("fresh ceiling = %d, want 1 (one fresh Claude account)", ceiling)
	}
	res := dispatchApplyFreshSeatCeiling(freshSeatPreflightFixture(t, 0), ceiling)
	if !res.OK || res.Verdict != dispatchtick.PreflightOKVerdict {
		t.Fatalf("downsized wave = %+v, want still SPAWN_OK", res)
	}
	if res.Cap != 1 || res.Headroom != 1 || res.CapTerms.EffectiveCap != 1 {
		t.Fatalf("cap/headroom/effective = %d/%d/%d, want 1/1/1 (cap bound to the fresh ceiling, not the 6 session slots)",
			res.Cap, res.Headroom, res.CapTerms.EffectiveCap)
	}
	if res.CapTerms.Limiting != dispatchFreshSeatLimiting {
		t.Fatalf("limiting = %q, want %q recorded as the binding term", res.CapTerms.Limiting, dispatchFreshSeatLimiting)
	}
}

// With the fresh ceiling at/above the session-slot cap (a healthy pool) the fold is
// byte-identical; a no-signal ceiling (codex / absent roster) and an already-refused
// preflight each abstain.
func TestDispatchFreshSeatCeilingHealthyPoolByteIdentical(t *testing.T) {
	base := freshSeatPreflightFixture(t, 1)
	for _, ceiling := range []int{6, 9} {
		if got := dispatchApplyFreshSeatCeiling(base, ceiling); !reflect.DeepEqual(got, base) {
			t.Fatalf("ceiling %d changed the preflight: %+v", ceiling, got)
		}
	}
	if got := dispatchApplyFreshSeatCeiling(base, 0); !reflect.DeepEqual(got, base) {
		t.Fatalf("no-signal ceiling changed the preflight: %+v", got)
	}
	refused := base
	refused.OK = false
	refused.Verdict = dispatchtick.PreflightRefuseAtCap
	if got := dispatchApplyFreshSeatCeiling(refused, 1); !reflect.DeepEqual(got, refused) {
		t.Fatalf("fold touched an already-refused preflight: %+v", got)
	}
}

// When the binding ceiling leaves no headroom above the live count, the wave is
// refused -- REFUSE_NO_SEAT naming the fresh-seat ceiling -- instead of over-admitting
// launches that would wall on seats that cannot serve.
func TestDispatchFreshSeatCeilingRefusesLiveAtCeiling(t *testing.T) {
	res := dispatchApplyFreshSeatCeiling(freshSeatPreflightFixture(t, 2), 2)
	if res.OK || res.Verdict != dispatchtick.PreflightRefuseNoSeat {
		t.Fatalf("verdict = %q (ok=%v), want %q", res.Verdict, res.OK, dispatchtick.PreflightRefuseNoSeat)
	}
	if res.Cap != 2 || res.Headroom != 0 || res.CapTerms.Limiting != dispatchFreshSeatLimiting {
		t.Fatalf("cap/headroom/limiting = %d/%d/%q, want 2/0/%q", res.Cap, res.Headroom, res.CapTerms.Limiting, dispatchFreshSeatLimiting)
	}
	if !strings.Contains(res.Reason, "fresh-seat ceiling 2") {
		t.Fatalf("reason = %q, want the fresh-seat ceiling named", res.Reason)
	}
}
