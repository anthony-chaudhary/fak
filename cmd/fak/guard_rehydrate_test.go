package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// guard_rehydrate_test.go — the #1834 witness: a headless `fak accounts launch` /
// `fak guard` with an expired-but-refreshable OAuth credential must refresh and serve (no
// surfaced 401); an expired-and-unrefreshable credential must refuse with a clear re-auth
// route (STALE_CRED), never a raw upstream_unauthorized. Before #1834's fix,
// accounts.NewRehydrateCredRung (#1183) had zero production call sites — nothing in the
// guard launch path ever ran it — so BOTH cases below would have fallen through to
// resolveGuardUpstream's pin-on-intent branch and hit the wrapped agent's first request with
// a stale bearer, relying solely on the reactive 3s poll (internal/agent's
// defaultAuthRefreshWindow, since raised to 10s as a backstop) to save a headless launch that
// has no interactive `claude` process to rewrite the credential at all. These tests exercise
// guardRunHeadlessRehydrate / guardHeadlessCredCheck directly against that OLD absence: they
// fail against a no-op wiring (the pre-fix state) and pass against the rung actually running.

func writeCred(t *testing.T, path, tok string, expiresAtMs int64) {
	t.Helper()
	body := fmt.Sprintf(`{"claudeAiOauth":{"accessToken":%q,"expiresAt":%d}}`, tok, expiresAtMs)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

// TestGuardHeadlessCredCheck_LiveCredentialNoWait proves the common case adds zero latency:
// a credential that is not yet expired reports fresh=true on the very first read, no polling.
func TestGuardHeadlessCredCheck_LiveCredentialNoWait(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	now := func() time.Time { return time.UnixMilli(nowMs) }
	writeCred(t, credPath, "sk-ant-oat01-live", nowMs+3_600_000)

	slept := 0
	check := guardHeadlessCredCheck(credPath, now, func(time.Duration) { slept++ })
	fresh, refreshed := check(context.Background())
	if !fresh || refreshed {
		t.Fatalf("live credential: got fresh=%v refreshed=%v, want fresh=true refreshed=false", fresh, refreshed)
	}
	if slept != 0 {
		t.Fatalf("live credential must not poll/sleep at all; slept %d times", slept)
	}
}

// TestGuardHeadlessCredCheck_ExpiredThenRefreshedInPlace proves the PROACTIVE half of #1834:
// a credential that is expired at check time but gets rewritten with a live expiry partway
// through the wait window (simulating Claude Code — or an operator's re-auth cron — rotating
// the token concurrently) is picked up as refreshed=true, without ever needing a 401 to fire.
func TestGuardHeadlessCredCheck_ExpiredThenRefreshedInPlace(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	startMs := int64(1_700_000_000_000)
	elapsed := time.Duration(0)
	now := func() time.Time { return time.UnixMilli(startMs).Add(elapsed) }
	writeCred(t, credPath, "sk-ant-oat01-expired", startMs-1)

	polls := 0
	sleep := func(d time.Duration) {
		polls++
		elapsed += d
		if polls == 2 {
			// The rotation lands on the second poll: a fresh, live token appears on disk.
			writeCred(t, credPath, "sk-ant-oat01-rotated", now().Add(time.Hour).UnixMilli())
		}
	}
	check := guardHeadlessCredCheck(credPath, now, sleep)
	fresh, refreshed := check(context.Background())
	if fresh || !refreshed {
		t.Fatalf("mid-poll rotation: got fresh=%v refreshed=%v, want fresh=false refreshed=true", fresh, refreshed)
	}
	if polls < 2 {
		t.Fatalf("expected the check to poll at least twice before the rotation landed; polled %d times", polls)
	}
}

// TestGuardHeadlessCredCheck_ExpiredNeverRefreshesWithinWindow proves the fail-closed half:
// a credential that stays expired for the whole wait window reports fresh=false,
// refreshed=false — never blocking forever, never silently claiming success.
func TestGuardHeadlessCredCheck_ExpiredNeverRefreshesWithinWindow(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	startMs := int64(1_700_000_000_000)
	elapsed := time.Duration(0)
	now := func() time.Time { return time.UnixMilli(startMs).Add(elapsed) }
	writeCred(t, credPath, "sk-ant-oat01-expired", startMs-1)

	sleep := func(d time.Duration) { elapsed += d } // credential is never rewritten
	check := guardHeadlessCredCheck(credPath, now, sleep)
	fresh, refreshed := check(context.Background())
	if fresh || refreshed {
		t.Fatalf("never-refreshed credential: got fresh=%v refreshed=%v, want both false", fresh, refreshed)
	}
	if elapsed < guardHeadlessRehydrateWindowDuration() {
		t.Fatalf("expected the check to exhaust the full wait window (%s); only elapsed %s", guardHeadlessRehydrateWindowDuration(), elapsed)
	}
}

// TestGuardHeadlessCredCheck_NoCredentialFile proves a missing/unreadable credentials file
// (nothing to check or refresh) fails closed rather than blocking or guessing fresh.
func TestGuardHeadlessCredCheck_NoCredentialFile(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json") // never written
	check := guardHeadlessCredCheck(credPath, nil, func(time.Duration) {
		t.Fatal("must not poll when there is no credential file to begin with")
	})
	fresh, refreshed := check(context.Background())
	if fresh || refreshed {
		t.Fatalf("missing credential file: got fresh=%v refreshed=%v, want both false", fresh, refreshed)
	}
}

// TestGuardRunHeadlessRehydrate is the #1834 done-condition witness end to end, at the level
// cmdGuard actually calls: it wires accounts.NewRehydrateCredRung through
// guardRunHeadlessRehydrate exactly as the guard launch path does, and asserts the two
// required outcomes plus the "must not apply" guards. Before the fix (no call site for
// NewRehydrateCredRung anywhere in cmd/ or internal/ outside its own package and tests),
// there was no function to call here at all — this test's assertions on a REFRESHING vs
// NON-REFRESHING credential are the fail-before/pass-after witness the issue asks for: they
// fail if this wiring is ever removed (Ran would stay false and Refused would stay false for
// the stale-and-unrefreshable case, masking the exact 401 regression #1834 reports).
func TestGuardRunHeadlessRehydrate(t *testing.T) {
	t.Run("headless_refreshable_credential_proceeds_no_401", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		// Expired at "now" -- but accounts still holds a valid login, so guardHeadlessCredCheck
		// (the real check cmdGuard builds) will poll; here we pre-seed a LIVE token so the very
		// first read already clears, standing in for "the rehydrate check found it fresh or
		// refreshed it before serving."
		writeCred(t, credPath, "sk-ant-oat01-live", time.Now().Add(time.Hour).UnixMilli())

		v := guardRunHeadlessRehydrate(false /* headless */, true /* pinUpstream */, credPath)
		if !v.Ran {
			t.Fatal("expected the rung to run on a headless pinned-subscription launch")
		}
		if v.Refused {
			t.Fatalf("a fresh/refreshable credential must NOT refuse; got Detail=%q", v.Detail)
		}
	})

	t.Run("headless_unrefreshable_credential_refuses_stale_cred_not_401", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		// Expired and never rewritten -- nothing will refresh it within the wait window. This
		// exercises the REAL time.Now/time.Sleep path (guardRunHeadlessRehydrate always builds
		// its check with nil/nil), so shrink the window via its documented env-var override —
		// otherwise this subtest would burn the full 30s ceiling waiting for a rotation that
		// never comes. FAK_GUARD_PARK_BUDGET=0 keeps the #2260 park out of this subtest (it
		// would otherwise wait for a re-login that never comes); the park's own wiring is
		// witnessed by the park subtests below.
		t.Setenv("FAK_AUTH_REFRESH_WINDOW", "50ms")
		t.Setenv("FAK_GUARD_PARK_BUDGET", "0")
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())

		v := guardRunHeadlessRehydrate(false /* headless */, true /* pinUpstream */, credPath)
		if !v.Ran {
			t.Fatal("expected the rung to run on a headless pinned-subscription launch")
		}
		if !v.Refused {
			t.Fatal("an expired, unrefreshable credential must refuse (STALE_CRED) rather than let a raw upstream 401 happen")
		}
		if v.Detail == "" {
			t.Fatal("a STALE_CRED refusal must carry a re-auth-routing detail, not a bare refusal")
		}
	})

	t.Run("park_exhaustion_still_refuses_stale_cred", func(t *testing.T) {
		// The #2260 park wiring, give-up half: an expired credential that never rotates
		// parks for the (shrunk) budget and then still refuses STALE_CRED — the park delays
		// the refusal, it never converts a genuine staleness into a silent pass.
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		t.Setenv("FAK_AUTH_REFRESH_WINDOW", "50ms")
		t.Setenv("FAK_GUARD_PARK_BUDGET", "80ms")
		t.Setenv("FAK_GUARD_PARK_POLL", "1s") // clamped to min 1s; one poll spends the 80ms budget
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())

		v := guardRunHeadlessRehydrate(false, true, credPath)
		if !v.Ran || !v.Refused {
			t.Fatalf("park exhaustion must still refuse: got %+v", v)
		}
	})

	t.Run("park_recovers_when_relogin_lands", func(t *testing.T) {
		// The #2260 park wiring, recovery half (the fleet's actual self-heal): the launch
		// finds an expired credential, parks, a re-login rewrites the file mid-park, and the
		// launch PROCEEDS instead of dying — the pre-#2260 behavior was a refusal inside the
		// 30s rehydrate ceiling no matter what landed minutes later.
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		t.Setenv("FAK_AUTH_REFRESH_WINDOW", "50ms")
		t.Setenv("FAK_GUARD_PARK_BUDGET", "30s") // ample; recovery ends the park well before this
		t.Setenv("FAK_GUARD_PARK_POLL", "1s")    // min-clamped poll keeps the test ~1s, not minutes
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())
		go func() {
			time.Sleep(200 * time.Millisecond) // the human runs `claude` once, mid-park
			writeCred(t, credPath, "sk-ant-oat01-fresh", time.Now().Add(time.Hour).UnixMilli())
		}()

		v := guardRunHeadlessRehydrate(false, true, credPath)
		if !v.Ran || v.Refused {
			t.Fatalf("a re-login landing mid-park must let the launch proceed: got %+v", v)
		}
	})

	t.Run("interactive_launch_is_left_alone", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())

		v := guardRunHeadlessRehydrate(true /* stdinInteractive */, true, credPath)
		if v.Ran {
			t.Fatal("an interactive launch must not run the proactive rung — the existing reactive per-request re-read already covers it")
		}
	})

	t.Run("not_pinning_subscription_is_left_alone", func(t *testing.T) {
		dir := t.TempDir()
		credPath := filepath.Join(dir, ".credentials.json")
		writeCred(t, credPath, "sk-ant-oat01-dead", time.Now().Add(-time.Hour).UnixMilli())

		v := guardRunHeadlessRehydrate(false, false /* pinUpstream */, credPath)
		if v.Ran {
			t.Fatal("a launch not pinning the Claude subscription OAuth token has no credential this rung understands and must not run")
		}
	})
}
