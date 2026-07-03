package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
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
	// This test exercises the WAIT-for-rotation path (a concurrent `claude` rewriting the file),
	// so disable the active-refresh trigger — otherwise the default-on branch would spawn a real
	// `claude` before the poll loop is ever reached.
	t.Setenv("FAK_GUARD_AUTO_REFRESH", "0")
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
	// Wait-path test: disable the active-refresh trigger so we reach the poll loop deterministically.
	t.Setenv("FAK_GUARD_AUTO_REFRESH", "0")
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
	// These subtests witness the wait/refuse/park outcomes on the real time.Now/nil-spawn path.
	// The active-refresh trigger (default-on) would spawn a real `claude` against an expired
	// credential here, so disable it for the whole group; its own behavior is covered separately
	// by the credrefresh tests and the active-refresh subtest below.
	t.Setenv("FAK_GUARD_AUTO_REFRESH", "0")
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

// TestGuardCredCheck_ActiveRefreshShortCircuitsPoll is the active-refresh witness: an expired
// credential is refreshed in place by a spawn (standing in for `claude -p` rotating the token),
// and the check returns refreshed=true WITHOUT ever entering the poll/wait loop. Before this
// wiring the only recovery was to wait for someone else to rewrite the file.
func TestGuardCredCheck_ActiveRefreshShortCircuitsPoll(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	now := func() time.Time { return time.UnixMilli(nowMs) }
	writeCred(t, credPath, "sk-ant-oat01-expired", nowMs-1)

	slept := 0
	var spawn accounts.RefreshSpawn = func(ctx context.Context, cfgDir string) error {
		// the spawn rotates the on-disk token forward, as `claude -p` would
		writeCred(t, filepath.Join(cfgDir, ".credentials.json"), "sk-ant-oat01-rotated", nowMs+3_600_000)
		return nil
	}
	check := guardCredCheckWithRefresh(credPath, guardHeadlessRehydrateWindowDuration(), now,
		func(time.Duration) { slept++ }, spawn)

	fresh, refreshed := check(context.Background())
	if fresh || !refreshed {
		t.Fatalf("active refresh: got fresh=%v refreshed=%v, want fresh=false refreshed=true", fresh, refreshed)
	}
	if slept != 0 {
		t.Fatalf("a successful active refresh must not enter the poll/wait loop; slept %d times", slept)
	}
}

// TestGuardCredCheck_AutoRefreshDisabledFallsThroughToPoll pins the opt-out safety: with
// FAK_GUARD_AUTO_REFRESH=0 the spawn is never called and the check behaves exactly like the old
// wait-for-rotation poll (here: never rotates, so it exhausts the window and refuses).
func TestGuardCredCheck_AutoRefreshDisabledFallsThroughToPoll(t *testing.T) {
	t.Setenv("FAK_GUARD_AUTO_REFRESH", "0")
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	startMs := int64(1_700_000_000_000)
	elapsed := time.Duration(0)
	now := func() time.Time { return time.UnixMilli(startMs).Add(elapsed) }
	writeCred(t, credPath, "sk-ant-oat01-expired", startMs-1)

	spawnCalled := false
	var spawn accounts.RefreshSpawn = func(ctx context.Context, cfgDir string) error {
		spawnCalled = true
		return nil
	}
	sleep := func(d time.Duration) { elapsed += d }
	check := guardCredCheckWithRefresh(credPath, guardHeadlessRehydrateWindowDuration(), now, sleep, spawn)

	fresh, refreshed := check(context.Background())
	if fresh || refreshed {
		t.Fatalf("disabled auto-refresh, never-rotated: got fresh=%v refreshed=%v, want both false", fresh, refreshed)
	}
	if spawnCalled {
		t.Fatal("FAK_GUARD_AUTO_REFRESH=0 must not call the refresh spawn at all")
	}
}

// TestGuardCredCheck_SkewRefreshesBeforeExpiry proves the proactive skew: a token that is still
// live NOW but expires within the skew window is treated as needing refresh, and a rotating spawn
// clears it as refreshed rather than letting a request race the imminent expiry.
func TestGuardCredCheck_SkewRefreshesBeforeExpiry(t *testing.T) {
	t.Setenv("FAK_GUARD_REFRESH_SKEW", "5m")
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	now := func() time.Time { return time.UnixMilli(nowMs) }
	// expires in 2 minutes: still live now, but inside the 5-minute skew window
	writeCred(t, credPath, "sk-ant-oat01-soon", nowMs+2*60*1000)

	rotated := false
	var spawn accounts.RefreshSpawn = func(ctx context.Context, cfgDir string) error {
		rotated = true
		writeCred(t, filepath.Join(cfgDir, ".credentials.json"), "sk-ant-oat01-rotated", nowMs+3_600_000)
		return nil
	}
	check := guardCredCheckWithRefresh(credPath, guardHeadlessRehydrateWindowDuration(), now,
		func(time.Duration) {}, spawn)

	fresh, refreshed := check(context.Background())
	if !rotated {
		t.Fatal("a token expiring within the skew window must trigger a proactive refresh")
	}
	if fresh || !refreshed {
		t.Fatalf("skew refresh: got fresh=%v refreshed=%v, want fresh=false refreshed=true", fresh, refreshed)
	}
}

// TestGuardCredCheck_SkewButRefreshFailsStillServesLiveToken guards the regression I was careful
// to avoid: inside the skew window, if refresh does NOT land but the token is still live right
// now, the check must return fresh=true immediately, never block on a wait for a not-yet-due
// rotation.
func TestGuardCredCheck_SkewButRefreshFailsStillServesLiveToken(t *testing.T) {
	t.Setenv("FAK_GUARD_REFRESH_SKEW", "5m")
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")
	nowMs := int64(1_700_000_000_000)
	now := func() time.Time { return time.UnixMilli(nowMs) }
	writeCred(t, credPath, "sk-ant-oat01-soon", nowMs+2*60*1000) // live now, within skew

	slept := 0
	var spawn accounts.RefreshSpawn = func(ctx context.Context, cfgDir string) error { return nil } // no rotation
	check := guardCredCheckWithRefresh(credPath, guardHeadlessRehydrateWindowDuration(), now,
		func(time.Duration) { slept++ }, spawn)

	fresh, refreshed := check(context.Background())
	if !fresh || refreshed {
		t.Fatalf("skew, refresh failed, token still live: got fresh=%v refreshed=%v, want fresh=true refreshed=false", fresh, refreshed)
	}
	if slept != 0 {
		t.Fatalf("a still-live token must not block on the wait loop; slept %d times", slept)
	}
}
