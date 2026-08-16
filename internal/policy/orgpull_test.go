package policy

// orgpull_test.go is the acceptance suite for #5321 — the pull, the last-good
// cache, and the offline refuse-to-widen fallback.
//
// Every test drives the fetch through an IN-PROCESS double. Nothing here opens a
// socket, and nothing reads the wall clock: the clock is a field, so "twelve
// hours later" is an assignment rather than a sleep. That matters because the
// property under test is entirely about the passage of time.
//
// The load-bearing assertion, restated in several shapes below, is that no path
// through this code can leave the box MORE permissive than the compiled-in
// floor. A grant that arrived over the wire survives exactly as long as the
// freshness bound allows and not one second longer — and every way of failing
// (unreachable, forged, corrupt, expired, revoked, clock-skewed) lands on a
// posture that widens nothing.

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// pullEnv builds a signed envelope whose validity window is wide enough that
// expiry never fires, so a test that floors can only have floored on staleness.
func pullEnv(t *testing.T, priv ed25519.PrivateKey, issuer string, version uint64) []byte {
	t.Helper()
	return envAt(t, priv, issuer, version, fixedNow.Unix()-3600, fixedNow.Unix()+30*86400)
}

type pullFix struct {
	puller *OrgPuller
	pub    ed25519.PublicKey
	priv   ed25519.PrivateKey
	cache  string
	enroll string
	// clock is what p.Now reads; assign to it to move time.
	clock *time.Time
	// served is what the transport double returns; swap it per phase.
	served []byte
	// fetchErr, when set, makes the double fail instead of answering.
	fetchErr error
	// fetches counts transport calls, so a test can prove a posture was reached
	// WITHOUT touching the network.
	fetches int
}

// newPullFix wires an enrolled box with a working transport double.
func newPullFix(t *testing.T, seed byte) *pullFix {
	t.Helper()
	dir := t.TempDir()
	pub, priv := testKey(t, seed)
	clock := fixedNow

	f := &pullFix{
		pub:    pub,
		priv:   priv,
		cache:  filepath.Join(dir, "cache", "org-policy-lastgood.json"),
		enroll: filepath.Join(dir, "enroll", "org-enrollment.json"),
		clock:  &clock,
	}
	if _, err := EnrollOrg(f.enroll, OrgEnrollRequest{
		OrgURL:   "https://policy.acme.example/fak",
		Issuer:   "acme-corp",
		RootKey:  pub,
		DeviceID: "node-a",
		Groups:   []string{"eng"},
		Now:      fixedNow,
	}); err != nil {
		t.Fatalf("enroll: %v", err)
	}
	f.served = pullEnv(t, priv, "acme-corp", 5)
	f.puller = &OrgPuller{
		URL:            "https://policy.acme.example/fak/policy.json",
		CachePath:      f.cache,
		EnrollmentPath: f.enroll,
		RunningVersion: "0.9.0",
		Now:            func() time.Time { return *f.clock },
		Fetch: func(context.Context) ([]byte, error) {
			f.fetches++
			if f.fetchErr != nil {
				return nil, f.fetchErr
			}
			return f.served, nil
		},
	}
	return f
}

func (f *pullFix) pull(t *testing.T) OrgPullResult {
	t.Helper()
	res := f.puller.Pull(context.Background())
	// The redundancy invariant, checked on EVERY pull in the suite rather than
	// in one place: a non-widening posture must never hand back a Runtime, so
	// a caller that ignores Widen still cannot apply anything.
	if !res.Widen && res.Runtime != nil {
		t.Fatalf("posture %q does not widen but still returned a Runtime — a caller ignoring Widen would apply it", res.Posture)
	}
	if res.Widen && res.Runtime == nil {
		t.Fatalf("posture %q claims to widen but carries no Runtime", res.Posture)
	}
	return res
}

// grants reports whether the org overlay actually widened the allowlist. The
// fixture body carries Allow:["search_web"], so this IS the grant whose survival
// the staleness bound governs.
func grants(res OrgPullResult) bool {
	return res.Runtime != nil && res.Runtime.Adjudicator.Allow["search_web"]
}

func (f *pullFix) advance(d time.Duration) { *f.clock = f.clock.Add(d) }

// --- the three DoD postures -------------------------------------------------

func TestOrgPullFreshFetchAppliesAndCaches(t *testing.T) {
	f := newPullFix(t, 9)

	res := f.pull(t)
	if res.Posture != OrgPostureFresh {
		t.Fatalf("posture = %q, want %q (err: %v)", res.Posture, OrgPostureFresh, res.Err)
	}
	if !res.Widen || !grants(res) {
		t.Fatalf("a verified envelope did not apply its grant: widen=%v runtime=%+v", res.Widen, res.Runtime)
	}
	if res.Version != 5 || res.Issuer != "acme-corp" {
		t.Errorf("result identity = v%d/%q, want v5/acme-corp", res.Version, res.Issuer)
	}
	if res.Err != nil {
		t.Errorf("clean fetch reported err = %v", res.Err)
	}

	rec, ok, err := LoadOrgLastGood(f.cache)
	if err != nil || !ok {
		t.Fatalf("cache not written: ok=%v err=%v", ok, err)
	}
	if rec.Version != 5 || rec.Issuer != "acme-corp" {
		t.Errorf("cache identity = v%d/%q", rec.Version, rec.Issuer)
	}
	// ReceivedAt is OUR clock, not any field the envelope chose.
	if rec.ReceivedAt != fixedNow.Unix() {
		t.Errorf("received_at = %d, want our clock %d", rec.ReceivedAt, fixedNow.Unix())
	}
}

func TestOrgPullOfflineWithinBoundServesLastGood(t *testing.T) {
	f := newPullFix(t, 10)
	if res := f.pull(t); res.Posture != OrgPostureFresh {
		t.Fatalf("seed pull posture = %q", res.Posture)
	}

	// The source goes dark, well inside the bound.
	f.fetchErr = errors.New("dial tcp: connection refused")
	f.advance(MaxOrgPolicyStaleness - time.Minute)

	res := f.pull(t)
	if res.Posture != OrgPostureLastGood {
		t.Fatalf("posture = %q, want %q (err: %v)", res.Posture, OrgPostureLastGood, res.Err)
	}
	if !grants(res) {
		t.Error("last-good served but the grant did not survive")
	}
	if res.Reason != "unreachable" {
		t.Errorf("reason = %q, want unreachable", res.Reason)
	}
	if res.Err == nil {
		t.Error("the transport failure was swallowed — a serving-from-cache posture must still report why")
	}
	if res.Age <= 0 {
		t.Errorf("age = %v, want the cache's real age", res.Age)
	}
}

// TestOrgPullOfflinePastBoundFallsToFloor is THE issue's assertion: past the
// staleness bound, no grant survives.
func TestOrgPullOfflinePastBoundFallsToFloor(t *testing.T) {
	f := newPullFix(t, 11)
	seed := f.pull(t)
	if !grants(seed) {
		t.Fatalf("seed pull did not grant; nothing to lose later")
	}

	f.fetchErr = errors.New("dial tcp: connection refused")
	f.advance(MaxOrgPolicyStaleness + time.Second)

	res := f.pull(t)
	if res.Posture != OrgPostureFloor {
		t.Fatalf("posture = %q, want %q", res.Posture, OrgPostureFloor)
	}
	if res.Widen {
		t.Fatal("REFUSE-TO-WIDEN VIOLATED: a source past the staleness bound still widened")
	}
	if res.Runtime != nil {
		t.Fatal("floor posture handed back a Runtime — the grant is still reachable")
	}
	if grants(res) {
		t.Fatal("the central grant survived past the staleness bound")
	}
	if res.Reason != "stale" {
		t.Errorf("reason = %q, want stale", res.Reason)
	}
}

// TestOrgPullStalenessBoundaryIsExactlyOneSided pins the comparison so a later
// refactor cannot turn > into >= and quietly shift the window by a tick.
func TestOrgPullStalenessBoundaryIsExactlyOneSided(t *testing.T) {
	for _, tc := range []struct {
		name string
		at   time.Duration
		want string
	}{
		{"one second inside", MaxOrgPolicyStaleness - time.Second, OrgPostureLastGood},
		{"exactly at the bound", MaxOrgPolicyStaleness, OrgPostureLastGood},
		{"one second past", MaxOrgPolicyStaleness + time.Second, OrgPostureFloor},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newPullFix(t, 12)
			f.pull(t)
			f.fetchErr = errors.New("offline")
			f.advance(tc.at)
			if got := f.pull(t).Posture; got != tc.want {
				t.Fatalf("at %v posture = %q, want %q", tc.at, got, tc.want)
			}
		})
	}
}

// --- the staleness window cannot be widened ---------------------------------

// TestOrgPullMaxStalenessCannotBeExtended is the anti-self-extension property:
// asking for a LONGER window than the compiled ceiling must not produce one.
// If it did, a captured envelope could keep a box widened indefinitely.
func TestOrgPullMaxStalenessCannotBeExtended(t *testing.T) {
	f := newPullFix(t, 13)
	f.puller.MaxStaleness = 30 * 24 * time.Hour // far past the ceiling
	f.pull(t)

	f.fetchErr = errors.New("offline")
	f.advance(MaxOrgPolicyStaleness + time.Minute)

	res := f.pull(t)
	if res.Posture != OrgPostureFloor || res.Widen {
		t.Fatalf("a 30-day MaxStaleness extended the %v ceiling: posture=%q widen=%v",
			MaxOrgPolicyStaleness, res.Posture, res.Widen)
	}
}

// TestOrgPullTighterStalenessIsHonoured is the other half: tightening is always
// allowed, because making yourself stricter can never fail open.
func TestOrgPullTighterStalenessIsHonoured(t *testing.T) {
	f := newPullFix(t, 14)
	f.puller.MaxStaleness = time.Minute
	f.pull(t)

	f.fetchErr = errors.New("offline")
	f.advance(2 * time.Minute) // still hours inside the compiled ceiling

	res := f.pull(t)
	if res.Posture != OrgPostureFloor || res.Widen {
		t.Fatalf("a 1-minute MaxStaleness was not honoured: posture=%q widen=%v", res.Posture, res.Widen)
	}
}

func TestClampStalenessIsAOneWayValve(t *testing.T) {
	for _, tc := range []struct {
		in   time.Duration
		want time.Duration
	}{
		{0, MaxOrgPolicyStaleness},
		{-time.Hour, MaxOrgPolicyStaleness},
		{MaxOrgPolicyStaleness, MaxOrgPolicyStaleness},
		{MaxOrgPolicyStaleness + time.Nanosecond, MaxOrgPolicyStaleness},
		{365 * 24 * time.Hour, MaxOrgPolicyStaleness},
		{time.Minute, time.Minute},
	} {
		if got := clampStaleness(tc.in); got != tc.want {
			t.Errorf("clampStaleness(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// --- never-fetched, un-enrolled, damaged ------------------------------------

func TestOrgPullNeverFetchedIsFloorNotWiden(t *testing.T) {
	f := newPullFix(t, 15)
	f.fetchErr = errors.New("dial tcp: connection refused")

	res := f.pull(t)
	if res.Posture != OrgPostureFloor || res.Widen {
		t.Fatalf("first-ever pull against a dead endpoint: posture=%q widen=%v", res.Posture, res.Widen)
	}
	if res.Reason != "no_cache" {
		t.Errorf("reason = %q, want no_cache", res.Reason)
	}
	// The transport failure must still be reported, not lost behind "no cache".
	if res.Err == nil {
		t.Error("the unreachable-source error was dropped")
	}
}

// TestOrgPullUnenrolledIsInertAndTouchesNothing keeps the plane opt-in: a box
// that never enrolled does not fetch, does not write, and does not widen.
func TestOrgPullUnenrolledIsInertAndTouchesNothing(t *testing.T) {
	f := newPullFix(t, 16)
	if _, err := RevokeOrgEnrollment(f.enroll); err != nil {
		t.Fatalf("revoke: %v", err)
	}

	res := f.pull(t)
	if res.Posture != OrgPostureInert || res.Widen {
		t.Fatalf("un-enrolled posture = %q widen=%v, want inert/false", res.Posture, res.Widen)
	}
	if f.fetches != 0 {
		t.Errorf("un-enrolled box made %d fetch(es) — the plane must be inert, not quiet", f.fetches)
	}
	if _, err := os.Stat(f.cache); !os.IsNotExist(err) {
		t.Errorf("un-enrolled box wrote a cache (stat err = %v)", err)
	}
}

// TestOrgPullDamagedEnrollmentIsFloorNotInert separates the two "no anchor"
// states. Inert is a CHOICE (opted out); a store we cannot read is a FAILURE,
// and collapsing the second into the first is how a corrupted anchor would come
// to look like a deliberate opt-out.
func TestOrgPullDamagedEnrollmentIsFloorNotInert(t *testing.T) {
	f := newPullFix(t, 17)
	f.pull(t) // seed a good cache first, so "floor" cannot be an artifact of having none
	if err := os.WriteFile(f.enroll, []byte("{not json"), 0o600); err != nil {
		t.Fatalf("corrupt enrollment: %v", err)
	}

	res := f.pull(t)
	if res.Posture != OrgPostureFloor {
		t.Fatalf("damaged anchor posture = %q, want floor", res.Posture)
	}
	if res.Widen {
		t.Fatal("a damaged anchor still widened")
	}
	if res.Reason != "enrollment_unreadable" || res.Err == nil {
		t.Errorf("damaged anchor reported reason=%q err=%v", res.Reason, res.Err)
	}
	if f.fetches != 1 {
		t.Errorf("fetched %d time(s) after the anchor broke; want no fetch on the second pull", f.fetches)
	}
}

// --- the cache is not trusted, only remembered ------------------------------

// TestOrgPullRefusedEnvelopeDoesNotPoisonCache: an endpoint that starts serving
// something that will not verify must not be able to overwrite the good record
// it previously served, nor widen on the bad one.
func TestOrgPullRefusedEnvelopeDoesNotPoisonCache(t *testing.T) {
	f := newPullFix(t, 18)
	f.pull(t)
	before, err := os.ReadFile(f.cache)
	if err != nil {
		t.Fatalf("read cache: %v", err)
	}

	// Same shape, different signing key — a forgery.
	_, evil := testKey(t, 200)
	f.served = pullEnv(t, evil, "acme-corp", 6)
	f.advance(time.Hour)

	res := f.pull(t)
	if res.Posture != OrgPostureLastGood {
		t.Fatalf("posture = %q, want last_good (the good record is still fresh)", res.Posture)
	}
	if res.Reason != "refused" || res.Err == nil {
		t.Errorf("a forged envelope was not reported as refused: reason=%q err=%v", res.Reason, res.Err)
	}
	if res.Version != 5 {
		t.Errorf("served version %d; the forged v6 must not be what applied", res.Version)
	}
	after, err := os.ReadFile(f.cache)
	if err != nil {
		t.Fatalf("read cache after: %v", err)
	}
	if string(before) != string(after) {
		t.Fatal("a REFUSED envelope rewrote the last-good cache")
	}
}

// TestOrgPullRevokedAnchorInvalidatesTheCache is why the cache is re-verified
// rather than replayed: revoking or re-pinning must retract everything the old
// anchor authorized, without anyone having to remember to clear a file.
func TestOrgPullRevokedAnchorInvalidatesTheCache(t *testing.T) {
	f := newPullFix(t, 19)
	f.pull(t)

	// Re-pin onto a DIFFERENT org root key (revoke is the sanctioned route).
	if _, err := RevokeOrgEnrollment(f.enroll); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	otherPub, _ := testKey(t, 201)
	if _, err := EnrollOrg(f.enroll, OrgEnrollRequest{
		OrgURL:   "https://policy.other.example/fak",
		Issuer:   "other-corp",
		RootKey:  otherPub,
		DeviceID: "node-a",
		Now:      fixedNow,
	}); err != nil {
		t.Fatalf("re-enroll: %v", err)
	}

	f.fetchErr = errors.New("offline")
	f.advance(time.Hour) // still well within the staleness bound

	res := f.pull(t)
	if res.Posture != OrgPostureFloor || res.Widen {
		t.Fatalf("a cache signed by the PREVIOUS anchor still applied: posture=%q widen=%v",
			res.Posture, res.Widen)
	}
	if res.Reason != "cache_no_longer_verifies" {
		t.Errorf("reason = %q, want cache_no_longer_verifies", res.Reason)
	}
}

func TestOrgPullExpiredEnvelopeFloorsInsideTheStalenessBound(t *testing.T) {
	f := newPullFix(t, 20)
	// A short-lived envelope: it expires long before the staleness bound does.
	f.served = envAt(t, f.priv, "acme-corp", 5, fixedNow.Unix()-60, fixedNow.Unix()+600)
	if res := f.pull(t); res.Posture != OrgPostureFresh {
		t.Fatalf("seed posture = %q", res.Posture)
	}

	f.fetchErr = errors.New("offline")
	f.advance(30 * time.Minute) // past expiry, far inside MaxOrgPolicyStaleness

	res := f.pull(t)
	if res.Posture != OrgPostureFloor || res.Widen {
		t.Fatalf("an EXPIRED cached envelope still applied: posture=%q widen=%v", res.Posture, res.Widen)
	}
	if res.Reason != "expired" {
		t.Errorf("reason = %q, want expired", res.Reason)
	}
}

// TestOrgPullFutureDatedCacheFloors: a record stamped ahead of our clock has an
// unmeasurable age, and an unmeasurable age cannot satisfy a freshness bound.
func TestOrgPullFutureDatedCacheFloors(t *testing.T) {
	f := newPullFix(t, 21)
	f.pull(t)

	rec, ok, err := LoadOrgLastGood(f.cache)
	if err != nil || !ok {
		t.Fatalf("load cache: ok=%v err=%v", ok, err)
	}
	rec.ReceivedAt = fixedNow.Add(24 * time.Hour).Unix()
	rec.Sum = lastGoodSum(rec) // re-seal, so this tests the CLOCK rule, not the checksum
	b, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if err := os.WriteFile(f.cache, b, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	f.fetchErr = errors.New("offline")
	res := f.pull(t)
	if res.Posture != OrgPostureFloor || res.Widen {
		t.Fatalf("future-dated cache applied: posture=%q widen=%v", res.Posture, res.Widen)
	}
	if res.Reason != "cache_future_dated" {
		t.Errorf("reason = %q, want cache_future_dated", res.Reason)
	}
}

// TestOrgPullCacheFailsClosed: every way of damaging the record lands on floor.
// A cache that cannot be read must never be able to widen, and must never be
// mistaken for a cache that was never written.
func TestOrgPullCacheFailsClosed(t *testing.T) {
	damage := []struct {
		name    string
		breakIt func(t *testing.T, path string)
	}{
		{name: "garbage", breakIt: func(t *testing.T, path string) {
			if err := os.WriteFile(path, []byte("{nope"), 0o600); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "truncated", breakIt: func(t *testing.T, path string) {
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, b[:len(b)/2], 0o600); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "tampered version", breakIt: func(t *testing.T, path string) {
			rewriteJSON(t, path, func(m map[string]any) { m["version"] = 99 })
		}},
		{name: "tampered envelope", breakIt: func(t *testing.T, path string) {
			rewriteJSON(t, path, func(m map[string]any) { m["envelope"] = "AAAA" })
		}},
		{name: "stripped sum", breakIt: func(t *testing.T, path string) {
			rewriteJSON(t, path, func(m map[string]any) { m["sum"] = "" })
		}},
		{name: "wrong schema", breakIt: func(t *testing.T, path string) {
			rewriteJSON(t, path, func(m map[string]any) { m["schema"] = "fak-org-policy-lastgood/v2" })
		}},
		{name: "unknown field", breakIt: func(t *testing.T, path string) {
			rewriteJSON(t, path, func(m map[string]any) { m["widen_everything"] = true })
		}},
		{name: "oversize", breakIt: func(t *testing.T, path string) {
			pad := make([]byte, maxOrgLastGoodBytes+1)
			for i := range pad {
				pad[i] = ' '
			}
			if err := os.WriteFile(path, pad, 0o600); err != nil {
				t.Fatal(err)
			}
		}},
	}

	for _, damageCase := range damage {
		t.Run(damageCase.name, func(t *testing.T) {
			f := newPullFix(t, 22)
			f.pull(t)
			damageCase.breakIt(t, f.cache)

			f.fetchErr = errors.New("offline")
			res := f.pull(t)
			if res.Posture != OrgPostureFloor {
				t.Fatalf("damaged cache posture = %q, want floor", res.Posture)
			}
			if res.Widen || grants(res) {
				t.Fatal("a damaged cache still widened")
			}
			if res.Err == nil {
				t.Error("a damaged cache was not reported — it is indistinguishable from having none")
			}
			if _, _, err := LoadOrgLastGood(f.cache); err == nil {
				t.Error("LoadOrgLastGood accepted the damaged record")
			}
		})
	}
}

func rewriteJSON(t *testing.T, path string, edit func(map[string]any)) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	edit(m)
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
}

// --- ordinary states --------------------------------------------------------

func TestLoadOrgLastGoodAbsentIsNotAnError(t *testing.T) {
	rec, ok, err := LoadOrgLastGood(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil {
		t.Fatalf("a missing cache is an ordinary state, got err = %v", err)
	}
	if ok {
		t.Error("a missing cache reported ok=true")
	}
	if rec.Version != 0 || rec.Envelope != "" {
		t.Errorf("missing cache produced a non-zero record: %+v", rec)
	}
	if _, _, err := LoadOrgLastGood("  "); err == nil {
		t.Error("an empty path was accepted")
	}
}

// TestOrgPullRepollSameVersionRefreshesFreshness is the anti-rollback corner:
// re-fetching the SAME version changes no knob, but it IS new evidence that the
// org is still reachable, so it must re-stamp freshness. Without this, a fleet
// on a stable policy would age out and floor itself while perfectly online.
func TestOrgPullRepollSameVersionRefreshesFreshness(t *testing.T) {
	f := newPullFix(t, 23)
	f.puller.Ledger = mustOpenLedger(t, filepath.Join(t.TempDir(), "ledger.json"))
	f.pull(t)

	// Re-poll the identical envelope most of the way through the window.
	f.advance(MaxOrgPolicyStaleness - time.Minute)
	if res := f.pull(t); res.Posture != OrgPostureFresh {
		t.Fatalf("re-poll of the same version posture = %q, want fresh (err %v)", res.Posture, res.Err)
	}

	// Now go dark for a stretch that WOULD have been past the bound measured
	// from the first pull, but is inside it measured from the re-poll.
	f.fetchErr = errors.New("offline")
	f.advance(2 * time.Minute)

	res := f.pull(t)
	if res.Posture != OrgPostureLastGood {
		t.Fatalf("posture = %q, want last_good — the re-poll did not refresh received_at", res.Posture)
	}
	if !grants(res) {
		t.Error("grant lost despite a refreshed last-good")
	}
}

func mustOpenLedger(t *testing.T, path string) *OrgLedger {
	t.Helper()
	l, err := OpenOrgLedger(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	return l
}

func TestOrgPolicyCachePathIsOverridable(t *testing.T) {
	want := filepath.Join(t.TempDir(), "elsewhere.json")
	t.Setenv(OrgPolicyCachePathEnv, want)
	if got := OrgPolicyCachePath(); got != want {
		t.Fatalf("OrgPolicyCachePath() = %q, want the env override %q", got, want)
	}
	t.Setenv(OrgPolicyCachePathEnv, "")
	if got := OrgPolicyCachePath(); got == "" {
		t.Error("OrgPolicyCachePath() resolved to empty — a caller could read that as 'off'")
	}
}

// TestOrgPullEmptyResponseIsNotASuccess: a 200 with a zero-length body is a
// broken endpoint, not an empty policy. Treating it as the latter would silently
// clear every grant AND overwrite the cache that could have carried them.
func TestOrgPullEmptyResponseIsNotASuccess(t *testing.T) {
	f := newPullFix(t, 24)
	f.pull(t)
	before, err := os.ReadFile(f.cache)
	if err != nil {
		t.Fatal(err)
	}

	f.served = []byte{}
	f.advance(time.Hour)
	res := f.pull(t)
	if res.Posture != OrgPostureLastGood {
		t.Fatalf("posture = %q, want last_good", res.Posture)
	}
	if res.Reason != "empty_response" || res.Err == nil {
		t.Errorf("empty body reported reason=%q err=%v", res.Reason, res.Err)
	}
	after, err := os.ReadFile(f.cache)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("an empty response overwrote the last-good cache")
	}
}
