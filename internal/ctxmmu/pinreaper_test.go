package ctxmmu_test

// pinreaper_test.go — proofs for the TTL leaked-pin reaper (issue #3385).
//
// The count-cap bound (leak_test.go) frees a leaked pin only when NEWER
// quarantines push it out; these tests pin the TIME bound: an idle pin past the
// TTL is force-unpinned (counted in ForcedUnpins, distinct from Evicted), a
// repin resets the keepalive so an actively-used pin is never reaped, and the
// sweep is idempotent. Everything runs on INJECTED millis — no wall clock in
// any reap decision, no sleeps.

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

const (
	reapTTL = int64(60_000)    // the TTL under test, millis
	reapT0  = int64(1_000_000) // synthetic epoch the injected clock starts from
)

// quarantinePin admits one distinct poison result through m (the same held-pin
// setup the leak tests use) and re-stamps its keepalive to the INJECTED stamp via
// TouchPin, so every subsequent reap decision is pure injected-millis arithmetic.
func quarantinePin(t *testing.T, m *ctxmmu.MMU, i int, stamp int64) string {
	t.Helper()
	c := call("read_file")
	r := result(c, poison(i))
	if v := m.Admit(context.Background(), c, r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("admit %d: want VerdictQuarantine, got %v", i, v.Kind)
	}
	id := r.Meta["quarantine_id"]
	if id == "" {
		t.Fatalf("admit %d: no quarantine_id stamped", i)
	}
	if !m.TouchPin(id, stamp) {
		t.Fatalf("TouchPin(%q) reported unheld immediately after quarantine", id)
	}
	return id
}

// TestReapExpiredPinIsForceUnpinned: a held pin idle past the TTL is reaped —
// dropped from the ledger, counted in ForcedUnpins (NOT in the count-cap
// Evicted), and degraded fail-closed exactly like an eviction: no clear, touch,
// or page-in can resurrect it.
func TestReapExpiredPinIsForceUnpinned(t *testing.T) {
	ctx := context.Background()
	m := ctxmmu.NewWithLimit(64)
	id := quarantinePin(t, m, 100, reapT0)

	if got := m.ReapExpiredPins(reapT0+reapTTL+1, reapTTL); got != 1 {
		t.Fatalf("ReapExpiredPins = %d, want 1 (pin idle past TTL)", got)
	}
	if got := m.ForcedUnpins(); got != 1 {
		t.Fatalf("ForcedUnpins = %d, want 1", got)
	}
	if got := m.Evicted(); got != 0 {
		t.Fatalf("Evicted = %d, want 0 — the TTL reap must not count as a cap eviction", got)
	}
	if hl := m.HeldLen(); hl != 0 {
		t.Fatalf("HeldLen = %d after reap, want 0", hl)
	}
	// Fail-closed degradation, mirroring the evicted-id contract: a re-Clear is a
	// no-op, PageIn refuses, and a touch reports unheld (never resurrects).
	m.Clear(id)
	if _, err := m.PageIn(ctx, id); err == nil {
		t.Fatalf("reaped id %q must refuse PageIn, but it returned bytes", id)
	}
	if m.TouchPin(id, reapT0+reapTTL+2) {
		t.Fatalf("TouchPin on reaped id %q must report unheld", id)
	}
}

// TestRepinResetsKeepalive: a repin (TouchPin) resets the TTL countdown, so at a
// now that is past the ORIGINAL stamp's TTL but within the REPIN's TTL the pin
// survives — and the reset countdown then expires on its own schedule.
func TestRepinResetsKeepalive(t *testing.T) {
	m := ctxmmu.NewWithLimit(64)
	id := quarantinePin(t, m, 200, reapT0)

	t1 := reapT0 + reapTTL/2 // repin well within the original TTL
	if !m.TouchPin(id, t1) {
		t.Fatalf("repin TouchPin(%q) reported unheld", id)
	}
	// now-reapT0 > TTL (would have been reaped) but now-t1 <= TTL (repin holds it).
	now := reapT0 + reapTTL + reapTTL/4
	if got := m.ReapExpiredPins(now, reapTTL); got != 0 {
		t.Fatalf("ReapExpiredPins = %d, want 0 — repin at %d must reset the keepalive", got, t1)
	}
	if got := m.ForcedUnpins(); got != 0 {
		t.Fatalf("ForcedUnpins = %d, want 0 after a keepalive save", got)
	}
	if hl := m.HeldLen(); hl != 1 {
		t.Fatalf("HeldLen = %d, want 1 (repinned entry survives)", hl)
	}
	// The countdown runs from the REPIN: once t1's TTL lapses the pin is reaped.
	if got := m.ReapExpiredPins(t1+reapTTL+1, reapTTL); got != 1 {
		t.Fatalf("ReapExpiredPins = %d, want 1 once the repin's own TTL lapses", got)
	}
	if got := m.ForcedUnpins(); got != 1 {
		t.Fatalf("ForcedUnpins = %d, want 1 after the repin's TTL lapsed", got)
	}
}

// TestPinWithinTTLSurvivesReap: a pin whose idle interval has not EXCEEDED the
// TTL survives the sweep (the boundary now-stamp == ttl is not expired) and stays
// fully usable — a witness Clear still pages the original bytes back in.
func TestPinWithinTTLSurvivesReap(t *testing.T) {
	ctx := context.Background()
	m := ctxmmu.NewWithLimit(64)
	id := quarantinePin(t, m, 300, reapT0)

	if got := m.ReapExpiredPins(reapT0+reapTTL, reapTTL); got != 0 {
		t.Fatalf("ReapExpiredPins = %d, want 0 — idle exactly TTL is not expired (strict >)", got)
	}
	if got := m.ForcedUnpins(); got != 0 {
		t.Fatalf("ForcedUnpins = %d, want 0", got)
	}
	if hl := m.HeldLen(); hl != 1 {
		t.Fatalf("HeldLen = %d, want 1", hl)
	}
	m.Clear(id)
	got, err := m.PageIn(ctx, id)
	if err != nil {
		t.Fatalf("surviving pin should page in after a witness clear: %v", err)
	}
	if !bytes.Equal(got, poison(300)) {
		t.Fatalf("page-in returned wrong bytes")
	}
}

// TestReapIsIdempotentAtSameNow: a second sweep with the same (now, ttl) reaps
// nothing — the expired entries are already gone and the counter does not move.
func TestReapIsIdempotentAtSameNow(t *testing.T) {
	m := ctxmmu.NewWithLimit(64)
	quarantinePin(t, m, 400, reapT0)
	quarantinePin(t, m, 401, reapT0)

	now := reapT0 + reapTTL + 1
	if got := m.ReapExpiredPins(now, reapTTL); got != 2 {
		t.Fatalf("first ReapExpiredPins = %d, want 2", got)
	}
	if got := m.ReapExpiredPins(now, reapTTL); got != 0 {
		t.Fatalf("second ReapExpiredPins at same now = %d, want 0 (idempotent)", got)
	}
	if got := m.ForcedUnpins(); got != 2 {
		t.Fatalf("ForcedUnpins = %d, want 2 (second sweep must not double-count)", got)
	}
	if hl := m.HeldLen(); hl != 0 {
		t.Fatalf("HeldLen = %d, want 0", hl)
	}
}

// TestHoldStampsKeepaliveOnQuarantine proves the HOLD path stamps the keepalive
// itself (no explicit TouchPin). The hold stamp is wall-clock, so the test
// samples the clock AFTER the hold as an upper bound: now = after+TTL+1 provably
// exceeds any stamp taken at-or-before `after` by more than the TTL — a
// deterministic inequality, no sleeps. Were the stamp-on-hold hook missing, the
// reaper would ADOPT the unstamped pin (earning it a fresh TTL) and reap 0, so a
// reap of 1 is a genuine witness of the hook.
func TestHoldStampsKeepaliveOnQuarantine(t *testing.T) {
	m := ctxmmu.NewWithLimit(64)
	c := call("read_file")
	r := result(c, poison(500))
	if v := m.Admit(context.Background(), c, r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("want VerdictQuarantine, got %v", v.Kind)
	}
	after := time.Now().UnixMilli()
	if got := m.ReapExpiredPins(after+reapTTL+1, reapTTL); got != 1 {
		t.Fatalf("ReapExpiredPins = %d, want 1 — the hold path must stamp the keepalive", got)
	}
}

// TestReapReleasesCASPin is the abi.UnpinResolved witness, mirroring
// caspin_test: while held, the quarantine's CAS pin survives byte-bound churn;
// after the TTL reap the SAME churn evicts the bytes — proving the reap actually
// released the pin rather than just forgetting the ledger entry.
func TestReapReleasesCASPin(t *testing.T) {
	ctx := context.Background()
	// A >256B secret-shaped body unique to this test: quarantines AND pages out
	// to the CAS (not inline), so the handle carries a pinnable digest.
	secret := []byte("sk-reaperpin" + strings.Repeat("q", 400))
	// Budget: everything currently resident (earlier tests leave PINNED bytes
	// behind in the shared store) plus the secret plus a small slack — so the
	// secret's own Put cannot self-evict before quarantineResult pins it, while
	// churn can still overrun the slack and cycle out every unpinned blob.
	old := blob.Default.MaxBytes()
	budget := blob.Default.Bytes() + int64(len(secret)) + 4096
	blob.Default.SetMaxBytes(budget)
	defer blob.Default.SetMaxBytes(old)

	m := ctxmmu.NewWithLimit(64)
	c := call("read_secret")
	r := result(c, secret)
	if v := m.Admit(ctx, c, r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("want Quarantine, got %v", v.Kind)
	}
	id := r.Meta["quarantine_id"]
	handle := m.Held()[id]
	if handle.Digest == "" {
		t.Fatalf("quarantine handle carries no CAS digest — body did not page out")
	}

	// churn inserts MORE fresh distinct bytes than the whole budget, so the byte
	// bound provably cycles every unpinned blob out of the store — older
	// unpinned residents (whatever earlier tests left) evict before any fresh
	// churn blob does, so nothing unpinned that predates the churn survives it.
	churn := func(phase string) {
		t.Helper()
		for i, n := 0, int(budget/1024)+8; i < n; i++ {
			body := bytes.Repeat([]byte{0xCC}, 1024) // >256B: lands in the CAS
			copy(body, fmt.Sprintf("churn-%s-%d", phase, i))
			if _, err := blob.Default.Put(ctx, body); err != nil {
				t.Fatalf("churn put: %v", err)
			}
		}
	}

	// While held, the pin protects the bytes from the byte bound.
	churn("held")
	if _, err := blob.Default.Resolve(ctx, handle); err != nil {
		t.Fatalf("pinned quarantine bytes evicted while still held: %v", err)
	}

	if !m.TouchPin(id, reapT0) {
		t.Fatalf("TouchPin(%q) reported unheld", id)
	}
	if got := m.ReapExpiredPins(reapT0+reapTTL+1, reapTTL); got != 1 {
		t.Fatalf("ReapExpiredPins = %d, want 1", got)
	}

	// After the reap the pin is RELEASED: the same churn now evicts the bytes.
	churn("reaped")
	if _, err := blob.Default.Resolve(ctx, handle); err == nil {
		t.Fatalf("reaped pin still protects its CAS bytes — UnpinResolved was not called")
	}
}

// TestReapComposesWithCountCapEviction: the count bound (Evicted) and the time
// bound (ForcedUnpins) stay distinct and compose — cap overflow evicts the
// oldest, the reap then frees the idle survivors, each drop counted exactly
// once; and the stale order slots the reap leaves behind are skipped (not
// re-counted) by later cap evictions.
func TestReapComposesWithCountCapEviction(t *testing.T) {
	const cap = 4
	m := ctxmmu.NewWithLimit(cap)
	for i := 0; i < cap*2; i++ {
		quarantinePin(t, m, 600+i, reapT0)
	}
	if got := m.Evicted(); got != int64(cap) {
		t.Fatalf("Evicted = %d, want %d after overflowing the cap", got, cap)
	}
	if got := m.ForcedUnpins(); got != 0 {
		t.Fatalf("ForcedUnpins = %d, want 0 before any reap", got)
	}
	if got := m.ReapExpiredPins(reapT0+reapTTL+1, reapTTL); got != cap {
		t.Fatalf("ReapExpiredPins = %d, want %d (every survivor idle past TTL)", got, cap)
	}
	if got, want := m.ForcedUnpins(), int64(cap); got != want {
		t.Fatalf("ForcedUnpins = %d, want %d", got, want)
	}
	if got := m.Evicted(); got != int64(cap) {
		t.Fatalf("Evicted = %d, want %d — the reap must not move the cap counter", got, cap)
	}
	if hl := m.HeldLen(); hl != 0 {
		t.Fatalf("HeldLen = %d, want 0 after the reap", hl)
	}
	// A second wave over the emptied ledger: cap eviction must skip the reaped
	// ids' stale order slots without counting them, and bound held as before.
	for i := 0; i < cap*2; i++ {
		quarantinePin(t, m, 700+i, reapT0)
	}
	if hl := m.HeldLen(); hl != cap {
		t.Fatalf("HeldLen = %d, want %d after the second wave", hl, cap)
	}
	if got, want := m.Evicted(), int64(2*cap); got != want {
		t.Fatalf("Evicted = %d, want %d — stale reaped slots must not be counted as evictions", got, want)
	}
	if got, want := m.ForcedUnpins(), int64(cap); got != want {
		t.Fatalf("ForcedUnpins = %d, want %d (unchanged by cap evictions)", got, want)
	}
}
