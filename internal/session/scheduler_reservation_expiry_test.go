package session

// Test-Context: ses_tester_13384_a1   Impl-Context: ses_worker_13384_b1   Separation-Verdict: SEPARATED
//
// Adversarial, spec-only witness for pub#13384: a known-coming reservation for an
// unchanged (trace, prefix) intent must NOT be resurrected once it has expired.
//
// Oracle: _scratch/13384/SPEC.md R-1..R-7 / E1..E16 and _scratch/13384/INTERFACES.md.
// This file was authored WITHOUT reading the implementation diff or scheduler.go body;
// it asserts the SPEC's required behavior, not whatever the implementation happens to do.
//
// Determinism: `now` is always an explicit parameter. No sleeps, no wall-clock reads.

import (
	"math"
	"testing"
	"time"
)

// reservationTestBase is a fixed, realistic epoch (well clear of the int64 ns range
// tails) so every `now` derived from it is deterministic.
func reservationTestBase() time.Time { return time.Unix(1000, 0).UTC() }

// reservationTestIntent is the canonical live "known-coming" hint used by the witness.
func reservationTestIntent(prefix string) TurnIntent {
	return TurnIntent{ArrivingInMillis: 10, Prefix: prefix}
}

// reservationExpiry is the expiry stamp R-1 requires to stay fixed: the first observation
// at `now` establishes arrival `now+10ms` and expiry `arrival+DefaultReservationGrace`.
func reservationExpiry(now time.Time) time.Time {
	return now.Add(10 * time.Millisecond).Add(DefaultReservationGrace)
}

// reservationMaxArrivingMillis is the STATIC multiply ceiling for an ArrivingInMillis
// hint: the largest ms whose `ms * time.Millisecond` stays inside int64. It is defined
// HERE, in the test, rather than borrowed from the implementation, so the oracle below
// is derived from the int64 arithmetic alone and cannot move with the code under test.
const reservationMaxArrivingMillis = math.MaxInt64 / int64(time.Millisecond)

// reservationCanRepresent reports whether an ArrivingInMillis hint, observed at the
// absolute nanosecond clock `nowNs`, can be represented WITHOUT any int64 wrap. It is
// the R-6 oracle derived straight from the stated invariant `Reserved <= Arrives <=
// Expires` plus the representable-nanosecond range, so it is independent of whatever
// predicate the implementation happens to use:
//
//   - the hint must be strictly positive (a zero/negative hint is "no opinion", R-6);
//   - the multiply `ms * 1e6` must not overflow (ms <= reservationMaxArrivingMillis);
//   - `nowNs + delay` must not overflow, and neither must the grace-expired stamp
//     `nowNs + delay + graceNs`.
//
// A hint is ACCEPTED by the spec exactly when it is representable; anything else must
// be rejected. This is what makes the near-boundary assertion non-vacuous: a hint such
// as reservationMaxArrivingMillis-1 can be representable at one `now` and not another.
func reservationCanRepresent(ms, nowNs int64) bool {
	if ms <= 0 || ms > reservationMaxArrivingMillis {
		return false
	}
	delayNs := ms * int64(time.Millisecond) // <= MaxInt64 by the bound above
	graceNs := int64(DefaultReservationGrace)
	return nowNs+delayNs >= nowNs && nowNs+delayNs+graceNs >= nowNs
}

const reservationTestPrefixP = "sha256:P"
const reservationTestPrefixQ = "sha256:Q"

// findReservation returns the reservation matching (trace, prefix) together with whether
// it was present. It probes the caller-visible slice, not internal state, so it exercises
// the same surface an embedder sees.
func findReservation(rs []SlotReservation, trace, prefix string) (SlotReservation, bool) {
	for _, r := range rs {
		if r.TraceID == trace && r.Prefix == prefix {
			return r, true
		}
	}
	return SlotReservation{}, false
}

// TestReservationDoesNotResurrect is the named witness. It must FAIL on the pre-fix
// scheduler (which rebases a fresh reservation onto `now` after pruning the old one)
// and PASS once the first deadline wins and expiry is terminal.
func TestReservationDoesNotResurrect(t *testing.T) {
	base := reservationTestBase()

	t.Run("E1_repeated_in_window_scans_keep_original_stamps", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}

		r0, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP)
		if !ok {
			t.Fatalf("first scan minted nothing: %+v", s.Reservations(base))
		}
		if len(s.reservations) != 1 {
			t.Fatalf("len(s.reservations) = %d, want 1 (one row per (trace,prefix))", len(s.reservations))
		}

		// Several later scans, all strictly inside [base, base+10ms+grace), must each
		// return the IDENTICAL original stamps (R-1), never a rebased copy.
		for _, step := range []time.Duration{time.Millisecond, 2 * time.Millisecond, 5 * time.Millisecond} {
			now := base.Add(step)
			got, ok := findReservation(s.ReserveKnownComing(now), "t", reservationTestPrefixP)
			if !ok {
				t.Fatalf("scan@+%v lost the reservation: %+v", step, s.Reservations(now))
			}
			if got != r0 {
				t.Fatalf("scan@+%v rebased the reservation:\n got  %+v\n want %+v", step, got, r0)
			}
			if got.ReservedAtUnixNano != r0.ReservedAtUnixNano ||
				got.ArrivesAtUnixNano != r0.ArrivesAtUnixNano ||
				got.ExpiresAtUnixNano != r0.ExpiresAtUnixNano {
				t.Fatalf("scan@+%v moved a stamp: got stamps (%d,%d,%d), want (%d,%d,%d)",
					step,
					got.ReservedAtUnixNano, got.ArrivesAtUnixNano, got.ExpiresAtUnixNano,
					r0.ReservedAtUnixNano, r0.ArrivesAtUnixNano, r0.ExpiresAtUnixNano)
			}
			if len(s.reservations) != 1 {
				t.Fatalf("scan@+%v duplicated bookkeeping: len(s.reservations)=%d, want 1", step, len(s.reservations))
			}
		}
	})

	t.Run("E2_E16_scan_and_promote_exactly_at_expiry", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}
		if _, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP); !ok {
			t.Fatal("first scan minted nothing")
		}

		expires := reservationExpiry(base)

		// R-2: the prune predicate is `now >= expiry`, so the exact boundary is terminal.
		if got, ok := findReservation(s.ReserveKnownComing(expires), "t", reservationTestPrefixP); ok {
			t.Fatalf("scan at now==expiry resurrected the reservation: %+v (want none)", got)
		}
		if got := s.Reservations(expires); len(got) != 0 {
			t.Fatalf("Reservations at now==expiry = %+v, want none", got)
		}

		// E16: promotion at the boundary must fail (the row is pruned first).
		if promo, ok := s.PromoteReservation("t", reservationTestPrefixP, expires); ok {
			t.Fatalf("PromoteReservation at now==expiry = (%+v, true), want (_, false)", promo)
		}
		// And no half-minted row is left behind by the failed promotion.
		if len(s.reservations) != 0 {
			t.Fatalf("boundary scan/promote left bookkeeping: len(s.reservations)=%d, want 0", len(s.reservations))
		}
	})

	t.Run("E3_far_past_expiry_never_resurrects", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}
		if _, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP); !ok {
			t.Fatal("first scan minted nothing")
		}

		far := base.Add(10*time.Millisecond + DefaultReservationGrace + time.Hour)
		if got, ok := findReservation(s.ReserveKnownComing(far), "t", reservationTestPrefixP); ok {
			t.Fatalf("scan 1h past expiry resurrected a NEW reservation: %+v (want none)", got)
		}
		if got := s.Reservations(far); len(got) != 0 {
			t.Fatalf("Reservations far past expiry = %+v, want none", got)
		}
		if len(s.reservations) != 0 {
			t.Fatalf("far scan left bookkeeping: len(s.reservations)=%d, want 0", len(s.reservations))
		}
	})

	t.Run("E4_unrelated_rev_bump_is_not_renewal", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}
		r0, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP)
		if !ok {
			t.Fatal("first scan minted nothing")
		}
		revBefore := tbl.Get("t").Rev

		// A byte-identical intent re-published via a DIFFERENT mutator (SetPriority) bumps
		// State.Rev but must not be read as an intent publication stamp (R-3).
		if _, ok := tbl.SetPriority("t", 71); !ok {
			t.Fatal("SetPriority rejected a live session")
		}
		if tbl.Get("t").Rev <= revBefore {
			t.Fatalf("SetPriority did not bump Rev (%d -> %d): test premise broken", revBefore, tbl.Get("t").Rev)
		}

		now := base.Add(5 * time.Millisecond) // strictly inside the original window
		got, ok := findReservation(s.ReserveKnownComing(now), "t", reservationTestPrefixP)
		if !ok {
			t.Fatalf("in-window scan after unrelated Rev bump lost the row: %+v", s.Reservations(now))
		}
		if got != r0 {
			t.Fatalf("unrelated Rev bump renewed/re-stamped the reservation:\n got  %+v\n want %+v", got, r0)
		}
		if got.ExpiresAtUnixNano != reservationExpiry(base).UnixNano() {
			t.Fatalf("expiry moved to %d, want original %d", got.ExpiresAtUnixNano, reservationExpiry(base).UnixNano())
		}

		// R-3 tail: after the ORIGINAL expiry, still gone — a Rev bump must not have
		// created a fresh generation that outlives the first deadline.
		if got, ok := findReservation(s.ReserveKnownComing(reservationExpiry(base)), "t", reservationTestPrefixP); ok {
			t.Fatalf("Rev bump + boundary scan resurrected: %+v (want none)", got)
		}
	})

	t.Run("E5_prefix_change_mints_Q_and_drops_P", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}
		if _, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP); !ok {
			t.Fatal("first scan minted nothing")
		}

		// R-4a: a changed prefix is a demonstrably changed identity -> a fresh row for Q
		// is allowed, and the stale P row must be dropped.
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixQ)); !ok {
			t.Fatal("SetTurnIntent (prefix P->Q) rejected a live session")
		}
		got := s.ReserveKnownComing(base.Add(5 * time.Millisecond))
		if _, ok := findReservation(got, "t", reservationTestPrefixQ); !ok {
			t.Fatalf("prefix change did not mint (t,Q): %+v", got)
		}
		if _, ok := findReservation(got, "t", reservationTestPrefixP); ok {
			t.Fatalf("prefix change left stale (t,P) live: %+v", got)
		}
		if len(s.reservations) != 1 {
			t.Fatalf("len(s.reservations)=%d, want 1 (only the current prefix)", len(s.reservations))
		}
		if _, ok := s.PromoteReservation("t", reservationTestPrefixP, base.Add(6*time.Millisecond)); ok {
			t.Fatal("stale prefix P promoted after intent change")
		}
	})

	t.Run("E6_clear_scan_republish_renews", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}
		r0, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP)
		if !ok {
			t.Fatal("first scan minted nothing")
		}

		// R-4b: clear, then a scan WITNESSES the hint gone (and releases the row). The
		// intervening scan is mandatory; without it the implementation MAY decline.
		if _, ok := tbl.SetTurnIntent("t", TurnIntent{}); !ok {
			t.Fatal("clear (zero intent) rejected a live session")
		}
		clearedAt := base.Add(2 * time.Millisecond)
		if got := s.ReserveKnownComing(clearedAt); len(got) != 0 {
			t.Fatalf("scan after clear = %+v, want none (hint gone)", got)
		}
		if len(s.reservations) != 0 {
			t.Fatalf("clear did not release bookkeeping: len(s.reservations)=%d, want 0", len(s.reservations))
		}

		// Re-publish a fresh hint AFTER the witnessed clear -> a NEW generation is allowed.
		republished := 25
		republishAt := clearedAt.Add(2 * time.Millisecond)
		if _, ok := tbl.SetTurnIntent("t", TurnIntent{ArrivingInMillis: int64(republished), Prefix: reservationTestPrefixP}); !ok {
			t.Fatal("re-publish rejected a live session")
		}
		r1, ok := findReservation(s.ReserveKnownComing(republishAt), "t", reservationTestPrefixP)
		if !ok {
			t.Fatalf("re-publish after witnessed clear minted nothing: %+v", s.Reservations(republishAt))
		}
		if r1.ExpiresAtUnixNano == r0.ExpiresAtUnixNano {
			t.Fatalf("republished row reused the pre-clear expiry %d; want a fresh generation", r0.ExpiresAtUnixNano)
		}
		wantExpiry := republishAt.Add(time.Duration(republished) * time.Millisecond).Add(DefaultReservationGrace)
		if r1.ExpiresAtUnixNano != wantExpiry.UnixNano() {
			t.Fatalf("fresh expiry = %d, want %d", r1.ExpiresAtUnixNano, wantExpiry.UnixNano())
		}
	})

	t.Run("E7_will_discard_drops_and_mints_none", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}
		if _, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP); !ok {
			t.Fatal("first scan minted nothing")
		}

		// R-5: WillDiscard is a terminal hint — release the row, mint nothing.
		if _, ok := tbl.SetTurnIntent("t", TurnIntent{ArrivingInMillis: 10, Prefix: reservationTestPrefixP, WillDiscard: true}); !ok {
			t.Fatal("WillDiscard update rejected a live session")
		}
		now := base.Add(5 * time.Millisecond)
		if got, ok := findReservation(s.ReserveKnownComing(now), "t", reservationTestPrefixP); ok {
			t.Fatalf("WillDiscard hint minted a reservation: %+v (want none)", got)
		}
		if len(s.reservations) != 0 {
			t.Fatalf("WillDiscard left bookkeeping: len(s.reservations)=%d, want 0", len(s.reservations))
		}
	})

	t.Run("E8_trace_removed_releases_row_on_next_scan", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected a live session")
		}
		if _, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP); !ok {
			t.Fatal("first scan minted nothing")
		}

		// R-5: the trace leaves the snapshot (Reset = LRU/eviction analogue) -> its rows
		// are released on the next scan, which mints nothing.
		tbl.Reset("t")
		now := base.Add(5 * time.Millisecond)
		if got := s.ReserveKnownComing(now); len(got) != 0 {
			t.Fatalf("scan after trace removal = %+v, want none", got)
		}
		if len(s.reservations) != 0 {
			t.Fatalf("trace removal left bookkeeping: len(s.reservations)=%d, want 0", len(s.reservations))
		}
	})

	t.Run("E9_E10_pathological_arriving_millis_reject_safely", func(t *testing.T) {
		base := reservationTestBase()

		// Exactly the values the R-6 / E9+E10 contract names, PLUS the near-boundary
		// triple around the test-local reservationMaxArrivingMillis. The boundary values
		// are the point of this subtest: an implementation that guards only the MULTIPLY
		// rejects the wild values below by accident, yet still mints an inverted row for
		// `reservationMaxArrivingMillis-1` once `now` is near the int64 nanosecond tail
		// (the real defect class this subtest must be able to catch).
		hints := []int64{0, -1, math.MinInt64, math.MaxInt64}
		hints = append(hints, reservationMaxArrivingMillis, reservationMaxArrivingMillis-1, reservationMaxArrivingMillis+1)
		// A modest positive hint that stays representable even at the near-tail point, so
		// the ACCEPT branch (and its ordering invariant) is exercised there as well, not
		// only at base.
		hints = append(hints, 10)

		// `base` is a realistic `now` for the first arm. tailNs is a synthetic near-tail
		// `now` (well clear of the representable end so its own UnixNano never wraps) where
		// the ADD hazard is actually reachable and reservationMaxArrivingMillis-1 is a high-but-valid
		// millisecond hint whose arrival wraps past MaxInt64. Both are fixed constants, so
		// the subtest stays deterministic (no sleeps, no wall-clock reads).
		// Half the representable nanosecond range: near enough to the int64 tail that the
		// ADD hazard is reachable, while still leaving a small positive window so a valid
		// short hint exists there too. (2*halfRange <= MaxInt64 by construction.)
		const halfRangeNs = int64(math.MaxInt64 / 2)
		tailNs := int64(math.MaxInt64) - halfRangeNs

		type e9e10Point struct {
			name string
			now  time.Time
		}
		// Exercise every input at BOTH points. At base every non-positive / oversized value
		// is refused up front; at the near-tail point the `reservationMaxArrivingMillis-1` class is
		// accepted only if the implementation is genuinely order-preserving.
		points := []e9e10Point{
			{"base", base},
			{"near-tail", time.Unix(0, tailNs).UTC()},
		}

		for _, pt := range points {
			nowNs := pt.now.UnixNano()
			for _, ms := range hints {
				tbl, s := attachedScheduler(t, StrictPriority)
				if _, ok := tbl.SetTurnIntent("t", TurnIntent{ArrivingInMillis: ms, Prefix: reservationTestPrefixP}); !ok {
					t.Fatalf("[%s] ArrivingInMillis=%d: SetTurnIntent rejected a live session", pt.name, ms)
				}

				// Oracle derived from the R-6 spec and the stated invariant (NOT from any
				// implementation diff): representable iff the multiply itself does not wrap
				// AND both the arrival and the grace-expired stamps stay on the int64
				// nanosecond timeline. Any accepted row must order Reserved <= Arrives <=
				// Expires; any value the math cannot represent must be rejected outright.
				canMint := reservationCanRepresent(ms, nowNs)
				got := s.ReserveKnownComing(pt.now)
				listed := s.Reservations(pt.now)

				if canMint {
					r, found := findReservation(got, "t", reservationTestPrefixP)
					if !found {
						t.Fatalf("[%s] ArrivingInMillis=%d is representable at now=%d but no row was minted (returned %+v)",
							pt.name, ms, nowNs, got)
					}
					wantArrives := nowNs + ms*int64(time.Millisecond)
					wantExpires := wantArrives + int64(DefaultReservationGrace)
					if r.ReservedAtUnixNano != nowNs {
						t.Fatalf("[%s] ArrivingInMillis=%d ReservedAtUnixNano=%d, want %d",
							pt.name, ms, r.ReservedAtUnixNano, nowNs)
					}
					if r.ArrivesAtUnixNano != wantArrives {
						t.Fatalf("[%s] ArrivingInMillis=%d ArrivesAtUnixNano=%d, want %d (now+delay)",
							pt.name, ms, r.ArrivesAtUnixNano, wantArrives)
					}
					if r.ExpiresAtUnixNano != wantExpires {
						t.Fatalf("[%s] ArrivingInMillis=%d ExpiresAtUnixNano=%d, want %d (arrival+grace)",
							pt.name, ms, r.ExpiresAtUnixNano, wantExpires)
					}
					if len(got) != 1 || len(listed) != 1 {
						t.Fatalf("[%s] ArrivingInMillis=%d minted len(got)=%d len(Reservations)=%d, want 1 each",
							pt.name, ms, len(got), len(listed))
					}
					if _, ok := findReservation(listed, "t", reservationTestPrefixP); !ok {
						t.Fatalf("[%s] ArrivingInMillis=%d: accepted row missing from Reservations(%+v)",
							pt.name, ms, listed)
					}
				} else {
					if len(got) != 0 {
						t.Fatalf("[%s] ArrivingInMillis=%d is unrepresentable at now=%d but minted %+v",
							pt.name, ms, nowNs, got)
					}
					// A rejection must also leave no row hiding in the map or the listed
					// slice; asserting only `len(got)==0` would let a bogus row hide.
					if len(listed) != 0 {
						t.Fatalf("[%s] ArrivingInMillis=%d rejected but Reservations=%+v, want none",
							pt.name, ms, listed)
					}
					if len(s.reservations) != 0 {
						t.Fatalf("[%s] ArrivingInMillis=%d rejected but len(s.reservations)=%d, want 0",
							pt.name, ms, len(s.reservations))
					}
				}

				// Invariant check on EVERY returned and EVERY listed row, independent of the
				// accept/reject branch above: no row may be inverted and no stamp may be
				// negative. This is the check the old body could never reach, because every
				// row it scanned was rejected first; here it executes on the accepted
				// near-boundary rows too, so an inverted row is caught as a row (not just
				// as a wrong count).
				for _, r := range append(append([]SlotReservation{}, got...), listed...) {
					if r.ArrivesAtUnixNano < r.ReservedAtUnixNano {
						t.Fatalf("[%s] ArrivingInMillis=%d produced a row arriving before it was reserved: %+v",
							pt.name, ms, r)
					}
					if r.ExpiresAtUnixNano < r.ArrivesAtUnixNano {
						t.Fatalf("[%s] ArrivingInMillis=%d produced an expiry before arrival: %+v",
							pt.name, ms, r)
					}
					if r.ArrivesAtUnixNano < 0 || r.ExpiresAtUnixNano < 0 {
						t.Fatalf("[%s] ArrivingInMillis=%d produced a negative stamp: %+v",
							pt.name, ms, r)
					}
				}
			}
		}
	})

	t.Run("E11_churn_leaves_bookkeeping_at_zero", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		const n = 1000
		for i := 0; i < n; i++ {
			trace := "churn-" + itoa(i)
			if _, ok := tbl.SetTurnIntent(trace, reservationTestIntent(reservationTestPrefixP)); !ok {
				t.Fatalf("SetTurnIntent(%s) rejected", trace)
			}
		}
		// One scan mints a row per live hint.
		if got := s.ReserveKnownComing(base); len(got) != n {
			t.Fatalf("reserve over %d live traces = %d rows, want %d", n, len(got), n)
		}

		// Every trace leaves; a later scan must release ALL bookkeeping (R-5).
		for i := 0; i < n; i++ {
			tbl.Reset("churn-" + itoa(i))
		}
		if got := s.ReserveKnownComing(base.Add(time.Millisecond)); len(got) != 0 {
			t.Fatalf("scan after full churn = %d rows, want 0", len(got))
		}
		if len(s.reservations) != 0 {
			t.Fatalf("churn leaked bookkeeping: len(s.reservations)=%d, want 0", len(s.reservations))
		}
	})

	t.Run("E12_two_traces_sharing_prefix_are_distinct_rows", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("a", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent(a) rejected")
		}
		if _, ok := tbl.SetTurnIntent("b", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent(b) rejected")
		}
		got := s.ReserveKnownComing(base)
		if len(got) != 2 {
			t.Fatalf("shared-prefix reserve = %+v, want two distinct rows", got)
		}
		if _, ok := findReservation(got, "a", reservationTestPrefixP); !ok {
			t.Fatalf("(a,P) missing from %+v", got)
		}
		if _, ok := findReservation(got, "b", reservationTestPrefixP); !ok {
			t.Fatalf("(b,P) missing from %+v", got)
		}

		// Each promotes by trace; promoting `a` must not consume `b`'s row (distinct keys).
		if _, ok := s.PromoteReservation("a", reservationTestPrefixP, base.Add(5*time.Millisecond)); !ok {
			t.Fatal("PromoteReservation(a,P) failed")
		}
		if _, ok := s.PromoteReservation("b", reservationTestPrefixP, base.Add(5*time.Millisecond)); !ok {
			t.Fatal("PromoteReservation(b,P) failed after a's promotion; rows not keyed by trace")
		}
	})

	t.Run("E13_only_current_prefix_row_survives", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent(P) rejected")
		}
		if _, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP); !ok {
			t.Fatal("first scan minted nothing")
		}

		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixQ)); !ok {
			t.Fatal("SetTurnIntent(Q) rejected")
		}
		got := s.ReserveKnownComing(base.Add(5 * time.Millisecond))
		if len(got) != 1 {
			t.Fatalf("after two prefixes = %+v, want exactly one (current) row", got)
		}
		if got[0].Prefix != reservationTestPrefixQ {
			t.Fatalf("surviving prefix = %q, want %q", got[0].Prefix, reservationTestPrefixQ)
		}
	})

	t.Run("E14_promote_never_expired_returns_same_stamps", func(t *testing.T) {
		tbl, s := attachedScheduler(t, StrictPriority)
		if _, ok := tbl.SetTurnIntent("t", reservationTestIntent(reservationTestPrefixP)); !ok {
			t.Fatal("SetTurnIntent rejected")
		}
		r0, ok := findReservation(s.ReserveKnownComing(base), "t", reservationTestPrefixP)
		if !ok {
			t.Fatal("first scan minted nothing")
		}

		promoAt := base.Add(5 * time.Millisecond)
		promo, ok := s.PromoteReservation("t", reservationTestPrefixP, promoAt)
		if !ok {
			t.Fatal("PromoteReservation on a live row returned ok=false")
		}
		if promo.SlotReservation != r0 {
			t.Fatalf("promotion adopted a different reservation:\n got  %+v\n want %+v", promo.SlotReservation, r0)
		}
		if promo.PromotedAtUnixNano != promoAt.UnixNano() {
			t.Fatalf("PromotedAtUnixNano = %d, want %d", promo.PromotedAtUnixNano, promoAt.UnixNano())
		}
		if _, ok := s.PromoteReservation("t", reservationTestPrefixP, promoAt.Add(time.Millisecond)); ok {
			t.Fatal("reservation promoted twice; want a single adoption")
		}
	})

	t.Run("E15_zero_hint_table_mints_nothing_and_grows_no_map", func(t *testing.T) {
		// A genuinely empty table (no SetTurnIntent at all): a scan must mint nothing and
		// must not grow bookkeeping (R-7 zero-hint compatibility).
		_, s := attachedScheduler(t, StrictPriority)
		if got := s.ReserveKnownComing(base); len(got) != 0 {
			t.Fatalf("zero-hint reserve = %+v, want none", got)
		}
		if len(s.reservations) != 0 {
			t.Fatalf("zero-hint reserve grew the map: len(s.reservations)=%d, want 0", len(s.reservations))
		}
		// A read-only Reservations must never grow bookkeeping either (R-5/R-7).
		if got := s.Reservations(base); len(got) != 0 {
			t.Fatalf("zero-hint Reservations = %+v, want none", got)
		}
		if len(s.reservations) != 0 {
			t.Fatalf("Reservations grew the map: len(s.reservations)=%d, want 0", len(s.reservations))
		}
	})
}
