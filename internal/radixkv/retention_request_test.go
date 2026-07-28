package radixkv

import (
	"errors"
	"testing"
)

// TestRetentionRequestReclaim is the #5259 witness: the pure client-declared per-request
// retention core — a (priority, TTL-window) descriptor over an INJECTED logical clock — and
// the deterministic reclaim ordering it drives. Sub-tests cover every acceptance case:
// TTL-expired entries are reclaim-eligible; among live entries the lowest priority reclaims
// first; ties break deterministically by older admission then ID; a never-expire max-priority
// entry is retained; and validation fails closed on negative TTL / out-of-range priority.
func TestRetentionRequestReclaim(t *testing.T) {
	// Expired-by-TTL entries land in the Expired partition; in-window entries stay Live. The
	// clock is injected, so the boundary (now > Admitted+TTL) is exact and replayable.
	t.Run("expired_by_ttl_is_reclaim_eligible", func(t *testing.T) {
		entries := []RetentionEntry{
			{ID: "past", RetentionRequest: RetentionRequest{Priority: 50, TTL: 10, Admitted: 0}},   // ends at 10
			{ID: "edge", RetentionRequest: RetentionRequest{Priority: 50, TTL: 10, Admitted: 5}},   // ends at 15
			{ID: "fresh", RetentionRequest: RetentionRequest{Priority: 50, TTL: 10, Admitted: 20}}, // ends at 30
		}
		v := ReclaimOrder(entries, 15) // now=15: "past" ended at 10 (expired), "edge" ends at 15 (still live), "fresh" future
		if len(v.Expired) != 1 || v.Expired[0].ID != "past" {
			t.Fatalf("Expired=%v, want exactly [past]", ids(v.Expired))
		}
		if len(v.Live) != 2 {
			t.Fatalf("Live=%v, want 2 live entries", ids(v.Live))
		}
		// Boundary check: at now == windowEnd the entry is NOT yet expired (strictly beyond).
		if entries[1].Expired(15) {
			t.Errorf("edge entry expired at now=15 (window end); want live until now>15")
		}
		if !entries[1].Expired(16) {
			t.Errorf("edge entry live at now=16; want expired strictly beyond window end")
		}
	})

	// Among live entries the lowest priority reclaims first: Live is sorted ascending priority.
	t.Run("lowest_priority_reclaims_first", func(t *testing.T) {
		entries := []RetentionEntry{
			{ID: "hi", RetentionRequest: RetentionRequest{Priority: 90, TTL: 0, Admitted: 0}},
			{ID: "lo", RetentionRequest: RetentionRequest{Priority: 10, TTL: 0, Admitted: 0}},
			{ID: "mid", RetentionRequest: RetentionRequest{Priority: 50, TTL: 0, Admitted: 0}},
		}
		v := ReclaimOrder(entries, 100)
		got := ids(v.Live)
		want := []string{"lo", "mid", "hi"}
		if !equalIDs(got, want) {
			t.Fatalf("reclaim order=%v, want %v (ascending priority)", got, want)
		}
	})

	// Ties broken deterministically: equal priority -> older admission reclaims first; equal
	// priority AND admission -> lexicographic ID. The order is total and replayable.
	t.Run("ties_broken_by_older_time_then_id", func(t *testing.T) {
		entries := []RetentionEntry{
			{ID: "b_newer", RetentionRequest: RetentionRequest{Priority: 40, TTL: 0, Admitted: 30}},
			{ID: "a_older", RetentionRequest: RetentionRequest{Priority: 40, TTL: 0, Admitted: 10}},
			{ID: "c_same2", RetentionRequest: RetentionRequest{Priority: 40, TTL: 0, Admitted: 10}},
			{ID: "a_same1", RetentionRequest: RetentionRequest{Priority: 40, TTL: 0, Admitted: 10}},
		}
		v := ReclaimOrder(entries, 100)
		got := ids(v.Live)
		// Admitted=10 group first (older time), tie-broken by ID lexicographic:
		// "a_older" < "a_same1" < "c_same2"; then the Admitted=30 entry last.
		want := []string{"a_older", "a_same1", "c_same2", "b_newer"}
		if !equalIDs(got, want) {
			t.Fatalf("tie-break order=%v, want %v (older admission, then ID)", got, want)
		}
	})

	// A never-expire (TTL=0), max-priority entry is retained: never in Expired, and last in the
	// live reclaim order (reclaimed only after every weaker entry).
	t.Run("never_expire_max_priority_retained", func(t *testing.T) {
		pinned := RetentionEntry{ID: "pinned", RetentionRequest: RetentionRequest{Priority: MaxRetentionPriority, TTL: retainForever, Admitted: 0}}
		weak := RetentionEntry{ID: "weak", RetentionRequest: RetentionRequest{Priority: MinRetentionPriority, TTL: 5, Admitted: 0}}
		// At a very large clock the pinned entry is still live; the weak one has long expired.
		v := ReclaimOrder([]RetentionEntry{pinned, weak}, 1_000_000)
		if len(v.Expired) != 1 || v.Expired[0].ID != "weak" {
			t.Fatalf("Expired=%v, want [weak]", ids(v.Expired))
		}
		if len(v.Live) != 1 || v.Live[0].ID != "pinned" {
			t.Fatalf("Live=%v, want [pinned] retained forever", ids(v.Live))
		}
		if pinned.Expired(1 << 62) {
			t.Errorf("retainForever entry reported expired; want never expired on window")
		}
	})

	// Validation fails CLOSED on out-of-range fields; a well-formed descriptor validates.
	t.Run("validation_fails_closed", func(t *testing.T) {
		cases := []struct {
			name string
			req  RetentionRequest
			want error
		}{
			{"priority_too_high", RetentionRequest{Priority: 101, TTL: 0, Admitted: 0}, ErrRetentionPriorityRange},
			{"priority_negative", RetentionRequest{Priority: -1, TTL: 0, Admitted: 0}, ErrRetentionPriorityRange},
			{"ttl_negative", RetentionRequest{Priority: 50, TTL: -1, Admitted: 0}, ErrRetentionTTLNegative},
			{"admitted_negative", RetentionRequest{Priority: 50, TTL: 0, Admitted: -1}, ErrRetentionAdmittedNegative},
		}
		for _, c := range cases {
			if err := c.req.Validate(); !errors.Is(err, c.want) {
				t.Errorf("%s: Validate()=%v, want %v", c.name, err, c.want)
			}
		}
		ok := RetentionRequest{Priority: DefaultRetentionPriority, TTL: 10, Admitted: 5}
		if err := ok.Validate(); err != nil {
			t.Errorf("well-formed descriptor: Validate()=%v, want nil", err)
		}
	})
}

// ids projects entry IDs for readable assertions.
func ids(entries []RetentionEntry) []string {
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.ID
	}
	return out
}

func equalIDs(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
