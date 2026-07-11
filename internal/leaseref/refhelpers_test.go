package leaseref

import (
	"testing"
	"time"
)

// refhelpers_test.go pins the two PURE helpers in refhelpers.go directly. Both are
// exercised only indirectly elsewhere (expired() via Record.Expired/IntentRecord.Expired;
// liveExpire() via Live/LiveIntents/LiveSessions), so their boundary behavior — the exact
// instant a lease lapses, and the ordering of the live-vs-expired split — was never asserted
// on its own. A drift here silently reaps a still-live lease out from under its worker or lets
// a dead lease keep refusing a peer, so the boundaries are asserted to the second.

func rhUnix(sec int64) time.Time { return time.Unix(sec, 0) }

// rhStub is a minimal expirable used only to exercise liveExpire's partition/ordering,
// isolated from expired()'s own TTL arithmetic: it is expired exactly once now reaches its
// deadline (the same inclusive >= boundary expired() uses).
type rhStub struct {
	id       string
	deadline int64 // unix seconds; expired once now >= deadline
}

func (s rhStub) Expired(now time.Time) bool { return now.Unix() >= s.deadline }

func rhIDs(recs []rhStub) []string {
	out := make([]string, 0, len(recs))
	for _, r := range recs {
		out = append(out, r.id)
	}
	return out
}

func rhEq(got []string, want ...string) bool {
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

// TestExpiredTTLBoundary pins the shared TTL primitive behind Record.Expired and
// IntentRecord.Expired (both delegate to expired()). Two load-bearing rules:
//   - a non-positive TTL never expires (a 0/absent TTL lease is held until explicitly
//     released — the "forever" case the cascade path leans on), and
//   - a positive TTL lapses at the INCLUSIVE instant now >= effectiveActiveAt+ttl,
//     not one second later.
func TestExpiredTTLBoundary(t *testing.T) {
	const activeAt = int64(1_000_000)
	tests := []struct {
		name       string
		now        int64
		ttlSeconds int64
		want       bool
	}{
		{"zero ttl never expires (far-future now)", activeAt + 1_000_000, 0, false},
		{"negative ttl never expires", activeAt + 1_000_000, -5, false},
		{"positive ttl: well before deadline", activeAt + 10, 60, false},
		{"positive ttl: one second before deadline", activeAt + 59, 60, false},
		{"positive ttl: exactly at deadline is expired (inclusive)", activeAt + 60, 60, true},
		{"positive ttl: one second past deadline", activeAt + 61, 60, true},
		{"positive ttl: now before activeAt is not expired", activeAt - 100, 60, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := expired(rhUnix(tc.now), tc.ttlSeconds, activeAt); got != tc.want {
				t.Fatalf("expired(now=%d, ttl=%d, activeAt=%d) = %v, want %v",
					tc.now, tc.ttlSeconds, activeAt, got, tc.want)
			}
		})
	}
}

// TestLiveExpirePartition pins liveExpire — the shared body of Live/LiveIntents/LiveSessions
// that splits records into the live set (returned to the arbiter) and the expired ids (handed
// to a reaper). Three invariants: expired records leave the live set, their ids are collected
// via idOf in input order, and the live set preserves input order (the arbiter's disjointness
// walk is order-stable).
func TestLiveExpirePartition(t *testing.T) {
	now := rhUnix(1000)
	idOf := func(s rhStub) string { return s.id }

	t.Run("empty input yields empty live and expired", func(t *testing.T) {
		var none []rhStub
		live, exp := liveExpire(none, now, idOf)
		if len(live) != 0 || len(exp) != 0 {
			t.Fatalf("liveExpire(nil) = live %v, expired %v; want both empty", rhIDs(live), exp)
		}
	})

	t.Run("all live: every record kept in order, nothing expired", func(t *testing.T) {
		all := []rhStub{{"a", 2000}, {"b", 1500}, {"c", 1001}}
		live, exp := liveExpire(all, now, idOf)
		if len(exp) != 0 {
			t.Fatalf("expired = %v, want none", exp)
		}
		if !rhEq(rhIDs(live), "a", "b", "c") {
			t.Fatalf("live order = %v, want [a b c]", rhIDs(live))
		}
	})

	t.Run("all expired: nothing live, ids in input order", func(t *testing.T) {
		all := []rhStub{{"a", 1000}, {"b", 500}, {"c", 999}} // every deadline <= now
		live, exp := liveExpire(all, now, idOf)
		if len(live) != 0 {
			t.Fatalf("live = %v, want none", rhIDs(live))
		}
		if !rhEq(exp, "a", "b", "c") {
			t.Fatalf("expired ids = %v, want [a b c]", exp)
		}
	})

	t.Run("mixed: split and order preserved on both sides", func(t *testing.T) {
		all := []rhStub{
			{"live1", 2000},
			{"dead1", 1000}, // exactly now -> expired (inclusive), matches expired()'s >=
			{"live2", 1001},
			{"dead2", 10},
		}
		live, exp := liveExpire(all, now, idOf)
		if !rhEq(rhIDs(live), "live1", "live2") {
			t.Fatalf("live = %v, want [live1 live2]", rhIDs(live))
		}
		if !rhEq(exp, "dead1", "dead2") {
			t.Fatalf("expired = %v, want [dead1 dead2]", exp)
		}
	})
}
