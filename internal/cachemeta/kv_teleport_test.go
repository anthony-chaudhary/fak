package cachemeta

import "testing"

// fastLink is a representative transport profile: line-rate streaming with a small
// first-byte latency, so a modest span crosses it well under a normal re-route budget.
func fastLink() TierProfile {
	return TierProfile{Tier: TierRemote, ReadLatencyNanos: 2_000, BandwidthMBPerSec: 12_000}
}

func TestResolveKVTeleport_TeleportBeatsRecompute(t *testing.T) {
	// A large warm prefix: re-prefilling 4k tokens at 20us/token is 80ms, while pushing a
	// 4 MiB span over a 12 GB/s link is well under a millisecond. Teleport must win.
	req := KVTeleportRequest{
		SpanBytes:            4 << 20,
		Tokens:               4096,
		Link:                 fastLink(),
		PerTokenPrefillNanos: 20_000,
		DeadlineNanos:        50_000_000, // 50ms budget
	}
	v := ResolveKVTeleport(req)
	if !v.Teleported() {
		t.Fatalf("expected teleport, got %q (%s)", v.Outcome, v.Reason)
	}
	if v.Outcome != TeleportKV || v.Reason != "teleport_beats_recompute" {
		t.Fatalf("unexpected verdict: %+v", v)
	}
	if !v.WithinDeadline {
		t.Fatalf("expected within deadline, got %+v", v)
	}
	if v.TeleportNanos >= v.RecomputeNanos {
		t.Fatalf("teleport (%d) should be cheaper than recompute (%d)", v.TeleportNanos, v.RecomputeNanos)
	}
}

func TestResolveKVTeleport_DeadlineFailsClosedToRecompute(t *testing.T) {
	// The transfer would beat recompute on raw cost, but the re-route deadline is tighter
	// than the transfer time — fail-closed to recompute rather than blow the budget.
	req := KVTeleportRequest{
		SpanBytes:            4 << 20,
		Tokens:               4096,
		Link:                 fastLink(),
		PerTokenPrefillNanos: 20_000,
		DeadlineNanos:        100, // 100ns budget: nothing crosses the link that fast
	}
	v := ResolveKVTeleport(req)
	if v.Outcome != RecomputePrefix {
		t.Fatalf("expected recompute, got %q (%s)", v.Outcome, v.Reason)
	}
	if v.Reason != "teleport_exceeds_deadline" {
		t.Fatalf("expected teleport_exceeds_deadline, got %q", v.Reason)
	}
	if v.WithinDeadline {
		t.Fatalf("teleport should NOT fit the deadline: %+v", v)
	}
}

func TestResolveKVTeleport_RecomputeCheaperForTinyPrefix(t *testing.T) {
	// A tiny warm prefix (8 tokens) is cheaper to rebuild than to move over the wire, even
	// with a generous deadline — recompute is the right call.
	req := KVTeleportRequest{
		SpanBytes:            4 << 20,
		Tokens:               8,
		Link:                 fastLink(),
		PerTokenPrefillNanos: 20_000,
		DeadlineNanos:        50_000_000,
	}
	v := ResolveKVTeleport(req)
	if v.Outcome != RecomputePrefix {
		t.Fatalf("expected recompute, got %q (%s)", v.Outcome, v.Reason)
	}
	if v.Reason != "recompute_cheaper_than_teleport" {
		t.Fatalf("expected recompute_cheaper_than_teleport, got %q", v.Reason)
	}
	if !v.WithinDeadline {
		t.Fatalf("transfer still fits the deadline even when it does not pay: %+v", v)
	}
}

func TestResolveKVTeleport_EmptySpanRecomputes(t *testing.T) {
	for _, tc := range []struct {
		name string
		req  KVTeleportRequest
	}{
		{"no_bytes", KVTeleportRequest{SpanBytes: 0, Tokens: 4096, Link: fastLink(), PerTokenPrefillNanos: 20_000}},
		{"no_tokens", KVTeleportRequest{SpanBytes: 4 << 20, Tokens: 0, Link: fastLink(), PerTokenPrefillNanos: 20_000}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := ResolveKVTeleport(tc.req)
			if v.Outcome != RecomputePrefix || v.Reason != "no_warm_span" {
				t.Fatalf("expected no_warm_span recompute, got %+v", v)
			}
		})
	}
}

func TestResolveKVTeleport_UnprofiledLinkFailsClosed(t *testing.T) {
	// A link with no bandwidth (unregistered/unknown transport) yields a sentinel transfer
	// time, so even with no deadline the teleport never looks cheap: fail-closed recompute.
	req := KVTeleportRequest{
		SpanBytes:            4 << 20,
		Tokens:               4096,
		Link:                 TierProfile{Tier: TierRemote, BandwidthMBPerSec: 0},
		PerTokenPrefillNanos: 20_000,
		DeadlineNanos:        0, // no deadline imposed
	}
	v := ResolveKVTeleport(req)
	if v.Outcome != RecomputePrefix {
		t.Fatalf("expected recompute for unprofiled link, got %q (%s)", v.Outcome, v.Reason)
	}
	// With no deadline the sentinel is "within" budget, so the guard that fires is the
	// cost comparison, not the deadline.
	if v.Reason != "recompute_cheaper_than_teleport" {
		t.Fatalf("expected recompute_cheaper_than_teleport, got %q", v.Reason)
	}
}

func TestResolveKVTeleport_Deterministic(t *testing.T) {
	req := KVTeleportRequest{
		SpanBytes:            4 << 20,
		Tokens:               4096,
		Link:                 fastLink(),
		PerTokenPrefillNanos: 20_000,
		DeadlineNanos:        50_000_000,
	}
	first := ResolveKVTeleport(req)
	for i := 0; i < 100; i++ {
		if got := ResolveKVTeleport(req); got != first {
			t.Fatalf("non-deterministic verdict at %d: %+v != %+v", i, got, first)
		}
	}
}
