package gateway

import "testing"

// TestPeerFetchTierChooseFetchWhenCheaper: the peer holds the full needed prefix and
// pulling the bytes is strictly cheaper than a local recompute → fetch verdict.
func TestPeerFetchTierChooseFetchWhenCheaper(t *testing.T) {
	peer := PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}
	// 4 needed blocks * 100 bytes = 400 bytes over a 200 bytes/unit link = 2.0 cost.
	// Recompute: 4 blocks * 10.0/block = 40.0 cost. Fetch is far cheaper.
	v := peer.Choose(PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10})
	if !v.Fetch {
		t.Fatalf("expected fetch, got %+v", v)
	}
	if v.Reason != reasonFetch {
		t.Fatalf("reason = %q, want %q", v.Reason, reasonFetch)
	}
	if v.FetchCost != 2.0 || v.RecomputeCost != 40.0 {
		t.Fatalf("costs = fetch %v / recompute %v, want 2.0 / 40.0", v.FetchCost, v.RecomputeCost)
	}
}

// TestPeerFetchTierPeerHoldsNothing: the peer holds no blocks of the needed prefix →
// recompute (a read only tier cannot fill the gap).
func TestPeerFetchTierPeerHoldsNothing(t *testing.T) {
	peer := PeerKVTier{HeldBlocks: 0, BytesPerBlock: 100}
	v := peer.Choose(PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10})
	if v.Fetch {
		t.Fatalf("expected recompute when peer holds nothing, got %+v", v)
	}
	if v.Reason != reasonPeerShort {
		t.Fatalf("reason = %q, want %q", v.Reason, reasonPeerShort)
	}
}

// TestPeerFetchTierPartialHoldRecomputes: the peer holds SOME but not the full needed
// prefix → recompute (read only: no partial top-up).
func TestPeerFetchTierPartialHoldRecomputes(t *testing.T) {
	peer := PeerKVTier{HeldBlocks: 3, BytesPerBlock: 100}
	v := peer.Choose(PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10})
	if v.Fetch || v.Reason != reasonPeerShort {
		t.Fatalf("expected peer_short recompute, got %+v", v)
	}
}

// TestPeerFetchTierFetchCostExceedsRecompute: the peer holds the full prefix but the
// fetch is more expensive than recompute → recompute (fetch never chosen when slower).
func TestPeerFetchTierFetchCostExceedsRecompute(t *testing.T) {
	peer := PeerKVTier{HeldBlocks: 4, BytesPerBlock: 1000}
	// Fetch: 4 * 1000 / 1.0 = 4000. Recompute: 4 * 10 = 40. Fetch is far dearer.
	v := peer.Choose(PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 1}, RecomputeLocal{PerBlock: 10})
	if v.Fetch {
		t.Fatalf("expected recompute when fetch dearer, got %+v", v)
	}
	if v.Reason != reasonRecomputeCheaper {
		t.Fatalf("reason = %q, want %q", v.Reason, reasonRecomputeCheaper)
	}
	if v.FetchCost <= v.RecomputeCost {
		t.Fatalf("expected fetch cost %v > recompute %v", v.FetchCost, v.RecomputeCost)
	}
}

// TestPeerFetchTierTieRecomputes: equal fetch and recompute cost keeps the local move
// (fetch is only chosen on a strict improvement).
func TestPeerFetchTierTieRecomputes(t *testing.T) {
	peer := PeerKVTier{HeldBlocks: 2, BytesPerBlock: 10}
	// Fetch: 2 * 10 / 1 = 20. Recompute: 2 * 10 = 20. Tie → recompute.
	v := peer.Choose(PrefixNeed{Blocks: 2}, FetchLink{BytesPerUnit: 1}, RecomputeLocal{PerBlock: 10})
	if v.Fetch {
		t.Fatalf("expected recompute on tie, got %+v", v)
	}
	if v.FetchCost != v.RecomputeCost {
		t.Fatalf("expected equal costs, got fetch %v / recompute %v", v.FetchCost, v.RecomputeCost)
	}
}

// TestPeerFetchTierReadOnlyInvariant: making a choice must not mutate the peer tier's
// held state — the receiver is by value, so the decision is read only by construction.
func TestPeerFetchTierReadOnlyInvariant(t *testing.T) {
	peer := PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}
	before := peer
	_ = peer.Choose(PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10})
	if peer != before {
		t.Fatalf("choice mutated peer tier: before %+v after %+v", before, peer)
	}
}

// TestPeerFetchTierDegenerateFailsClosed: every degenerate input falls back to the safe
// local recompute, never a fetch.
func TestPeerFetchTierDegenerateFailsClosed(t *testing.T) {
	inf := 1.0
	for {
		inf *= 1e300
		if inf > 1e307 {
			break
		}
	}
	posInf := inf * inf
	nan := posInf - posInf

	cases := []struct {
		name   string
		peer   PeerKVTier
		need   PrefixNeed
		link   FetchLink
		local  RecomputeLocal
		reason string
	}{
		{"no need", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: 0}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10}, reasonNoNeed},
		{"negative need", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: -3}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10}, reasonNoNeed},
		{"zero bytes per block", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 0}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10}, reasonBadInput},
		{"negative bytes per block", PeerKVTier{HeldBlocks: 8, BytesPerBlock: -5}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: 10}, reasonBadInput},
		{"zero link rate", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 0}, RecomputeLocal{PerBlock: 10}, reasonBadInput},
		{"negative link rate", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: -1}, RecomputeLocal{PerBlock: 10}, reasonBadInput},
		{"nan link rate", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: nan}, RecomputeLocal{PerBlock: 10}, reasonBadInput},
		{"inf link rate", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: posInf}, RecomputeLocal{PerBlock: 10}, reasonBadInput},
		{"negative recompute", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: -1}, reasonBadInput},
		{"nan recompute", PeerKVTier{HeldBlocks: 8, BytesPerBlock: 100}, PrefixNeed{Blocks: 4}, FetchLink{BytesPerUnit: 200}, RecomputeLocal{PerBlock: nan}, reasonBadInput},
	}
	for _, c := range cases {
		v := c.peer.Choose(c.need, c.link, c.local)
		if v.Fetch {
			t.Fatalf("%s: expected recompute, got fetch %+v", c.name, v)
		}
		if v.Reason != c.reason {
			t.Fatalf("%s: reason = %q, want %q", c.name, v.Reason, c.reason)
		}
	}
}
