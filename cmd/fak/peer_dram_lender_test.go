package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// TestProbedLadderGatesOnRegisteredLender is the #5083 witness: a CapacityProbe carrying a
// registered, active peer-DRAM lender yields a ProbedTierProfiles ladder that INCLUDES the
// remote_dram rung with the lent capacity, while an unregistered probe does NOT. This is the
// producer half the issue names — proving the rung is reachable once a lender is folded in,
// and drops out otherwise (prove-it-or-drop-it), with no RDMA fabric.
func TestProbedLadderGatesOnRegisteredLender(t *testing.T) {
	const now = int64(1_000_000)
	base := cachemeta.CapacityProbe{DRAMBytes: 64 << 30, DiskBytes: 1 << 40}

	// Unregistered: no lender folded in → no remote_dram rung.
	bare := cachemeta.ProbedTierProfiles(applyPeerDRAMLenders(base, nil, now))
	if _, ok := bare[cachemeta.TierRemoteDRAM]; ok {
		t.Fatalf("unregistered probe must NOT yield a remote_dram rung, got one")
	}

	// Registered active lender: fold it in → remote_dram rung present, sized to the lent bytes.
	const lent = int64(48 << 30)
	lenders := []peerDRAMLender{{PeerID: "peer-b", LendableBytes: lent, GrantedAtMillis: now}}
	withLender := cachemeta.ProbedTierProfiles(applyPeerDRAMLenders(base, lenders, now))
	prof, ok := withLender[cachemeta.TierRemoteDRAM]
	if !ok {
		t.Fatalf("registered lender must yield a remote_dram rung, got none")
	}
	if prof.CapacityBytes != lent {
		t.Fatalf("remote_dram rung sized %d, want the lent %d", prof.CapacityBytes, lent)
	}
}

// TestPeerDRAMLenderActiveAt covers the fail-closed lease gate: only an identified lender with
// positive bytes, unreclaimed, and (if it carries an expiry) unexpired is active.
func TestPeerDRAMLenderActiveAt(t *testing.T) {
	const now = int64(5_000)
	cases := []struct {
		name string
		l    peerDRAMLender
		want bool
	}{
		{"active no expiry", peerDRAMLender{PeerID: "p", LendableBytes: 1}, true},
		{"active before expiry", peerDRAMLender{PeerID: "p", LendableBytes: 1, ExpiresAtMillis: now + 1}, true},
		{"expired", peerDRAMLender{PeerID: "p", LendableBytes: 1, ExpiresAtMillis: now}, false},
		{"reclaimed", peerDRAMLender{PeerID: "p", LendableBytes: 1, Released: true}, false},
		{"no peer", peerDRAMLender{LendableBytes: 1}, false},
		{"zero bytes", peerDRAMLender{PeerID: "p", LendableBytes: 0}, false},
		{"negative bytes", peerDRAMLender{PeerID: "p", LendableBytes: -1}, false},
	}
	for _, tc := range cases {
		if got := tc.l.activeAt(now); got != tc.want {
			t.Errorf("%s: activeAt=%v want %v", tc.name, got, tc.want)
		}
	}
}

// TestActiveLentDRAMBytesSumsOnlyActive proves the fold sums only active leases, so a reclaim
// or expiry shrinks the offered rung — the fail-closed reclaim, folded.
func TestActiveLentDRAMBytesSumsOnlyActive(t *testing.T) {
	const now = int64(100)
	lenders := []peerDRAMLender{
		{PeerID: "a", LendableBytes: 10, GrantedAtMillis: now},
		{PeerID: "b", LendableBytes: 20, ExpiresAtMillis: now - 1}, // expired
		{PeerID: "c", LendableBytes: 40, Released: true},           // reclaimed
		{PeerID: "d", LendableBytes: 8, ExpiresAtMillis: now + 50}, // active
	}
	if got := activeLentDRAMBytes(lenders, now); got != 18 {
		t.Fatalf("active lent bytes = %d, want 10+8=18 (expired/reclaimed excluded)", got)
	}
}

// TestPeerDRAMRosterRegisterReleaseRefresh covers the mutable registration seam: a refresh
// replaces (never double-counts) a same-peer offer, and release drops it (fail-closed reclaim).
func TestPeerDRAMRosterRegisterReleaseRefresh(t *testing.T) {
	const now = int64(0)
	r := newPeerDRAMRoster()

	r.register(peerDRAMLender{PeerID: "peer-a", LendableBytes: 100})
	r.register(peerDRAMLender{PeerID: "peer-a", LendableBytes: 250}) // refresh same peer
	if got := activeLentDRAMBytes(r.snapshot(), now); got != 250 {
		t.Fatalf("after refresh want 250 (replace, not sum), got %d", got)
	}

	r.register(peerDRAMLender{PeerID: "peer-b", LendableBytes: 50})
	if got := activeLentDRAMBytes(r.snapshot(), now); got != 300 {
		t.Fatalf("two distinct lenders want 300, got %d", got)
	}

	r.release("peer-a")
	if got := activeLentDRAMBytes(r.snapshot(), now); got != 50 {
		t.Fatalf("after release peer-a want 50, got %d", got)
	}

	// An empty-peer offer is ignored (nothing to key it by).
	r.register(peerDRAMLender{PeerID: "", LendableBytes: 999})
	if got := activeLentDRAMBytes(r.snapshot(), now); got != 50 {
		t.Fatalf("empty-peer offer must be ignored, got %d", got)
	}
}

// TestParsePeerDRAMLenderSpec covers the operator/test declaration parser, including the
// optional TTL and fail-closed skipping of malformed entries.
func TestParsePeerDRAMLenderSpec(t *testing.T) {
	const now = int64(1_000)
	got := parsePeerDRAMLenderSpec("peer-x:1024, peer-y:2048:500 , bad, peer-z:0, :99, peer-w:-5", now)
	if len(got) != 2 {
		t.Fatalf("want 2 valid lenders (peer-x, peer-y), got %d: %+v", len(got), got)
	}
	if got[0].PeerID != "peer-x" || got[0].LendableBytes != 1024 || got[0].ExpiresAtMillis != 0 {
		t.Errorf("peer-x parsed wrong: %+v", got[0])
	}
	if got[1].PeerID != "peer-y" || got[1].LendableBytes != 2048 || got[1].ExpiresAtMillis != now+500 {
		t.Errorf("peer-y parsed wrong (want ttl→expiry %d): %+v", now+500, got[1])
	}
}
