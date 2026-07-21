package main

// peer_dram_lender.go — issue #5083, the PRODUCER half of the peer-DRAM-over-RDMA paging
// rung (#4306, cachemeta commit 1719f3006). cachemeta.ProbedTierProfiles admits
// TierRemoteDRAM ONLY when CapacityProbe.RemoteDRAMPresent is true with positive
// RemoteDRAMBytes — the prove-it-or-drop-it gate every far tier shares. Before this file
// nothing set those two fields: probedTierProfilesForHost built the live probe from the
// HBM/DRAM/Disk/NUMA-far/CXL compute HAL and never offered a lender, so the borrowed-DRAM
// rung was correct and unit-proven but unreachable in a live serve (grep -rn TierRemoteDRAM
// returned hits only inside internal/cachemeta). This is that missing spine half: a
// neighbor's lendable free DRAM registered under a lease, folded into the host probe so the
// pager can prefer borrowed peer RAM over local disk.
//
// FENCES. The roster lives ABOVE cachemeta so the policy plane never imports the RDMA fabric
// HAL (the same posture ProbedTierProfiles takes — "pure and witnessable with no GPU"). The
// lease/reclaim shape mirrors internal/cachemeta/nixl_lease.go: a registration is held under
// a lease with an expiry instant, and the moment the lease lapses OR the lender reclaims its
// RAM the offer folds to zero and probedTierProfilesForHost drops the rung — fail-closed, so
// a span never pages to memory that is no longer on offer (cachemeta.ReclaimRemoteDRAM is the
// placement-side twin that re-pages an already-borrowed span the instant the borrow vanishes).
// This ships the registration seam and the env-seeded operator declaration; the LIVE RDMA
// discovery transport that would auto-register a neighbor is #3199, and the 2-node hardware
// page-in measurement that turns the modeled advantage MEASURED is #5066 — both out of scope
// here, which is the wiring, not the fabric and not the measurement.

import (
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// peerDRAMLender is a field-only record of one neighbor's offer to lend fak its idle DRAM as
// a paging target. It names the peer, how many bytes it offers, and the lease that bounds the
// offer in time; a lender is HONORED only while its lease is active (see activeAt). It is the
// registration-seam twin of cachemeta.NIXLLease — that witnesses an external K/V lease; this
// witnesses a peer-memory *capacity* lease — kept in cmd/fak so cachemeta stays HAL-free.
type peerDRAMLender struct {
	// PeerID identifies the neighbor host/engine lending the DRAM. An empty PeerID is never
	// active (fail-closed: an unidentified lender can never back the rung).
	PeerID string
	// LendableBytes is the free DRAM the peer offers to borrow. A non-positive value is never
	// active (a borrow of nothing gains nothing).
	LendableBytes int64
	// GrantedAtMillis is when the offer was registered/last refreshed; ExpiresAtMillis is the
	// instant it lapses absent a refresh. A non-positive ExpiresAtMillis means "no expiry
	// clock" — the offer is governed only by an explicit release (the lender taking its RAM
	// back), exactly like NIXLLease's non-positive expiry.
	GrantedAtMillis int64
	ExpiresAtMillis int64
	// Released records the lender reclaiming its RAM. A released lender is never active — this
	// is the fail-closed reclaim the #4306 sketch names ("lease + fail-closed reclaim when the
	// lender needs its RAM back"), at the capacity-registration plane.
	Released bool
}

// activeAt reports whether the lender's offer stands at nowMillis: it names a peer, offers
// positive bytes, has not been reclaimed, and (when it carries an expiry clock) has not
// lapsed. Every other case folds to inactive — fail-closed, so the rung is offered ONLY on a
// live, unexpired, unreclaimed lease. Mirrors NIXLLease.StateAt's Active arm.
func (l peerDRAMLender) activeAt(nowMillis int64) bool {
	if l.PeerID == "" || l.LendableBytes <= 0 || l.Released {
		return false
	}
	if l.ExpiresAtMillis > 0 && nowMillis >= l.ExpiresAtMillis {
		return false
	}
	return true
}

// activeLentDRAMBytes sums the lendable DRAM of every lender whose lease is active at
// nowMillis. A reclaimed, lapsed, unidentified, or empty offer contributes zero, so a reclaim
// or expiry silently shrinks (or removes) the borrowed rung — the fail-closed reclaim, folded.
func activeLentDRAMBytes(lenders []peerDRAMLender, nowMillis int64) int64 {
	var total int64
	for _, l := range lenders {
		if l.activeAt(nowMillis) {
			total += l.LendableBytes
		}
	}
	return total
}

// applyPeerDRAMLenders folds the active lent DRAM into a CapacityProbe, setting the
// RemoteDRAMPresent/RemoteDRAMBytes fields cachemeta.ProbedTierProfiles gates TierRemoteDRAM
// on. When no lender is active it leaves the probe untouched (Present stays false), so the
// ladder ProbedTierProfiles returns has NO remote-DRAM rung — the "unregistered probe yields
// no rung" half of the witness. It is pure and clock-injected so a test replays the whole
// registered-vs-unregistered ladder decision deterministically, with no RDMA fabric.
func applyPeerDRAMLenders(probe cachemeta.CapacityProbe, lenders []peerDRAMLender, nowMillis int64) cachemeta.CapacityProbe {
	if total := activeLentDRAMBytes(lenders, nowMillis); total > 0 {
		probe.RemoteDRAMPresent = true
		probe.RemoteDRAMBytes = total
	}
	return probe
}

// peerDRAMRoster is the process-wide set of registered peer-DRAM lenders, keyed by PeerID so
// a refreshed offer from the same neighbor replaces its prior one rather than double-counting.
// It is the mutable registration seam the live serve consults; the fold above stays pure so
// the roster's concurrency is the only stateful surface.
type peerDRAMRoster struct {
	mu      sync.Mutex
	lenders map[string]peerDRAMLender
}

func newPeerDRAMRoster() *peerDRAMRoster {
	return &peerDRAMRoster{lenders: map[string]peerDRAMLender{}}
}

// register records (or refreshes) a lender's offer. An offer naming no peer is ignored (there
// is nothing to key it by, and an unidentified lender can never be active anyway).
func (r *peerDRAMRoster) register(l peerDRAMLender) {
	if l.PeerID == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lenders[l.PeerID] = l
}

// release drops a lender's offer entirely — the lender reclaimed its RAM. The next probe fold
// then omits it (fail-closed reclaim). Releasing an unknown peer is a no-op.
func (r *peerDRAMRoster) release(peerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.lenders, peerID)
}

// snapshot copies the current lenders for a pure fold, so a probe never holds the lock across
// cachemeta.ProbedTierProfiles.
func (r *peerDRAMRoster) snapshot() []peerDRAMLender {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]peerDRAMLender, 0, len(r.lenders))
	for _, l := range r.lenders {
		out = append(out, l)
	}
	return out
}

// defaultPeerDRAMRoster is the process-global roster probedTierProfilesForHost consults. It is
// empty until a lender registers, so a box with no neighbor lending offers no remote-DRAM rung
// — byte-identical to the pre-#5083 behavior.
var defaultPeerDRAMRoster = newPeerDRAMRoster()

// peerDRAMLenderEnvVar is the operator/test declaration of a neighbor's lendable DRAM, used
// until the live RDMA discovery transport (#3199) can auto-register one. It gives the issue's
// live witness ("a fak run whose probed ladder contains remote_dram with the lent capacity") a
// reachable path without inventing the fabric: an operator who knows a peer is lending declares
// it, and the rung enters the live serve's ladder.
const peerDRAMLenderEnvVar = "FAK_PEER_DRAM_LENDER"

// parsePeerDRAMLenderSpec parses the FAK_PEER_DRAM_LENDER value into lender records. The format
// is comma-separated entries "peerID:bytes[:ttlMillis]", where ttlMillis (optional) sets an
// expiry relative to nowMillis so a stale declaration self-reclaims instead of stranding spans
// on a peer that has gone away. Malformed or non-positive entries are skipped (fail-closed: a
// declaration fak cannot parse into a positive, identified offer registers nothing).
func parsePeerDRAMLenderSpec(spec string, nowMillis int64) []peerDRAMLender {
	var out []peerDRAMLender
	for _, entry := range strings.Split(spec, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		parts := strings.Split(entry, ":")
		if len(parts) < 2 {
			continue
		}
		peerID := strings.TrimSpace(parts[0])
		if peerID == "" {
			continue
		}
		bytes, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil || bytes <= 0 {
			continue
		}
		l := peerDRAMLender{PeerID: peerID, LendableBytes: bytes, GrantedAtMillis: nowMillis}
		if len(parts) >= 3 {
			if ttl, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64); err == nil && ttl > 0 {
				l.ExpiresAtMillis = nowMillis + ttl
			}
		}
		out = append(out, l)
	}
	return out
}

// registerPeerDRAMLendersFromEnv seeds defaultPeerDRAMRoster from FAK_PEER_DRAM_LENDER at serve
// startup, so an operator declaration is live before the first post-decode capacity sweep runs
// probedTierProfilesForHost. It returns the number of lenders registered so the caller can log
// the wire. Absent the env var it registers nothing (the rung stays out of the ladder).
func registerPeerDRAMLendersFromEnv() int {
	spec := strings.TrimSpace(os.Getenv(peerDRAMLenderEnvVar))
	if spec == "" {
		return 0
	}
	n := 0
	for _, l := range parsePeerDRAMLenderSpec(spec, time.Now().UnixMilli()) {
		defaultPeerDRAMRoster.register(l)
		n++
	}
	return n
}
