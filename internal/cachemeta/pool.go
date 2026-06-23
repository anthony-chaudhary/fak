package cachemeta

import "github.com/anthony-chaudhary/fak/internal/abi"

// pool.go models a SWITCH-POOLED, multi-host SHARED cache tier — the economics and
// the trust gate a fabric-attached memory pool (a CXL.mem / CXL-switch pool, a
// coherent far-memory fabric) turns on once a KV cache is SHARED across a fleet of
// tenants rather than living host-private.
//
// hardware.go gave each tier a physical character (latency/bandwidth/capacity/
// coherence) and a zero-copy share kind; placement.go demotes a hot prefix one tier
// colder instead of evicting it. Both reason about ONE host's ladder. The part a
// pooled tier adds is what happens when the SAME hot prefix is wanted by N tenants:
//
//   - host-private far memory holds N copies and pays N prefills (each tenant builds
//     its own);
//   - a reachable-but-copy-only pool pays ONE prefill (the rest stage a copy) but
//     still holds N copies;
//   - a COHERENT, zero-copy pool (CXL.mem / a CXL-switch memory pool) pays ONE
//     prefill AND holds ONE copy — every other tenant attends it in place, zero-copy.
//
// That last regime is the one a shared memory fabric exists to enable, and it is the
// only one that saves on BOTH axes (compute and capacity) at fleet scale. This file
// makes the three regimes a deterministic, payload-free calculation (PlanFleetReuse),
// and adds the cross-tenant reuse GATE (PoolReuseVerdict) that keeps the dedup honest:
// a pooled cell may be attended by a DIFFERENT tenant only when its materialization
// key matches AND the producer declared it shareable AND it is adjudicated trusted —
// a shared address space is not a license to alias mismatched or poisoned bytes.
//
// Like the rest of cachemeta this plane owns no cache and touches no bytes: it decides
// what a pooled tier is WORTH and WHO may reuse a cell in it; the engine adapter that
// maps the CXL region performs the physical sharing. It is the placement counterpart
// of materialization.go's correctness gate, extended with the trust/scope axis a
// SHARED pool — unlike a single host's private ladder — is forced to answer.

// PoolProfile describes the POOLING character of a residency tier: how many hosts can
// reach it, whether it is cache-coherent (a consumer attends a cell in place rather
// than staging a copy), and the native zero-copy share kind across the fabric. It is
// the pooling counterpart of hardware.go's TierProfile (which describes the per-host
// physical character of the same tier).
type PoolProfile struct {
	Tier     ResidencyTier
	Hosts    int       // hosts/sockets that can attend this tier (1 = host-private)
	Coherent bool      // cache-coherent across the fabric: attend in place, no copy
	Share    ShareKind // native zero-copy share kind across the fabric
}

// Reachable reports whether a cell in this tier can be reached at all by a host other
// than the one that wrote it — i.e. it is a shared pool (Hosts > 1), not host-private
// memory. A reachable pool lets a non-owner STAGE a copy instead of re-prefilling;
// only a FabricShareable pool lets it attend zero-copy.
func (p PoolProfile) Reachable() bool { return p.Hosts > 1 }

// FabricShareable reports whether ONE resident copy is attendable zero-copy by every
// host in the pool: the tier is pooled (Hosts > 1), cache-coherent, and advertises a
// real zero-copy share kind. This is the property that turns N per-host copies into a
// single shared copy — the reason a coherent CXL pool changes the fleet economics.
func (p PoolProfile) FabricShareable() bool {
	return p.Hosts > 1 && p.Coherent && p.Share.ZeroCopy()
}

// DefaultPoolProfiles returns a representative pooling profile for each tier, matching
// the DefaultTierProfiles ladder. HBM/DRAM/NUMA-far are host-private (one host attends
// them); CXL is the SWITCH-POOLED, coherent, zero-copy tier shared across a pod of
// hosts (the value a CXL.mem memory pool adds); disk is a host-local store; a remote
// tier is reachable but copy-only (an RDMA transfer, not a coherent attend). The Hosts
// counts are order-of-magnitude stand-ins (a pod-sized CXL pool); an operator
// overrides them with their fabric's real topology, exactly as TierProfile numbers are
// overridden — the FabricShareable logic is identical against measured values.
func DefaultPoolProfiles() map[ResidencyTier]PoolProfile {
	return map[ResidencyTier]PoolProfile{
		TierHBM:     {Tier: TierHBM, Hosts: 1, Coherent: true, Share: ShareDmabuf},
		TierDRAM:    {Tier: TierDRAM, Hosts: 1, Coherent: true, Share: ShareMmap},
		TierNUMAFar: {Tier: TierNUMAFar, Hosts: 1, Coherent: true, Share: ShareMmap},
		TierCXL:     {Tier: TierCXL, Hosts: 8, Coherent: true, Share: ShareCXLHDM},
		TierDisk:    {Tier: TierDisk, Hosts: 1, Coherent: false, Share: ShareCopy},
		TierRemote:  {Tier: TierRemote, Hosts: 8, Coherent: false, Share: ShareRDMA},
	}
}

// FleetReuseRequest describes a hot prefix that a FLEET of tenants each want to attend,
// and the pooled tier it would be published to. SizeBytes is the KV bytes of the
// prefix (drives memory + stage cost), Tokens its length (drives prefill cost),
// PerTokenPrefillNanos the model/hardware cost of re-prefilling one token. Profile is
// the tier's physical character (for the per-tenant attend/stage cost); Pool is its
// pooling character.
type FleetReuseRequest struct {
	Tenants              int
	Tokens               int64
	SizeBytes            int64
	PerTokenPrefillNanos int64
	Profile              TierProfile
	Pool                 PoolProfile
}

// FleetReuseResult places the per-host-PRIVATE baseline (every tenant prefills and
// holds its own copy) next to the GIVEN pool's behavior, and reports the savings
// between them on both axes — prefill tokens (compute) and resident bytes (capacity).
// Reachable/Shareable record which pooling regime applied, so a reader can see WHY the
// savings are what they are (zero for host-private, prefill-only for a copy-only pool,
// both for a coherent fabric pool).
type FleetReuseResult struct {
	Tenants int

	// Per-host-private baseline: every tenant builds and holds its own copy.
	PrivatePrefillTokens int64
	PrivateResidentBytes int64

	// The given pool's behavior.
	PooledPrefillTokens int64
	PooledResidentBytes int64
	Reachable           bool // non-owners can reach the pool (stage instead of re-prefill)
	Shareable           bool // non-owners attend zero-copy (one shared copy serves all)

	// Savings of the pool over the per-host-private baseline.
	PrefillTokensSaved int64
	BytesDeduplicated  int64

	// Per-non-owner-tenant cost of OBTAINING the prefix under the pool, vs rebuilding it.
	PerTenantAttendNanos    int64
	PerTenantRecomputeNanos int64
}

// PlanFleetReuse computes the three-way fleet economics for one hot prefix shared by
// Tenants tenants, given the destination tier's physical and pooling character. The
// regime is chosen purely from the PoolProfile:
//
//   - FabricShareable (coherent, zero-copy, pooled): one owner prefills, the rest
//     attend the single pooled copy IN PLACE — one prefill, one resident copy;
//   - Reachable but copy-only (a shared but non-coherent pool): one owner prefills,
//     the rest STAGE a copy (cheaper than re-prefill for a big span) — one prefill, N
//     resident copies;
//   - host-private (unreachable by other hosts): each tenant rebuilds its own — N
//     prefills, N resident copies (identical to the private baseline, savings zero).
//
// It is pure and deterministic (no clock, no bytes moved): the same inputs yield the
// same result on every machine and in CI.
func PlanFleetReuse(req FleetReuseRequest) FleetReuseResult {
	n := int64(req.Tenants)
	if n < 1 {
		n = 1
	}
	res := FleetReuseResult{
		Tenants:                 int(n),
		PrivatePrefillTokens:    n * req.Tokens,
		PrivateResidentBytes:    n * req.SizeBytes,
		Reachable:               req.Pool.Reachable(),
		Shareable:               req.Pool.FabricShareable(),
		PerTenantRecomputeNanos: recomputeNanos(req.Tokens, req.PerTokenPrefillNanos),
	}
	switch {
	case res.Shareable:
		// One owner prefills; every other tenant attends the single pooled copy in
		// place. A coherent attend stages nothing — its cost is the tier's access
		// latency, not a bandwidth-bound copy.
		res.PooledPrefillTokens = req.Tokens
		res.PooledResidentBytes = req.SizeBytes
		res.PerTenantAttendNanos = req.Profile.ReadLatencyNanos
	case res.Reachable:
		// Reachable but copy-only: the owner prefills once; every other tenant stages a
		// copy from the pool (a one-time bytes/bandwidth move) but still holds its own.
		res.PooledPrefillTokens = req.Tokens
		res.PooledResidentBytes = n * req.SizeBytes
		res.PerTenantAttendNanos = stageNanos(req.SizeBytes, req.Profile)
	default:
		// Host-private: unreachable by other hosts, so each tenant rebuilds its own.
		res.PooledPrefillTokens = n * req.Tokens
		res.PooledResidentBytes = n * req.SizeBytes
		res.PerTenantAttendNanos = res.PerTenantRecomputeNanos
	}
	res.PrefillTokensSaved = res.PrivatePrefillTokens - res.PooledPrefillTokens
	res.BytesDeduplicated = res.PrivateResidentBytes - res.PooledResidentBytes
	return res
}

// PoolReuseVerdict is the cross-tenant reuse GATE for a SHARED pool: it decides whether
// a tenant may attend a pooled cell another tenant wrote. It is the placement
// counterpart of materialization.go's MaterializeVerdict, adding the trust/scope axis a
// SHARED pool turns on — a single host's private ladder never has to answer "may a
// DIFFERENT tenant alias this cell." A pooled cell is reusable across a tenant boundary
// only when, in order:
//
//  1. it is not poisoned — a quarantined cell must LEAVE the pool, never be re-served
//     (a shared address space would otherwise hand poisoned bytes to every reader);
//  2. its producer declared it shareable beyond itself — the fail-closed default
//     (ScopeAgent, private to one agent) refuses cross-tenant reuse;
//  3. it is adjudicated trusted (TaintTrusted) — only proven-trusted bytes may be
//     aliased across a tenant boundary;
//  4. its materialization key MATCHES on every axis (model/tokenizer/serializer/
//     position/policy/admitter) — a KV span built under one model is garbage under
//     another; an incomplete key fails closed.
//
// Any failure is a typed, NON-serveable verdict (CanServe() == false), so dedup across
// the pool is honest: only a trusted, shareable, key-matched cell is ever aliased.
func PoolReuseVerdict(stored Entry, want MaterializationKey) LookupVerdict {
	// (1) A quarantined/poisoned cell must leave the pool, never be re-served.
	if stored.Security.Taint == abi.TaintQuarantined || stored.Security.AdmissionVerdict == AdmissionQuarantine {
		return Quarantine(stored, ReasonTaintDenied)
	}
	// (2) The producer must have declared the cell shareable beyond itself.
	if stored.Security.Scope == abi.ScopeAgent {
		return Miss(ReasonScopeDenied)
	}
	// (3) Only an adjudicated-trusted cell may be aliased across a tenant boundary.
	if stored.Security.Taint != abi.TaintTrusted {
		return Miss(ReasonTaintDenied)
	}
	// (4) Correctness: a KV span built under one model/tokenizer/position regime is
	// garbage under another. The materialization gate makes the per-axis match (and the
	// incomplete-key fail-closed) mechanical; MatKVSpan is exact, so no quality evidence
	// is required.
	return MaterializeVerdict(MatKVSpan, stored, want, QualityEvidence{})
}
