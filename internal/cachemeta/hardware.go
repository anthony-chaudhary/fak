package cachemeta

// hardware.go makes the cache metadata plane HARDWARE-AWARE from the foundation up.
//
// cachemeta already named WHERE a payload lives (ResidencyTier) and how a KV span
// MOVES between tiers (kvtransfer.go). What it could not express is the part a
// modern memory hierarchy turns on: the physical CHARACTER of each tier (latency,
// bandwidth, capacity, byte-addressability, coherence, persistence) and whether a
// payload can be handed to another consumer ZERO-COPY (a coherent CXL address, a
// shared mmap, an RDMA-registered region, a GPU dma-buf) or must be memcpy'd.
//
// Those two facts are what let a placement policy be co-optimized with the hardware
// instead of running blind LRU: under memory pressure a hot prefix can be DEMOTED to
// CXL-attached far memory (byte-addressable, still attendable-in-place, far cheaper
// than recompute) rather than EVICTED (which forces a full re-prefill later). This
// file adds the tier characteristics and the zero-copy share descriptor; lifecycle.go
// adds the per-tier TTL + state machine that times the moves, and placement.go adds
// the cost model that chooses demote-vs-evict. Together they are the "hardware-aware
// from day 1" layer — payload-free, deterministic, and below the engine that performs
// the physical movement (cachemeta still owns no cache and touches no bytes).
//
// CXL and NUMA-far are added as FIRST-CLASS tiers, slotting into the existing
// HBM/DRAM/Disk/Remote ladder between local DRAM and disk — exactly where a coherent,
// byte-addressable, high-capacity expansion tier belongs.

const (
	// TierNUMAFar is byte-addressable, cache-coherent DRAM on a REMOTE NUMA node
	// (another socket): same load/store semantics as local DRAM, modestly higher
	// latency and lower effective bandwidth. A KV span here is still attended in
	// place — it never needs recompute, only a NUMA hop.
	TierNUMAFar ResidencyTier = "numa_far"

	// TierCXL is CXL-attached memory (CXL.mem / Type-3 expander or a fabric-attached
	// pool): byte-addressable and cache-coherent like DRAM, but a few times the
	// latency and a fraction of the bandwidth — in exchange for very large, poolable,
	// shareable capacity. It is the demote target that makes "don't evict, relocate"
	// pay: a span demoted to CXL stays reusable WITHOUT recompute, and (with a
	// coherent CXL.mem region) can be shared zero-copy across hosts in a pod.
	TierCXL ResidencyTier = "cxl"

	// TierRemoteDRAM is a peer node's idle DRAM lent as a PAGING tier over RDMA
	// (Infiniswap / far-memory, #4306): a memory-starved box pages a cold-but-expensive
	// KV span into a neighbor's spare RAM by one-sided RDMA instead of spilling it to
	// local SSD, because a random RDMA page-in (~2 µs) is an order of magnitude faster
	// than an NVMe random read (~10 µs). It turns "this box does not have enough RAM for
	// the KV pool" into "borrow the rack's spare RAM." It is OFF-BOX and NON-coherent:
	// the span is paged back on access (like disk), so it is a SPILL target, not a
	// demote-in-place tier — but it sits ABOVE local disk in the paging order
	// (HBM > DRAM > NUMA-far > CXL > remote-DRAM > disk > object-store). The bytes move
	// zero-copy by the NIC's DMA engine (ShareRDMA); the lender's RAM is borrowed under a
	// lease and reclaimed fail-closed when the lender needs it back (nixl_lease.go models
	// the lease/reclaim seam). Like the local far tiers it is prove-it-or-drop-it: it
	// enters the ladder ONLY when a peer has registered a lendable region
	// (CapacityProbe.RemoteDRAMPresent), never by default — a paging target that is not
	// actually on offer would strand every span spilled to it.
	TierRemoteDRAM ResidencyTier = "remote_dram"
)

// ShareKind names HOW a payload can be handed to another consumer (another session,
// model, process, or host) without copying its bytes. Zero-copy sharing is the
// difference between a reuse that costs a pointer and one that costs a memcpy of the
// whole KV span — the single biggest lever once a prefix is hot across many requests.
//
// The zero value is ShareCopy: an entry that has not declared a zero-copy capability
// is assumed to require a copy. That fail-safe default means a missing/empty share
// descriptor never tricks a consumer into aliasing memory it may not alias.
type ShareKind string

const (
	// ShareCopy (zero value) — the payload must be memcpy'd to be reused elsewhere.
	ShareCopy ShareKind = ""
	// ShareMmap — a shared file/anonymous mapping: zero-copy across processes on the
	// same host (the payload is faulted in, not duplicated).
	ShareMmap ShareKind = "mmap"
	// ShareCXLHDM — a coherent CXL Host-managed Device Memory region: zero-copy
	// load/store sharing across sockets, and across hosts on a coherent CXL fabric.
	ShareCXLHDM ShareKind = "cxl_hdm"
	// ShareRDMA — an RDMA-registered region: zero-copy transfer over the wire by the
	// NIC's DMA engine (the bytes are never touched by a CPU on either side).
	ShareRDMA ShareKind = "rdma"
	// ShareDmabuf — an exported GPU dma-buf: zero-copy GPU<->GPU or GPU<->NIC handoff
	// of device-resident KV without a host bounce.
	ShareDmabuf ShareKind = "dmabuf"
)

// ZeroCopy reports whether this share kind hands the payload over without a copy.
func (s ShareKind) ZeroCopy() bool { return s != ShareCopy }

// ShareDescriptor is the zero-copy share capability advertised by a resident payload:
// the kind of sharing available, an opaque handle the owning engine can resolve into
// a mapping (an mmap path, a CXL HDM base address, an RDMA rkey, a dma-buf fd — never
// resolved here; cachemeta stays payload-free), and whether the region is cache-
// coherent (a consumer may dereference it directly) versus merely transferable.
type ShareDescriptor struct {
	Kind     ShareKind
	Handle   string // opaque to cachemeta; the owning engine resolves it
	Coherent bool   // true => a consumer can load/store the region in place
}

// ZeroCopy reports whether the descriptor advertises a real zero-copy capability.
func (d ShareDescriptor) ZeroCopy() bool { return d.Kind.ZeroCopy() }

// WithShare sets the zero-copy share descriptor on an entry's residency. Use it to
// declare that a resident span is shareable in place (e.g. a CXL-resident prefix a
// fleet of sessions can attend without each cloning it).
func WithShare(d ShareDescriptor) Option {
	return func(e *Entry) { e.Residency.Share = d }
}

// TierProfile is the physical character of one residency tier. It is the table a
// placement policy reads to be hardware-aware: where capacity is, how far each tier
// is from the compute (latency/bandwidth), whether a payload there is attendable in
// place (ByteAddressable + Coherent) or must be staged back first, whether it
// survives a process/power cycle (Persistent), and the native zero-copy share kind
// the tier supports.
//
// The numbers in DefaultTierProfiles are REPRESENTATIVE order-of-magnitude defaults,
// not measurements of any particular box. An operator overrides them with values
// measured for their hardware (the same posture experiments/benchmark/catalog.json
// takes for the machine table); the placement math is identical either way, so the
// policy is exercised against whatever profile it is handed.
type TierProfile struct {
	Tier              ResidencyTier
	ReadLatencyNanos  int64 // typical random-read latency to first byte
	BandwidthMBPerSec int64 // sustained streaming bandwidth (MB/s) for staging
	CapacityBytes     int64 // usable capacity of this tier (0 = unknown/unbounded)
	ByteAddressable   bool  // true => load/store addressable (not block-only)
	Coherent          bool  // true => CPU-cache-coherent; a span is attendable in place
	Persistent        bool  // true => survives a process/power cycle
	Share             ShareKind
}

// AttendableInPlace reports whether a span resident in this tier can be read by the
// model WITHOUT first staging it into a hotter tier — true exactly when the tier is
// byte-addressable AND coherent. This is the property that makes CXL/NUMA-far demotion
// cheap: the span stays usable where it is, so demotion never implies recompute.
func (p TierProfile) AttendableInPlace() bool { return p.ByteAddressable && p.Coherent }

// DefaultTierProfiles returns a representative profile for every tier in the local
// memory hierarchy plus the off-box tiers, ordered hottest to coldest:
// HBM -> DRAM -> NUMA-far -> CXL -> Disk -> Remote. The values are order-of-magnitude
// stand-ins (see TierProfile); the point is the SHAPE — each step is colder, larger,
// and (past CXL) no longer attendable in place.
//
// The peer-DRAM-over-RDMA paging rung (TierRemoteDRAM, #4306) is deliberately NOT in
// this map: unlike the local tiers it exists ONLY when a peer registers a lendable
// region, so ProbedTierProfiles adds it prove-it-or-drop-it (its representative physics
// live in remoteDRAMProfile).
func DefaultTierProfiles() map[ResidencyTier]TierProfile {
	return map[ResidencyTier]TierProfile{
		TierHBM: {
			Tier: TierHBM, ReadLatencyNanos: 200, BandwidthMBPerSec: 2_000_000,
			CapacityBytes: 80 << 30, ByteAddressable: true, Coherent: true,
			Persistent: false, Share: ShareDmabuf,
		},
		TierDRAM: {
			Tier: TierDRAM, ReadLatencyNanos: 90, BandwidthMBPerSec: 300_000,
			CapacityBytes: 512 << 30, ByteAddressable: true, Coherent: true,
			Persistent: false, Share: ShareMmap,
		},
		TierNUMAFar: {
			Tier: TierNUMAFar, ReadLatencyNanos: 140, BandwidthMBPerSec: 200_000,
			CapacityBytes: 512 << 30, ByteAddressable: true, Coherent: true,
			Persistent: false, Share: ShareMmap,
		},
		TierCXL: {
			Tier: TierCXL, ReadLatencyNanos: 300, BandwidthMBPerSec: 64_000,
			CapacityBytes: 2 << 40, ByteAddressable: true, Coherent: true,
			Persistent: false, Share: ShareCXLHDM,
		},
		TierDisk: {
			Tier: TierDisk, ReadLatencyNanos: 10_000, BandwidthMBPerSec: 7_000,
			CapacityBytes: 8 << 40, ByteAddressable: false, Coherent: false,
			Persistent: true, Share: ShareCopy,
		},
		TierRemote: {
			Tier: TierRemote, ReadLatencyNanos: 100_000, BandwidthMBPerSec: 12_000,
			CapacityBytes: 0, ByteAddressable: false, Coherent: false,
			Persistent: false, Share: ShareRDMA,
		},
	}
}

// remoteDRAMProfile is the representative physical profile of the peer-DRAM-over-RDMA
// paging rung (TierRemoteDRAM, #4306). Latency and bandwidth are order-of-magnitude
// stand-ins (see TierProfile) for a one-sided RDMA page-in over a 100 Gb NIC: a ~2 µs
// random read-to-first-byte — an order of magnitude under the ~10 µs of an NVMe random
// read (TierDisk), the whole point of paging to borrowed RAM instead of local SSD — at
// line-rate streaming bandwidth (above disk, below local coherent memory). The region
// is byte-TRANSFERABLE but NOT coherent, so a span here is paged back before use (a
// SPILL target, never attended in place), and NOT persistent (peer RAM is volatile and
// reclaimable). CapacityBytes is 0 (borrowed/unbounded) until a probe sizes the lent
// region. Kept out of DefaultTierProfiles because, unlike the local far tiers, a
// remote-DRAM rung exists only when a peer lender is registered — ProbedTierProfiles
// adds it prove-it-or-drop-it, sizing this profile from CapacityProbe.RemoteDRAMBytes.
func remoteDRAMProfile() TierProfile {
	return TierProfile{
		Tier: TierRemoteDRAM, ReadLatencyNanos: 2_000, BandwidthMBPerSec: 12_000,
		CapacityBytes: 0, ByteAddressable: false, Coherent: false,
		Persistent: false, Share: ShareRDMA,
	}
}

// CapacityProbe carries the live per-tier capacity readings ProbedTierProfiles needs to
// turn DefaultTierProfiles' representative stand-ins into the ladder THIS box can prove.
// It is plain data, so ProbedTierProfiles stays pure and witnessable with no GPU and no
// host inspection: the startup caller fills it from the compute probes (DeviceMemoryInfo
// for HBM, HostSystemMemoryInfo for DRAM, the spill-filesystem free-space probe for Disk,
// NUMAFarMemoryInfo/CXLMemoryInfo for the far tiers)
// — that wiring lives ABOVE cachemeta so the policy plane never imports the compute HAL —
// and a test injects synthetic readings to assert the chosen ladder.
type CapacityProbe struct {
	// HBMBytes is the device (GPU) memory total; HBMPresent gates whether the box has a
	// provable HBM tier at all. A no-GPU box leaves HBMPresent false and ProbedTierProfiles
	// omits TierHBM entirely, so placement never targets device memory the box does not
	// have. When present, a positive HBMBytes sizes the tier.
	HBMBytes   int64
	HBMPresent bool
	// DRAMBytes is the host RAM total. The host always has DRAM, so the tier is always in
	// the ladder; a non-positive reading keeps the representative default rather than
	// claiming a measurement it does not have.
	DRAMBytes int64
	// DiskBytes is the usable capacity of the spill filesystem. Disk is always present; a
	// non-positive reading keeps the representative default.
	DiskBytes int64
	// NUMAFarBytes sizes the far-NUMA tier (another socket's DRAM); NUMAFarPresent gates
	// whether the box PROVED one (#1470, the far-memory probe). Unlike DRAM/Disk the far
	// tiers have no always-present default to fall back to: absent a confirming probe the
	// tier stays out of the proved ladder entirely, exactly as HBM does on a no-GPU box.
	NUMAFarBytes   int64
	NUMAFarPresent bool
	// CXLBytes / CXLPresent size the CXL tier (CPU-less expansion memory, probed as
	// memory-only NUMA nodes) the same prove-it-or-drop-it way.
	CXLBytes   int64
	CXLPresent bool
	// RemoteDRAMBytes / RemoteDRAMPresent size the peer-DRAM-over-RDMA paging rung
	// (TierRemoteDRAM, #4306) the same prove-it-or-drop-it way as the far tiers: the rung
	// enters the ladder ONLY when a peer has registered a lendable region (Present with
	// positive bytes). Unlike DRAM/Disk there is no always-present default — absent a
	// registered lender the box has no remote-DRAM rung and a starved span spills to
	// local disk, not to memory that is not on offer. The registering caller lives ABOVE
	// cachemeta (a peer-memory transport / NIXL adapter), so the policy plane never
	// imports the RDMA fabric HAL.
	RemoteDRAMBytes   int64
	RemoteDRAMPresent bool
}

// ProbedTierProfiles turns the representative DefaultTierProfiles ladder into the one THIS
// box can prove it has: it sizes each locally-probeable physical tier (HBM, DRAM, Disk)
// from the live CapacityProbe and DROPS a tier the box cannot prove. A no-GPU box
// (p.HBMPresent == false) gets a ladder with no TierHBM, so the planner never places a
// span on device memory that is not there. The far-memory tiers — NUMA-far and CXL —
// follow the same prove-it-or-drop-it rule via the far-memory probe (#1470): they enter
// the ladder exactly when the probe confirmed them (Present with positive bytes) and stay
// out otherwise. The peer-DRAM-over-RDMA rung (#4306) follows the same rule, gated on a
// registered lender (RemoteDRAMPresent). Only the off-box Remote (object-store) pool still
// has no local probe and is always left OUT; an operator who has provisioned one re-adds it
// the same way they override any other profile. Only CapacityBytes is taken from the probe — latency, bandwidth, and
// addressability stay at their representative values, because the probe sizes the ladder,
// it does not re-measure the physics. The returned map is independent of
// DefaultTierProfiles' (callers may mutate it).
func ProbedTierProfiles(p CapacityProbe) map[ResidencyTier]TierProfile {
	defaults := DefaultTierProfiles()
	out := make(map[ResidencyTier]TierProfile, 3)

	// DRAM and Disk are always present; size them from the probe when it read a real
	// number, else keep the representative default.
	dram := defaults[TierDRAM]
	if p.DRAMBytes > 0 {
		dram.CapacityBytes = p.DRAMBytes
	}
	out[TierDRAM] = dram

	disk := defaults[TierDisk]
	if p.DiskBytes > 0 {
		disk.CapacityBytes = p.DiskBytes
	}
	out[TierDisk] = disk

	// HBM is in the ladder only when the box proved a device with real capacity.
	if p.HBMPresent && p.HBMBytes > 0 {
		hbm := defaults[TierHBM]
		hbm.CapacityBytes = p.HBMBytes
		out[TierHBM] = hbm
	}

	// The far tiers are in the ladder only when the far-memory probe confirmed them
	// (#1470) — same prove-it-or-drop-it rule as HBM, because a demote target that does
	// not exist is worse than none: the planner would relocate spans into it forever.
	if p.NUMAFarPresent && p.NUMAFarBytes > 0 {
		far := defaults[TierNUMAFar]
		far.CapacityBytes = p.NUMAFarBytes
		out[TierNUMAFar] = far
	}
	if p.CXLPresent && p.CXLBytes > 0 {
		cxl := defaults[TierCXL]
		cxl.CapacityBytes = p.CXLBytes
		out[TierCXL] = cxl
	}

	// The peer-DRAM-over-RDMA paging rung (#4306) is in the ladder only when a peer
	// registered a lendable region (RemoteDRAMPresent with positive bytes) — same
	// prove-it-or-drop-it rule as the far tiers, and for the same reason: a paging target
	// that is not actually on offer would strand every span spilled to it. Its physics
	// come from remoteDRAMProfile (not DefaultTierProfiles); the probe sizes the lent
	// region, it does not re-measure the physics.
	if p.RemoteDRAMPresent && p.RemoteDRAMBytes > 0 {
		rdma := remoteDRAMProfile()
		rdma.CapacityBytes = p.RemoteDRAMBytes
		out[TierRemoteDRAM] = rdma
	}

	return out
}

// localTierLadder is the demote/promote order of the LOCAL memory hierarchy, hottest
// to coldest. Off-box tiers (Remote/Provider, and the peer-DRAM-over-RDMA paging rung
// TierRemoteDRAM, #4306) and the synthetic Recompute sentinel are not part of the
// IN-BOX ladder: remote-DRAM is a colder-than-CXL SPILL target reached by demotion
// (NextColderTier threads it in) and re-entered by page-in on access, never a local
// promote tier — so it stays out of localTierLadder / IsLocalTier while remaining a
// first-class relocation target. Demotion past Disk means Recompute (drop the resident
// copy and re-prefill on demand).
var localTierLadder = []ResidencyTier{TierHBM, TierDRAM, TierNUMAFar, TierCXL, TierDisk}

// TierRank orders tiers from hottest (0) to coldest by access cost, so a policy can
// compare two tiers without a profile table. Lower rank == closer to the compute.
// Off-ladder tiers sort after the local hierarchy; an unknown tier sorts last.
func TierRank(t ResidencyTier) int {
	switch t {
	case TierHBM:
		return 0
	case TierDRAM:
		return 1
	case TierNUMAFar:
		return 2
	case TierCXL:
		return 3
	case TierRemoteDRAM:
		return 4
	case TierDisk:
		return 5
	case TierRemote:
		return 6
	case TierProvider:
		return 7
	case TierRecompute:
		return 8
	default:
		return 9
	}
}

// NextColderTier returns the next tier down the relocation/paging ladder
// (HBM->DRAM->NUMA-far->CXL->remote-DRAM->Disk->Recompute). The peer-DRAM-over-RDMA rung
// (#4306) threads between CXL and Disk, so a pager that walks this ladder reaches
// borrowed peer RAM before it spills to local SSD (it is only actually chosen when a
// lender is profiled — coldestColderWithRoom skips a rung the box does not have). Past
// Disk the only "colder" option is to stop holding the bytes and recompute later, so it
// returns TierRecompute. For an off-ladder tier (Remote/Provider/Recompute/Unknown)
// there is no colder relocation tier, so it returns TierUnknown.
func NextColderTier(t ResidencyTier) ResidencyTier {
	switch t {
	case TierHBM:
		return TierDRAM
	case TierDRAM:
		return TierNUMAFar
	case TierNUMAFar:
		return TierCXL
	case TierCXL:
		return TierRemoteDRAM
	case TierRemoteDRAM:
		return TierDisk
	case TierDisk:
		return TierRecompute
	default:
		return TierUnknown
	}
}

// NextWarmerTier returns the next tier UP the local relocation ladder (the promote
// direction). For the hottest tier or an off-ladder tier it returns TierUnknown.
func NextWarmerTier(t ResidencyTier) ResidencyTier {
	for i, tt := range localTierLadder {
		if tt == t && i > 0 {
			return localTierLadder[i-1]
		}
	}
	return TierUnknown
}

// IsLocalTier reports whether a tier is part of the in-box relocation ladder
// (HBM/DRAM/NUMA-far/CXL/Disk) — i.e. a tier a placement policy may demote into or
// promote out of, as opposed to an off-box or synthetic tier.
func IsLocalTier(t ResidencyTier) bool {
	for _, tt := range localTierLadder {
		if tt == t {
			return true
		}
	}
	return false
}
