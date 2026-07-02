package compute

// farmem.go — the far-memory capacity probes behind the NUMA-far and CXL cache tiers
// (#1470, phase-2 child P2-4 of the L2/L3 cache storage spine #1463). The tier model
// (internal/cachemeta/hardware.go) made NUMA-far and CXL first-class demote targets —
// byte-addressable, attendable in place, the tiers that make "don't evict, relocate"
// pay — but until this probe existed nothing could confirm a box actually HAS far
// memory, so the placement plane planned those tiers against an assumed-empty
// placeholder pressure of 0 and a representative capacity.
//
// These are the far-memory siblings of HostSystemMemoryInfo (DRAM) and DiskInfo
// (disk): backend-free process-host probes that report (total, free, known) and FAIL
// OPEN — a platform without a probe, or a box without far memory, reports known=false
// and the caller keeps today's behavior. No number is ever fabricated: a tier this
// probe cannot confirm simply stays unconfirmed (the #1470 fence).
//
// Sourcing (linux only today, farmem_linux.go): the kernel's NUMA topology under
// /sys/devices/system/node. CPU-ful nodes beyond the first are the far-NUMA tier
// (another socket's DRAM); MEMORY-ONLY nodes (a node with memory but no CPUs) are how
// the kernel represents CPU-less expansion memory onlined as system RAM — CXL.mem
// being the canonical instance — and feed the CXL tier. Other platforms return
// known=false (farmem_other.go).

// NUMAFarMemoryInfo reports total/free bytes of byte-addressable memory on FAR NUMA
// nodes: CPU-ful nodes other than the local (lowest-numbered CPU-ful) node. It is for
// far-memory-tier cache placement against real fullness; a single-socket box, a box
// whose topology cannot be read, or an unsupported platform returns known=false,
// preserving the fail-open capacity contract.
func NUMAFarMemoryInfo() (total, free int64, known bool) {
	return numaFarMemory()
}

// CXLMemoryInfo reports total/free bytes of CPU-less expansion memory: memory-only
// NUMA nodes, the kernel's representation of CXL.mem (and kin) onlined as system RAM.
// It is for CXL-tier cache placement against real fullness; a box with no memory-only
// node or an unsupported platform returns known=false, preserving the fail-open
// capacity contract.
func CXLMemoryInfo() (total, free int64, known bool) {
	return cxlMemory()
}
