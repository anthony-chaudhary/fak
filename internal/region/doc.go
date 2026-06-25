// Package region provides a typed, in-process one-sided shared window over
// abi.Ref values. It is the MPI_Win-shaped primitive for fak's existing
// Resolver seam: Put stores bytes through Resolver.Put, Get materializes them
// through Resolver.Resolve, and Accumulate performs a deterministic
// read-modify-write fold over a Ref.
//
// The important boundary is the kernel admission point. Put and Accumulate build
// synthetic region.put / region.accumulate ToolCalls and submit them through the
// caller's abi.Kernel before the Resolver write happens. Get is submitted as a
// read-shaped region.get call before Resolve. The leaf performs the local effect
// after an Allow verdict; denied calls do not write.
//
// Scope and taint ride on the Ref. A write can never widen a Window's current
// scope or exceed ScopeFleet, so the default ScopeAgent remains fail-closed and
// an explicit fleet share is the widest region write this package admits.
// Successful writes emit a write-shaped completion to the configured vDSO
// coherence observer, which bumps the epoch and strands stale tier-2 reads.
//
// This deliberately borrows one-sided window vocabulary, not RDMA semantics. It
// is not remote DMA, not a hardware zero-copy path, and not an MPI memory-model
// guarantee. The v0 implementation is a kernel-adjudicated Go call over the
// copy-CAS Resolver; Accumulate is an in-process deterministic fold, not a
// hardware atomic, and it carries no HPC bandwidth claim.
package region
