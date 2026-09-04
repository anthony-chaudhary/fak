// Package cdb provides the context debugger and core image inspection interface.
//
// Invariant: context database page operations are fail-closed and sealed-safe.
// Sealed or quarantined pages are never demand-paged into active context without
// an explicit witness clearance and fresh gate re-screening.
//
// Guard: page table lookups and demand-paging enforce boundary integrity,
// ensuring content-addressed CAS blobs match digest addresses and refusing
// tampered swap data.
package cdb
