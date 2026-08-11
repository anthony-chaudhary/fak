// Package fabricmap plans transfers over a directed graph of storage, memory,
// compute, and fabric endpoints without assigning semantic meaning to tier names.
//
// Tier: mechanism (2) - see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
//
// Reservable links declare both capacity and shared-resource identity. Allocator
// performs capacity-aware route selection and all resource debits atomically; its
// FIFO ticket order, explicit expiry clock, and idempotent release are deterministic.
package fabricmap
