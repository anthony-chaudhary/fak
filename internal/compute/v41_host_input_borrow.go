package compute

// v41HostInputBorrowBackend is sealed to compute. It certifies synchronous
// read-only consumption of host activation inputs by the exact backend, not a
// general zero-copy interface. Caller ownership and callback admission remain
// the model continuation's responsibility.
type v41HostInputBorrowBackend interface {
	v41HostInputBorrowSafe(Backend) bool
}

// V41HostInputBorrowSafe admits only a canonical backend whose Upload copies
// host values without retaining or mutating the source. Embedded wrappers must
// be refused even when they promote the sealed receiver method.
func V41HostInputBorrowSafe(backend Backend) bool {
	admitted, ok := backend.(v41HostInputBorrowBackend)
	return ok && admitted.v41HostInputBorrowSafe(backend)
}
