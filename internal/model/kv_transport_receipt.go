package model

import "fmt"

// ReceiptsEquivalent reports whether two paged-KV transfer receipts describe the
// SAME moved span under the native byte contract: identical span checksum
// (SpanDigest), moved-byte count, serializer axis, and restored positions. It
// returns nil when the two receipts are equivalent and a typed diff otherwise.
//
// This is the loopback oracle the KV transport matrix's witness column names
// ("registered backend must emit the same checksum/lease receipt", see the
// ucx-rdma / nvme-of / object rows in kv_transport_registry.go). A UCX/RDMA,
// NVMe-oF, or object backend registered behind its fail-closed row is correct
// ONLY if, for the same source span, its receipt is ReceiptsEquivalent to the
// shipped shm/tcp loopback receipt. The shipped LocalKVTransport (shm) and
// TCPKVTransport already agree by construction; this makes that agreement a
// reusable, callable check so a build-tagged hardware backend proves parity with
// the loopback receipt instead of asserting it ad hoc.
//
// The check moves zero bytes and reads only the descriptor that already crossed
// with the span, so it is deterministic, wall-clock-free, and witnessable on one
// box with no RDMA fabric.
func ReceiptsEquivalent(a, b PagedKVTransferReceipt) error {
	if a.Transfer.SpanDigest == "" || b.Transfer.SpanDigest == "" {
		return fmt.Errorf("model: KV transfer receipt has empty span digest (a=%q b=%q)", a.Transfer.SpanDigest, b.Transfer.SpanDigest)
	}
	if a.Transfer.SpanDigest != b.Transfer.SpanDigest {
		return fmt.Errorf("model: KV transfer receipt span digest mismatch: %q != %q", a.Transfer.SpanDigest, b.Transfer.SpanDigest)
	}
	if a.Transfer.BytesMoved != b.Transfer.BytesMoved {
		return fmt.Errorf("model: KV transfer receipt moved-bytes mismatch: %d != %d", a.Transfer.BytesMoved, b.Transfer.BytesMoved)
	}
	if a.Transfer.SerializerID != b.Transfer.SerializerID {
		return fmt.Errorf("model: KV transfer receipt serializer mismatch: %q != %q", a.Transfer.SerializerID, b.Transfer.SerializerID)
	}
	if len(a.Positions) != len(b.Positions) {
		return fmt.Errorf("model: KV transfer receipt position count mismatch: %d != %d", len(a.Positions), len(b.Positions))
	}
	for i := range a.Positions {
		if a.Positions[i] != b.Positions[i] {
			return fmt.Errorf("model: KV transfer receipt position[%d] mismatch: %d != %d", i, a.Positions[i], b.Positions[i])
		}
	}
	return nil
}
