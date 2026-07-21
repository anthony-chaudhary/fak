package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// openAndSendReceipt opens one registered backend, moves the given source span
// through it, and returns the receive-side receipt. It closes any transport-owned
// resources before returning; the reconstructed span lives in its own destination
// pool and stays valid.
func openAndSendReceipt(t *testing.T, backend KVTransferBackend, cfg Config, src *PagedKV, transfer cachemeta.KVTransfer) PagedKVTransferReceipt {
	t.Helper()
	h, err := OpenKVTransferBackend(KVTransportOpenRequest{Backend: backend, Pool: NewPagedKVPoolWithRaw(cfg, 2)})
	if err != nil {
		t.Fatalf("OpenKVTransferBackend(%s): %v", backend, err)
	}
	if h.Close != nil {
		defer func() { _ = h.Close() }()
	}
	got, err := h.Transport.Send(src, transfer, 0, src.Len())
	if err != nil {
		t.Fatalf("%s Send: %v", backend, err)
	}
	return got
}

// TestReceiptsEquivalentAcrossBackends witnesses the loopback oracle: the shipped
// shm (LocalKVTransport) and tcp (TCPKVTransport) backends move the SAME source
// span to byte-identical receipts. This is the parity a build-tagged hardware
// backend must reproduce; here it is proven on one box with no RDMA fabric.
func TestReceiptsEquivalentAcrossBackends(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	ref := m.NewSession()
	ref.Prefill([]int{5, 8, 13, 21, 34})

	srcPool := NewPagedKVPoolWithRaw(cfg, 2)
	src := snapshotCacheToPaged(srcPool, ref.Cache)

	transfer := cachemeta.KVTransfer{
		Direction: cachemeta.KVMigrate,
		Tokens:    int64(src.Len()),
		ModelID:   "synthetic",
		FromTier:  cachemeta.TierDRAM,
		ToTier:    cachemeta.TierRemote,
	}

	shm := openAndSendReceipt(t, KVTransferBackendSHM, cfg, src, transfer)
	tcp := openAndSendReceipt(t, KVTransferBackendTCP, cfg, src, transfer)

	if shm.Transfer.SpanDigest == "" {
		t.Fatal("shm receipt carried no span digest")
	}
	if err := ReceiptsEquivalent(shm, tcp); err != nil {
		t.Fatalf("shm and tcp receipts are not equivalent: %v", err)
	}
	if err := ReceiptsEquivalent(tcp, shm); err != nil {
		t.Fatalf("receipt equivalence is not symmetric: %v", err)
	}
}

// TestReceiptsEquivalentRejectsMismatch pins the negative direction: any drift in
// the byte contract (digest, moved bytes, serializer, positions) or an empty
// digest is refused, so a wrong backend cannot pass the oracle silently.
func TestReceiptsEquivalentRejectsMismatch(t *testing.T) {
	base := PagedKVTransferReceipt{
		Positions: []int{0, 1, 2},
		Transfer: cachemeta.KVTransfer{
			SpanDigest:   "abc123",
			BytesMoved:   128,
			SerializerID: PagedKVTransferSerializerID,
		},
	}

	if err := ReceiptsEquivalent(base, base); err != nil {
		t.Fatalf("identical receipts must be equivalent: %v", err)
	}

	digest := base
	digest.Transfer.SpanDigest = "def456"
	if err := ReceiptsEquivalent(base, digest); err == nil {
		t.Fatal("span digest mismatch must be refused")
	}

	moved := base
	moved.Transfer.BytesMoved = 256
	if err := ReceiptsEquivalent(base, moved); err == nil {
		t.Fatal("moved-bytes mismatch must be refused")
	}

	serde := base
	serde.Transfer.SerializerID = "foreign-serializer"
	if err := ReceiptsEquivalent(base, serde); err == nil {
		t.Fatal("serializer mismatch must be refused")
	}

	pos := base
	pos.Positions = []int{0, 1, 9}
	if err := ReceiptsEquivalent(base, pos); err == nil {
		t.Fatal("position mismatch must be refused")
	}

	empty := base
	empty.Transfer.SpanDigest = ""
	other := empty
	if err := ReceiptsEquivalent(empty, other); err == nil {
		t.Fatal("an empty span digest must be refused even when both sides are equal")
	}
}
