package cachemeta

import "errors"

// TransferChannel types the seam a KV transfer backend speaks for moving spans
// between residency tiers. It ADAPTS LMCache's four-method transfer-channel
// contract (LMCache/LMCache lmcache/v1/transfer_channel/abstract.py:21-285@
// 1b7dff2cd83fc634326b5989eb71aa5f05b4f426, Apache-2.0; verdict ADAPT): the four
// batched operations split by peer topology — BatchedSend/BatchedRecv are
// PAIRED (the span only lands when both endpoints act) while BatchedWrite/
// BatchedRead are ONE-SIDED (only the initiator acts against a store that
// outlives the call). This file is the interface and its in-memory fake only:
// no concrete channel moves bytes here and nothing wires it into an existing
// transport; a real backend adapts down to this shape exactly as
// FromKVTransfer does for events, and keeps the house posture of field-only
// records and typed outcomes.

// TransferMode names the peer topology a transfer channel supports.
type TransferMode string

const (
	// TransferModePaired channels move a span between two live peers: a
	// BatchedSend on one side completes only against a BatchedRecv on the other.
	TransferModePaired TransferMode = "paired"
	// TransferModeOneSided channels move a span against a durable store: a
	// BatchedWrite or BatchedRead needs only its initiator.
	TransferModeOneSided TransferMode = "one_sided"
)

// TransferOperation names one of the four batched channel operations. The set
// is closed: Paired() partitions it into the two topologies the guard enforces.
type TransferOperation string

const (
	OpBatchedSend  TransferOperation = "batched_send"
	OpBatchedRecv  TransferOperation = "batched_recv"
	OpBatchedWrite TransferOperation = "batched_write"
	OpBatchedRead  TransferOperation = "batched_read"
)

// Paired reports whether op belongs to the two-peer topology. An operation
// outside the closed set is fail-closed: it is treated as not paired, so the
// only mode that can admit it is one-sided, and even there a channel that
// recognizes no such operation must refuse it itself.
func (o TransferOperation) Paired() bool {
	switch o {
	case OpBatchedSend, OpBatchedRecv:
		return true
	default:
		return false
	}
}

var (
	// ErrPairedCallOnOneSided refuses a BatchedSend/BatchedRecv on a channel
	// whose topology has no second peer to complete it against.
	ErrPairedCallOnOneSided = errors.New("cachemeta: paired transfer call refused on one-sided channel")
	// ErrOneSidedCallOnPaired refuses a BatchedWrite/BatchedRead on a channel
	// whose topology moves spans only between two live peers.
	ErrOneSidedCallOnPaired = errors.New("cachemeta: one-sided transfer call refused on paired channel")
	// ErrUnknownTransferMode refuses every call on a channel reporting a mode
	// outside the closed set: an unrecognized topology is never silently
	// treated as either of the two governed ones.
	ErrUnknownTransferMode = errors.New("cachemeta: unknown transfer channel mode")
)

// CheckTransferCall is the pure guard every compliant channel consults before
// acting: it reports whether op may run on a channel of mode. A mismatch is a
// typed refusal (sentinel error, stable under errors.Is), never a silent
// no-op — sending into a store that only ever receives, or writing into a
// channel that only ever pairs, would otherwise lose the span without trace.
func CheckTransferCall(mode TransferMode, op TransferOperation) error {
	switch mode {
	case TransferModePaired:
		if !op.Paired() {
			return ErrOneSidedCallOnPaired
		}
		return nil
	case TransferModeOneSided:
		if op.Paired() {
			return ErrPairedCallOnOneSided
		}
		return nil
	default:
		return ErrUnknownTransferMode
	}
}

// TransferSpanRequest names one span of a batched channel operation. Field-only
// like KVTransfer: it carries span identity and the residency endpoints the
// span moves between, never payload bytes.
type TransferSpanRequest struct {
	SpanDigest string
	FromTier   ResidencyTier
	ToTier     ResidencyTier
	Owner      string
	Lease      string
	Bytes      int64
}

// TransferSpanResult is the typed outcome of one span in a batch, reusing the
// shared KVTransferOutcome trichotomy so a channel fault stays a FAULT and a
// read that found nothing stays a MISS — never a silent zero.
type TransferSpanResult struct {
	SpanDigest  string
	Outcome     KVTransferOutcome
	FaultReason string
}

// TransferChannel is the seam every KV transfer backend speaks. A compliant
// channel reports its Mode and consults CheckTransferCall before acting,
// refusing the off-topology operation with the matching sentinel error.
// Interface and contract only: implementations decide how (or whether) bytes
// actually move.
type TransferChannel interface {
	Mode() TransferMode
	BatchedSend(reqs []TransferSpanRequest) ([]TransferSpanResult, error)
	BatchedRecv(reqs []TransferSpanRequest) ([]TransferSpanResult, error)
	BatchedWrite(reqs []TransferSpanRequest) ([]TransferSpanResult, error)
	BatchedRead(reqs []TransferSpanRequest) ([]TransferSpanResult, error)
}

// MemoryTransferChannel is the in-memory fake of TransferChannel for tests and
// offline planning: it enforces the mode guard exactly like a real channel
// must, records each accepted operation in call order, and returns an OK result
// per span without moving any bytes. Refusals record nothing, so Calls is
// evidence of what the channel admitted.
type MemoryTransferChannel struct {
	mode  TransferMode
	Calls []TransferOperation
}

// NewMemoryTransferChannel builds a fake channel of the given mode.
func NewMemoryTransferChannel(mode TransferMode) *MemoryTransferChannel {
	return &MemoryTransferChannel{mode: mode}
}

// Mode implements TransferChannel.
func (c *MemoryTransferChannel) Mode() TransferMode { return c.mode }

// ok runs the guard for op and, when admitted, records the call and returns an
// OK result per requested span.
func (c *MemoryTransferChannel) ok(op TransferOperation, reqs []TransferSpanRequest) ([]TransferSpanResult, error) {
	if err := CheckTransferCall(c.mode, op); err != nil {
		return nil, err
	}
	c.Calls = append(c.Calls, op)
	out := make([]TransferSpanResult, len(reqs))
	for i, r := range reqs {
		out[i] = TransferSpanResult{SpanDigest: r.SpanDigest, Outcome: KVTransferOK}
	}
	return out, nil
}

// BatchedSend implements TransferChannel: paired topology only.
func (c *MemoryTransferChannel) BatchedSend(reqs []TransferSpanRequest) ([]TransferSpanResult, error) {
	return c.ok(OpBatchedSend, reqs)
}

// BatchedRecv implements TransferChannel: paired topology only.
func (c *MemoryTransferChannel) BatchedRecv(reqs []TransferSpanRequest) ([]TransferSpanResult, error) {
	return c.ok(OpBatchedRecv, reqs)
}

// BatchedWrite implements TransferChannel: one-sided topology only.
func (c *MemoryTransferChannel) BatchedWrite(reqs []TransferSpanRequest) ([]TransferSpanResult, error) {
	return c.ok(OpBatchedWrite, reqs)
}

// BatchedRead implements TransferChannel: one-sided topology only.
func (c *MemoryTransferChannel) BatchedRead(reqs []TransferSpanRequest) ([]TransferSpanResult, error) {
	return c.ok(OpBatchedRead, reqs)
}
