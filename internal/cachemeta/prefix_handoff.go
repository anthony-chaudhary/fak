package cachemeta

// prefix_handoff.go — prefix-delta KV handoff (sglang-study #5288).
//
// On a node-to-node KV handoff (a teleport #4301 / heterogeneous P/D split #4302),
// the prefix is usually the largest, most-shared part of the KV — re-sending a prefix
// the destination already holds dominates handoff bandwidth. sglang's prefill->decode
// disaggregation avoids that: the DESTINATION advertises how much of the prefix it
// already caches, and the SENDER skips that held prefix and ships only the uncached
// suffix (prefill.py:297-300 / decode.py:974-980 @b8ec5449).
//
// This file is the deterministic, no-hardware half of that handshake: given the
// destination's held-prefix descriptor (a per-block digest chain + a held token count)
// and the sender's full prefix (the same shape), it computes the DELTA — the longest
// common held prefix (in positions) and the suffix range [from, from+n) the sender
// must actually transfer. The value it yields feeds directly into the existing
// MarshalPagedKVTransfer(seq, transfer, from, n) span mover: `from` stops being a bare
// caller guess and becomes the derived divergence offset the destination witnessed.
//
// It moves no bytes and reads no clock: every quantity is a pure function of the two
// injected descriptors, so a replay is reproducible from the descriptors alone. It is
// FAIL-CLOSED — it never claims a longer common prefix than the block digests prove.
// A divergence mid-prefix (a digest mismatch) stops the common run AT that block even
// when the destination advertises a larger held token count, because the tokens past
// the first mismatched block are NOT proven identical and re-sending them is the safe
// choice. This is the sibling of AlmostHit: AlmostHit measures the warm gap ABOVE a
// hit within one store; this measures the delta the destination does not yet hold at
// all, so the sender transfers exactly the missing suffix and no more.

// HeldPrefix is the destination's advertisement of the KV prefix it already holds. The
// destination fills it from its own paged store and sends it to the prospective sender
// BEFORE the handoff, so the sender can skip the overlap. It is the field-only shape;
// no transport is implied.
type HeldPrefix struct {
	// BlockDigests are the per-block content digests of the prefix the destination
	// already holds, in prefix order (block 0 first). Two blocks with equal digests
	// over the same token span are proven identical content.
	BlockDigests []string
	// Tokens is the exact prefix length in positions the destination holds. It may be
	// shorter than len(BlockDigests)*BlockTokens when the final held block is partial;
	// it caps how much common prefix a digest match is allowed to credit.
	Tokens int64
	// BlockTokens is the tokens-per-block paging granularity the digests are computed
	// over. Must match the sender's granularity or the handshake fails closed.
	BlockTokens int
}

// SenderPrefix is the sender's FULL prefix — the span it would otherwise transfer in
// its entirety. It carries the same per-block digest chain so the two sides can be
// compared block-for-block. Tokens is the full prefix length in positions.
type SenderPrefix struct {
	BlockDigests []string
	Tokens       int64
	BlockTokens  int
}

// PrefixDeltaVerdict is the resolved handshake: how much prefix the two sides share,
// where they first diverged, and the exact suffix range the sender must move. It
// records the compared quantities so an observing sink sees WHY the range was chosen
// without re-deriving the walk.
type PrefixDeltaVerdict struct {
	// CommonTokens is the longest PROVEN-common held prefix in positions: the number of
	// leading tokens both sides hold with byte-identical content (block digests equal
	// over the same token span), clamped to what the destination actually advertises
	// holding and to what the sender actually has.
	CommonTokens int64
	// DivergeBlock is the index of the first block whose digests differ (the divergence
	// point the common run stopped at), or -1 when one side's blocks are a clean prefix
	// of the other's with no mismatch (a pure length difference, not a divergence).
	DivergeBlock int
	// TransferFrom is the sender-side offset to start the transfer at — exactly
	// CommonTokens. It is the value that feeds MarshalPagedKVTransfer's `from`.
	TransferFrom int64
	// TransferTokens is the suffix length in positions the sender must actually move:
	// SenderPrefix.Tokens - CommonTokens. Zero means the destination already holds the
	// whole prefix (nothing to send).
	TransferTokens int64
	// DestHoldsNothing is true when the common prefix is empty — the destination is cold
	// for this prefix, so the sender ships the entire prefix.
	DestHoldsNothing bool
	// DestHoldsAll is true when the sender has nothing to send — the destination already
	// holds the whole sender prefix (a full hit).
	DestHoldsAll bool
	// Reason is a stable, metric-readable tag for the verdict.
	Reason string
}

// SendRange returns the [from, n) span the sender should hand to MarshalPagedKVTransfer.
// It is the single value the wiring spends: from = the destination-held divergence
// offset, n = the uncached suffix length.
func (v PrefixDeltaVerdict) SendRange() (from, n int) {
	return int(v.TransferFrom), int(v.TransferTokens)
}

// ResolvePrefixDelta computes the prefix-delta handoff verdict from the destination's
// held-prefix advertisement and the sender's full prefix. It walks the two block-digest
// chains from the front, counting a block as common only while both digests are equal;
// the first mismatch (or the shorter chain running out) ends the common run. The common
// prefix in positions is then clamped to what the destination advertises holding and to
// the sender's own length, so a matching digest can never credit more tokens than either
// side actually has. The sender transfers the suffix [CommonTokens, SenderPrefix.Tokens).
//
// Fail-closed cases:
//   - mismatched or non-positive BlockTokens: the two chains are not comparable, so
//     nothing is proven common and the sender ships the whole prefix.
//   - a digest mismatch mid-prefix: the common run stops AT that block even if the
//     destination advertises a larger held token count — the tokens past the first
//     mismatch are not proven identical, so re-sending them is the safe choice.
func ResolvePrefixDelta(held HeldPrefix, sender SenderPrefix) PrefixDeltaVerdict {
	senderTokens := sender.Tokens
	if senderTokens < 0 {
		senderTokens = 0
	}
	full := func(reason string) PrefixDeltaVerdict {
		return PrefixDeltaVerdict{
			CommonTokens:     0,
			DivergeBlock:     -1,
			TransferFrom:     0,
			TransferTokens:   senderTokens,
			DestHoldsNothing: true,
			DestHoldsAll:     senderTokens == 0,
			Reason:           reason,
		}
	}

	// Not comparable: the block granularities disagree or are degenerate. Nothing is
	// proven common; ship the whole prefix.
	if sender.BlockTokens <= 0 || held.BlockTokens != sender.BlockTokens {
		return full("block_granularity_mismatch")
	}
	blockTokens := int64(sender.BlockTokens)

	heldTokens := held.Tokens
	if heldTokens < 0 {
		heldTokens = 0
	}

	// Walk the two chains front-to-back, counting proven-common leading blocks.
	matchable := len(held.BlockDigests)
	if len(sender.BlockDigests) < matchable {
		matchable = len(sender.BlockDigests)
	}
	commonBlocks := 0
	divergeBlock := -1
	for i := 0; i < matchable; i++ {
		if held.BlockDigests[i] != sender.BlockDigests[i] {
			divergeBlock = i
			break
		}
		commonBlocks++
	}

	// Common prefix in positions: the token end of the last common block, clamped so a
	// digest match never credits more than either side actually holds. This also folds
	// the partial-final-block case: if one side's final block covers fewer tokens, the
	// clamp stops the common extent at the shorter side's boundary.
	commonTokens := int64(commonBlocks) * blockTokens
	if commonTokens > heldTokens {
		commonTokens = heldTokens
	}
	if commonTokens > senderTokens {
		commonTokens = senderTokens
	}
	if commonTokens < 0 {
		commonTokens = 0
	}

	transferTokens := senderTokens - commonTokens
	if transferTokens < 0 {
		transferTokens = 0
	}

	v := PrefixDeltaVerdict{
		CommonTokens:     commonTokens,
		DivergeBlock:     divergeBlock,
		TransferFrom:     commonTokens,
		TransferTokens:   transferTokens,
		DestHoldsNothing: commonTokens == 0,
		DestHoldsAll:     transferTokens == 0,
	}
	switch {
	case commonTokens == 0:
		v.Reason = "dest_holds_nothing_send_all"
	case transferTokens == 0:
		v.Reason = "dest_holds_all_send_nothing"
	case divergeBlock >= 0:
		v.Reason = "diverged_send_suffix_from_divergence"
	default:
		v.Reason = "partial_overlap_send_suffix"
	}
	return v
}
