package vcachechain

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// warm_descriptor.go — the warm-prefix descriptor (issue #1611, child of the
// managed-context epic #1570). A reset that silently destroys cache value or makes
// cost unpredictable is not a survivable reset. This file gives a reset-seed builder
// a small, self-contained object that names the STABLE prefix a fresh session should
// be able to replay/rehydrate from the vCache prefix-DAG instead of re-paying a cold
// prefill for the system preamble + tools + any sealed span it already warmed.
//
// SCOPE. This is a descriptor, not a mover: it records what the prefix WAS (a content
// digest, its byte length, and the span boundaries within it) so a consumer can prove
// a later replay reproduced it byte-for-byte. It does not itself move KV state — that
// remains the session.WarmKVStore live-splice follow-on sessionreset's warmPrefix
// contributor already documents as "deferred" until wired (MarkLiveKVReuse).
//
// DIGEST REUSE. The digest is cachemeta.DigestBytes (SHA-256, lowercase hex) — the
// SAME canonical content digest cachemeta/vcachegov already use for manifest binding
// and attention-index binding — so a warm-prefix descriptor's digest is comparable
// against any other cachemeta-digested payload instead of inventing a second hash
// convention for the same job.
//
// Pure and deterministic: no clock, no network, no randomness. Tier: mechanism (2),
// same as the rest of vcachechain — it imports only cachemeta (tier 1) and stdlib.

// SpanKind classifies one contiguous span of the descriptor's stable prefix. It
// mirrors cachemeta's SegmentKind naming (stable / tool schema / sealed) so a reader
// already familiar with the §A3 prefix-stability linter recognizes the vocabulary,
// without importing cachemeta's PromptSegment type itself (this descriptor only
// needs boundaries, not the §A3 divergence machinery).
type SpanKind string

const (
	// SpanSystem is the system preamble / static instructions — the bulk of a stable
	// prefix, expected byte-identical across resets.
	SpanSystem SpanKind = "system"
	// SpanTools is the tool/function schema block — stable, belongs ahead of any
	// per-turn content.
	SpanTools SpanKind = "tools"
	// SpanSealed is a span fak quarantined before the reset. It is included in the
	// digest (a sealed span still participates in prefix identity) but a consumer
	// must not re-serve it verbatim across a trust boundary — the same refusal rule
	// §A3/§A4 already enforce for the live prefix-stability linter.
	SpanSealed SpanKind = "sealed"
)

// SpanBoundary marks one named span within the descriptor's stable-prefix bytes, as
// a half-open byte range [Start, End). Boundaries let a consumer verify not just
// "the whole prefix matches" but which part of a mismatch is responsible (e.g. the
// tool schema changed but the system preamble did not).
type SpanBoundary struct {
	Kind  SpanKind `json:"kind"`
	Start int      `json:"start"`
	End   int      `json:"end"`
}

// WarmPrefixDescriptor is the reset-seed's warm-prefix carryover object: enough
// metadata to let a consumer verify a replayed/rehydrated prefix is byte-identical
// to the original by comparing digests, without carrying the prefix bytes themselves
// in the descriptor (the descriptor is metadata, not a payload cache).
type WarmPrefixDescriptor struct {
	// Digest is the lowercase hex SHA-256 (cachemeta.DigestBytes) over the full
	// stable-prefix bytes at capture time — the content address a replay must
	// reproduce to count as byte-identical.
	Digest string `json:"digest"`
	// ByteLen is len(prefix) in bytes at capture time, redundant with Digest but
	// cheap to check first (a length mismatch is a digest mismatch without hashing).
	ByteLen int `json:"byte_len"`
	// Spans are the named sub-ranges within the prefix (system / tools / sealed),
	// in byte order. Optional: a caller with only an undifferentiated prefix (e.g.
	// the system preamble alone) may pass a single SpanSystem span covering the
	// whole range, or none at all — Spans does not affect Digest/ByteLen.
	Spans []SpanBoundary `json:"spans,omitempty"`
}

// DescribeWarmPrefix computes a WarmPrefixDescriptor over prefix — the stable prefix
// bytes BEFORE a reset — and the caller-supplied span boundaries (pass nil/empty when
// the caller has no finer-grained span breakdown). It is pure: the same bytes always
// yield the same descriptor, so two independent captures of an identical prefix are
// digest-equal.
func DescribeWarmPrefix(prefix []byte, spans []SpanBoundary) WarmPrefixDescriptor {
	return WarmPrefixDescriptor{
		Digest:  cachemeta.DigestBytes(prefix),
		ByteLen: len(prefix),
		Spans:   append([]SpanBoundary(nil), spans...),
	}
}

// VerifyWarmPrefixReplay reports whether replayed reproduces the descriptor's
// original stable prefix byte-for-byte: it recomputes the SAME digest function over
// replayed and compares against want.Digest (and, as a cheap first rung, ByteLen).
// This is the witness the reset seed's warm-prefix descriptor exists to make
// possible — a consumer that rehydrated a prefix from the vCache prefix-DAG (or any
// other replay path) can prove the rehydration was faithful without diffing raw
// bytes against a copy it may not have kept.
//
// The comparison is deliberately over CONTENT only (digest), never identity or
// length alone accepted as sufficient: two different byte strings of the same length
// must not be accepted as a match, which is exactly what makes this a genuine
// discriminator rather than an always-true rubber stamp.
func VerifyWarmPrefixReplay(want WarmPrefixDescriptor, replayed []byte) bool {
	if len(replayed) != want.ByteLen {
		return false
	}
	return cachemeta.DigestBytes(replayed) == want.Digest
}
