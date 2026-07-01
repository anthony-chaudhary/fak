package vcachechain

import "testing"

// warm_descriptor_test.go pins the #1611 witness at the vcachechain layer: a
// WarmPrefixDescriptor computed over a prefix must digest-equal a descriptor
// recomputed over a byte-identical replay, and must NOT digest-equal one recomputed
// over a genuinely different replay (the discrimination the digest exists to prove).

// TestWarmPrefixDescriptorDigestEqualityOnIdenticalReplay proves the core witness:
// capture a descriptor from an original prefix, "replay" the identical bytes, and
// assert VerifyWarmPrefixReplay accepts it.
func TestWarmPrefixDescriptorDigestEqualityOnIdenticalReplay(t *testing.T) {
	original := []byte("You are a helpful coding assistant for the fak repo.\n[tool schema block]")
	spans := []SpanBoundary{
		{Kind: SpanSystem, Start: 0, End: 54},
		{Kind: SpanTools, Start: 54, End: len(original)},
	}
	desc := DescribeWarmPrefix(original, spans)

	if desc.Digest == "" {
		t.Fatal("descriptor digest is empty")
	}
	if desc.ByteLen != len(original) {
		t.Fatalf("descriptor ByteLen = %d, want %d", desc.ByteLen, len(original))
	}
	if len(desc.Spans) != 2 {
		t.Fatalf("descriptor Spans = %d, want 2", len(desc.Spans))
	}

	// Simulate a reset + rehydrate: an independent byte copy, identical content.
	replayed := append([]byte(nil), original...)
	redigested := DescribeWarmPrefix(replayed, nil)

	if redigested.Digest != desc.Digest {
		t.Fatalf("digest not stable across an identical replay: original=%s replayed=%s",
			desc.Digest, redigested.Digest)
	}
	if !VerifyWarmPrefixReplay(desc, replayed) {
		t.Fatal("VerifyWarmPrefixReplay rejected a byte-identical replay")
	}
}

// TestWarmPrefixDescriptorDiscriminatesDifferentReplay proves the negative case: the
// digest is a genuine discriminator, not an always-true rubber stamp. A replay that
// produces DIFFERENT bytes (even same length, even a one-byte change) must NOT verify.
func TestWarmPrefixDescriptorDiscriminatesDifferentReplay(t *testing.T) {
	original := []byte("You are a helpful coding assistant for the fak repo.")
	desc := DescribeWarmPrefix(original, nil)

	sameLenDifferentContent := []byte("You are a HELPFUL coding assistant for the fak repo.")
	if len(sameLenDifferentContent) != len(original) {
		t.Fatalf("test fixture bug: fixtures must be same length, got %d vs %d",
			len(sameLenDifferentContent), len(original))
	}
	if VerifyWarmPrefixReplay(desc, sameLenDifferentContent) {
		t.Fatal("VerifyWarmPrefixReplay accepted a same-length but content-different replay")
	}

	truncated := original[:len(original)-1]
	if VerifyWarmPrefixReplay(desc, truncated) {
		t.Fatal("VerifyWarmPrefixReplay accepted a truncated replay")
	}

	extended := append(append([]byte(nil), original...), '!')
	if VerifyWarmPrefixReplay(desc, extended) {
		t.Fatal("VerifyWarmPrefixReplay accepted an extended replay")
	}

	empty := []byte{}
	if VerifyWarmPrefixReplay(desc, empty) {
		t.Fatal("VerifyWarmPrefixReplay accepted an empty replay against a non-empty original")
	}
}

// TestWarmPrefixDescriptorSpansDoNotAffectDigest proves Spans is pure metadata: two
// descriptors over identical bytes digest-equal regardless of how (or whether) the
// caller annotated span boundaries.
func TestWarmPrefixDescriptorSpansDoNotAffectDigest(t *testing.T) {
	prefix := []byte("stable system preamble plus tool schema")
	noSpans := DescribeWarmPrefix(prefix, nil)
	withSpans := DescribeWarmPrefix(prefix, []SpanBoundary{
		{Kind: SpanSystem, Start: 0, End: 23},
		{Kind: SpanTools, Start: 23, End: len(prefix)},
	})
	if noSpans.Digest != withSpans.Digest {
		t.Fatalf("span annotations changed the digest: %s vs %s", noSpans.Digest, withSpans.Digest)
	}
	if noSpans.ByteLen != withSpans.ByteLen {
		t.Fatalf("span annotations changed ByteLen: %d vs %d", noSpans.ByteLen, withSpans.ByteLen)
	}
}
