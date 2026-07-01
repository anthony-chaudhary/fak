package sessionreset

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachechain"
)

// warm_prefix_replay_test.go is the #1611 witness: a reset seed's warm-prefix
// descriptor must let a consumer verify a REPLAYED prefix is byte-identical to the
// ORIGINAL prefix captured before the reset — and must genuinely discriminate a
// replay that is NOT byte-identical, not rubber-stamp every replay as a match.

// buildOriginalSession returns a transcript with a system preamble stable enough to
// be worth warm-prefix replay, plus ordinary user/assistant turns so the other
// contributors also have something to fold.
func buildOriginalSession() []Msg {
	return []Msg{
		{Role: "system", Content: "You are a helpful coding assistant for the fak repo. Follow the repo conventions exactly."},
		{Role: "user", Content: "Add a warm-prefix descriptor to the reset seed."},
		{Role: "assistant", Content: "On it."},
		{Role: "user", Content: "Make sure the digest actually discriminates."},
	}
}

// rehydratePrefix simulates a consumer replaying/rehydrating the stable prefix from
// the vCache prefix-DAG after a reset: it reconstructs the prefix bytes the fresh
// session would splice back in. Here that is simply re-deriving the system preamble
// from the (still-held, for test purposes) original messages — standing in for a
// real rehydration off the prefix-DAG, which this package does not itself perform
// (that is the session.WarmKVStore live-splice follow-on).
func rehydratePrefix(msgs []Msg) []byte {
	return []byte(systemPreamble(msgs))
}

// TestWarmPrefixDescriptorWitnessesByteIdenticalReplay is the done-condition witness:
// build an original prefix, compute its descriptor via BuildSeed, simulate a reset,
// rehydrate/replay the prefix, recompute the descriptor, and assert digest equality.
func TestWarmPrefixDescriptorWitnessesByteIdenticalReplay(t *testing.T) {
	original := buildOriginalSession()

	// 1. Compute the descriptor BEFORE the reset, as part of building the seed.
	seed := BuildSeed(Input{Messages: original})
	if seed.WarmPrefix == nil {
		t.Fatal("BuildSeed produced no WarmPrefix descriptor for a transcript with a system preamble")
	}
	originalDesc := *seed.WarmPrefix
	if originalDesc.Digest == "" {
		t.Fatal("warm-prefix descriptor has an empty digest")
	}
	if originalDesc.ByteLen != len(systemPreamble(original)) {
		t.Fatalf("warm-prefix descriptor ByteLen = %d, want %d", originalDesc.ByteLen, len(systemPreamble(original)))
	}

	// 2. Simulate the reset: the drained transcript is gone; only the seed (and its
	// descriptor) survives into the fresh session.
	survivingDescriptor := originalDesc

	// 3. Rehydrate/replay the stable prefix (e.g. from the vCache prefix-DAG) in the
	// fresh session. In this test the replay path reconstructs identical bytes.
	replayed := rehydratePrefix(original)

	// 4. Recompute the descriptor over the replayed bytes and assert digest equality
	// — the witness the issue title names.
	replayedDesc := vcachechain.DescribeWarmPrefix(replayed, nil)
	if replayedDesc.Digest != survivingDescriptor.Digest {
		t.Fatalf("replayed prefix digest %s does not equal original prefix digest %s",
			replayedDesc.Digest, survivingDescriptor.Digest)
	}
	if !vcachechain.VerifyWarmPrefixReplay(survivingDescriptor, replayed) {
		t.Fatal("VerifyWarmPrefixReplay rejected a byte-identical rehydrated prefix")
	}
}

// TestWarmPrefixDescriptorRejectsDivergedReplay is the negative witness: a replay
// that produces a prefix DIFFERENT from the original (e.g. the system preamble drifted,
// or a rehydration bug spliced the wrong span) must NOT show equal digests. Without
// this, a descriptor that always "verified" would be worthless — this proves the
// digest is a genuine discriminator.
func TestWarmPrefixDescriptorRejectsDivergedReplay(t *testing.T) {
	original := buildOriginalSession()
	seed := BuildSeed(Input{Messages: original})
	if seed.WarmPrefix == nil {
		t.Fatal("BuildSeed produced no WarmPrefix descriptor for a transcript with a system preamble")
	}
	desc := *seed.WarmPrefix

	// A "replay" that rehydrated a DIFFERENT session's preamble (the failure mode a
	// buggy consumer could actually hit: wrong cache key, stale prefix, cross-session
	// bleed).
	diverged := buildOriginalSession()
	diverged[0].Content = "You are a helpful coding assistant for a DIFFERENT repo entirely."
	replayed := rehydratePrefix(diverged)

	if vcachechain.VerifyWarmPrefixReplay(desc, replayed) {
		t.Fatal("VerifyWarmPrefixReplay accepted a replay of a different prefix — the digest must discriminate")
	}

	replayedDesc := vcachechain.DescribeWarmPrefix(replayed, nil)
	if replayedDesc.Digest == desc.Digest {
		t.Fatal("digest equality held across two genuinely different prefixes")
	}

	// A byte-for-byte truncation of the ORIGINAL prefix (a partial rehydration) must
	// also fail to verify.
	truncated := rehydratePrefix(original)
	truncated = truncated[:len(truncated)/2]
	if vcachechain.VerifyWarmPrefixReplay(desc, truncated) {
		t.Fatal("VerifyWarmPrefixReplay accepted a truncated replay of the original prefix")
	}
}

// TestWarmPrefixDescriptorAbsentWithoutSystemPreamble proves BuildSeed does not
// fabricate a descriptor when the transcript has nothing stable to describe —
// mirroring the warmPrefix contributor's own decline case.
func TestWarmPrefixDescriptorAbsentWithoutSystemPreamble(t *testing.T) {
	seed := BuildSeed(Input{Messages: []Msg{{Role: "user", Content: "hi"}}})
	if seed.WarmPrefix != nil {
		t.Fatalf("expected no WarmPrefix descriptor without a system preamble, got %+v", *seed.WarmPrefix)
	}
}
