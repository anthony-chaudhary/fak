package cachemeta

import "testing"

// Test the four prefix-delta handoff cases from #5288: destination holds nothing (send
// everything), destination holds all (send nothing), partial overlap (send exactly the
// suffix), and a mid-prefix digest divergence (fail-closed to the divergence point).

func TestResolvePrefixDeltaHoldsNothing(t *testing.T) {
	sender := SenderPrefix{
		BlockDigests: []string{"s0", "s1", "s2"},
		Tokens:       10,
		BlockTokens:  4,
	}
	held := HeldPrefix{BlockTokens: 4} // no blocks, no tokens
	v := ResolvePrefixDelta(held, sender)

	if !v.DestHoldsNothing || v.DestHoldsAll {
		t.Fatalf("cold destination: got DestHoldsNothing=%v DestHoldsAll=%v", v.DestHoldsNothing, v.DestHoldsAll)
	}
	if v.CommonTokens != 0 {
		t.Fatalf("common tokens = %d, want 0", v.CommonTokens)
	}
	from, n := v.SendRange()
	if from != 0 || n != 10 {
		t.Fatalf("send range = [%d,+%d), want [0,+10)", from, n)
	}
}

func TestResolvePrefixDeltaHoldsAll(t *testing.T) {
	digests := []string{"s0", "s1", "s2"}
	sender := SenderPrefix{BlockDigests: digests, Tokens: 10, BlockTokens: 4}
	held := HeldPrefix{BlockDigests: digests, Tokens: 10, BlockTokens: 4}
	v := ResolvePrefixDelta(held, sender)

	if !v.DestHoldsAll {
		t.Fatalf("full hit: got DestHoldsAll=%v reason=%q", v.DestHoldsAll, v.Reason)
	}
	if v.CommonTokens != 10 {
		t.Fatalf("common tokens = %d, want 10", v.CommonTokens)
	}
	from, n := v.SendRange()
	if from != 10 || n != 0 {
		t.Fatalf("send range = [%d,+%d), want [10,+0)", from, n)
	}
	if v.DivergeBlock != -1 {
		t.Fatalf("diverge block = %d, want -1 (no divergence)", v.DivergeBlock)
	}
}

func TestResolvePrefixDeltaPartialOverlap(t *testing.T) {
	sender := SenderPrefix{
		BlockDigests: []string{"s0", "s1", "s2"}, // 10 tokens over 4-token blocks: [0,4)[4,8)[8,10)
		Tokens:       10,
		BlockTokens:  4,
	}
	held := HeldPrefix{
		BlockDigests: []string{"s0", "s1"}, // holds the first two blocks
		Tokens:       8,
		BlockTokens:  4,
	}
	v := ResolvePrefixDelta(held, sender)

	if v.DestHoldsNothing || v.DestHoldsAll {
		t.Fatalf("partial overlap should be neither cold nor full: %+v", v)
	}
	if v.CommonTokens != 8 {
		t.Fatalf("common tokens = %d, want 8", v.CommonTokens)
	}
	from, n := v.SendRange()
	if from != 8 || n != 2 {
		t.Fatalf("send range = [%d,+%d), want [8,+2) (exactly the suffix)", from, n)
	}
}

func TestResolvePrefixDeltaDivergenceFailsClosed(t *testing.T) {
	sender := SenderPrefix{
		BlockDigests: []string{"s0", "s1", "s2"},
		Tokens:       10,
		BlockTokens:  4,
	}
	// Destination ADVERTISES 8 held tokens, but its block 1 digest diverges from the
	// sender's. Fail-closed: credit only the proven-common block 0 (4 tokens), NOT the
	// advertised 8 — the tokens past the first mismatch are not proven identical.
	held := HeldPrefix{
		BlockDigests: []string{"s0", "DIFFERENT", "s2"},
		Tokens:       8,
		BlockTokens:  4,
	}
	v := ResolvePrefixDelta(held, sender)

	if v.DivergeBlock != 1 {
		t.Fatalf("diverge block = %d, want 1", v.DivergeBlock)
	}
	if v.CommonTokens != 4 {
		t.Fatalf("common tokens = %d, want 4 (fail-closed to divergence, not advertised 8)", v.CommonTokens)
	}
	from, n := v.SendRange()
	if from != 4 || n != 6 {
		t.Fatalf("send range = [%d,+%d), want [4,+6)", from, n)
	}
	// A later matching block (s2 == s2) must NOT be credited across the divergence.
	if v.CommonTokens+v.TransferTokens != sender.Tokens {
		t.Fatalf("common(%d)+transfer(%d) != sender tokens(%d)", v.CommonTokens, v.TransferTokens, sender.Tokens)
	}
}

func TestResolvePrefixDeltaGranularityMismatchFailsClosed(t *testing.T) {
	sender := SenderPrefix{BlockDigests: []string{"s0", "s1"}, Tokens: 8, BlockTokens: 4}
	held := HeldPrefix{BlockDigests: []string{"s0", "s1"}, Tokens: 8, BlockTokens: 8} // different granularity
	v := ResolvePrefixDelta(held, sender)

	if v.Reason != "block_granularity_mismatch" {
		t.Fatalf("reason = %q, want block_granularity_mismatch", v.Reason)
	}
	from, n := v.SendRange()
	if from != 0 || n != 8 {
		t.Fatalf("send range = [%d,+%d), want [0,+8) (ship everything)", from, n)
	}
}
