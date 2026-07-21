package decodemigrate

import (
	"bytes"
	"testing"
)

// TestCheckpointMarshalRoundTrip proves the wire form is a faithful, deterministic
// serde: a captured checkpoint survives Marshal→Unmarshal unchanged, the bytes are
// stable across repeated marshals, and a wrong-length payload is refused rather
// than silently zero-filled.
func TestCheckpointMarshalRoundTrip(t *testing.T) {
	s := NewDecodeStream(20260720)
	for i := 0; i < 40; i++ {
		s.Next()
	}
	cp := s.Capture()

	wire := cp.Marshal()
	if got := len(wire); got != checkpointBytes {
		t.Fatalf("marshal length = %d, want %d", got, checkpointBytes)
	}
	if !bytes.Equal(wire, cp.Marshal()) {
		t.Fatalf("marshal is not deterministic: two marshals of the same checkpoint differ")
	}

	back, err := UnmarshalCheckpoint(wire)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back != cp {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", back, cp)
	}

	if _, err := UnmarshalCheckpoint(wire[:checkpointBytes-1]); err == nil {
		t.Fatalf("expected error on truncated checkpoint, got nil")
	}
	if _, err := UnmarshalCheckpoint(append(wire, 0)); err == nil {
		t.Fatalf("expected error on over-long checkpoint, got nil")
	}
}

// reference runs a single un-migrated instance for n steps, returning the token
// stream and the per-step checkpoint stream (the full RNG/history state captured
// after each draw). This is the ground truth a faithful migration must reproduce.
func reference(seed int64, n int) (tokens []int, states [][]byte) {
	s := NewDecodeStream(seed)
	tokens = make([]int, 0, n)
	states = make([][]byte, 0, n)
	for i := 0; i < n; i++ {
		tokens = append(tokens, s.Next())
		states = append(states, s.Capture().Marshal())
	}
	return tokens, states
}

// TestDecodeMigrationReplayEquivalent is the headline replay-equivalence witness
// for #4307: a 200-token generation is migrated mid-stream from instance A onto a
// FRESH instance B, and the full post-migration continuation is asserted
// bit-identical to a run that never migrated — same token stream AND same
// per-step RNG/history state. The migration is invisible in the output.
func TestDecodeMigrationReplayEquivalent(t *testing.T) {
	const (
		seed  = int64(4307)
		total = 200 // the issue's 200-token in-flight generation
		hopAt = 137 // migrate mid-stream, on a boundary that isn't step 0 or the end
	)

	refTokens, refStates := reference(seed, total)

	// Instance A: decode up to the mid-stream hop point.
	srcA := NewDecodeStream(seed)
	migTokens := make([]int, 0, total)
	migStates := make([][]byte, 0, total)
	for i := 0; i < hopAt; i++ {
		migTokens = append(migTokens, srcA.Next())
		migStates = append(migStates, srcA.Capture().Marshal())
	}

	// The hop: capture A's live state, teleport the bytes, rehydrate a fresh
	// instance B. A is dead past this point — only what round-tripped through the
	// wire survives, exactly as on another machine.
	dstB, err := Migrate(srcA)
	if err != nil {
		t.Fatalf("migrate: %v", err)
	}
	if dstB == srcA {
		t.Fatalf("destination instance must be fresh, not aliased to the source")
	}
	if got := dstB.Step(); got != hopAt {
		t.Fatalf("restored step index = %d, want %d", got, hopAt)
	}

	// Instance B: resume decoding to completion.
	for i := hopAt; i < total; i++ {
		migTokens = append(migTokens, dstB.Next())
		migStates = append(migStates, dstB.Capture().Marshal())
	}

	// Bit-identical token stream: the migration left no seam.
	if len(migTokens) != len(refTokens) {
		t.Fatalf("migrated stream length = %d, want %d", len(migTokens), len(refTokens))
	}
	for i := range refTokens {
		if migTokens[i] != refTokens[i] {
			t.Fatalf("token stream diverged at step %d (hop at %d): migrated=%d, un-migrated=%d",
				i, hopAt, migTokens[i], refTokens[i])
		}
	}

	// Bit-identical RNG/history state at every step, most importantly after the
	// hop: proves it is not merely the visible tokens that match but the entire
	// resumable state — the stream could be migrated again and still stay exact.
	for i := range refStates {
		if !bytes.Equal(migStates[i], refStates[i]) {
			t.Fatalf("resumable state diverged at step %d (hop at %d)", i, hopAt)
		}
	}

	t.Logf("replay-equivalent: migrated %d-token generation at step %d onto a fresh instance; "+
		"token stream and per-step RNG state are bit-identical to the un-migrated run", total, hopAt)
}

// TestMigrationAtEveryStepEquivalent strengthens the headline: migrating at ANY
// step k reproduces the un-migrated stream exactly. A correct hop is invisible for
// every k, so the equivalence cannot depend on a lucky boundary.
func TestMigrationAtEveryStepEquivalent(t *testing.T) {
	const (
		seed  = int64(99)
		total = 32
	)
	refTokens, _ := reference(seed, total)

	for k := 0; k <= total; k++ {
		src := NewDecodeStream(seed)
		got := make([]int, 0, total)
		for i := 0; i < k; i++ {
			got = append(got, src.Next())
		}
		dst, err := Migrate(src)
		if err != nil {
			t.Fatalf("k=%d migrate: %v", k, err)
		}
		for i := k; i < total; i++ {
			got = append(got, dst.Next())
		}
		for i := range refTokens {
			if got[i] != refTokens[i] {
				t.Fatalf("k=%d: token stream diverged at step %d: migrated=%d, un-migrated=%d",
					k, i, got[i], refTokens[i])
			}
		}
	}
}

// TestDroppedHistoryFieldDiverges gives the equivalence assertion teeth: a
// migration that drops the carried-history fold (acc) produces a DIFFERENT
// resumable state at the hop, so it can no longer match the un-migrated run. This
// is the exact corruption class the checkpoint must transport intact — the test
// fails only if acc stopped being load-bearing.
func TestDroppedHistoryFieldDiverges(t *testing.T) {
	const (
		seed  = int64(4307)
		hopAt = 50
	)

	src := NewDecodeStream(seed)
	for i := 0; i < hopAt; i++ {
		src.Next()
	}
	good := src.Capture()

	// Precondition: by mid-stream the carried history is genuinely non-zero, so
	// dropping it is a real change (guards against a vacuous test).
	if good.Acc == 0 {
		t.Fatalf("carried history is zero at step %d; pick a hop where it is populated", hopAt)
	}

	// The lossy teleport: acc field lost in the handoff (restored as zero).
	lossy := good
	lossy.Acc = 0

	faithful := RestoreDecodeStream(good)
	broken := RestoreDecodeStream(lossy)

	if faithful.Capture() != good {
		t.Fatalf("faithful restore did not reproduce the captured state")
	}
	if broken.Capture() == good {
		t.Fatalf("dropping the carried-history field left the state unchanged; acc is not load-bearing")
	}
	// And the divergence is observable in the very next serialized state — a
	// re-migration of the broken instance would carry the corruption forward.
	if bytes.Equal(broken.Capture().Marshal(), faithful.Capture().Marshal()) {
		t.Fatalf("lossy and faithful restored states serialize identically; the drop is invisible")
	}
}
