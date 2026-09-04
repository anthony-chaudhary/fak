package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"
)

// TestGoalContinuationPrefixStability pins #10671:
// Multi-turn goal continuation must serialize consecutive turns with byte-stable prefix hashes.
// Per-turn progress deltas trail at the end, and world_state is partial/diff after turn 0.
func TestGoalContinuationPrefixStability(t *testing.T) {
	session := NewGoalContinuationSession("goal-1234", "Complete top stability and functional issues")

	env := map[string]string{
		"CWD":  "/work/fak",
		"OS":   "windows",
		"TIER": "T1",
	}

	// Turn 1
	up1, ws1 := session.FormatWorldState(env, false)
	if !up1.Full {
		t.Fatalf("turn 1 must emit full world_state snapshot")
	}

	turn1Messages := []Message{
		session.StableGoalMessage(),
		ws1,
		session.TrailingProgressDeltaMessage(1, 100000, "in_progress"),
	}

	// Turn 2
	up2, ws2 := session.FormatWorldState(env, false)
	if up2.Full {
		t.Fatalf("turn 2 with unchanged env must emit partial diff, not full")
	}

	turn2Messages := []Message{
		session.StableGoalMessage(),
		ws2,
		session.TrailingProgressDeltaMessage(2, 95000, "in_progress"),
	}

	// Assert StableGoalMessage is byte-identical across turns
	if turn1Messages[0].Content != turn2Messages[0].Content {
		t.Fatalf("goal message mutated between turns: %q vs %q", turn1Messages[0].Content, turn2Messages[0].Content)
	}

	// Verify serialization prefix stability
	b1, err := json.Marshal(turn1Messages[0])
	if err != nil {
		t.Fatalf("marshal turn1 goal: %v", err)
	}
	b2, err := json.Marshal(turn2Messages[0])
	if err != nil {
		t.Fatalf("marshal turn2 goal: %v", err)
	}

	h1 := sha256.Sum256(b1)
	h2 := sha256.Sum256(b2)
	if hex.EncodeToString(h1[:]) != hex.EncodeToString(h2[:]) {
		t.Fatalf("prefix hash mismatch between turns: %x vs %x", h1, h2)
	}

	// Turn 3 with detected drift: should force full snapshot
	envDrift := map[string]string{
		"CWD":  "/work/fak/subdir",
		"OS":   "windows",
		"TIER": "T1",
	}
	up3, _ := session.FormatWorldState(envDrift, false)
	if !up3.Full || !up3.DriftSeen {
		t.Fatalf("turn 3 with drift must emit full snapshot with DriftSeen=true")
	}
}
