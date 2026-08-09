package memq

import "testing"

func TestWitnessExemplarValueKeepsOnlyPositiveNetDelta(t *testing.T) {
	// Captured scripted fixture: the wall is cleared in three turns from the raw
	// dump, two from verified notes, and one when the matching concrete command
	// trajectory is present. Input bytes include each injected block.
	w := WitnessExemplarValue(
		RecallArmResult{Arm: "A_raw_MEMORY.md", RediscoveryTurns: 3, InputBytes: 420},
		RecallArmResult{Arm: "B_verified_notes", RediscoveryTurns: 2, InputBytes: 190},
		RecallArmResult{Arm: "C_notes_plus_exemplars", RediscoveryTurns: 1, InputBytes: 90},
	)
	if !w.KeepExemplars || w.TurnsSavedVsNotes != 1 || w.TurnsSavedVsRaw != 2 || w.IncrementalBytes != -100 {
		t.Fatalf("positive fixture witness = %+v", w)
	}

	noGain := WitnessExemplarValue(w.RawMemory, w.NotesOnly,
		RecallArmResult{Arm: "C_notes_plus_exemplars", RediscoveryTurns: 2, InputBytes: 90})
	if noGain.KeepExemplars {
		t.Fatalf("zero incremental turn gain must fail closed: %+v", noGain)
	}
}
