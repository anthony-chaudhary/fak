package memq

// RecallArmResult is one fixture-loop arm in the contextual-replay net-value
// witness. RediscoveryTurns is the number of turns spent re-deriving known
// facts; InputBytes includes the memory injected by that arm.
type RecallArmResult struct {
	Arm              string `json:"arm"`
	RediscoveryTurns int    `json:"rediscovery_turns"`
	InputBytes       int    `json:"input_bytes"`
}

// ExemplarValueWitness compares the real alternatives required by the memory
// value contract: raw MEMORY.md, verified notes, and verified notes plus
// concrete exemplars. The exemplar arm is enabled only when it saves at least
// one rediscovery turn beyond notes and does not regress total input bytes per
// avoided turn. This is deliberately a measured gate, not a product claim.
type ExemplarValueWitness struct {
	RawMemory            RecallArmResult `json:"raw_memory"`
	NotesOnly            RecallArmResult `json:"notes_only"`
	NotesExemplars       RecallArmResult `json:"notes_exemplars"`
	TurnsSavedVsRaw      int             `json:"turns_saved_vs_raw"`
	TurnsSavedVsNotes    int             `json:"turns_saved_vs_notes"`
	IncrementalBytes     int             `json:"incremental_bytes"`
	NotesBytesPerTurn    float64         `json:"notes_bytes_per_turn"`
	ExemplarBytesPerTurn float64         `json:"exemplar_bytes_per_turn"`
	KeepExemplars        bool            `json:"keep_exemplars"`
}

// WitnessExemplarValue folds externally observed fixture-loop results. It does
// not invent measurements: malformed or non-positive exemplar arms fail closed.
func WitnessExemplarValue(raw, notes, exemplars RecallArmResult) ExemplarValueWitness {
	w := ExemplarValueWitness{
		RawMemory: raw, NotesOnly: notes, NotesExemplars: exemplars,
		TurnsSavedVsRaw:   raw.RediscoveryTurns - exemplars.RediscoveryTurns,
		TurnsSavedVsNotes: notes.RediscoveryTurns - exemplars.RediscoveryTurns,
		IncrementalBytes:  exemplars.InputBytes - notes.InputBytes,
	}
	if notes.RediscoveryTurns > 0 {
		w.NotesBytesPerTurn = float64(notes.InputBytes) / float64(notes.RediscoveryTurns)
	}
	if exemplars.RediscoveryTurns > 0 {
		w.ExemplarBytesPerTurn = float64(exemplars.InputBytes) / float64(exemplars.RediscoveryTurns)
	}
	valid := raw.RediscoveryTurns >= 0 && notes.RediscoveryTurns > 0 && exemplars.RediscoveryTurns >= 0
	netInputOK := exemplars.InputBytes <= notes.InputBytes
	if exemplars.RediscoveryTurns > 0 {
		netInputOK = w.ExemplarBytesPerTurn <= w.NotesBytesPerTurn
	}
	w.KeepExemplars = valid && w.TurnsSavedVsNotes > 0 && w.TurnsSavedVsRaw > 0 && netInputOK
	return w
}
