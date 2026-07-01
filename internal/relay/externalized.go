// Rung F1 (issue #1884): the transcript-only load-bearing state detector. A fact the
// current leg relies on that exists ONLY in the growing transcript — not backed by a
// durable pointer (a commit, issue, memory slug, ledger row, or file) — is exactly what a
// rotation would silently drop. This detector flags such facts so a later rung (F2) can
// refuse to rotate until they are externalized. It only DETECTS; it does not refuse, and it
// reads no clock and does no I/O.
package relay

// LoadBearingFact is a claim the current leg depends on, paired with the durable pointer
// that backs it (if any). A fact with no backing durable pointer lives only in the
// transcript and would be lost on rotation. Label is display-only — the fact's identity,
// never its bytes.
type LoadBearingFact struct {
	Label   string   `json:"label"`
	Backing Artifact `json:"backing"`
}

// IsExternalized reports whether the fact is backed by a durable pointer: a non-empty ref
// whose kind is in the closed ArtifactKind vocabulary. A pointer with an empty ref, or a
// kind outside the vocabulary, does NOT externalize the fact — the state is still
// transcript-only and would not survive a rotation.
func (f LoadBearingFact) IsExternalized() bool {
	if f.Backing.Ref == "" {
		return false
	}
	switch ArtifactKind(f.Backing.Kind) {
	case ArtifactCommit, ArtifactIssue, ArtifactMemory, ArtifactLedger, ArtifactFile:
		return true
	default:
		return false
	}
}

// TranscriptOnly returns the subset of facts that are NOT externalized — the load-bearing
// state a rotation would drop, in input order. An empty (nil) result means every fact is
// backed by a durable pointer: a clean, safe-to-rotate state.
func TranscriptOnly(facts []LoadBearingFact) []LoadBearingFact {
	var out []LoadBearingFact
	for _, f := range facts {
		if !f.IsExternalized() {
			out = append(out, f)
		}
	}
	return out
}

// FullyExternalized reports whether every fact is backed by a durable pointer — the
// safe-point precondition F2 will gate rotation on. It is exactly len(TranscriptOnly)==0,
// named for the call site that asks the question positively.
func FullyExternalized(facts []LoadBearingFact) bool {
	return len(TranscriptOnly(facts)) == 0
}
