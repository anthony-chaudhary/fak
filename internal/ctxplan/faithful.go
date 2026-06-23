package ctxplan

// Witness is the faithfulness audit of a plan — the property that separates a planned
// VIEW from lossy COMPACTION. A plan is Faithful iff (a) its resident and elided sets
// PARTITION every candidate (no span silently vanished, none double-counted) and (b)
// every elided span carries a recovery handle (a content address / id) so it can be
// paged back in on demand. A faithful plan never DESTROYS history; it only decides which
// spans are resident right now. Compaction, by contrast, replaces elided spans with a
// lossy summary and drops the originals — Audit reports that UNFAITHFUL.
//
// This is the honesty gate the whole concept rests on: it is what lets ctxplan claim
// "O(1) resident AND exact recall," because every byte the view leaves out is still
// recoverable. A regression that constructed an Elision with no handle, or that lost a
// candidate from both sets, turns Faithful false — exactly the silent-loss bug it exists
// to catch.
type Witness struct {
	Candidates     int      `json:"candidates"`              // spans the planner considered
	Resident       int      `json:"resident"`                // spans kept in the view
	Elided         int      `json:"elided"`                  // spans kept cold (out of the view)
	Recoverable    int      `json:"recoverable"`             // elided spans WITH a page-back-in handle
	Unrecoverable  []string `json:"unrecoverable,omitempty"` // elided spans with NO handle (destroyed — a compaction tell)
	ResidentTokens int      `json:"resident_tokens"`
	ElidedTokens   int      `json:"elided_tokens"` // tokens out of the window but still recoverable
	Partition      bool     `json:"partition"`     // Resident+Elided == Candidates AND the two sets are disjoint
	Faithful       bool     `json:"faithful"`      // Partition AND every elided span is recoverable
}

// Audit computes the faithfulness witness of a plan. It reads only the plan's own
// accounting (no store access), so it is a cheap, deterministic check a caller can run
// before trusting a plan — or a gate can run to REFUSE one that dropped a span with no
// recovery path.
func Audit(p Plan) Witness {
	w := Witness{
		Candidates: p.Candidates,
		Resident:   len(p.Selected),
		Elided:     len(p.Elided),
	}
	selectedIDs := make(map[string]bool, len(p.Selected))
	for _, s := range p.Selected {
		w.ResidentTokens += s.Cost
		if s.ID != "" {
			selectedIDs[s.ID] = true
		}
	}
	disjoint := true
	for _, e := range p.Elided {
		w.ElidedTokens += e.Cost
		if e.ID != "" && selectedIDs[e.ID] {
			disjoint = false // a span both resident AND elided — a double-count, not a partition
		}
		if e.ID == "" && e.Digest == "" {
			// No handle: this span cannot be paged back in. It was DESTROYED, not elided —
			// the defining property of compaction. Record its (best-effort) identity.
			id := e.Digest
			if id == "" {
				id = e.Reason // fall back to the reason so the report is not blank
			}
			w.Unrecoverable = append(w.Unrecoverable, id)
		} else {
			w.Recoverable++
		}
	}
	w.Partition = disjoint && (w.Resident+w.Elided == w.Candidates)
	w.Faithful = w.Partition && len(w.Unrecoverable) == 0
	return w
}

// CompactionView models what LOSSY COMPACTION does to the same plan: it keeps the same
// resident view but strips the recovery handles off every elided span — i.e. it drops
// the originals and would replace them with a summary. It exists to make the faithful-vs-
// compaction distinction a CHECKABLE contrast, not a slogan: Audit(p) is Faithful while
// Audit(CompactionView(p)) is not, with identical residency. The token savings look the
// same; only recoverability differs, and that is the whole point.
func CompactionView(p Plan) Plan {
	out := p
	out.Objective = "compaction"
	out.Elided = make([]Elision, len(p.Elided))
	for i, e := range p.Elided {
		e.ID = ""     // the original span id is gone
		e.Digest = "" // the bytes are gone — no page-back-in handle
		e.Reason = "compacted_away"
		out.Elided[i] = e
	}
	return out
}
