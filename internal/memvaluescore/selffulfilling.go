package memvaluescore

import (
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// selffulfilling.go — #2914: the self-fulfilling-metric detector (#2818),
// extended from the cache surface to the skill/memory LEARNING surface.
//
// Hermes' learning loop rewards *activity* ("most sessions produce at least one
// skill update") with no counter-metric, so the loop can raise "skills kept"
// without raising net task value — the reward-hack shape fak already names on
// the cache surface (docs/notes/CACHE-HIT-VANITY-METRIC-SELF-FULFILLING). This
// detector pairs the gameable learning-loop metric with an INDEPENDENT net
// witness and flags divergence: a loop optimizing "skills kept" MUST move net
// value, or a skill that only games the witness (grows the store, delivers no
// witnessed value) is flagged and scores worse — not better.
//
// The pairing reuses this card's existing axes, no new bookkeeping:
//   - the gameable metric  = store_notes           — "skills kept": the count of
//     notes retained in the store, raisable by keeping ANY note (Hermes'
//     "be ACTIVE, produce a skill" nudge inflates exactly this);
//   - the net witness       = memory_value_frontier — accumulated WITNESSED
//     recall value, which by construction moves ONLY on realized ledger events,
//     never on store size (see score.go) — the value the loop CANNOT forge by
//     hoarding notes.
//
// The detector compares the two across a window (before -> after) and flags a
// self-fulfilling raise: skills_kept went UP while net witnessed value did NOT.

// LearningMetric is the gameable/witness pair the detector reads: the
// learning-loop metric a loop can inflate (SkillsKept) alongside its
// independent net witness (NetValue). Both are read straight off this card's
// control-pane corpus (see MetricFromPayload), so the detector and the
// scorecard can never disagree about what "value" means.
type LearningMetric struct {
	SkillsKept int `json:"skills_kept"` // gameable: store_notes (notes retained)
	NetValue   int `json:"net_value"`   // witness:  memory_value_frontier
}

// The detector verdicts.
const (
	VerdictSelfFulfilling = "self_fulfilling" // metric rose; net value did not
	VerdictWitnessed      = "witnessed"       // metric rose WITH a net-value rise
	VerdictNoRaise        = "no_raise"        // the metric did not rise; nothing to check
)

// SelfFulfilling is the divergence verdict over a before->after window.
type SelfFulfilling struct {
	Before     LearningMetric `json:"before"`
	After      LearningMetric `json:"after"`
	KeptDelta  int            `json:"kept_delta"`  // Δ skills kept (the gameable metric)
	ValueDelta int            `json:"value_delta"` // Δ net witnessed value
	Flagged    bool           `json:"flagged"`     // true == a self-fulfilling raise
	Verdict    string         `json:"verdict"`     // one of the Verdict* constants
	Reason     string         `json:"reason"`
}

// DetectSelfFulfilling pairs the "skills kept" learning-loop metric with its
// independent net witness across a window and flags a self-fulfilling raise:
// skills kept went UP while net witnessed value did NOT. This is the
// learning-surface analogue of the cache-side self-fulfilling-metric detector
// (#2818, #2914). A window where the loop grew the store but the frontier
// stayed flat is exactly the reward hack — a gaming skill that raises the
// witness metric without raising net task value — so it is flagged (scores
// worse), never rewarded. Net value rising on its own (a recall delivering
// value without hoarding) is never flagged: raising the metric alone is what
// the detector refuses to reward, not delivering value.
func DetectSelfFulfilling(before, after LearningMetric) SelfFulfilling {
	s := SelfFulfilling{
		Before:     before,
		After:      after,
		KeptDelta:  after.SkillsKept - before.SkillsKept,
		ValueDelta: after.NetValue - before.NetValue,
	}
	switch {
	case s.KeptDelta <= 0:
		s.Verdict = VerdictNoRaise
		s.Reason = fmt.Sprintf("skills_kept did not rise (Δ%d); no learning-loop metric was raised to check", s.KeptDelta)
	case s.ValueDelta > 0:
		s.Verdict = VerdictWitnessed
		s.Reason = fmt.Sprintf("skills_kept rose by %d WITH a paired net-value rise (Δ%d) — witnessed learning, not a self-fulfilling raise", s.KeptDelta, s.ValueDelta)
	default:
		s.Flagged = true
		s.Verdict = VerdictSelfFulfilling
		s.Reason = fmt.Sprintf("skills_kept rose by %d but the net-value frontier did not rise (Δ%d): a learning-loop metric raised without a paired net-value rise — a skill that games the witness (grew the store, delivered no witnessed value) scores worse, not better", s.KeptDelta, s.ValueDelta)
	}
	return s
}

// MetricFromPayload reads the detector's gameable/witness pair straight off a
// control-pane payload this card emits (BuildWith): store_notes is the gameable
// "skills kept" metric, memory_value_frontier is the net witness. It accepts
// both the in-memory ints and the float64 a JSON round-trip yields (the
// `fak memory-value-scorecard --compare` seam), so a prior-snapshot payload and
// a live payload pair cleanly.
func MetricFromPayload(p scorecard.Payload) LearningMetric {
	return LearningMetric{
		SkillsKept: corpusInt(p.Corpus["store_notes"]),
		NetValue:   corpusInt(p.Corpus["memory_value_frontier"]),
	}
}

// corpusInt coerces a control-pane corpus value to int across the in-memory
// (int) and JSON-decoded (float64 / json.Number) shapes.
func corpusInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	default:
		return 0
	}
}
