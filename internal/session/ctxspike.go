package session

import "fmt"

// ctxspike.go — the CONTEXT-SPIKE advisory (issue #2197): the first consumer of the
// per-session cost ring (#756). The ring records each turn's context/output cost and
// its doc names the runaway tell — the per-turn cost that jumps ~200x — but until this
// file nothing READ it: no renderer, no advisory, no nudge. The budget arms in Decide
// catch absolute DRAIN (ContextTokensLeft hit zero); nothing catches the SHAPE of the
// approach — a session whose context window jumps 30k→120k in ONE turn is burning its
// budget in sudden steps (one giant tool result, a whole-file re-read, an unwindowed
// log dump), and by the time the drain fires the spend already happened.
//
// THE FOLD. SpikeAdvisory compares the LATEST debited turn's context tokens to the
// turn before it — sudden means turn-over-turn, so the fold is latest-vs-previous by
// construction, and it self-extinguishes: after one non-spiking turn (a plateau, a
// shrink, a normal step) the latest pair no longer spikes and the advisory goes quiet.
// A spike must be BOTH sudden and material: Ratio ≥ MinRatio (suddenness — a +30k step
// on a 200k base is heavy but not sudden) AND Delta ≥ MinDeltaTokens (materiality — a
// 2k→8k jump quadruples but costs nothing). Requiring both is what keeps the nudge
// rare enough to be read rather than tuned out.
//
// THE NUDGE. Nudge() renders the advisory as one deterministic model-facing paragraph:
// the observed numbers plus the concrete context-hygiene moves (windowed reads,
// summarize-don't-carry, justify the next large ingest before doing it). The agent
// loop splices it into the next turn's input at the same boundary the #850 steer
// splice uses — the model is TOLD its context just spiked and how to spend the window
// from here, each turn it matters. That is the cheapest prevention level: the model
// self-corrects before any gate has to act.
//
// FENCE: advisory, never trust — same law as the ring itself. Nothing here GATES: no
// Decide verdict changes, no budget debits, no run-state transition. The zero ring, a
// single-turn ring, a paused/terminal session all fold to the quiet zero advisory.
// The opt-in HOLD (pause-for-permission on breach) is a separate rung of #2197 and
// must arrive as an explicit operator-set policy, not a default.

// Default spike thresholds. Conservative on purpose: both must hold, so the nudge
// fires on "a third of a typical window arrived in one turn" and stays silent on
// ordinary growth. Not tuned constants — seeds for the operator-set policy rung.
const (
	// DefaultSpikeMinRatio is the suddenness floor: the latest turn's context must be
	// at least this multiple of the previous turn's. 1.5 means "grew by half or more
	// in one turn".
	DefaultSpikeMinRatio = 1.5
	// DefaultSpikeMinDeltaTokens is the materiality floor: the one-turn context growth
	// in tokens. 16384 (~16k) is a large tool result or an unwindowed file read —
	// exactly the ingests the nudge exists to make deliberate.
	DefaultSpikeMinDeltaTokens = 16384
)

// SpikePolicy is the two-axis threshold a context spike must clear. The zero value is
// valid and means the defaults — a caller opts into stricter or looser thresholds by
// setting either field; non-positive/garbage fields fall back to the defaults, the same
// fail-closed posture PaceBudget applies to a poisoned ratio.
type SpikePolicy struct {
	MinRatio       float64 `json:"min_ratio"`        // latest/previous context floor; <=1 (or NaN via zero) => default
	MinDeltaTokens int     `json:"min_delta_tokens"` // one-turn context growth floor; <=0 => default
}

// withDefaults returns the policy with every unset/degenerate field filled.
func (p SpikePolicy) withDefaults() SpikePolicy {
	if !(p.MinRatio > 1) { // catches <=1 and NaN: a ratio floor at or below 1 would fire on every turn
		p.MinRatio = DefaultSpikeMinRatio
	}
	if p.MinDeltaTokens <= 0 {
		p.MinDeltaTokens = DefaultSpikeMinDeltaTokens
	}
	return p
}

// SpikeAdvisory is the fold's result: the latest turn-over-turn context comparison and
// whether it cleared both spike floors. The observed numbers are carried even when
// Spiked is false so a renderer (`fak ps`, /metrics — the #2197 observability rung) can
// show the growth shape without re-deriving it. The zero value is the quiet "nothing to
// say" advisory an empty or single-turn ring folds to.
type SpikeAdvisory struct {
	Spiked        bool    `json:"spiked"`
	PrevContext   int     `json:"prev_context"`   // previous turn's context tokens
	LatestContext int     `json:"latest_context"` // latest turn's context tokens
	DeltaTokens   int     `json:"delta_tokens"`   // LatestContext - PrevContext
	Ratio         float64 `json:"ratio"`          // LatestContext / PrevContext
}

// SpikeAdvisory folds the ring's two most recent turns into the context-growth
// advisory under the given policy. It needs a real baseline: fewer than two recorded
// turns, or a previous turn with no context accounting (ContextTokens 0 — an
// output-only debit), folds to the zero advisory — a session's first big window
// cannot be SUDDEN, and inventing a baseline would fabricate a ratio. Pure: same
// (ring, policy) => identical advisory.
func (r CostRing) SpikeAdvisory(p SpikePolicy) SpikeAdvisory {
	if r.Count < 2 {
		return SpikeAdvisory{}
	}
	latest, _ := r.at(0)
	prev, _ := r.at(1)
	if prev.ContextTokens <= 0 || latest.ContextTokens <= 0 {
		return SpikeAdvisory{}
	}
	p = p.withDefaults()
	a := SpikeAdvisory{
		PrevContext:   prev.ContextTokens,
		LatestContext: latest.ContextTokens,
		DeltaTokens:   latest.ContextTokens - prev.ContextTokens,
		Ratio:         float64(latest.ContextTokens) / float64(prev.ContextTokens),
	}
	a.Spiked = a.DeltaTokens >= p.MinDeltaTokens && a.Ratio >= p.MinRatio
	return a
}

// Nudge renders the model-facing advisory line, or "" when there is nothing to say
// (no spike). Deterministic — same advisory, byte-identical string — so a transcript
// diff or a test can bind to it. The text leads with the observed numbers (the model
// should see the cost, not just an instruction), names the concrete hygiene moves,
// and asks the model to make the NEXT large ingest deliberate — the ask-first posture
// #2197's example names — while stating honestly that nothing was blocked.
func (a SpikeAdvisory) Nudge() string {
	if !a.Spiked {
		return ""
	}
	return fmt.Sprintf(
		"[fak context advisory] Context grew %.1fx in one turn (%d -> %d tokens, +%d). "+
			"To avoid another spike: read large files in windows (offset/limit) instead of whole, "+
			"summarize or drop bulky tool results rather than carrying them verbatim, and do not "+
			"re-read content already in context. If the next step genuinely needs another large "+
			"ingest, say so and why before doing it. Advisory only - nothing was blocked.",
		a.Ratio, a.PrevContext, a.LatestContext, a.DeltaTokens)
}

// ContextNudge is the turn-boundary read the agent loop makes right after Decide: the
// rendered spike nudge for this session's latest debited turn, "" when there is
// nothing to say. Read-only — no debit, no state transition, no LRU-order guarantee
// beyond Get's own — and nil-safe (a loop with no table wired behaves byte-identically
// to the historical path, the same posture Decide's nil receiver takes). Uses the
// default SpikePolicy; the operator-set per-session policy is a follow-on rung of #2197.
func (t *Table) ContextNudge(trace string) string {
	if t == nil || trace == "" {
		return ""
	}
	return t.Get(trace).Cost.SpikeAdvisory(SpikePolicy{}).Nudge()
}
