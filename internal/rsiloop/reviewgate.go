package rsiloop

// reviewgate.go — VALUE-GATE the periodic self-improvement review fire (#2837,
// part of Track A #2834). Spawn a forked review turn only when its estimated
// expected value exceeds its estimated cost, and SELF-TUNE the threshold from the
// realized outcomes of prior fires.
//
// THE PROBLEM (the Hermes mechanism this improves on). A background "review this
// session for extractable skill/memory" fork costs real tokens. Hermes fires it on
// a FIXED cadence — a memory nudge every N user turns, a skill nudge every N tool
// iterations (defaults 10) — so every eligible session pays for the fork regardless
// of whether it is likely to produce anything worth keeping.
//
// WHAT THIS DOES. fak already PRICES token/cache economics per fire (the compaction-
// economics / cache-value work, #2810/#2817). This gate applies the same discipline
// to the review-fire decision: estimate the expected value of a fire (in token-
// equivalents) from the session's tool trace — session novelty, unseen tool
// sequences, error density — price the forked turn's COST on the SAME compaction-
// economics basis (a re-read prefix billed at the cache-read marginal plus generated
// output, cacheprice.ReadMultiplier), and spawn ONLY when expected value > cost.
//
// SELF-TUNING, ANTI-REWARD-HACK (the load-bearing fence, #2816). The expected value
// is pKeep × the token-equiv worth of a KEPT skill/memory, and pKeep is anchored on
// a base keep-rate the gate reads back from its OWN net-value ledger — the REALIZED
// fraction of past fires that produced a kept artifact. Crucially the feedback reads
// the REALIZED outcome recorded per fire (ReviewOutcome rows), NEVER the estimator's
// own prediction: a gate that games itself into always-spawning by inflating its
// estimates cannot win, because the low-value fires it lets through come back
// NOT-kept, which DROPS the realized keep-rate, which tightens the gate. The metric
// the loop optimizes is the realized keep-rate, and it is non-forgeable by the
// estimator that drives spawning — the same self-fulfilling-metric guard #2816 built.
//
// RELATION TO #2910 (internal/nightrun/learningnudge.go). #2910 is the sibling
// novelty/friction THRESHOLD gate proven by an OFFLINE ablation whose kept-skill
// label the gate never reads. This is its complement: a LIVE, ledger-fed, expected-
// value-vs-cost gate that reads realized outcomes back to self-tune. They coordinate
// on the same Track A review-fire path; this file changes only the WHEN (spawn)
// decision, never what the review itself does once spawned (#2835/#2836).
//
// Pure and deterministic except the append-only ledger I/O: the same trace + corpus
// + config scores identically every time, so the anti-reward-hack witness below is a
// fixed test.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// ReviewFireLedgerSchema tags each durable net-value ledger row so a reader can
// never confuse it for another rsiloop journal (the keep-gate journal, the curator
// ledger). The "/1" is the row-shape version.
const ReviewFireLedgerSchema = "fak-review-fire-ledger/1"

// DefaultReviewFireLedgerRel is the sibling ledger this gate writes, next to the
// other per-fire economics ledgers (docs/nightrun/cache-value.jsonl,
// memory-value.jsonl), so the review-fire net-value account has its own file and
// never shares a row with a cache-economics one.
const DefaultReviewFireLedgerRel = "docs/nightrun/review-fire.jsonl"

// ReviewTrace is the value-relevant summary of a session's tool trace the gate
// estimates from — structured signal only (tool NAMES and an error count), never
// raw prompt/result prose, so a corpus of these stays committable (the same
// discipline internal/sessionobs and #2910's NudgeSession hold).
type ReviewTrace struct {
	// ToolCalls is the ordered sequence of tool NAMES the session issued — its shape.
	// Session novelty is scored over its distinct tool unigrams; unseen tool
	// sequences over its consecutive-pair bigrams.
	ToolCalls []string
	// Errors is the count of failed/is_error tool results in the trace — the density
	// signal (a session that kept failing is where a guardrail memory is worth keeping).
	Errors int
}

// ReviewCost prices the forked review turn on the SAME compaction-economics basis the
// cache-value work uses (#2810/#2817/#2798). A review fork RE-READS the session
// prefix — the provider bills those tokens at the cache-read marginal, not fresh
// input — and GENERATES output at the full rate. TokenEquiv reuses the canonical
// cacheprice.ReadMultiplier (the ONE 0.1× anchor, #2798) so the gate's cost and the
// compaction report's shed marginal value an identical cached token identically by
// CONSTRUCTION, not by a mirrored literal.
type ReviewCost struct {
	// PrefixTokens is the session prefix the forked turn re-reads (billed cache_read).
	PrefixTokens uint64
	// OutputTokens is the forked turn's generated output (billed at the full rate).
	OutputTokens uint64
}

// TokenEquiv is the cost of the fork in input-token-equivalents: the re-read prefix
// at the cache-read marginal plus the generated output at 1.0×.
func (c ReviewCost) TokenEquiv() float64 {
	return float64(c.PrefixTokens)*cacheprice.ReadMultiplier + float64(c.OutputTokens)
}

// ReviewGateConfig is the gate's tunables. KeptValueTokenEquiv is the estimated
// downstream worth (in input-token-equivalents) of a review that produces a KEPT
// skill/memory — a modeling prior the operator tunes; only its RATIO to the fork
// cost decides a spawn, so its exact value is not load-bearing. The keep-rate self-
// tune replaces PriorKeepRate with the ledger's REALIZED rate once at least
// MinOutcomes fires have a recorded outcome.
type ReviewGateConfig struct {
	KeptValueTokenEquiv float64 // token-equiv worth of a fire that yields a kept artifact
	PriorKeepRate       float64 // base keep-rate before the ledger has enough realized outcomes
	MinOutcomes         int     // realized outcomes required before the ledger rate replaces the prior
	WNovelty            float64 // weight on session novelty (new tool types)
	WUnseenSequence     float64 // weight on unseen tool sequences (new tool order)
	WErrorDensity       float64 // weight on error density (unresolved friction)
}

// DefaultReviewGateConfig is the shipped gate: a kept skill/memory is priced at 50k
// token-equiv of avoided future re-derivation (a prior, tuned by the operator), a
// conservative 0.2 base keep-rate until 8 fires have a realized outcome, and equal
// weight across the three trace signals.
func DefaultReviewGateConfig() ReviewGateConfig {
	return ReviewGateConfig{
		KeptValueTokenEquiv: 50_000,
		PriorKeepRate:       0.2,
		MinOutcomes:         8,
		WNovelty:            1.0 / 3,
		WUnseenSequence:     1.0 / 3,
		WErrorDensity:       1.0 / 3,
	}
}

func (c ReviewGateConfig) withDefaults() ReviewGateConfig {
	d := DefaultReviewGateConfig()
	if c.KeptValueTokenEquiv <= 0 {
		c.KeptValueTokenEquiv = d.KeptValueTokenEquiv
	}
	if c.PriorKeepRate <= 0 {
		c.PriorKeepRate = d.PriorKeepRate
	}
	if c.MinOutcomes <= 0 {
		c.MinOutcomes = d.MinOutcomes
	}
	if c.WNovelty <= 0 && c.WUnseenSequence <= 0 && c.WErrorDensity <= 0 {
		c.WNovelty, c.WUnseenSequence, c.WErrorDensity = d.WNovelty, d.WUnseenSequence, d.WErrorDensity
	}
	return c
}

// ReviewEventKind is the CLOSED set of net-value ledger row kinds. A decision row
// records the spawn/skip and the estimate that drove it; an outcome row resolves one
// prior spawned decision with its REALIZED kept/not result. Keeping them distinct is
// the confusion-risk fence of #2837: "review fired" (the spawn decision) and "review
// produced a kept skill/memory" (the realized-value signal) must never collapse.
type ReviewEventKind string

const (
	// ReviewDecisionKind is a spawn/skip decision row (Spawned + the estimate fields).
	ReviewDecisionKind ReviewEventKind = "decision"
	// ReviewOutcomeKind is a realized-outcome row that resolves one spawned decision
	// (Resolves names its Seq; Kept is the realized result).
	ReviewOutcomeKind ReviewEventKind = "outcome"
)

// ReviewRow is one append-only net-value ledger record. A decision row carries the
// spawn bit and the estimate; an outcome row carries Resolves + Kept and is the
// REALIZED signal the self-tune reads back.
type ReviewRow struct {
	Schema string          `json:"schema"`
	Seq    int             `json:"seq"`
	Kind   ReviewEventKind `json:"kind"`

	// Decision-row fields.
	Spawned      bool    `json:"spawned,omitempty"`
	EstValueTeq  float64 `json:"est_value_teq,omitempty"`
	EstCostTeq   float64 `json:"est_cost_teq,omitempty"`
	Novelty      float64 `json:"novelty,omitempty"`
	UnseenSeqs   float64 `json:"unseen_sequence_ratio,omitempty"`
	ErrorDensity float64 `json:"error_density,omitempty"`
	KeepRate     float64 `json:"keep_rate_used,omitempty"` // the self-tuned base rate this decision used

	// Outcome-row fields.
	Resolves int    `json:"resolves,omitempty"` // the decision Seq this outcome resolves
	Kept     bool   `json:"kept,omitempty"`     // the REALIZED result: did the fire produce a kept artifact?
	Note     string `json:"note,omitempty"`
}

// ReviewEstimate is the folded verdict for one candidate fire: whether to spawn, the
// expected value and cost that decided it (in token-equivalents), and the signal +
// self-tuned base rate that produced the expected value — everything a caller needs
// to journal and audit the decision.
type ReviewEstimate struct {
	Spawn        bool
	ExpValueTeq  float64 // pKeep × KeptValueTokenEquiv
	CostTeq      float64 // the fork cost on the compaction-economics basis
	KeepRate     float64 // the self-tuned base keep-rate used (realized or prior)
	PKeep        float64 // the signal-adjusted keep probability
	SignalScore  float64 // the blended [0,1] trace signal
	Novelty      float64
	UnseenSeqs   float64
	ErrorDensity float64
}

// ReviewLedger is the append-only JSONL net-value ledger plus the folded read-path
// that self-tunes the gate. It mirrors CuratorLedger's discipline: every decision and
// every outcome is one durable row, the governing keep-rate is FOLDED from the rows,
// and the load is corruption-tolerant (a torn final line from an O_APPEND crash is
// skipped, not fatal).
type ReviewLedger struct {
	path string
	cfg  ReviewGateConfig
	rows []ReviewRow
}

// OpenReviewLedger opens (or creates) the JSONL ledger at path and loads existing
// rows so Seq assignment and the keep-rate fold continue across restarts. A path of
// "" keeps the ledger in-memory (a fast test). A missing file is not an error.
func OpenReviewLedger(path string, cfg ReviewGateConfig) (*ReviewLedger, error) {
	l := &ReviewLedger{path: path, cfg: cfg.withDefaults()}
	if path == "" {
		return l, nil
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return l, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r ReviewRow
		if err := json.Unmarshal(line, &r); err != nil {
			continue // skip a torn / non-JSON line rather than fail the whole read
		}
		l.rows = append(l.rows, r)
	}
	return l, nil
}

// RealizedKeepRate folds the ledger's OUTCOME rows into the realized fraction of
// resolved fires that produced a kept artifact, and the count of resolved fires. It
// reads ONLY the recorded Kept bit (the realized result), NEVER any decision-row
// estimate — the non-forgeable input to the self-tune (#2816). An empty history is
// (0, 0).
func (l *ReviewLedger) RealizedKeepRate() (rate float64, n int) {
	kept := 0
	for _, r := range l.rows {
		if r.Kind != ReviewOutcomeKind {
			continue
		}
		n++
		if r.Kept {
			kept++
		}
	}
	if n == 0 {
		return 0, 0
	}
	return float64(kept) / float64(n), n
}

// EffectiveKeepRate is the SELF-TUNE: the ledger's REALIZED keep-rate once at least
// MinOutcomes fires have a recorded outcome, else the configured prior. This is the
// base rate the estimator anchors pKeep on, so the gate's threshold moves with the
// realized outcomes of its own prior fires and nothing else.
func (l *ReviewLedger) EffectiveKeepRate() float64 {
	if rate, n := l.RealizedKeepRate(); n >= l.cfg.MinOutcomes {
		return rate
	}
	return l.cfg.PriorKeepRate
}

// Estimate is the PURE value-gate fold for one candidate fire: it scores the trace
// signal, anchors pKeep on the self-tuned base keep-rate, prices expected value and
// cost in token-equivalents, and decides spawn iff expected value STRICTLY exceeds
// cost. It never appends — Decide does that — so a caller can preview a decision.
func (l *ReviewLedger) Estimate(t ReviewTrace, seen map[string]bool, cost ReviewCost) ReviewEstimate {
	novelty := t.Novelty(seen)
	unseen := t.UnseenSequenceRatio(seen)
	errDensity := t.ErrorDensity()
	signal := l.cfg.blendSignal(novelty, unseen, errDensity)

	base := l.EffectiveKeepRate()
	// The signal MODULATES the self-tuned base rate around its midpoint: a maximally
	// novel/frictional session lifts pKeep to 1.5× base, a maximally repetitive clean
	// one drops it to 0.5× base, all clamped to [0,1]. The base rate — not the signal —
	// sets the operating point, so the self-tune governs the gate.
	pKeep := clampUnit(base * (0.5 + signal))
	expValue := pKeep * l.cfg.KeptValueTokenEquiv
	costTeq := cost.TokenEquiv()

	return ReviewEstimate{
		Spawn:        expValue > costTeq,
		ExpValueTeq:  expValue,
		CostTeq:      costTeq,
		KeepRate:     base,
		PKeep:        pKeep,
		SignalScore:  signal,
		Novelty:      novelty,
		UnseenSeqs:   unseen,
		ErrorDensity: errDensity,
	}
}

// Decide estimates the candidate fire, appends the decision as one durable net-value
// row, and returns the estimate plus the decision's Seq — which the caller later
// passes to RecordOutcome once the (spawned) review resolves. A skip is journaled
// too, so the ledger records every decision, not only the fires.
func (l *ReviewLedger) Decide(t ReviewTrace, seen map[string]bool, cost ReviewCost) (ReviewEstimate, int, error) {
	est := l.Estimate(t, seen, cost)
	seq := len(l.rows) + 1
	row := ReviewRow{
		Schema:       ReviewFireLedgerSchema,
		Seq:          seq,
		Kind:         ReviewDecisionKind,
		Spawned:      est.Spawn,
		EstValueTeq:  est.ExpValueTeq,
		EstCostTeq:   est.CostTeq,
		Novelty:      est.Novelty,
		UnseenSeqs:   est.UnseenSeqs,
		ErrorDensity: est.ErrorDensity,
		KeepRate:     est.KeepRate,
	}
	if err := l.append(row); err != nil {
		return ReviewEstimate{}, 0, err
	}
	return est, seq, nil
}

// RecordOutcome resolves ONE prior spawned decision with its REALIZED result: did the
// spawned review produce a kept skill/memory? It refuses an unknown Seq, a non-spawned
// (skipped) decision — a fire that never ran has no realized outcome — a Seq that is
// not a decision row, and a decision already resolved (an outcome must map one-to-one
// to a fire, or the realized keep-rate would double-count). The appended outcome row
// is what the self-tune reads back.
func (l *ReviewLedger) RecordOutcome(decisionSeq int, kept bool) (int, error) {
	dec, ok := l.rowBySeq(decisionSeq)
	if !ok {
		return 0, fmt.Errorf("reviewgate: cannot resolve unknown decision seq %d", decisionSeq)
	}
	if dec.Kind != ReviewDecisionKind {
		return 0, fmt.Errorf("reviewgate: seq %d is not a decision row", decisionSeq)
	}
	if !dec.Spawned {
		return 0, fmt.Errorf("reviewgate: decision seq %d was a SKIP (no fire) — nothing to resolve", decisionSeq)
	}
	if l.isResolved(decisionSeq) {
		return 0, fmt.Errorf("reviewgate: decision seq %d already has a realized outcome", decisionSeq)
	}
	seq := len(l.rows) + 1
	row := ReviewRow{
		Schema:   ReviewFireLedgerSchema,
		Seq:      seq,
		Kind:     ReviewOutcomeKind,
		Resolves: decisionSeq,
		Kept:     kept,
	}
	if err := l.append(row); err != nil {
		return 0, err
	}
	return seq, nil
}

// PendingOutcomes returns the Seqs of spawned decisions that do not yet have a
// realized outcome — the fires whose value the ledger is still waiting to learn.
func (l *ReviewLedger) PendingOutcomes() []int {
	var out []int
	for _, r := range l.rows {
		if r.Kind == ReviewDecisionKind && r.Spawned && !l.isResolved(r.Seq) {
			out = append(out, r.Seq)
		}
	}
	return out
}

// Rows returns a copy of the append-only ledger for inspection/telemetry.
func (l *ReviewLedger) Rows() []ReviewRow {
	return append([]ReviewRow(nil), l.rows...)
}

func (l *ReviewLedger) rowBySeq(seq int) (ReviewRow, bool) {
	for _, r := range l.rows {
		if r.Seq == seq {
			return r, true
		}
	}
	return ReviewRow{}, false
}

func (l *ReviewLedger) isResolved(decisionSeq int) bool {
	for _, r := range l.rows {
		if r.Kind == ReviewOutcomeKind && r.Resolves == decisionSeq {
			return true
		}
	}
	return false
}

// append records the row in memory and, if file-backed, durably appends it as one
// JSON line so the ledger survives a restart (the CuratorLedger discipline).
func (l *ReviewLedger) append(r ReviewRow) error {
	return appendLedgerRow(l.path, &l.rows, r)
}

// --- trace signals (session novelty / unseen tool sequences / error density) ---

// Novelty is the SESSION-NOVELTY signal: the fraction of the session's DISTINCT tool
// TYPES (unigrams) that have never been seen before, in [0,1]. A session that only
// re-runs known tools is 0-novel (nothing new to learn); an empty session is 0.
func (t ReviewTrace) Novelty(seen map[string]bool) float64 {
	uniq := map[string]bool{}
	for _, tool := range t.ToolCalls {
		uniq[unigramKey(tool)] = true
	}
	if len(uniq) == 0 {
		return 0
	}
	novel := 0
	for k := range uniq {
		if !seen[k] {
			novel++
		}
	}
	return float64(novel) / float64(len(uniq))
}

// UnseenSequenceRatio is the UNSEEN-TOOL-SEQUENCE signal: the fraction of the
// session's DISTINCT consecutive tool-pair bigrams that have never been seen before,
// in [0,1]. This is the ORDER signal a unigram novelty misses — a known tool set run
// in a genuinely new order still reads as an unseen workflow. Fewer than two calls
// has no sequence, so it is 0.
func (t ReviewTrace) UnseenSequenceRatio(seen map[string]bool) float64 {
	uniq := map[string]bool{}
	for i := 1; i < len(t.ToolCalls); i++ {
		uniq[bigramKey(t.ToolCalls[i-1], t.ToolCalls[i])] = true
	}
	if len(uniq) == 0 {
		return 0
	}
	unseen := 0
	for k := range uniq {
		if !seen[k] {
			unseen++
		}
	}
	return float64(unseen) / float64(len(uniq))
}

// ErrorDensity is the friction signal: failed tool results over total tool calls, in
// [0,1]. A trace with no calls has no density (0).
func (t ReviewTrace) ErrorDensity() float64 {
	if len(t.ToolCalls) == 0 {
		return 0
	}
	return clampUnit(float64(t.Errors) / float64(len(t.ToolCalls)))
}

// FoldTrace records a trace's shape (unigrams + bigrams) into seen, so the next
// candidate's novelty and unseen-sequence signals are measured against everything
// before it — the online history a live loop maintains. Callers pass the SAME seen
// map to Estimate/Decide and this fold.
func FoldTrace(seen map[string]bool, t ReviewTrace) {
	for _, tool := range t.ToolCalls {
		seen[unigramKey(tool)] = true
	}
	for i := 1; i < len(t.ToolCalls); i++ {
		seen[bigramKey(t.ToolCalls[i-1], t.ToolCalls[i])] = true
	}
}

func (c ReviewGateConfig) blendSignal(novelty, unseen, errDensity float64) float64 {
	wsum := c.WNovelty + c.WUnseenSequence + c.WErrorDensity
	if wsum <= 0 {
		return 0
	}
	blended := (c.WNovelty*novelty + c.WUnseenSequence*unseen + c.WErrorDensity*errDensity) / wsum
	return clampUnit(blended)
}

func unigramKey(tool string) string     { return "1\x1f" + tool }
func bigramKey(prev, cur string) string { return "2\x1f" + prev + "\x1f" + cur }

func clampUnit(x float64) float64 {
	switch {
	case x < 0:
		return 0
	case x > 1:
		return 1
	default:
		return x
	}
}
