package ablate

// The DETERMINISTIC-REPLAY rung of the MODEL-STRENGTH axis (#5413, follow-on to #4412).
//
// WHAT WAS MISSING. #4412 shipped the TierScorer seam, the ladder replay, the
// LOAD_BEARING / REDUNDANT / HOBBLING / UNMEASURED classifier and the flat verdicts[]
// payload — then bound StubTierScorer in production. The stub measures NOTHING and says
// so, which is an honest refusal (ClassifyStrength returns UNMEASURED, and the #4414 debt
// router drops UNMEASURED, so a stub can never auto-file a deletion issue) — but it also
// means the axis earns nothing. Nothing on trunk can return Measured=true.
//
// WHAT THIS FILE LANDS. The two halves of a real scorer that can be gated at $0 and
// byte-exact in CI, both behind the SAME TierScorer interface production already binds:
//
//	ReplayTierScorer   serves per-rung outcomes from a RECORDED score table, so a
//	                   measurement made ONCE — however expensive it was to produce —
//	                   replays deterministically forever after. Same table => same
//	                   verdict, with no model, no network, and no per-run spend.
//	BudgetTierScorer   a COST FENCE that wraps any TierScorer and stops after N scored
//	                   rungs. It exists for the live caller: a per-rung model call is the
//	                   expensive thing, and an N-arm sweep across a 3-rung ladder is
//	                   3N calls, so the budget is what keeps "grade everything" from
//	                   being unbounded spend.
//
// WHAT IS STILL NOT WIRED — and this file does not pretend otherwise. The scorer that
// actually CALLS a model per rung. That rung needs a tier -> model binding and live
// account access, neither of which can be gated deterministically in CI. It drops in
// behind this same interface: a live caller wrapped in BudgetTierScorer, recording into a
// TierScoreTable that ReplayTierScorer then serves back for free.
//
// THE HONESTY FENCE STILL HOLDS. A rung the table has no record for returns
// Measured=false with a Detail naming the miss — NEVER a zero score. An absent
// measurement and a measured zero must never be the same value on the page, because
// #4414 files real code-deletion issues off these grades. Every refusal path here is a
// Measured=false, not a fabricated 0.0.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
)

// TierScoreRecord is ONE recorded measurement: an arm's outcome at one rung of the
// weak -> strong ladder. Detail carries the provenance of that number (which model, which
// harness, which run) so a replayed verdict can always be traced back to what produced it.
type TierScoreRecord struct {
	Tier   string  `json:"tier"`
	ArmID  string  `json:"arm_id"`
	Score  float64 `json:"score"`
	Detail string  `json:"detail,omitempty"`
}

// TierScoreTable is the on-disk replay artifact: a flat list of recorded rungs plus a
// Source naming where the measurements came from. Flat rather than nested tier->arm maps
// on purpose — the same shape the report's verdicts[] uses, so it diffs readably in git
// and a partial table (some rungs measured, some not) is expressible without null holes.
type TierScoreTable struct {
	// Source names what produced these numbers (a harness id, a run id, a date). It is
	// echoed into every replayed TierOutcome's Detail, so a verdict graded off this table
	// carries its provenance rather than appearing from nowhere.
	Source  string            `json:"source,omitempty"`
	Records []TierScoreRecord `json:"records"`
}

// ReplayTierScorer serves recorded per-rung outcomes. It is the deterministic-replay
// TierScorer: construction validates the whole table up front, so a malformed or
// ambiguous artifact fails LOUD at bind time rather than silently grading a sweep off
// half a table.
type ReplayTierScorer struct {
	source string
	scores map[string]TierScoreRecord
}

// compile-time assertion: the replay scorer satisfies the seam production binds.
var _ TierScorer = (*ReplayTierScorer)(nil)

// replayKey is the (tier, arm) lookup key. The NUL separator cannot occur in either
// component, so no arm id can collide with a tier boundary.
func replayKey(tier, armID string) string { return tier + "\x00" + armID }

// NewReplayTierScorer validates a table and returns the scorer that replays it.
//
// It refuses, rather than repairs, three things — each of which would otherwise turn into
// a wrong grade downstream:
//
//   - an unknown tier, because a typo'd rung would silently never be found, and every arm
//     would replay as UNMEASURED for a reason nothing on the page names;
//   - an empty arm id, which can never match a run and is always a producer bug;
//   - the SAME (tier, arm) recorded twice with DIFFERENT scores, because then "replay" has
//     no single answer and determinism is a coin flip on map order.
//
// A duplicate carrying the IDENTICAL score is collapsed, not refused: re-recording the
// same measurement is idempotent, not ambiguous.
func NewReplayTierScorer(table TierScoreTable) (*ReplayTierScorer, error) {
	scores := make(map[string]TierScoreRecord, len(table.Records))
	for i, rec := range table.Records {
		tier := strings.ToLower(strings.TrimSpace(rec.Tier))
		armID := strings.TrimSpace(rec.ArmID)
		if !validModelTier(tier) {
			return nil, fmt.Errorf("ablate: tier score table record %d: unknown model tier %q (known: %s)",
				i, rec.Tier, strings.Join(modelTierLadder, ", "))
		}
		if armID == "" {
			return nil, fmt.Errorf("ablate: tier score table record %d (tier %q): empty arm_id", i, tier)
		}
		key := replayKey(tier, armID)
		if prev, dup := scores[key]; dup && prev.Score != rec.Score {
			return nil, fmt.Errorf("ablate: tier score table records arm %q at tier %q twice with different scores (%v and %v): replay would not be deterministic",
				armID, tier, prev.Score, rec.Score)
		}
		rec.Tier, rec.ArmID = tier, armID
		scores[key] = rec
	}
	return &ReplayTierScorer{source: strings.TrimSpace(table.Source), scores: scores}, nil
}

// LoadReplayTierScorer reads a TierScoreTable JSON artifact and binds a scorer to it.
// This is the production-reachable form: the path an operator points at a recorded sweep.
func LoadReplayTierScorer(path string) (*ReplayTierScorer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("ablate: read tier score table: %w", err)
	}
	var table TierScoreTable
	if err := json.Unmarshal(raw, &table); err != nil {
		return nil, fmt.Errorf("ablate: parse tier score table %s: %w", path, err)
	}
	if table.Source == "" {
		table.Source = path
	}
	return NewReplayTierScorer(table)
}

// Rungs reports how many (tier, arm) measurements the table carries, for a caller that
// wants to say "replaying N recorded rungs" before grading anything.
func (r *ReplayTierScorer) Rungs() int {
	if r == nil {
		return 0
	}
	return len(r.scores)
}

// Tiers reports the ladder rungs this table actually has records for, in canonical
// weak -> strong order. A caller can pass this straight to AnnotateModelStrength rather
// than guessing a ladder the table cannot serve.
func (r *ReplayTierScorer) Tiers() []string {
	if r == nil {
		return nil
	}
	seen := map[string]bool{}
	for _, rec := range r.scores {
		seen[rec.Tier] = true
	}
	return canonicalTiers(seen)
}

// ScoreArm replays the recorded outcome for one rung.
//
// A HIT is Measured=true carrying the recorded score and its provenance. A MISS is
// Measured=false naming exactly which (tier, arm) the table lacks — never a zero score,
// because the classifier must be able to tell "measured no effect" from "nobody measured
// this", and #4414 deletes code off that distinction.
func (r *ReplayTierScorer) ScoreArm(_ context.Context, tier, armID string, _ map[string]string) (TierOutcome, error) {
	rec, ok := r.scores[replayKey(strings.ToLower(strings.TrimSpace(tier)), strings.TrimSpace(armID))]
	if !ok {
		return TierOutcome{
			Detail: fmt.Sprintf("no recorded measurement for arm %q at tier %q in the replay table (%s)", armID, tier, r.sourceLabel()),
		}, nil
	}
	detail := rec.Detail
	if detail == "" {
		detail = "replayed from " + r.sourceLabel()
	}
	return TierOutcome{Score: rec.Score, Measured: true, Detail: detail}, nil
}

func (r *ReplayTierScorer) sourceLabel() string {
	if r == nil || r.source == "" {
		return "unnamed tier score table"
	}
	return r.source
}

// RecordTierScores folds an already-graded report back into a replay table, so a sweep
// that WAS measured (by whatever expensive path produced it) becomes a $0 artifact the
// next run replays. Only measured rungs are recorded: an unmeasured rung must not be
// frozen into the table as a score, or the next replay would serve a fabricated zero.
//
// THE BASELINE ROW MATTERS. The baseline arm is deliberately left ungraded by
// AnnotateModelStrength, so it carries no ModelStrength card to harvest — but a rung only
// counts as measured when BOTH sides of the subtraction were, so a table without the
// baseline's own scores replays as entirely UNMEASURED. Its per-tier score is recovered
// from any graded arm's TierDelta.BaselineScore, which is the same number by construction.
//
// Records come out in a stable (tier-ladder, arm) order so the artifact diffs cleanly.
func RecordTierScores(rep *Report, source string) TierScoreTable {
	table := TierScoreTable{Source: strings.TrimSpace(source)}
	if rep == nil {
		return table
	}
	// baselineAt collapses the baseline's per-tier score, which every graded arm's
	// trajectory repeats, into the single record the replay needs.
	baselineAt := map[string]float64{}
	for i := range rep.Runs {
		run := &rep.Runs[i]
		if run.ModelStrength == nil {
			continue
		}
		for _, td := range run.ModelStrength.Tiers {
			if !td.Measured {
				continue
			}
			baselineAt[td.Tier] = td.BaselineScore
			table.Records = append(table.Records, TierScoreRecord{
				Tier:   td.Tier,
				ArmID:  run.ArmID,
				Score:  td.Score,
				Detail: td.Detail,
			})
		}
	}
	for tier, score := range baselineAt {
		table.Records = append(table.Records, TierScoreRecord{
			Tier:   tier,
			ArmID:  rep.Baseline,
			Score:  score,
			Detail: "baseline reference score recovered from the graded trajectory",
		})
	}
	sort.SliceStable(table.Records, func(a, b int) bool {
		ta, tb := tierRank(table.Records[a].Tier), tierRank(table.Records[b].Tier)
		if ta != tb {
			return ta < tb
		}
		return table.Records[a].ArmID < table.Records[b].ArmID
	})
	return table
}

// tierRank orders a tier by its position on the weak -> strong ladder.
func tierRank(tier string) int {
	for i, t := range modelTierLadder {
		if t == tier {
			return i
		}
	}
	return len(modelTierLadder)
}

// BudgetTierScorer is the COST FENCE for per-rung scoring: it forwards at most Budget
// calls to the wrapped scorer and refuses every rung after that.
//
// It counts CALLS, not measured results, because that is the honest cost model for the
// live scorer this exists to bound — a model call that comes back unusable still cost
// money. Once the budget is spent the fence short-circuits WITHOUT calling the inner
// scorer at all, which is the whole point: the refusal has to happen before the spend.
//
// FAIL CLOSED. A budget below 1 refuses every rung. A cost fence whose zero value means
// "unlimited" is a footgun — the one caller who forgets to set it pays for the whole
// sweep — so the zero value here spends nothing and grades UNMEASURED, which is a
// refusal the classifier already handles honestly.
//
// DETERMINISM. Exhaustion is order-sensitive by nature, so the replay is deterministic
// exactly when the caller is sequential. AnnotateModelStrength is (baseline per tier,
// then each arm in report order across the ladder), so a given (report, ladder, budget)
// always yields the same verdicts. The mutex makes concurrent use safe, not ordered.
type BudgetTierScorer struct {
	inner  TierScorer
	budget int

	mu    sync.Mutex
	spent int
}

// compile-time assertion: the fence satisfies the seam it wraps.
var _ TierScorer = (*BudgetTierScorer)(nil)

// NewBudgetTierScorer wraps inner in a fence of at most budget scored rungs. A nil inner
// takes the stub, so the fence is never the thing that turns a missing scorer into a
// panic.
func NewBudgetTierScorer(inner TierScorer, budget int) *BudgetTierScorer {
	if inner == nil {
		inner = StubTierScorer{}
	}
	return &BudgetTierScorer{inner: inner, budget: budget}
}

// Spent reports how many rungs the fence has forwarded, for the caller that wants to
// print "scored N of a budget of M" after a sweep.
func (b *BudgetTierScorer) Spent() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.spent
}

// ScoreArm forwards one rung if the budget allows, and otherwise refuses it as
// explicitly unmeasured — naming the budget, so an operator reading a wall of UNMEASURED
// verdicts learns it was the fence and not a missing measurement.
func (b *BudgetTierScorer) ScoreArm(ctx context.Context, tier, armID string, features map[string]string) (TierOutcome, error) {
	b.mu.Lock()
	if b.spent >= b.budget {
		b.mu.Unlock()
		return TierOutcome{
			Detail: fmt.Sprintf("per-rung scoring budget of %d exhausted before arm %q at tier %q: raise the budget to grade the rest of the ladder", b.budget, armID, tier),
		}, nil
	}
	b.spent++
	b.mu.Unlock()
	return b.inner.ScoreArm(ctx, tier, armID, features)
}
