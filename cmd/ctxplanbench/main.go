// Command ctxplanbench measures the ctxplan planned VIEW over REAL Claude Code session
// transcripts — the empirical counterpart to internal/ctxplan/scaling.go's synthetic
// Params model (issue #559). scaling.go takes Params (a mean tokens/turn, a hit rate) and
// COMPUTES the resident-token curve; this command MEASURES it on the heaviest real
// transcripts on the box, so the headline "O(1) resident AND exact recall" is a number
// read off real sessions, not an assumption.
//
// It ingests each transcript through the SHIPPED cdb→recall ingest (so a result the
// write-time gate quarantines is sealed here too — the poison/suppression floor is the
// real gate's decision, not ours), bridges each recall page into a ctxplan Span, and
// replays the session turn-by-turn through the real planner (ctxplan.Materialize) and the
// real page-fault handler (ctxplan.DemandPage), reporting the three quantities the issue
// names:
//
//   - RESIDENT TOKENS — planned O(1) view vs linear Θ(N) vs compaction cap, priced in the
//     REAL per-span token sizes the transcript carries (ceil(bytes/4), the same proxy
//     ctxplan.TokenCost uses). Cumulative sums are the empirical analogue of scaling.go's
//     cumLinear / cumCapped; the peak planned resident is asserted ≤ the budget.
//   - FAULT RATE — the forecast-MISS rate. The ground-truth "reference" signal is real
//     lexical overlap: a turn references an older span when the arriving result's content
//     fingerprint appears in that span's descriptor. A miss is a referenced span the PRIOR
//     resident view had elided; each miss is SERVED through ctxplan.DemandPage, proving the
//     planned regime recovers it (a page fault, not a lost fact). The fault tax is the
//     served tokens — scaling.go's FaultTaxCum, measured.
//   - QUALITY VS COMPACTION — planned keeps exact recall every turn (ctxplan.Audit.Faithful
//     witnessed on the real plans); compaction drops the originals (ctxplan.CompactionView
//     strips the recovery handle, so Audit reports it UNFAITHFUL). The served faults ARE the
//     facts the planned regime recovers that compaction would lose; compaction's oldest-fact
//     survival is ρ^k priced on the REAL compaction count.
//
// The forecast is a deliberately cheap, real heuristic a deployment could actually run: the
// union of the last K turn descriptors as intents (a recency window), plus the durable spans
// pinned. It is NOT an oracle — a MISS costs one demand-page, never a lost fact, which is the
// whole point of measuring the fault rate on real data.
//
// Usage:
//
//	ctxplanbench -selfcheck                                  # synthetic known-answer pipeline check
//	ctxplanbench -transcripts a.jsonl,b.jsonl               # measure named transcripts
//	ctxplanbench -heaviest 5                                # auto-discover the 5 heaviest under ~/.claude
//	ctxplanbench -transcripts a.jsonl -budget 8000 -window 6 -out report.json
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/recall"

	// Register the context-MMU's page-out backend so the ingest runs in its full
	// production configuration (a quarantined/oversize result pages out to the CAS, the
	// path the live gate takes). recall already pulls this in; the blank import here keeps
	// the bench honest if that transitive edge ever changes.
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the synthetic known-answer pipeline check and exit")
	transcripts := flag.String("transcripts", "", "comma-separated Claude Code session .jsonl paths to measure")
	heaviest := flag.Int("heaviest", 0, "auto-discover and measure the N heaviest real transcripts under the claude projects dir")
	budget := flag.Int("budget", 8000, "resident-token working-set cap W (the O(1) bound)")
	window := flag.Int("window", 6, "forecast recency window K (last K turn descriptors as intents)")
	probeCap := flag.Int("probe-cap", 0, "bounded-probe candidate cap c (0 = ctxplan default 128); the per-turn planner-compute bound")
	probeRecency := flag.Int("probe-recency", 0, "bounded-probe recency tail (0 = ctxplan default 32)")
	out := flag.String("out", "", "optional JSON report path")
	flag.Parse()

	// The bounded probe's tuning. The zero value is valid — ctxplan.ProbeOptions.orDefaults
	// fills DefaultMaxCandidates/DefaultRecencyWindow — so an operator who passes nothing gets
	// the shipped defaults, and -probe-cap N demonstrates the bound biting on a real session.
	probe := ctxplan.ProbeOptions{MaxCandidates: *probeCap, RecencyWindow: *probeRecency}

	if *selfcheck {
		if err := runSelfcheck(*budget, *window, probe); err != nil {
			fmt.Fprintln(os.Stderr, "selfcheck FAIL:", err)
			os.Exit(1)
		}
		fmt.Println("selfcheck OK")
		return
	}

	paths, err := resolvePaths(*transcripts, *heaviest)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ctxplanbench:", err)
		os.Exit(2)
	}
	if len(paths) == 0 {
		fmt.Fprintln(os.Stderr, "ctxplanbench: no transcripts. Pass -transcripts a.jsonl,b.jsonl or -heaviest N.")
		os.Exit(2)
	}

	ctx := context.Background()
	results := make([]sessionResult, 0, len(paths))
	for _, p := range paths {
		r, err := measureFile(ctx, p, *budget, *window, probe)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  skip %s: %v\n", filepath.Base(p), err)
			continue
		}
		results = append(results, r)
		printSession(os.Stdout, r)
	}
	if len(results) == 0 {
		fmt.Fprintln(os.Stderr, "ctxplanbench: no transcript produced a measurement.")
		os.Exit(1)
	}
	printAggregate(os.Stdout, results, *budget, *window)

	if *out != "" {
		agg := aggregate(results)
		b, _ := json.MarshalIndent(map[string]any{
			"budget":   *budget,
			"window":   *window,
			"sessions": results,
			"total":    agg,
		}, "", "  ")
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write report:", err)
			os.Exit(1)
		}
		fmt.Printf("\nwrote %s\n", *out)
	}
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

// sessionResult is the per-transcript measurement. Every token field is priced in REAL
// per-span token sizes (ceil(bytes/4)); every count is read off the replay, never modeled.
type sessionResult struct {
	Source string `json:"source"`
	Pages  int    `json:"pages"`  // benign pages replayed (one per turn)
	Sealed int    `json:"sealed"` // pages the write-time gate quarantined (never a candidate)
	Turns  int    `json:"turns"`  // benign pages replayed as turns
	Budget int    `json:"budget"` // W
	Window int    `json:"window"` // K

	// RESIDENT TOKENS — cumulative across the session (the prefill/$ proxy).
	LinearCumTok    int64 `json:"linear_cum_tokens"`   // Σ turn: every span resident (Θ(N²) the linear transcript forces)
	PlannedCumTok   int64 `json:"planned_cum_tokens"`  // Σ turn: the planned O(1) view (the bent curve)
	CompactCumTok   int64 `json:"compact_cum_tokens"`  // Σ turn: the same view capped at W (compaction's resident cost)
	PeakPlannedTok  int   `json:"peak_planned_tokens"` // max planned resident over the session (≤ W unless pins overrun)
	OverBudgetTurns int   `json:"over_budget_turns"`   // turns pins alone exceeded W (they stay; honesty, not a bug)
	FaultTaxCum     int64 `json:"fault_tax_cum"`       // Σ served-fault tokens (the miss re-prefill tax, measured)

	// FAULT RATE — the forecast-MISS cost on the real reference signal.
	References int     `json:"references"` // older spans a turn actually referenced (real lexical overlap)
	Faults     int     `json:"faults"`     // referenced-but-elided-at-the-prior-turn (the misses)
	Served     int     `json:"served"`     // misses recovered via DemandPage (FaultServed)
	Refused    int     `json:"refused"`    // misses the gate refused (sealed — the floor held)
	FaultRate  float64 `json:"fault_rate"` // faults / references

	// QUALITY VS COMPACTION.
	PlannedFaithfulTurns  int     `json:"planned_faithful_turns"`  // turns Audit(plan).Faithful (exact recall held)
	CompactionLossTurns   int     `json:"compaction_loss_turns"`   // turns CompactionView destroyed ≥1 fact
	CompactionRecallFinal float64 `json:"compaction_recall_final"` // ρ^k survival of the oldest fact at session end
	FactsRecovered        int     `json:"facts_recovered"`         // == served: facts planned recovers, compaction loses

	// PLANNING COST — the per-turn planner WORK (candidate-scoring ops), measured. The
	// full-scan planner (ctxplan.Materialize → PlanCells) scores every span each turn (Σ N_t
	// = Θ(N²)); a persistent ctxplan.Index probes a BOUNDED candidate set (≤ cap each turn, Σ
	// = Θ(c·N), LINEAR). Both go through the same Optimize(ObjGreedy) under the same forecast
	// and budget, so the ONLY difference is the candidate set — and the resident plan is
	// compared turn-for-turn (PlanAgreeTurns). This is scaling.go's PlannerComputeCum vs
	// IndexBoundedPlannerCompute, measured on the real session instead of modeled.
	MaxCandidates      int   `json:"max_candidates"`       // the bounded-probe cap c (effective, after defaults)
	RecencyWindow      int   `json:"recency_window"`       // the bounded-probe recency tail (effective)
	FullScanCandCum    int64 `json:"fullscan_cand_cum"`    // Σ turn: candidates the full-scan planner scored (== N each turn) → Θ(N²)
	ProbeCandCum       int64 `json:"probe_cand_cum"`       // Σ turn: candidates the bounded probe scored (≤ cap) → Θ(c·N)
	PeakProbe          int   `json:"peak_probe"`           // max probe size over the session (≤ cap — the per-turn bound held)
	PlanAgreeTurns     int   `json:"plan_agree_turns"`     // turns the bounded plan's resident set+cost == the full-scan plan's
	BoundedTurns       int   `json:"bounded_turns"`        // turns the probe was actually smaller than N (cap bit) — the real fidelity test
	BoundedAgreeTurns  int   `json:"bounded_agree_turns"`  // agreement among the BoundedTurns (the cap pruned, the plan held)
	FullScanPlannedCum int64 `json:"fullscan_planned_cum"` // Σ full-scan resident tokens (the planned curve, matched entry point)
	ProbePlannedCum    int64 `json:"probe_planned_cum"`    // Σ bounded resident tokens (== full-scan on agreement turns)
}

// runSelfcheck drives the WHOLE pipeline (ingest → recall reload → replay) over a small
// synthetic transcript of known shape, and asserts the invariants the measurement rests on.
// It is the "did the bridge + the loop survive" gate, not a unit test of the planner (the
// planner has its own). The transcript is built so a tight budget forces ≥1 forecast miss
// that DemandPage must serve — exercising the recoverability claim end to end.
func runSelfcheck(_, window int, probe ctxplan.ProbeOptions) error {
	// A tight budget (enough for ~1 span) forces elision deterministically regardless of the
	// operator's -budget flag: the selfcheck is a fixed known-answer pipeline gate, so it
	// picks its own budget that the synthetic transcript is built to overflow.
	const budget = 20
	const (
		durable = `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"u0","content":"preference: always deploy from the release branch, never main"}]}}`
		read    = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u1","name":"Read"},{"type":"tool_result","tool_use_id":"u1","content":"auth token rotation runbook step one mint roll revoke"}]}}`
		noiseA  = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u2","name":"Bash"},{"type":"tool_result","tool_use_id":"u2","content":"build log compiled 412 files quietly on tuesday afternoon"}]}}`
		noiseB  = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u3","name":"Bash"},{"type":"tool_result","tool_use_id":"u3","content":"weather sunny light wind from the west nothing relevant"}]}}`
		// The arriving turn that REFERENCES the read ("auth token rotation") — a real
		// lexical back-reference, the ground-truth signal the fault rate is measured on.
		edit = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u4","name":"Edit"},{"type":"tool_result","tool_use_id":"u4","content":"applied the auth token rotation runbook edit to the billing service"}]}}`
	)
	tmp, err := os.MkdirTemp("", "ctxplanbench-selfcheck-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	tpath := filepath.Join(tmp, "selfcheck.jsonl")
	if err := os.WriteFile(tpath, []byte(strings.Join([]string{durable, read, noiseA, noiseB, edit}, "\n")), 0o644); err != nil {
		return err
	}
	ctx := context.Background()
	r, err := measureFile(ctx, tpath, budget, window, probe)
	if err != nil {
		return fmt.Errorf("measureFile: %w", err)
	}
	switch {
	case r.Turns < 3:
		return fmt.Errorf("too few turns replayed (%d) — ingest/reload lost spans", r.Turns)
	case r.PeakPlannedTok > r.Budget:
		return fmt.Errorf("planned resident (%d) exceeded the budget (%d) — the O(1) bound broke", r.PeakPlannedTok, r.Budget)
	case r.PlannedFaithfulTurns != r.Turns:
		return fmt.Errorf("planned faithful %d/%d turns — exact recall did not hold every turn", r.PlannedFaithfulTurns, r.Turns)
	case r.PlannedCumTok > r.LinearCumTok:
		return fmt.Errorf("planned cum (%d) exceeded linear cum (%d) — planned resident never exceeds the full transcript", r.PlannedCumTok, r.LinearCumTok)
	case r.CompactionLossTurns == 0:
		return fmt.Errorf("compaction never destroyed a fact (0 loss turns) — the tight budget did not force any elision, so the contrast is vacuous")
	}
	// The fault-tax + served counts must be internally consistent: served faults paged real bytes.
	if r.Served > r.Faults {
		return fmt.Errorf("served (%d) > faults (%d) — DemandPage over-counted", r.Served, r.Faults)
	}
	// Planning-cost invariants. The synthetic transcript is short (N < cap), so the bounded
	// probe sees every span: it must score no more than the full scan, never overrun its cap,
	// and reach the IDENTICAL resident plan every turn (the bounded probe changes the planner's
	// COST, not its answer — index.go's behavior-preservation, end to end here).
	switch {
	case r.ProbeCandCum > r.FullScanCandCum:
		return fmt.Errorf("bounded probe scored more candidates (%d) than the full scan (%d) — impossible", r.ProbeCandCum, r.FullScanCandCum)
	case r.PeakProbe > r.MaxCandidates:
		return fmt.Errorf("peak probe (%d) exceeded the cap (%d) — the per-turn planner-compute bound broke", r.PeakProbe, r.MaxCandidates)
	case r.PlanAgreeTurns != r.Turns:
		return fmt.Errorf("bounded plan diverged from full-scan on %d/%d turns at N<cap — the probe must reach the same answer when it sees every span", r.Turns-r.PlanAgreeTurns, r.Turns)
	case r.ProbePlannedCum != r.FullScanPlannedCum:
		return fmt.Errorf("bounded resident tokens (%d) != full-scan (%d) on identical plans — the cost win was not free", r.ProbePlannedCum, r.FullScanPlannedCum)
	}
	fmt.Printf("  selfcheck replay: %d turns, peak-planned %d/%d, faithful %d/%d, %d ref → %d faults (%d served), oldest-recall %.3f\n",
		r.Turns, r.PeakPlannedTok, r.Budget, r.PlannedFaithfulTurns, r.Turns, r.References, r.Faults, r.Served, r.CompactionRecallFinal)
	fmt.Printf("  planning cost:     full-scan %d cand, bounded-probe %d cand (peak %d/%d), plan-agree %d/%d turns\n",
		r.FullScanCandCum, r.ProbeCandCum, r.PeakProbe, r.MaxCandidates, r.PlanAgreeTurns, r.Turns)
	return nil
}

// measureFile ingests one transcript and replays it. The ingest drives the shipped
// write-time gate (so sealed decisions are real); Persist+Load lifts the result into a
// recall.Session whose Resolve re-screens on page-in (the rung-4 floor), so ctxplan's
// Materialize/DemandPage go through the real gate, not a stub.
func measureFile(ctx context.Context, path string, budget, window int, probe ctxplan.ProbeOptions) (sessionResult, error) {
	rec, ist, err := cdb.IngestSession(ctx, path, filepath.Base(path))
	if err != nil {
		return sessionResult{}, fmt.Errorf("ingest %s: %w", filepath.Base(path), err)
	}
	tmp, err := os.MkdirTemp("", "ctxplanbench-*")
	if err != nil {
		return sessionResult{}, err
	}
	defer os.RemoveAll(tmp)
	if err := rec.Persist(tmp); err != nil {
		return sessionResult{}, fmt.Errorf("persist: %w", err)
	}
	sess, err := recall.Load(tmp)
	if err != nil {
		return sessionResult{}, fmt.Errorf("reload: %w", err)
	}
	pages := sess.Pages()
	r := replay(ctx, sess, pages, budget, window, probe)
	r.Source = filepath.Base(path)
	if r.Pages == 0 && ist.Pages > 0 {
		r.Pages = ist.Pages // keep a non-zero denominator if every page re-quarantined on reload
	}
	return r, nil
}

// replay is the turn-by-turn measurement core. It grows the store one benign span per turn,
// plans the resident view with a recency-window forecast, and scores the four quantities
// (resident tokens, fault rate, quality vs compaction, and the per-turn planning COST).
func replay(ctx context.Context, sess *recall.Session, pages []recall.Page, budget, window int, probe ctxplan.ProbeOptions) sessionResult {
	r := sessionResult{Budget: budget, Window: window}
	store := &recallStore{sess: sess, pages: pages}
	for _, p := range pages {
		if p.Quarantined {
			r.Sealed++
		} else {
			r.Pages++
		}
	}
	r.Turns = r.Pages

	// The PERSISTENT bounded index — maintained incrementally across turns (the SessionPlanner
	// pattern, here over the recall-bridged spans the full scan also sees), so each turn probes
	// a BOUNDED candidate set instead of re-scoring all N. r.MaxCandidates / r.RecencyWindow
	// record the EFFECTIVE cap (ctxplan fills 0 with DefaultMaxCandidates / DefaultRecencyWindow).
	index := ctxplan.NewIndex()
	idxNext := 0 // pages already lowered into the persistent index (the append cursor)
	r.MaxCandidates = probe.MaxCandidates
	if r.MaxCandidates <= 0 {
		r.MaxCandidates = ctxplan.DefaultMaxCandidates
	}
	r.RecencyWindow = probe.RecencyWindow
	if r.RecencyWindow <= 0 {
		r.RecencyWindow = ctxplan.DefaultRecencyWindow
	}

	var (
		prevView     *ctxplan.View
		prevResident = map[string]bool{}
		linearPer    int64 // running resident tokens if the whole transcript stayed resident
		retain       = 0.7 // compaction per-event retain (scaling.go's default)
	)
	W := int64(budget)

	for t := 0; t < len(pages); t++ {
		p := pages[t]
		if p.Quarantined {
			continue // a sealed span is never a resident candidate; it is not a "turn"
		}
		store.upto = t + 1
		// Catch the persistent index up to include every span the full scan now sees (pages
		// [idxNext, t], INCLUDING any sealed pages skipped between turns — they enter the index
		// as Sealed spans exactly as recallStore.Spans surfaces them, so the two candidate
		// universes are identical and the only difference measured below is the access PATH).
		for ; idxNext <= t; idxNext++ {
			index.Add(pageToSpan(pages[idxNext]))
		}
		spans, _ := store.Spans(ctx)
		fc := recencyForecast(spans, window)
		view, err := ctxplan.Materialize(ctx, store, fc, ctxplan.Budget{Tokens: budget}, nil)
		if err != nil || view.Plan.CostUsed > budget && !view.Plan.OverBudget {
			// Materialize only errors on a store failure; a cost overrun without OverBudget
			// would be a planner bug. Either way, skip the turn rather than corrupt the tally.
			continue
		}

		// --- planning cost: the full scan (view.Plan, == ctxplan.PlanCells over all N spans)
		// vs the bounded probe over the persistent index, SAME forecast + budget + planner, so
		// the only variable is the candidate set. view.Plan.Candidates is N; idxPlan.Candidates
		// is the bounded probe size (≤ cap). The resident plans are compared turn-for-turn. ---
		idxPlan := index.PlanCells(fc, ctxplan.Budget{Tokens: budget}, nil, probe)
		r.FullScanCandCum += int64(view.Plan.Candidates)
		r.ProbeCandCum += int64(idxPlan.Candidates)
		if idxPlan.Candidates > r.PeakProbe {
			r.PeakProbe = idxPlan.Candidates
		}
		r.FullScanPlannedCum += int64(view.Plan.CostUsed)
		r.ProbePlannedCum += int64(idxPlan.CostUsed)
		agree := sameResident(view.Plan, idxPlan)
		if agree {
			r.PlanAgreeTurns++
		}
		if idxPlan.Candidates < view.Plan.Candidates {
			// The cap actually pruned candidates this turn — the real fidelity test of the
			// bounded probe (a turn where it sees fewer spans than the full scan yet must still
			// reach the same resident plan when the relevant+recent+durable set sufficed).
			r.BoundedTurns++
			if agree {
				r.BoundedAgreeTurns++
			}
		}

		// --- resident tokens (this turn's contribution) ---
		tok := int64(ctxplan.TokenCost(pageToSpan(p)))
		linearPer += tok
		r.LinearCumTok += linearPer
		r.PlannedCumTok += int64(view.Plan.CostUsed)
		if linearPer < W {
			r.CompactCumTok += linearPer
		} else {
			r.CompactCumTok += W
		}
		if view.Plan.CostUsed > r.PeakPlannedTok {
			r.PeakPlannedTok = view.Plan.CostUsed
		}
		if view.Plan.OverBudget {
			r.OverBudgetTurns++
		}

		// --- quality vs compaction (witnessed on the real plan) ---
		w := ctxplan.Audit(view.Plan)
		if w.Faithful {
			r.PlannedFaithfulTurns++
		}
		if len(ctxplan.Audit(ctxplan.CompactionView(view.Plan)).Unrecoverable) > 0 {
			r.CompactionLossTurns++
		}

		// --- fault rate: did the span arriving this turn reference an older span the
		// PRIOR resident view had elided? Each miss is served through DemandPage. ---
		if prevView != nil {
			for _, j := range referencesTo(pages, t) {
				r.References++
				id := spanID(j)
				if prevResident[id] {
					continue // already resident — a forecast HIT
				}
				r.Faults++
				_, fault, ferr := ctxplan.DemandPage(ctx, store, *prevView, id)
				if ferr != nil {
					continue
				}
				switch fault.Status {
				case ctxplan.FaultServed:
					r.Served++
					r.FaultTaxCum += int64(fault.Tokens)
				case ctxplan.FaultRefused:
					r.Refused++ // the gate held (sealed/tombstoned on re-screen)
				}
			}
		}

		prevView = &view
		prevResident = map[string]bool{}
		for _, rd := range view.Rendered {
			prevResident[rd.ID] = true
		}
	}

	if r.References > 0 {
		r.FaultRate = float64(r.Faults) / float64(r.References)
	}
	r.FactsRecovered = r.Served
	r.CompactionRecallFinal = compactionRecall(retain, r.LinearCumTok, W)
	return r
}

// compactionRecall is the empirical analogue of scaling.go's compactionRecall: the oldest
// surviving fact retains ρ per compaction event, and the compaction count is read off the
// REAL benign byte total (cumLinear tokens) vs the cap W — not a mean tokens/turn assumption.
// With ρ=1 (lossless summary) it is 1.0; with ρ<1 it decays toward 0, the irreversible loss
// the planned view avoids. cumLinear here is the FINAL per-turn resident total (linearPer).
func compactionRecall(rho float64, cumLinear, W int64) float64 {
	r := float64(rho)
	if r <= 0 {
		r = 0.5
	}
	if r > 1 {
		r = 1
	}
	if W <= 0 || cumLinear <= 0 {
		return r
	}
	compactions := float64(cumLinear) / float64(W)
	// floor via integer truncation on the byte ratio is the scaling-model intent; recompute
	// from the token ratio the model uses (cumCapped engages the cap at ceil(W/b) per turn).
	c := float64(int64(compactions))
	if c < 0 {
		c = 0
	}
	return pow(r, c)
}

func pow(base, exp float64) float64 {
	out := 1.0
	for i := 0; i < int(exp); i++ {
		out *= base
	}
	return out
}

// sameResident reports whether two plans keep the IDENTICAL resident set and charge the
// identical budget — the order-independent equality the agreement tally needs. The bounded
// probe is a SUBSET of the full candidate set, so when the budget-optimal selection is wholly
// inside the probe the two plans match exactly; this is the turn-for-turn witness of that. It
// compares resident span IDs (render order is irrelevant) and the total resident cost, so a
// plan that selected a different span — or the same spans at a different cost — is not "same".
func sameResident(a, b ctxplan.Plan) bool {
	if len(a.Selected) != len(b.Selected) || a.CostUsed != b.CostUsed {
		return false
	}
	seen := make(map[string]bool, len(a.Selected))
	for _, s := range a.Selected {
		seen[s.ID] = true
	}
	for _, s := range b.Selected {
		if !seen[s.ID] {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// recall → ctxplan bridge
// ---------------------------------------------------------------------------

// recallStore adapts a recall core image into a ctxplan.Store. It is the "thin, higher-tier
// follow-on" internal/ctxplan/doc.go names: it imports both recall and ctxplan, so it lives
// above the foundation leaf (here, in the bench cmd, the established home for the adapters
// that are not yet on the live path). `upto` grows the visible prefix turn-by-turn so the
// replay models a session in progress over a frozen image.
type recallStore struct {
	sess  *recall.Session
	pages []recall.Page
	upto  int
}

// Spans returns the visible prefix (pages [0,upto)) lowered into ctxplan.Spans — the session-in-progress view the replay grows turn by turn.
func (s *recallStore) Spans(_ context.Context) ([]ctxplan.Span, error) {
	n := s.upto
	if n > len(s.pages) {
		n = len(s.pages)
	}
	out := make([]ctxplan.Span, n)
	for i := 0; i < n; i++ {
		out[i] = pageToSpan(s.pages[i])
	}
	return out, nil
}

// Materialize resolves a span id's bytes through the recall session's re-screening Resolve, mapping a missing/sealed page to ErrSealed and a tombstoned one to ErrTombstoned.
func (s *recallStore) Materialize(ctx context.Context, id string) ([]byte, error) {
	step, ok := stepFromID(id)
	if !ok {
		return nil, fmt.Errorf("%w: %s", ctxplan.ErrSealed, id)
	}
	b, err := s.sess.Resolve(ctx, step)
	if err == nil {
		return b, nil
	}
	if errors.Is(err, recall.ErrTombstoned) {
		return nil, fmt.Errorf("%w: %s", ctxplan.ErrTombstoned, id)
	}
	// recall wraps its sealed/absent/re-screen-quarantine in ErrSealed; anything else is a
	// lookup miss the planner counts as a refused page-in.
	return nil, fmt.Errorf("%w: %s", ctxplan.ErrSealed, id)
}

// pageToSpan lowers a recall.Page into a ctxplan.Span. The fields are 1:1 by design (both
// are the "SAFE metadata + content address" shape); the utility scalar is carried in the
// open Attrs bag the planner's utility signal reads.
func pageToSpan(p recall.Page) ctxplan.Span {
	s := ctxplan.Span{
		ID:         spanID(p.Step),
		Step:       p.Step,
		Role:       p.Role,
		Descriptor: p.Descriptor,
		Digest:     p.Digest,
		Bytes:      p.Len,
		Durability: ctxplan.NormDurability(p.Durability),
		Sealed:     p.Quarantined,
	}
	if p.Utility > 0 {
		s.Attrs = map[string]string{"utility": strconv.FormatFloat(p.Utility, 'f', -1, 64)}
	}
	return s
}

func spanID(step int) string { return fmt.Sprintf("span:%d", step) }
func stepFromID(id string) (int, bool) {
	if !strings.HasPrefix(id, "span:") {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimPrefix(id, "span:"))
	return n, err == nil
}

// ---------------------------------------------------------------------------
// Forecast + reference signal (the real, cheap heuristics the replay drives)
// ---------------------------------------------------------------------------

// recencyForecast is the deployment-realistic forecast: the union of the last K turn
// descriptors as intents (a recency window — "more of what we just touched"), plus every
// durable span pinned (the always-resident facts). It predicts ONLY from past spans; it
// never peeks at the turn about to arrive.
func recencyForecast(spans []ctxplan.Span, window int) ctxplan.Forecast {
	fc := ctxplan.Forecast{Horizon: 1}
	start := len(spans) - window
	if start < 0 {
		start = 0
	}
	seen := map[string]bool{}
	for i := start; i < len(spans); i++ {
		for t := range tokenSet(spans[i].Role + " " + spans[i].Descriptor) {
			seen[t] = true
		}
		if !spans[i].Sealed && spans[i].Durability == ctxplan.DurabilityDurable {
			fc.Pins = append(fc.Pins, spans[i].ID)
		}
	}
	for t := range seen {
		fc.Intents = append(fc.Intents, t)
	}
	sort.Strings(fc.Intents)
	return fc
}

// referencesTo returns the OLDER benign spans whose descriptor content overlaps the span
// arriving at step t — the ground-truth "this turn referenced those facts" signal, derived
// from the real transcript content. It is the same stopword-aware extractive overlap the
// planner's relevance signal uses, so "referenced" means the same thing at measure time and
// plan time.
func referencesTo(pages []recall.Page, t int) []int {
	q := tokenSet(pages[t].Descriptor)
	if len(q) == 0 {
		return nil
	}
	var out []int
	for j := 0; j < t; j++ {
		if pages[j].Quarantined {
			continue // sealed spans can never be resident — referencing one is not a planning miss
		}
		doc := tokenSet(pages[j].Role + " " + pages[j].Descriptor)
		for tok := range q {
			if doc[tok] {
				out = append(out, j)
				break
			}
		}
	}
	return out
}

// tokenSet is the lowercased, length>2, content-token set — the extractive tokenization the
// planner's relevance ranker uses, kept local so this cmd adds no coupling to ctxplan's
// unexported helpers.
func tokenSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, t := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(t) > 2 {
			out[t] = true
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// discovery + reporting
// ---------------------------------------------------------------------------

func resolvePaths(list string, heaviest int) ([]string, error) {
	if list != "" {
		var out []string
		for _, p := range strings.Split(list, ",") {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			abs, err := filepath.Abs(p)
			if err != nil {
				return nil, err
			}
			if _, err := os.Stat(abs); err != nil {
				return nil, fmt.Errorf("stat %s: %w", p, err)
			}
			out = append(out, abs)
		}
		return out, nil
	}
	if heaviest > 0 {
		return discoverHeaviest(heaviest)
	}
	return nil, nil
}

// discoverHeaviest globs the Claude Code projects dir for session transcripts and returns
// the N largest — the "heaviest REAL session transcripts" the issue names. It mirrors the
// discovery cmd/fak's fak debug --list uses.
func discoverHeaviest(n int) ([]string, error) {
	roots := claudeProjectRoots()
	type ent struct {
		path string
		sz   int64
	}
	var ents []ent
	for _, root := range roots {
		matches, _ := filepath.Glob(filepath.Join(root, "*", "*.jsonl"))
		for _, m := range matches {
			fi, err := os.Stat(m)
			if err != nil {
				continue
			}
			ents = append(ents, ent{m, fi.Size()})
		}
	}
	sort.Slice(ents, func(i, j int) bool { return ents[i].sz > ents[j].sz })
	if n > len(ents) {
		n = len(ents)
	}
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, ents[i].path)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no session transcripts found under %s", strings.Join(roots, "; "))
	}
	return out, nil
}

func claudeProjectRoots() []string {
	var roots []string
	if cfg := os.Getenv("CLAUDE_CONFIG_DIR"); cfg != "" {
		roots = append(roots, filepath.Join(cfg, "projects"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots, filepath.Join(home, ".claude", "projects"))
	}
	return roots
}

func printSession(w *os.File, r sessionResult) {
	fmt.Fprintf(w, "\n== %s : %d turns (%d sealed skipped), budget=%d window=%d ==\n", r.Source, r.Turns, r.Sealed, r.Budget, r.Window)
	fmt.Fprintf(w, "  resident tokens (cum):  linear %s   planned %s   compact %s   peak-planned %s   fault-tax %s\n",
		human(r.LinearCumTok), human(r.PlannedCumTok), human(r.CompactCumTok), human(int64(r.PeakPlannedTok)), human(r.FaultTaxCum))
	fmt.Fprintf(w, "  fault rate:             %d ref → %d faults (%.1f%% miss), %d served, %d refused\n",
		r.References, r.Faults, r.FaultRate*100, r.Served, r.Refused)
	fmt.Fprintf(w, "  quality vs compaction:  planned faithful %d/%d turns   compaction lost facts %d turns   oldest-fact recall %.3f\n",
		r.PlannedFaithfulTurns, r.Turns, r.CompactionLossTurns, r.CompactionRecallFinal)
	fmt.Fprintf(w, "  planning cost (cum):    full-scan %s cand   bounded-probe %s cand   (%s less planner work)   peak-probe %d/%d   plan-agree %d/%d (bounded %d/%d)\n",
		human(r.FullScanCandCum), human(r.ProbeCandCum), ratioX(r.FullScanCandCum, r.ProbeCandCum), r.PeakProbe, r.MaxCandidates,
		r.PlanAgreeTurns, r.Turns, r.BoundedAgreeTurns, r.BoundedTurns)
}

// ratioX renders num/den as a "N.Nx" speedup string, or "—" when the denominator is zero
// (a degenerate session that never planned a turn). It is the planner-work reduction the
// bounded probe buys, the candidate-scoring analogue of the resident-token ratio.
func ratioX(num, den int64) string {
	if den <= 0 {
		return "—"
	}
	return fmt.Sprintf("%.1fx", float64(num)/float64(den))
}

type total struct {
	Sessions             int     `json:"sessions"`
	Turns                int     `json:"turns"`
	LinearCumTok         int64   `json:"linear_cum_tokens"`
	PlannedCumTok        int64   `json:"planned_cum_tokens"`
	CompactCumTok        int64   `json:"compact_cum_tokens"`
	FaultTaxCum          int64   `json:"fault_tax_cum"`
	References           int     `json:"references"`
	Faults               int     `json:"faults"`
	Served               int     `json:"served"`
	Refused              int     `json:"refused"`
	FaultRate            float64 `json:"fault_rate"`
	PlannedFaithfulTurns int     `json:"planned_faithful_turns"`
	CompactionLossTurns  int     `json:"compaction_loss_turns"`
	FactsRecovered       int     `json:"facts_recovered"`

	// PLANNING COST (cumulative across sessions).
	FullScanCandCum    int64 `json:"fullscan_cand_cum"`
	ProbeCandCum       int64 `json:"probe_cand_cum"`
	PeakProbe          int   `json:"peak_probe"`
	PlanAgreeTurns     int   `json:"plan_agree_turns"`
	BoundedTurns       int   `json:"bounded_turns"`
	BoundedAgreeTurns  int   `json:"bounded_agree_turns"`
	FullScanPlannedCum int64 `json:"fullscan_planned_cum"`
	ProbePlannedCum    int64 `json:"probe_planned_cum"`
}

func aggregate(rs []sessionResult) total {
	var t total
	for _, r := range rs {
		t.Sessions++
		t.Turns += r.Turns
		t.LinearCumTok += r.LinearCumTok
		t.PlannedCumTok += r.PlannedCumTok
		t.CompactCumTok += r.CompactCumTok
		t.FaultTaxCum += r.FaultTaxCum
		t.References += r.References
		t.Faults += r.Faults
		t.Served += r.Served
		t.Refused += r.Refused
		t.PlannedFaithfulTurns += r.PlannedFaithfulTurns
		t.CompactionLossTurns += r.CompactionLossTurns
		t.FactsRecovered += r.FactsRecovered
		t.FullScanCandCum += r.FullScanCandCum
		t.ProbeCandCum += r.ProbeCandCum
		t.PlanAgreeTurns += r.PlanAgreeTurns
		t.BoundedTurns += r.BoundedTurns
		t.BoundedAgreeTurns += r.BoundedAgreeTurns
		t.FullScanPlannedCum += r.FullScanPlannedCum
		t.ProbePlannedCum += r.ProbePlannedCum
		if r.PeakProbe > t.PeakProbe {
			t.PeakProbe = r.PeakProbe
		}
	}
	if t.References > 0 {
		t.FaultRate = float64(t.Faults) / float64(t.References)
	}
	return t
}

func printAggregate(w *os.File, rs []sessionResult, budget, window int) {
	t := aggregate(rs)
	linRatio := 0.0
	if t.PlannedCumTok > 0 {
		linRatio = float64(t.LinearCumTok) / float64(t.PlannedCumTok)
	}
	fmt.Fprintf(w, "\n== aggregate over %d session(s) : %d turns, budget=%d window=%d ==\n", t.Sessions, t.Turns, budget, window)
	fmt.Fprintf(w, "  resident tokens (cum):  linear %s   planned %s   compact %s   (%.1fx fewer resident than linear)\n",
		human(t.LinearCumTok), human(t.PlannedCumTok), human(t.CompactCumTok), linRatio)
	fmt.Fprintf(w, "  fault rate:             %d ref → %d faults (%.1f%% miss), %d served (%s re-prefill tax), %d refused\n",
		t.References, t.Faults, t.FaultRate*100, t.Served, human(t.FaultTaxCum), t.Refused)
	fmt.Fprintf(w, "  quality vs compaction:  planned faithful %d/%d turns   compaction lost facts %d turns   %d facts recovered compaction would lose\n",
		t.PlannedFaithfulTurns, t.Turns, t.CompactionLossTurns, t.FactsRecovered)
	fmt.Fprintf(w, "  planning cost (cum):    full-scan %s cand   bounded-probe %s cand   (%s less planner work)   peak-probe %d   plan-agree %d/%d turns (bounded %d/%d)\n",
		human(t.FullScanCandCum), human(t.ProbeCandCum), ratioX(t.FullScanCandCum, t.ProbeCandCum), t.PeakProbe,
		t.PlanAgreeTurns, t.Turns, t.BoundedAgreeTurns, t.BoundedTurns)
}

func human(n int64) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}
