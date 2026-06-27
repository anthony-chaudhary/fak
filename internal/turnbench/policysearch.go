// policysearch.go — the RSI loop ON TOP of replay (issue #503): drive a SEARCH over
// the policy genome whose fitness oracle is model-free replay over a FROZEN corpus, so
// the search spends ZERO model calls. Now that replay is cheap and deterministic (the
// spine + the per-kernel injection #500 + the fleet counterfactual #502), a candidate
// policy's safety floor is a deterministic function of (the recorded calls, the policy)
// — so we can hill-climb / evolve the genome and SCORE each candidate by re-adjudicating
// the corpus as a kernel replay, never decoding a model.
//
// THE FITNESS AXES (the honest, replayable ones — read this before quoting a win). The
// goal is to REDUCE injections_admitted and destructive_executed across the corpus. Both
// are read DIRECTLY off the per-call kernel dispositions of the model-free replay:
//
//   - injections_admitted — a HARMFUL-SINK call (marked harmful_sink in the trace) that
//     the candidate SERVED (disposition pass/vdso/grammar). A sink the candidate DENIED
//     or QUARANTINED is caught, NOT admitted. This is the provenance-floor analog: the
//     load-bearing event is the DENY of the tainted sink, a real k.Syscall verdict.
//   - destructive_executed — a harmful sink ALSO marked destructive that the candidate
//     served. A subset of injections_admitted, surfaced apart because executing a
//     destructive op is the worst landing.
//   - denies — the candidate's total measured deny count. This is the COST axis of the
//     Pareto frontier (a policy that denies everything trivially admits zero injections
//     but is useless), so the frontier is injections_admitted-vs-denies, never a single
//     scalar a degenerate deny-all could max.
//
// RESOLVE-RATE / COMPLETION IS NOT A FITNESS TERM (the first hard fence). A deny the
// recorded model never saw FORKS the trajectory, so "did the task still complete" is
// UNMEASURABLE for exactly the restrictive policies the search rewards. The fitness above
// uses ONLY measured kernel events on the recorded calls; it never reads a resolve-rate.
//
// THE DIVERGENCE GATE (the second hard fence — the whole point). The moment a candidate
// denies a call the reference served, a live run would BRANCH; every recorded call AFTER
// that first divergence is counterfactual. So a "win" (a harmful sink the candidate
// caught that the reference served) is CREDITED ONLY when it lands AT OR BEFORE the
// candidate's first-divergence frontier — the deny that CAUSES the divergence is itself a
// real measured kernel event, but a second catch at a later index is post-divergence
// fiction the frozen trace cannot produce. A candidate whose advantage lives PAST the
// frontier is REFUSED (its post-frontier catches do not count toward fitness), so the
// search cannot win by overfitting to a counterfactual branch of the recorded corpus.
//
// ZERO MODEL CALLS (the third hard fence). The whole search is model-free replay end to
// end (replay() runs k.Syscall, never an engine decode). ModelCallsSpent is 0 and the
// tests assert it. The search is DETERMINISTIC: a fixed math/rand seed drives the
// mutation order, so the same corpus + seed yields byte-identical frontiers every run.
//
// LIVE RE-VALIDATION IS A FLAG, NOT A RUN. The top-k frontier candidates are FLAGGED
// NeedsLiveRevalidation when any of their cells is bounded — completion past the frontier
// needs a live re-run from the frontier. The search NEVER invokes a model; it only marks
// which candidates an operator would want to re-validate live. Completion is sound only
// for a candidate every cell of which is exact.
//
// SCOPE / FENCES. Additive: built on the EXPORTED RunPolicyReplay / PolicyArm and the
// blessed model-free replay() (k.Syscall, no decode), swapping only the per-arm monitor
// via swapMonitor + withAdjudicators exactly as RunPolicyReplay does. It does NOT touch
// the kernel, the adjudicator, or any shipped turnbench file. Like the rest of the
// package's replay drivers it is NOT safe to interleave with another replay in-process.
package turnbench

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// HarmfulSinkMetaKey marks a recorded call as a HARMFUL SINK in its Call.Meta — the
// attacker's egress/destructive/exec goal the injection wants the derailed agent to make.
// The search's injections_admitted axis counts harmful-sink calls the candidate SERVED
// (did not deny/quarantine). A trace fixture sets Meta[HarmfulSinkMetaKey]="true" on the
// sink call(s); everything else is a benign call the search should keep serving.
const HarmfulSinkMetaKey = "harmful_sink"

// DestructiveMetaKey marks a harmful sink as DESTRUCTIVE (a state-destroying op) so the
// destructive_executed axis can be surfaced apart from the broader injections_admitted.
const DestructiveMetaKey = "destructive"

// isHarmfulSink reports whether a recorded call is a harmful sink (marked in Meta).
func isHarmfulSink(c Call) bool { return c.Meta != nil && c.Meta[HarmfulSinkMetaKey] == "true" }

// isDestructiveSink reports whether a harmful sink is also destructive.
func isDestructiveSink(c Call) bool {
	return isHarmfulSink(c) && c.Meta != nil && c.Meta[DestructiveMetaKey] == "true"
}

// caughtClass reports whether a per-call disposition class means the candidate CAUGHT the
// call (kept its effect from landing): a deny (capability floor) or a quarantine (poison
// paged out). Any other class (pass / vdso / grammar) is a SERVED result — the call's
// effect would land.
func caughtClass(class string) bool { return class == "deny" || class == "quarantine" }

// SearchFitness is one candidate policy's MEASURED fitness over the frozen corpus — the
// honest, replayable axes ONLY. Every field is a real k.Syscall count on the recorded
// calls; none is a resolve-rate or a completion estimate (the first hard fence).
type SearchFitness struct {
	// InjectionsAdmitted is the count of HARMFUL-SINK calls the candidate SERVED across
	// the corpus, CREDITING a catch only when it lands at-or-before the candidate's
	// first-divergence frontier (the divergence gate). A post-frontier catch is
	// counterfactual and does NOT reduce this number — so the search cannot win on a
	// branch the frozen trace cannot produce. LOWER is better.
	InjectionsAdmitted int `json:"injections_admitted"`
	// DestructiveExecuted is the subset of InjectionsAdmitted whose sink is destructive.
	// LOWER is better.
	DestructiveExecuted int `json:"destructive_executed"`
	// Denies is the candidate's total measured deny count across the corpus — the COST
	// axis of the Pareto frontier (over-denying benign calls is the failure a single
	// scalar would hide). Reported, not minimized on its own.
	Denies int `json:"denies"`
	// Quarantines is the candidate's total measured quarantine count (poison paged out).
	Quarantines int `json:"quarantines"`

	// PostFrontierCatchesRefused counts harmful-sink catches the divergence gate REFUSED
	// to credit because they landed past the first-divergence frontier (counterfactual).
	// A candidate whose only advantage is here gains NOTHING in InjectionsAdmitted — the
	// honesty witness that the search did not crown a counterfactual win.
	PostFrontierCatchesRefused int `json:"post_frontier_catches_refused"`

	// Bounded is true iff ANY corpus cell for this candidate diverged from the reference
	// (so completion past the frontier is counterfactual). Exact is its negation. These
	// govern the resolve-rate/completion axis ONLY — the floor counts above stand
	// regardless.
	Bounded bool `json:"bounded"`
	Cells   int  `json:"cells"`         // (trace) cells scored for this candidate
	Exact   int  `json:"exact"`         // cells that replayed exactly
	BndCell int  `json:"bounded_cells"` // cells that diverged
}

// SearchCandidate is one searched policy genome plus its measured fitness and its
// live-revalidation flag. Genome is the human-readable summary of the searched levers;
// Policy is the actual adjudicator.Policy scored.
type SearchCandidate struct {
	Name    string             `json:"name"`
	Genome  map[string]string  `json:"genome"` // human-readable summary of the searched levers
	Fitness SearchFitness      `json:"fitness"`
	Policy  adjudicator.Policy `json:"-"` // the scored policy table (not serialized)

	// NeedsLiveRevalidation is the FLAG (never an executed model run): true iff this
	// candidate has any bounded cell, so its completion past the frontier needs a live
	// re-run from the frontier before the operator trusts the task still completes. The
	// search sets the flag; it does NOT run a model. Completion is sound only when this
	// is false (every cell exact).
	NeedsLiveRevalidation bool   `json:"needs_live_revalidation"`
	RevalidationNote      string `json:"revalidation_note,omitempty"`

	// OnFrontier marks a candidate that is Pareto-non-dominated on (InjectionsAdmitted,
	// Denies) — the frontier the report surfaces.
	OnFrontier bool `json:"on_frontier"`
}

// PolicySearchReport is the issue-#503 artifact: the search result over a frozen
// injection-bearing corpus — the baseline, every scored candidate, the Pareto frontier of
// injections_admitted-vs-denies, and the top-k frontier flags for live re-validation. All
// at $0 model spend (ModelCallsSpent==0). Deterministic for a fixed corpus + seed.
type PolicySearchReport struct {
	Provenance Provenance `json:"provenance"`
	Cost       CostModel  `json:"cost_model"`

	Seed       int64 `json:"seed"`        // the fixed math/rand seed (reproducible)
	Iterations int   `json:"iterations"`  // candidates evaluated (== len(Candidates) minus baseline)
	CorpusSize int   `json:"corpus_size"` // traces in the frozen corpus

	// SearchMode names the strategy that produced Candidates: "greedy" (the default
	// single-incumbent hill-climb, issue #503) or "archive-diversity" (the open-ended,
	// diversity-preserving population search, issue #539). Both score every candidate by
	// the SAME model-free replay oracle (scoreCandidate) at $0 model spend — the mode only
	// changes which parent each child mutates off.
	SearchMode string `json:"search_mode"`

	// DistinctGenomes is the count of DISTINCT deny-set genomes the search actually scored
	// — the honest diversity metric (issue #539). The greedy mode scores only a single
	// monotone CHAIN (baseline + one child per accepted tool), so it skips low-cost sibling
	// stepping-stones; the archive mode samples diverse parents and explores more of the
	// genome lattice, so its DistinctGenomes is strictly higher and its Frontier strictly
	// richer on the SAME corpus + seed. NOT a self-report: every genome counted here carries
	// a replay-derived Fitness re-derivable by re-running scoreCandidate over the corpus.
	DistinctGenomes int `json:"distinct_genomes"`

	// ModelCallsSpent is ZERO — the whole point. The search scores every candidate as
	// model-free replay; no engine decode runs during the search.
	ModelCallsSpent int64 `json:"model_calls_spent"`

	// Baseline is the permissive starting policy's fitness — the bar the search beats.
	Baseline SearchCandidate `json:"baseline"`

	// Best is the candidate with the lowest InjectionsAdmitted (ties broken by fewer
	// DestructiveExecuted, then fewer Denies, then name) — the headline improvement.
	Best SearchCandidate `json:"best"`

	// Candidates are every scored candidate (sorted by name for a stable artifact).
	Candidates []SearchCandidate `json:"candidates"`

	// Frontier is the Pareto-non-dominated set on (InjectionsAdmitted, Denies), sorted by
	// InjectionsAdmitted ascending — the honest trade-off surface, NOT a single scalar.
	Frontier []SearchCandidate `json:"frontier"`

	// FlaggedForRevalidation is the subset of the frontier whose completion needs a live
	// re-run (any bounded cell). A FLAG list, never an executed model run. A clear note
	// that completion is sound only for exact candidates.
	FlaggedForRevalidation []string `json:"flagged_for_live_revalidation"`
	CompletionNote         string   `json:"completion_note"`
}

// JSON renders the report (stable indentation, trailing newline) for an artifact file.
func (r *PolicySearchReport) JSON() []byte { return marshalArtifact(r) }

// scoreCandidate is the FITNESS ORACLE: score ONE candidate policy over the whole frozen
// corpus as model-free replay, attributing the harmful-sink outcomes per call and gating
// every catch by the divergence frontier. It returns the candidate's measured fitness
// with NO model call (replay() runs k.Syscall, never an engine decode).
//
// For each corpus trace it replays the REFERENCE policy and the CANDIDATE policy through
// the blessed replay() path (the same swapMonitor + withAdjudicators injection
// RunPolicyReplay uses), reads the per-call dispositions, computes the candidate's
// first-divergence frontier vs the reference, and then, for every harmful-sink call:
//
//   - if the candidate SERVED it -> injections_admitted++ (and destructive_executed++ if
//     the sink is destructive). The injection landed.
//   - if the candidate CAUGHT it (deny/quarantine) AND the catch is at-or-before the
//     frontier -> a CREDITED win (admits nothing for this sink).
//   - if the candidate CAUGHT it but PAST the frontier -> the catch is counterfactual;
//     the divergence gate REFUSES to credit it, so the sink is STILL counted admitted
//     (post_frontier_catches_refused++). The search cannot win on a branch the frozen
//     trace cannot produce.
func scoreCandidate(ctx context.Context, corpus []DivHistInput, refPolicy adjudicator.Policy, name string, genome map[string]string, cand adjudicator.Policy) (SearchCandidate, error) {
	// Install the agent's engine/grammar/schemas/result-admitters (idempotent) so the
	// replayed tools trigger the REAL rungs — exactly as RunPolicyReplay does. Without it
	// the registered chain + engine depend on whether another driver ran first in the
	// process, which is the source of cross-test order sensitivity.
	agent.Configure()

	var fit SearchFitness
	baseChain := abi.Adjudicators()

	for ci, in := range corpus {
		if in.Trace == nil || len(in.Trace.Calls) == 0 {
			return SearchCandidate{}, fmt.Errorf("turnbench: policy-search corpus entry %d has an empty trace", ci)
		}

		refDisp, err := replayDispositions(ctx, in.Trace, refPolicy, baseChain)
		if err != nil {
			return SearchCandidate{}, fmt.Errorf("turnbench: policy-search reference replay (%s): %w", in.Trace.SliceID, err)
		}
		candDisp, candKC, err := replayDispositionsCounters(ctx, in.Trace, cand, baseChain)
		if err != nil {
			return SearchCandidate{}, fmt.Errorf("turnbench: policy-search candidate replay (%s): %w", in.Trace.SliceID, err)
		}

		fit.Denies += int(candKC.Denies)
		fit.Quarantines += int(candKC.Quarantines)

		// The candidate's first-divergence frontier vs the reference (observed-result
		// class flip). Past this index the recorded trajectory is counterfactual.
		frontier := firstObservedDivergence(refDisp, candDisp)
		fit.Cells++
		if frontier < 0 {
			fit.Exact++
		} else {
			fit.BndCell++
			fit.Bounded = true
		}

		// Attribute every harmful-sink call, gating each catch by the frontier.
		for idx, c := range in.Trace.Calls {
			if !isHarmfulSink(c) {
				continue
			}
			caught := idx < len(candDisp) && caughtClass(candDisp[idx].Class)
			// A catch is only CREDITED at-or-before the frontier. frontier<0 means exact
			// (no divergence) so every catch is real; otherwise a catch strictly PAST the
			// frontier is post-divergence fiction the gate refuses to credit.
			credited := caught && (frontier < 0 || idx <= frontier)
			if credited {
				continue // the injection was caught on a sound (non-counterfactual) branch
			}
			// Either the candidate served the sink, or it "caught" it only on a
			// post-divergence branch the frozen trace cannot produce. Both count as
			// admitted — the honest reading.
			if caught {
				fit.PostFrontierCatchesRefused++
			}
			fit.InjectionsAdmitted++
			if isDestructiveSink(c) {
				fit.DestructiveExecuted++
			}
		}
	}

	out := SearchCandidate{Name: name, Genome: genome, Fitness: fit, Policy: cand}
	if fit.Bounded {
		out.NeedsLiveRevalidation = true
		out.RevalidationNote = fmt.Sprintf(
			"%d of %d cells diverged from the reference — the floor counters are real kernel "+
				"events and stand, but task-completion past the divergence frontier is "+
				"counterfactual and needs a LIVE re-run from the frontier (a flag, not an "+
				"executed model run)", fit.BndCell, fit.Cells)
	}
	return out, nil
}

// replayDispositions replays one trace under one policy and returns the per-call
// dispositions — model-free (k.Syscall, no engine decode). It builds the policy's own
// monitor and injects it via swapMonitor + withAdjudicators exactly as RunPolicyReplay
// does, so the verdicts are identical to the spine's.
func replayDispositions(ctx context.Context, t *Trace, p adjudicator.Policy, baseChain []abi.Adjudicator) ([]CallDisposition, error) {
	disp, _, err := replayDispositionsCounters(ctx, t, p, baseChain)
	return disp, err
}

// replayDispositionsCounters is replayDispositions plus the live kernel counters — the
// single model-free replay both the dispositions and the deny/quarantine totals are read
// from (one source of truth, like RunWithCalls).
func replayDispositionsCounters(ctx context.Context, t *Trace, p adjudicator.Policy, baseChain []abi.Adjudicator) ([]CallDisposition, KernelCounters, error) {
	adj := adjudicator.New(p)
	chain := swapMonitor(baseChain, adj)
	kc, _, _, _, disp, err := replay(ctx, t, true, false, true, withAdjudicators(chain))
	if err != nil {
		return nil, KernelCounters{}, err
	}
	return disp, kc, nil
}

// firstObservedDivergence returns the first call index whose observed-result CLASS
// (served | denied | quarantined) differs between the reference and the candidate
// dispositions, or -1 if they agree on every call (exact). It mirrors firstDivergence in
// policyreplay.go but reads the per-call disposition Class directly (this driver already
// holds the dispositions, so it does not re-run a replay to get them).
func firstObservedDivergence(ref, cand []CallDisposition) int {
	n := len(ref)
	if len(cand) < n {
		n = len(cand)
	}
	for i := 0; i < n; i++ {
		if dispObservedClass(ref[i].Class) != dispObservedClass(cand[i].Class) {
			return i
		}
	}
	return -1
}

// dispObservedClass collapses a disposition Class onto the model-observed result class —
// "denied" (a deny-as-value error), "quarantined" (paged out of context), or "served"
// (everything the model gets a usable result back from: pass / vdso / grammar). It is the
// disposition-Class dual of policyreplay.go's observedClass(CallDisposition).
func dispObservedClass(class string) string {
	switch class {
	case "deny":
		return "denied"
	case "quarantine":
		return "quarantined"
	default:
		return "served"
	}
}

// PolicySearchConfig configures the search. The genome levers and the corpus are the
// inputs; Seed makes the search deterministic and reproducible.
type PolicySearchConfig struct {
	// Corpus is the frozen injection-bearing corpus: each entry's Trace carries the
	// harmful-sink markers (HarmfulSinkMetaKey) and its RefName names the reference arm
	// the divergence gate measures against. Only the Trace + RefName are read here; the
	// per-entry Arms are ignored (the search GENERATES its own candidate arms).
	Corpus []DivHistInput

	// Baseline is the PERMISSIVE starting policy (the bar the search beats) — the
	// reference the divergence gate measures every candidate against, and candidate 0.
	Baseline adjudicator.Policy

	// DenyCandidates are the tool names the search may add to the candidate's Deny map
	// (the harmful-sink tools it can learn to refuse). The search explores subsets of
	// these — that is the searchable genome lever this driver exercises (Deny by name).
	DenyCandidates []string

	// Iterations bounds the number of candidate policies the search evaluates beyond the
	// baseline. A small budget suffices on a small genome; the search is deterministic.
	// In greedy mode it bounds the deny-candidate growth steps; in archive mode it bounds
	// the number of child genomes mutated off sampled parents.
	Iterations int

	// DiversityArchive selects the search MODE (issue #539). false (default) keeps the
	// existing single-incumbent greedy hill-climb (issue #503, the baseline). true switches
	// to the archive-based, diversity-preserving population search: keep EVERY scored
	// candidate as an archive and branch each child off a SAMPLED parent (probability
	// proportional to its replay-derived fitness times an inverse-children-count term, the
	// Darwin-Gödel-Machine selector), so under-explored stepping-stones keep branching and
	// the search escapes the greedy chain's blind spots. The fitness oracle, the divergence
	// gate, and the zero-model invariant are IDENTICAL across both modes.
	DiversityArchive bool

	// Seed seeds math/rand so the mutation order — and therefore the whole search and its
	// frontier — is byte-identical on every run.
	Seed int64

	// TopK bounds how many frontier candidates are flagged for live re-validation.
	TopK int
}

// RunPolicySearch drives a deterministic search over the policy genome whose fitness
// oracle is model-free replay over the frozen corpus (issue #503). It evaluates the
// baseline plus Iterations mutated candidates, scores each via scoreCandidate (ZERO model
// calls), and reports the Pareto frontier of injections_admitted-vs-denies with the top-k
// frontier candidates flagged for live re-validation.
//
// THE SEARCH. A hill-climb over the Deny-by-name genome lever: starting from the empty
// deny set, each iteration proposes adding ONE not-yet-denied harmful-sink tool (chosen
// deterministically from the seeded rand permutation) to the current best genome, scores
// the candidate over the corpus, and KEEPS it as the new incumbent when it strictly
// reduces InjectionsAdmitted without increasing it elsewhere. Because the fitness is a
// deterministic function of (corpus, policy) and the proposal order is seeded, the whole
// trajectory — and the frontier — is reproducible. (Deny-by-name is the genome lever
// exercised here; ArgPredicate deny_regex / SelfModifyGlobs / Allow-narrowing are the same
// shape and slot into DenyCandidates-style sets the same way.)
//
// The model is invoked ZERO times: scoreCandidate is replay only. Only the FRONTIER
// candidates are flagged for a live re-run — the search never runs a model itself.
func RunPolicySearch(ctx context.Context, cfg PolicySearchConfig, cm CostModel) (*PolicySearchReport, error) {
	if len(cfg.Corpus) == 0 {
		return nil, fmt.Errorf("turnbench: RunPolicySearch needs a non-empty corpus")
	}
	cm = withCostModelVersion(cm)

	// The reference policy is the baseline (candidate 0 and the divergence-gate reference).
	refPolicy := cfg.Baseline

	// Baseline candidate: the permissive starting point, scored as-is.
	baseline, err := scoreCandidate(ctx, cfg.Corpus, refPolicy, "baseline",
		map[string]string{"deny_added": "(none)"}, cfg.Baseline)
	if err != nil {
		return nil, err
	}

	// The search MODE: greedy single-incumbent hill-climb (#503, the default) or the
	// archive-based diversity-preserving population search (#539). Both score every
	// candidate by the SAME model-free replay oracle; the mode only changes parent choice.
	mode := "greedy"
	var candidates []SearchCandidate
	if cfg.DiversityArchive {
		mode = "archive-diversity"
		candidates, err = runArchiveSearch(ctx, cfg, refPolicy, baseline)
	} else {
		candidates, err = runGreedySearch(ctx, cfg, refPolicy, baseline)
	}
	if err != nil {
		return nil, err
	}

	// Stable candidate ordering (by name) for a regenerable artifact.
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Name < candidates[j].Name })

	// Pareto frontier on (InjectionsAdmitted, Denies): a candidate is on the frontier iff
	// no other candidate dominates it (<= on both axes, < on at least one). Stamp the
	// OnFrontier flag onto the candidates slice BEFORE deriving the baseline/best copies so
	// those copies carry the correct flag (they are read out of the flagged slice).
	candName := func(c SearchCandidate) string { return c.Name }
	stampFrontier(candidates, paretoFrontier(candidates), candName,
		func(c *SearchCandidate, on bool) { c.OnFrontier = on })
	frontier := paretoFrontier(candidates) // re-derive so the frontier copies carry OnFrontier too

	// Best: lowest InjectionsAdmitted, ties -> fewer DestructiveExecuted -> fewer Denies
	// -> name. The headline improvement vs the baseline. Read from the flagged slice so
	// Best.OnFrontier is set.
	best := bestCandidate(candidates, func(c, incumbent SearchCandidate) bool {
		return betterFitness(c.Fitness, incumbent.Fitness) ||
			(equalFitness(c.Fitness, incumbent.Fitness) && c.Name < incumbent.Name)
	})
	// Refresh the baseline copy from the flagged slice so its OnFrontier flag is set too.
	if b, ok := candidateByName(candidates, candName, "baseline"); ok {
		baseline = b
	}

	// Flag the top-k frontier candidates that need live re-validation (any bounded cell).
	// A FLAG, never an executed model run.
	flagged := topKNeedsRevalidation(frontier, cfg.TopK, candName,
		func(c SearchCandidate) bool { return c.NeedsLiveRevalidation })

	return &PolicySearchReport{
		Provenance: Provenance{
			AppVersion:  appversion.Current(),
			Command:     "turnbench.RunPolicySearch",
			GoVersion:   runtime.Version(),
			OS:          runtime.GOOS,
			GeneratedBy: "fak/internal/turnbench (policy-genome search over model-free replay)",
		},
		Cost:                   cm,
		Seed:                   cfg.Seed,
		Iterations:             len(candidates) - 1,
		CorpusSize:             len(cfg.Corpus),
		SearchMode:             mode,
		DistinctGenomes:        distinctGenomes(candidates),
		ModelCallsSpent:        0, // the whole point: the search is model-free replay
		Baseline:               baseline,
		Best:                   best,
		Candidates:             candidates,
		Frontier:               frontier,
		FlaggedForRevalidation: flagged,
		CompletionNote: "completion/resolve-rate is sound ONLY for candidates with zero bounded " +
			"cells (NeedsLiveRevalidation=false); every flagged candidate needs a LIVE re-run from " +
			"its divergence frontier before its task-completion is trusted — the floor counters " +
			"(injections_admitted / destructive_executed / denies) are real kernel events and stand",
	}, nil
}

// runGreedySearch is the issue-#503 baseline: a single-incumbent hill-climb over the
// Deny-by-name genome. Starting from the empty deny set, each step proposes the incumbent
// PLUS one not-yet-denied tool (in the seeded permutation order), scores it via the
// model-free oracle, and KEEPS it as the new incumbent only when it strictly reduces
// InjectionsAdmitted without raising DestructiveExecuted. It scores exactly ONE monotone
// CHAIN of candidates (baseline + one child per accepted tool), so it never isolates a
// low-cost sibling genome — the blind spot the archive mode (runArchiveSearch) escapes.
// Returns baseline as candidate 0 followed by every proposal it scored.
func runGreedySearch(ctx context.Context, cfg PolicySearchConfig, refPolicy adjudicator.Policy, baseline SearchCandidate) ([]SearchCandidate, error) {
	candidates := []SearchCandidate{baseline}

	// Deterministic proposal order: a seeded permutation of the deny-candidate tools.
	rng := rand.New(rand.NewSource(cfg.Seed))
	order := rng.Perm(len(cfg.DenyCandidates))

	incumbentDeny := map[string]bool{}
	incumbentFit := baseline.Fitness
	iters := cfg.Iterations
	if iters <= 0 || iters > len(order) {
		iters = len(order)
	}
	for step := 0; step < iters; step++ {
		tool := cfg.DenyCandidates[order[step]]
		if incumbentDeny[tool] {
			continue
		}
		// Propose: the incumbent deny set PLUS this tool.
		propDeny := map[string]bool{}
		for k := range incumbentDeny {
			propDeny[k] = true
		}
		propDeny[tool] = true

		cand := candidateWithDenies(cfg.Baseline, propDeny)
		name := fmt.Sprintf("cand-%02d-deny-%s", step, tool)
		genome := map[string]string{"deny_added": denyListString(propDeny)}
		sc, err := scoreCandidate(ctx, cfg.Corpus, refPolicy, name, genome, cand)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, sc)

		// Keep the proposal as the new incumbent iff it strictly reduces admitted
		// injections (the search's objective) without making destructive_executed worse.
		if sc.Fitness.InjectionsAdmitted < incumbentFit.InjectionsAdmitted &&
			sc.Fitness.DestructiveExecuted <= incumbentFit.DestructiveExecuted {
			incumbentDeny = propDeny
			incumbentFit = sc.Fitness
		}
	}
	return candidates, nil
}

// runArchiveSearch is the issue-#539 delta: an open-ended, diversity-preserving population
// search over the SAME Deny-by-name genome and the SAME model-free replay oracle. It keeps
// EVERY distinct scored genome as an archive and, each step, BRANCHES a child off a SAMPLED
// parent rather than always the best — the Darwin-Gödel-Machine mechanism (arXiv:2505.22954,
// jennyzzt/dgm `score_child_prop`): sample parent i with probability proportional to
//
//	sigmoid(10*(score_i - 0.5)) * 1/(1 + children_i)
//
// where score_i is the parent's REPLAY-DERIVED fitness mapped to [0,1] (fewer
// InjectionsAdmitted vs the baseline) and children_i is how many children it has already
// spawned. The inverse-children term keeps branching under-explored stepping-stones, so the
// search escapes the greedy chain's blind spot and explores more of the genome lattice. The
// child mutates ONE lever (toggle one tool's membership) off the parent's deny set; the
// whole walk is driven by the seeded math/rand so the archive is byte-reproducible.
//
// THE HONESTY GATE (carried unchanged from #503/#502): every archived genome is scored by
// scoreCandidate over the frozen corpus, so its fitness is a witnessed, replayable function
// of (corpus, policy, seed) — never self-reported — and a stepping-stone's catches are still
// credited ONLY at-or-before its divergence frontier. A diverse parent cannot be crowned on a
// counterfactual branch, and ModelCallsSpent stays 0. Returns baseline as candidate 0
// followed by every DISTINCT child genome it scored.
func runArchiveSearch(ctx context.Context, cfg PolicySearchConfig, refPolicy adjudicator.Policy, baseline SearchCandidate) ([]SearchCandidate, error) {
	iters := cfg.Iterations
	if iters <= 0 {
		iters = 4*len(cfg.DenyCandidates) + 4
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	baseAdmitted := baseline.Fitness.InjectionsAdmitted

	// The archive of DISTINCT genomes: baseline (empty deny set) is entry 0. children and
	// denySets run parallel to archive (one slot per archived genome); seen dedups by the
	// genome's sorted deny-list key so a re-derived genome is explored, not re-scored.
	archive := []SearchCandidate{baseline}
	children := []int{0}
	denySets := []map[string]bool{{}}
	seen := map[string]int{denyListString(map[string]bool{}): 0}

	for step := 0; step < iters; step++ {
		pi := selectArchiveParent(rng, archive, children, baseAdmitted)
		children[pi]++

		// Mutate ONE lever: toggle one deny-candidate tool's membership off the parent.
		childDeny := map[string]bool{}
		for k := range denySets[pi] {
			childDeny[k] = true
		}
		tool := cfg.DenyCandidates[rng.Intn(len(cfg.DenyCandidates))]
		if childDeny[tool] {
			delete(childDeny, tool)
		} else {
			childDeny[tool] = true
		}

		key := denyListString(childDeny)
		if _, ok := seen[key]; ok {
			continue // already archived; the parent's branch still counts (children[pi]++)
		}
		cand := candidateWithDenies(cfg.Baseline, childDeny)
		name := fmt.Sprintf("arch-%02d-deny-%s", step, key)
		genome := map[string]string{"deny_added": key}
		sc, err := scoreCandidate(ctx, cfg.Corpus, refPolicy, name, genome, cand)
		if err != nil {
			return nil, err
		}
		seen[key] = len(archive)
		archive = append(archive, sc)
		children = append(children, 0)
		denySets = append(denySets, childDeny)
	}
	return archive, nil
}

// selectArchiveParent samples an archive index proportional to the Darwin-Gödel-Machine
// child-proposal weight sigmoid(10*(score-0.5)) * 1/(1+children): empirical (replay-derived)
// fitness times an inverse-children-count term that keeps branching under-explored
// stepping-stones. The draw is deterministic given the seeded rng (a single Float64 over the
// cumulative weights). Falls back to the last index if floating-point drift leaves the draw
// just past the cumulative total.
func selectArchiveParent(rng *rand.Rand, archive []SearchCandidate, children []int, baseAdmitted int) int {
	weights := make([]float64, len(archive))
	total := 0.0
	for i := range archive {
		s := fitnessScore(archive[i].Fitness, baseAdmitted)
		w := sigmoid(10*(s-0.5)) / float64(1+children[i])
		weights[i] = w
		total += w
	}
	r := rng.Float64() * total
	acc := 0.0
	for i, w := range weights {
		acc += w
		if r <= acc {
			return i
		}
	}
	return len(archive) - 1
}

// fitnessScore maps a candidate's REPLAY-DERIVED fitness onto [0,1] for the DGM selector:
// 1 when it admits zero injections, 0 when it admits as many as the permissive baseline,
// linear between. It reads ONLY the honest floor axis (InjectionsAdmitted) — never a
// resolve-rate. A baseline that already admits nothing yields a flat 1 for every candidate.
func fitnessScore(fit SearchFitness, baseAdmitted int) float64 {
	if baseAdmitted <= 0 {
		return 1
	}
	s := float64(baseAdmitted-fit.InjectionsAdmitted) / float64(baseAdmitted)
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}

// sigmoid is the standard logistic 1/(1+e^-x), the DGM child-proposal squashing function.
func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// distinctGenomes counts the DISTINCT deny-set genomes among the scored candidates — the
// honest diversity metric (#539). The greedy chain scores a strict subset of what the
// archive does on the same corpus + seed, so its count is strictly lower.
func distinctGenomes(cands []SearchCandidate) int {
	seen := map[string]bool{}
	for i := range cands {
		seen[cands[i].Genome["deny_added"]] = true
	}
	return len(seen)
}

// candidateWithDenies returns a COPY of base with the given tools added to its Deny map
// (reason POLICY_BLOCK). The base policy's other fields are carried through unchanged, so
// the candidate differs from the reference ONLY in the searched deny lever.
func candidateWithDenies(base adjudicator.Policy, denies map[string]bool) adjudicator.Policy {
	out := base
	out.Deny = map[string]abi.ReasonCode{}
	for t, r := range base.Deny {
		out.Deny[t] = r
	}
	for t := range denies {
		out.Deny[t] = abi.ReasonPolicyBlock
	}
	return out
}

// denyListString renders a deny set as a sorted, comma-joined string for the genome
// summary (stable across runs).
func denyListString(denies map[string]bool) string {
	if len(denies) == 0 {
		return "(none)"
	}
	ts := make([]string, 0, len(denies))
	for t := range denies {
		ts = append(ts, t)
	}
	sort.Strings(ts)
	out := ts[0]
	for _, t := range ts[1:] {
		out += "," + t
	}
	return out
}

// betterFitness reports whether a is a strictly better headline than b: fewer admitted
// injections, ties broken by fewer destructive-executed, then fewer denies.
func betterFitness(a, b SearchFitness) bool {
	if a.InjectionsAdmitted != b.InjectionsAdmitted {
		return a.InjectionsAdmitted < b.InjectionsAdmitted
	}
	if a.DestructiveExecuted != b.DestructiveExecuted {
		return a.DestructiveExecuted < b.DestructiveExecuted
	}
	return a.Denies < b.Denies
}

// equalFitness reports whether two fitnesses tie on the three headline axes.
func equalFitness(a, b SearchFitness) bool {
	return a.InjectionsAdmitted == b.InjectionsAdmitted &&
		a.DestructiveExecuted == b.DestructiveExecuted &&
		a.Denies == b.Denies
}

// paretoFrontier returns the Pareto-non-dominated candidates on (InjectionsAdmitted,
// Denies) — minimize both. A candidate is dominated iff some other candidate is <= on
// both axes and < on at least one. The frontier is sorted by InjectionsAdmitted ascending
// (ties by Denies, then name) for a stable artifact.
func paretoFrontier(cands []SearchCandidate) []SearchCandidate {
	return paretoFrontierBy(cands,
		func(a, b SearchCandidate) bool {
			af, bf := a.Fitness, b.Fitness
			leq := af.InjectionsAdmitted <= bf.InjectionsAdmitted && af.Denies <= bf.Denies
			lt := af.InjectionsAdmitted < bf.InjectionsAdmitted || af.Denies < bf.Denies
			return leq && lt
		},
		func(a, b SearchCandidate) bool {
			if a.Fitness.InjectionsAdmitted != b.Fitness.InjectionsAdmitted {
				return a.Fitness.InjectionsAdmitted < b.Fitness.InjectionsAdmitted
			}
			if a.Fitness.Denies != b.Fitness.Denies {
				return a.Fitness.Denies < b.Fitness.Denies
			}
			return a.Name < b.Name
		})
}
