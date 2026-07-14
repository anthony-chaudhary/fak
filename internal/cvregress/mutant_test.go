package cvregress

// mutant_test.go — mutant-sensitivity proof for this package's quality gates (#4573, parent
// #4509): plant one representative fault per engine-fault class (tokenization, logits, sampler,
// cache, stop, report-fact) and prove each critical gate KILLS its mutant — flags REGRESSED on
// the mutant corpus while staying OK on the clean one — or pin the blindness as a RECORDED GAP
// that names the owning gate. A pinned gap is load-bearing: if a later change closes the gap,
// the pin fails loudly and tells the author to promote the case from gap to kill.
//
// The captured witness the issue asks for is the go-test run itself: every kill case fails the
// gate against the planted defect, passes it on the clean corpus, and independently replays the
// kill from an emitted JSONL artifact in a fresh temp dir (ScoreLedgerFile round-trip). Every
// case records the #4509 shared provenance contract (engine/backend, model, tokenizer, seed or
// deterministic oracle, code revision binding, tolerance/baseline provenance, tier, runtime
// cost); TestMutantCases_RecordRequiredProvenance enforces the record as an invariant.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// mutantCase is one planted fault. The provenance fields are the #4573 acceptance contract:
// a case that cannot say what it tested, against which baseline, at which revision, is not
// evidence. This package's gates score WITNESSED ledger counters, not live engine output, so
// model/tokenizer record what is (and is not) visible at this layer rather than pretending a
// live engine ran.
type mutantCase struct {
	name       string
	faultClass string // tokenization | logits | sampler | cache | stop | report-fact

	engine    string // engine/backend under test
	model     string // model provenance (or why none is visible at this layer)
	tokenizer string // tokenizer provenance (or why none is visible at this layer)
	oracle    string // deterministic oracle (or seed) that decides the case
	revision  string // code/module revision binding
	baseline  string // tolerance/baseline provenance
	tier      string // pr | nightly | release
	cost      string // documented runtime/resource cost

	base            Baseline                                            // baseline the case scores against
	clean           func() []cachevalueledger.Row                       // pre-fault corpus; nil means healthyCorpus
	mutate          func([]cachevalueledger.Row) []cachevalueledger.Row // nil: no ledger-visible footprint
	kills           bool                                                // true: gate must flag REGRESSED
	gapOwner        string                                              // for gaps: the gate that owns this fault class
	firstDivergence string                                              // for kills: substring the flagged Reason must name
}

// cleanRows returns the case's pre-fault corpus (healthyCorpus unless the case pins its own).
func (tc mutantCase) cleanRows() []cachevalueledger.Row {
	if tc.clean != nil {
		return tc.clean()
	}
	return healthyCorpus()
}

// provRevision is the shared code-revision binding: the go-test binary compiles this exact
// working tree, so the run itself pins the revision without a drifting hand-written SHA.
const provRevision = "github.com/anthony-chaudhary/fak/internal/cvregress @ go-test build of the working tree under test"

const provEngine = "fak cache-value ledger (witnessed per-session usage counters; no live decode engine at this layer)"

// churnBase is the tightened consumer ratchet the write-amp mutant scores against. Under the
// DefaultBaseline pins the ceiling is geometrically redundant — write-amp 1.5 is exactly hit
// 40%, so the ceiling can never fire without the floor — hence isolating the write-amp kill
// requires a consumer baseline where the ceiling binds first, same as the existing
// TestFold_WriteAmpCeilingFiresIndependently.
func churnBase() Baseline {
	return Baseline{HitRatePctFloor: 10, WriteAmpCeiling: 0.5, MinPromptTokens: 100}
}

func mutantCases() []mutantCase {
	defaultBaselineProv := "DefaultBaseline pins (hit>=40%, write-amp<=1.5, min-prompt 300) calibrated on docs/nightrun/cache-value.jsonl — provenance in cvregress.go"
	return []mutantCase{
		{
			name:       "fleet-wide reuse collapse",
			faultClass: "cache",
			engine:     provEngine,
			model:      "synthetic multi-turn corpus mirroring docs/nightrun/cache-value.jsonl (healthyCorpus); no model weights visible at ledger layer",
			tokenizer:  "counts-only — tokenizer identity not visible at ledger layer",
			oracle:     "deterministic: Fold pinned-baseline hit-rate floor",
			revision:   provRevision,
			baseline:   defaultBaselineProv,
			tier:       TierPR,
			cost:       "pure in-memory fold over 6 rows, <1ms",
			base:       DefaultBaseline(),
			// A prefix-invalidation bug that quarters realized reuse fleet-wide: the exact
			// slow-rot the PIN exists to catch (a self-relative median slides with it).
			mutate: func(rows []cachevalueledger.Row) []cachevalueledger.Row {
				for i := range rows {
					rows[i].ReusedTokens /= 4
				}
				return rows
			},
			kills:           true,
			firstDivergence: "hit-rate",
		},
		{
			name:       "prefix churn re-prefill",
			faultClass: "cache",
			engine:     provEngine,
			model:      "single synthetic churny session (50% hit, write-amp 1.0); no model weights visible at ledger layer",
			tokenizer:  "counts-only — tokenizer identity not visible at ledger layer",
			oracle:     "deterministic: Fold pinned-baseline write-amp ceiling",
			revision:   provRevision,
			baseline:   "tightened consumer ratchet (floor 10%, ceiling 0.5, min-prompt 100); the default-pin geometry makes the ceiling redundant (1.5 ceiling == 40% floor), recorded here as provenance",
			tier:       TierPR,
			cost:       "pure in-memory fold over 1 row, <1ms",
			base:       churnBase(),
			// The pre-fault session reuses most of its prefix (write-amp 0.25, hit 80%) and
			// clears the ratchet; the ratchet is per-case because healthyCorpus's churniest
			// sessions (write-amp up to 1.23) never targeted a 0.5 ceiling.
			clean: func() []cachevalueledger.Row {
				return []cachevalueledger.Row{ledgerRow("2026-07-08", 4, 1000, 800, 8)}
			},
			// A cache-key churn bug that re-prefills the prefix faster than it reuses it while
			// hit-rate still clears the floor: only the write-amp ceiling sees it (0.25 -> 1.0).
			mutate: func(rows []cachevalueledger.Row) []cachevalueledger.Row {
				rows[0].ReusedTokens = 500
				return rows
			},
			kills:           true,
			firstDivergence: "write-amp",
		},
		{
			name:       "reused exceeds prompt",
			faultClass: "report-fact",
			engine:     provEngine,
			model:      "healthyCorpus with one double-counted row (reused = 2x prompt)",
			tokenizer:  "counts-only — tokenizer identity not visible at ledger layer",
			oracle:     "deterministic: Fold pinned-baseline (blind — pinned gap)",
			revision:   provRevision,
			baseline:   defaultBaselineProv,
			tier:       TierPR,
			cost:       "pure in-memory fold over 7 rows, <1ms",
			base:       DefaultBaseline(),
			// A ledger-writer double-count: reused > prompt is an impossible fact, but the fold
			// performs no fact validation — hit-rate 200% clears the floor and the write-amp
			// guard's prompt>reused branch zeroes out. RECORDED GAP: fact validation belongs to
			// the ledger schema (internal/cachevalueledger), not the fold.
			mutate: func(rows []cachevalueledger.Row) []cachevalueledger.Row {
				bad := ledgerRow("2026-07-09", 5, 900, 1800, 9)
				return append(rows, bad)
			},
			kills:    false,
			gapOwner: "internal/cachevalueledger row schema validation (fold performs no fact checks)",
		},
		{
			name:       "uniform token-count inflation",
			faultClass: "tokenization",
			engine:     provEngine,
			model:      "healthyCorpus with every count tripled (mis-tokenizing engine)",
			tokenizer:  "fault class UNDER TEST: a mis-tokenizer inflating prompt and reused counts uniformly",
			oracle:     "deterministic: Fold pinned-baseline (blind — pinned gap)",
			revision:   provRevision,
			baseline:   defaultBaselineProv,
			tier:       TierPR,
			cost:       "pure in-memory fold over 6 rows, <1ms",
			base:       DefaultBaseline(),
			// A tokenizer regression that inflates BOTH prompt and reused counts leaves the
			// hit-rate ratio and write-amp unchanged: the ledger gate is ratio-invariant and
			// therefore blind by construction. RECORDED GAP: owned by the tokenizer-parity gate.
			mutate: func(rows []cachevalueledger.Row) []cachevalueledger.Row {
				for i := range rows {
					rows[i].PromptTokens *= 3
					rows[i].ReusedTokens *= 3
				}
				return rows
			},
			kills:    false,
			gapOwner: "internal/quality tokenizer parity gate (tokenizer_parity.go)",
		},
		{
			name:       "logit corruption",
			faultClass: "logits",
			engine:     provEngine,
			model:      "no ledger-visible footprint — a logit fault changes token DISTRIBUTIONS, not witnessed usage counters",
			tokenizer:  "counts-only — tokenizer identity not visible at ledger layer",
			oracle:     "none at this layer (blind — pinned gap)",
			revision:   provRevision,
			baseline:   defaultBaselineProv,
			tier:       TierPR,
			cost:       "no runnable case at this layer; record only",
			base:       DefaultBaseline(),
			// mutate == nil: corrupted logits have NO representation in a cache-value ledger
			// row at all, so there is no ledger mutation to plant. RECORDED GAP by construction.
			mutate:   nil,
			kills:    false,
			gapOwner: "internal/quality logit tolerance gate (logit_tolerance.go)",
		},
	}
}

// ledgerRow mirrors the row helper in cvregress_test.go with an explicit name so the mutant
// table reads standalone.
func ledgerRow(date string, turns, prompt, reused uint64, unixMillis int64) cachevalueledger.Row {
	return row(date, turns, prompt, reused, unixMillis)
}

// replayKill re-proves a kill from the emitted artifact: write the mutant corpus to a JSONL in
// a fresh temp dir, assert the artifact is scrubbed (no home-directory path leaks into the
// replay evidence), and re-run the file-reading gate over it. A kill that only reproduces
// in-process is not independently replayable evidence.
func replayKill(t *testing.T, tc mutantCase, mutated []cachevalueledger.Row) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "mutant-replay.jsonl")
	var lines []string
	for _, r := range mutated {
		line, err := cachevalueledger.AppendLedgerLine(r)
		if err != nil {
			t.Fatalf("marshal mutant row for replay artifact: %v", err)
		}
		lines = append(lines, line)
	}
	blob := strings.Join(lines, "\n") + "\n"
	if home, err := os.UserHomeDir(); err == nil && home != "" && strings.Contains(blob, home) {
		t.Fatalf("replay artifact leaks the home directory path — artifact must be scrubbed")
	}
	if err := os.WriteFile(path, []byte(blob), 0o644); err != nil {
		t.Fatalf("write replay artifact: %v", err)
	}
	rep := ScoreLedgerFile(path, tc.base)
	if rep.Verdict != "REGRESSED" || rep.OK {
		t.Fatalf("replay from emitted artifact must reproduce the kill, got %s ok=%v: %s",
			rep.Verdict, rep.OK, rep.Finding)
	}
}

// TestMutants_DeterministicGate is the kill-or-gap sweep over the deterministic pinned fold:
// each kill case must pass clean AND fail mutated (the issue's witness shape), name the first
// actionable divergence in the flagged reason, and replay the kill from the emitted artifact;
// each gap case must stay blind exactly as recorded, so a future gap-closing change trips the
// pin and promotes the case.
func TestMutants_DeterministicGate(t *testing.T) {
	for _, tc := range mutantCases() {
		t.Run(tc.faultClass+"/"+tc.name, func(t *testing.T) {
			if clean := Fold(tc.cleanRows(), tc.base); !clean.OK {
				t.Fatalf("clean corpus must pass before a kill means anything, got %s: %s",
					clean.Verdict, clean.Finding)
			}
			if tc.mutate == nil {
				if tc.kills {
					t.Fatalf("a kill case must plant a mutation")
				}
				return // recorded gap with no ledger-visible footprint; provenance test covers the record
			}
			mutated := tc.mutate(tc.cleanRows())
			rep := Fold(mutated, tc.base)
			if tc.kills {
				if rep.Verdict != "REGRESSED" || rep.OK {
					t.Fatalf("gate must kill the %s mutant, got %s ok=%v: %s",
						tc.faultClass, rep.Verdict, rep.OK, rep.Finding)
				}
				if got := rep.Regressions[0].Reason; !strings.Contains(got, tc.firstDivergence) {
					t.Fatalf("first actionable divergence must name %q, got %q", tc.firstDivergence, got)
				}
				replayKill(t, tc, mutated)
				return
			}
			// Pinned gap: the gate is blind to this fault class TODAY. If this assert starts
			// failing, the gap has been closed — promote the case to kills=true instead of
			// deleting it.
			if !rep.OK {
				t.Fatalf("pinned gap regressed the gate unexpectedly — the %s gap appears closed; promote this case to a kill (got %s: %s)",
					tc.faultClass, rep.Verdict, rep.Finding)
			}
		})
	}
}

// TestMutants_StopFaultNeverPasses plants the stop-fault mutant: an early-stop bug that
// truncates every session after its first turn destroys the multi-turn evidence the gate
// scores. The fold falls open INSUFFICIENT by design (fall-open posture, CI stays green), so
// the kill lands at the EVIDENCE layer: the acceptance's "missing or inconclusive evidence is
// never pass" — the mutant can never obtain a conclusive OK.
func TestMutants_StopFaultNeverPasses(t *testing.T) {
	mutated := healthyCorpus()
	for i := range mutated {
		mutated[i].Turns = 1 // stop fault: every session dies after turn 1
	}
	rep := Fold(mutated, DefaultBaseline())
	if rep.Verdict != "INSUFFICIENT" || rep.Scored != 0 {
		t.Fatalf("stop-truncated corpus must be INSUFFICIENT with nothing scored, got %s scored=%d",
			rep.Verdict, rep.Scored)
	}
	if !strings.Contains(rep.Finding, "no multi-turn session") {
		t.Fatalf("finding must name the first actionable divergence (evidence destroyed), got %q", rep.Finding)
	}
	// The stochastic layer converts the destroyed evidence into an explicit never-pass: zero
	// scored sessions is UNDERPOWERED against any real spec, and an underpowered clean run is
	// INSUFFICIENT and NOT conclusive.
	spec := samplerMutantSpec()
	verdict, ok, conclusive := StochasticVerdict(Assess(spec, rep.Scored), false)
	if verdict != "INSUFFICIENT" || !ok || conclusive {
		t.Fatalf("stop mutant must never earn a conclusive pass, got verdict=%s ok=%v conclusive=%v",
			verdict, ok, conclusive)
	}
}

// samplerMutantSpec is the stochastic sampler-fault case: a sampler regression shifts the
// per-session hit-rate distribution's mean by 5 points against a 10-point per-session sigma.
// Every provenance field of the #4509 contract rides on the spec itself.
func samplerMutantSpec() PowerSpec {
	return PowerSpec{
		EffectSize: 5,  // hit-rate percentage points — smallest shift worth detecting
		StdDev:     10, // per-session hit-rate sigma
		Alpha:      0.05,
		Power:      0.80,
		Tails:      1,
		Seed:       4573, // deterministic replay — the issue number, so provenance names its contract
		Oracle:     "one-sided z-test on mean per-session hit-rate shift",
		Revision:   provRevision,
		Baseline:   "clean distribution N(0, 10) vs sampler-mutant distribution N(5, 10), hit-rate points",
		Tier:       TierPR,
	}
}

// TestMutants_SamplerShiftKilledAtPower proves the stochastic gate kills the sampler mutant at
// its promised rate AND can never be talked into a pass by an underpowered sample. The seeded
// Monte-Carlo replays identically everywhere, so the empirical kill rate is captured, portable
// provenance — not a one-off observation. Cost (documented per the tier contract): 4000 trials
// x 2 arms x n<=25 draws, well under a second — PR tier.
func TestMutants_SamplerShiftKilledAtPower(t *testing.T) {
	spec := samplerMutantSpec()
	if err := spec.Validate(); err != nil {
		t.Fatalf("sampler mutant spec must be valid: %v", err)
	}
	n, err := RequiredN(spec)
	if err != nil {
		t.Fatalf("RequiredN: %v", err)
	}

	// The kill: at the required sample size, the seeded z-test rejects the shifted-mean
	// sampler mutant at >= the promised power while holding the false-kill rate near alpha.
	sim := Simulate(spec, n, 4000, 0.03)
	if !sim.MeetsTargets {
		t.Fatalf("stochastic gate must kill the sampler mutant at promised power/alpha: %s", sim.Finding)
	}

	// The refusal: a sample too small to detect the shift is UNDERPOWERED, and a clean
	// underpowered run is never a verified pass — the mutant cannot hide in a thin sample.
	under := Assess(spec, n/2)
	if under.Adequate || under.Conclusive {
		t.Fatalf("half the required sample must be underpowered and inconclusive, got %+v", under)
	}
	verdict, _, conclusive := StochasticVerdict(under, false)
	if verdict != "INSUFFICIENT" || conclusive {
		t.Fatalf("underpowered clean run must fold to INSUFFICIENT/inconclusive, got %s conclusive=%v", verdict, conclusive)
	}
}

// TestMutantCases_RecordRequiredProvenance enforces the #4573 acceptance record as an
// invariant: every case names its engine/backend, model, tokenizer, deterministic oracle (or
// seed), code revision, baseline provenance, a valid tier, and a documented runtime cost — and
// every recorded gap names the gate that owns the fault class. A case that cannot say what it
// tested is not evidence.
func TestMutantCases_RecordRequiredProvenance(t *testing.T) {
	for _, tc := range mutantCases() {
		t.Run(tc.faultClass+"/"+tc.name, func(t *testing.T) {
			for field, v := range map[string]string{
				"engine": tc.engine, "model": tc.model, "tokenizer": tc.tokenizer,
				"oracle": tc.oracle, "revision": tc.revision, "baseline": tc.baseline,
				"cost": tc.cost,
			} {
				if strings.TrimSpace(v) == "" {
					t.Fatalf("case must record %s provenance (#4573 acceptance)", field)
				}
			}
			switch tc.tier {
			case TierPR, TierNightly, TierRelease:
			default:
				t.Fatalf("case must be assigned an explicit pr|nightly|release tier, got %q", tc.tier)
			}
			if !tc.kills && strings.TrimSpace(tc.gapOwner) == "" {
				t.Fatalf("a recorded gap must name the owning gate")
			}
			if tc.kills && strings.TrimSpace(tc.firstDivergence) == "" {
				t.Fatalf("a kill case must name the first actionable divergence it asserts")
			}
		})
	}
	// The stochastic case carries its provenance on the spec itself.
	spec := samplerMutantSpec()
	if spec.Seed == 0 || spec.Oracle == "" || spec.Revision == "" || spec.Baseline == "" || spec.Tier == "" {
		t.Fatalf("sampler mutant spec must record seed, oracle, revision, baseline, and tier, got %+v", spec)
	}
}
