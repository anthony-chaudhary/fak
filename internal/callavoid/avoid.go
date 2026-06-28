package callavoid

import (
	"fmt"
	"math"
)

// ProofStatus is the closed, two-valued verdict vocabulary, identical to the vcache
// proof leaves so a callavoid proof reads the same as a vcachegov/vcachechain one.
type ProofStatus string

const (
	// ProofProven — avoidance is net-positive: the work it saves exceeds the
	// validation, capture, and stale-miss overhead it costs.
	ProofProven ProofStatus = "PROVEN"
	// ProofRefuted — avoidance does not pay: a single-use entry, a too-volatile
	// world, or validation as costly as execution. The correct decision is to just
	// run the call.
	ProofRefuted ProofStatus = "REFUTED"
)

// neverBreakEven is the sentinel for "no finite reuse count makes memoization pay",
// matching vcache's formatBreakEven("never"). It is the max int.
const neverBreakEven = int(^uint(0) >> 1)

// DefaultMaxRedirectFanout bounds the futile-variant fan-out a single productive
// deny may credit. An UNBOUNDED fan-out claim would let amplification be inflated to
// any value by asserting a deny pruned an arbitrarily large sub-tree, so each
// productive deny credits at most this many naive round-trips. Surfaced, never
// silent: Account reports how many fan-outs were clamped.
const DefaultMaxRedirectFanout = 1024

// DefaultMaxSpeculativePrunedTurns bounds the AGGREGATE speculative pruning a window
// may report — the per-deny DefaultMaxRedirectFanout caps each entry but not the NUMBER
// of entries, so an unbounded Redirects slice could otherwise report an arbitrarily
// large pruned total (#816). The aggregate is saturated at this value and the
// saturation is surfaced (SpeculativeAggregateCapped), so the speculative upper bound is
// always finite and non-gameable. It is a reporting bound only: it never touches the
// grade, which is built from realized dispositions alone.
const DefaultMaxSpeculativePrunedTurns = 1 << 20 // 1,048,576 futile round-trips

// ValidateFloor is the minimum cost charged to a memo hit / stale validation: a
// vDSO entry is cheap to re-validate (a world-version check) but NEVER free. At the
// old zero floor a StaleMiss priced exactly 1 (validate 0 + re-run 1 + capture 0),
// so a window of pure cache thrash read break-even instead of the strict loss it is
// (a stale miss is overhead a naive agent never paid). The floor also caps a
// pure-cache window's amplification at 1/ValidateFloor (=100x) instead of +Inf.
const ValidateFloor = 0.01

// ---------------------------------------------------------------------------
// ProveMemo — the skeptical economics gate (the local-tool-call dual of
// vcachechain.ProveRecall's §11.0 cost gate).
// ---------------------------------------------------------------------------

// MemoInput is the per-key calibration for the memoization break-even proof. All
// costs are in EXECUTION-EQUIVALENTS: one full tool execution — engine dispatch plus
// the I/O, parse, and model round-trip it forces — is 1.0. The defaults a caller
// would draw from internal/vcachecal-style observation: a vDSO tier-2 entry validates
// by comparing a world-version (tiny) and captures by storing a result in an LRU
// (tiny), while the mutation rate is the share of reuses whose key was invalidated by
// an intervening write.
type MemoInput struct {
	Accesses     int     `json:"accesses"`      // k: times this exact pure call is proposed in the window (>=1).
	ValidateCost float64 `json:"validate_cost"` // v: cost to re-validate an entry on each reuse (a world-version / fingerprint check), in execution-equivalents.
	MutationRate float64 `json:"mutation_rate"` // m: probability the world changed between two accesses, invalidating the entry, in [0,1].
	CaptureCost  float64 `json:"capture_cost"`  // c: one-time cost to capture/store a fingerprint+result on each execution, in execution-equivalents.
}

// MemoProof is the self-describing output of ProveMemo: the verdict plus every number
// the gate used, so a human or CI line can audit the decision.
type MemoProof struct {
	Status               ProofStatus `json:"status"`
	Decision             string      `json:"decision"` // "memoize" | "always_execute"
	Reason               string      `json:"reason"`
	Accesses             int         `json:"accesses"`
	ValidateCost         float64     `json:"validate_cost"`
	MutationRate         float64     `json:"mutation_rate"`
	CaptureCost          float64     `json:"capture_cost"`
	NaiveCost            float64     `json:"naive_cost"`                 // k — run the call every time.
	MemoCost             float64     `json:"memo_cost"`                  // executions(1+c) + reuses·v.
	SavedCost            float64     `json:"saved_cost"`                 // NaiveCost - MemoCost.
	SavedPct             float64     `json:"saved_pct"`                  // 100·SavedCost/NaiveCost.
	PerReuseNetGain      float64     `json:"per_reuse_net_gain"`         // D = 1 - m - v - m·c, the gain each extra reuse adds.
	BreakEvenAccesses    int         `json:"break_even_accesses"`        // smallest k that clears the gate (neverBreakEven if D<=0).
	SingleUseLoss        float64     `json:"single_use_loss"`            // cost burned by a k=1 memo that never pays back (= c).
	CorrectnessDependsOn bool        `json:"correctness_depends_on_hit"` // always false — the law.
}

// ProveMemo runs the break-even gate over one key's calibration. It is pure and
// deterministic: no cache, no clock, no I/O. A green proof is direct evidence that
// memoizing this class of call repays its overhead; a red proof is the honest signal
// to leave it on the always-execute path.
//
// The arithmetic. For k accesses to one key, the first always executes and each of
// the (k-1) reuses validates (cost v) and, if the world mutated (prob m), re-executes
// and re-captures. So expected executions = 1+(k-1)m, captures ride every execution
// (cost c each), and every reuse validates:
//
//	memo  = (1+(k-1)m)·(1+c) + (k-1)·v
//	naive = k
//	saved = (k-1)·D - c,   where D = 1 - m - v - m·c
//
// D is the net each extra reuse buys. If D<=0 — validation plus mutation overhead is
// already >= the execution a reuse would save (volatile state, or a call too
// expensive to fingerprint) — no amount of reuse pays and the gate refutes. If D>0,
// the entry pays once reuse passes the break-even that amortizes the one-time capture.
func ProveMemo(in MemoInput) MemoProof {
	k := in.Accesses
	if k < 1 {
		k = 1
	}
	v := nonNeg(in.ValidateCost)
	c := nonNeg(in.CaptureCost)
	m := clamp01(in.MutationRate)

	d := 1 - m - v - m*c
	executions := 1 + float64(k-1)*m
	memo := executions*(1+c) + float64(k-1)*v
	naive := float64(k)
	saved := naive - memo

	p := MemoProof{
		Accesses:             k,
		ValidateCost:         v,
		MutationRate:         m,
		CaptureCost:          c,
		NaiveCost:            naive,
		MemoCost:             memo,
		SavedCost:            saved,
		PerReuseNetGain:      d,
		SingleUseLoss:        c,
		BreakEvenAccesses:    memoBreakEven(d, c),
		CorrectnessDependsOn: false,
	}
	if naive > 0 {
		p.SavedPct = 100 * saved / naive
	}

	switch {
	case saved > 0:
		p.Status = ProofProven
		p.Decision = "memoize"
		p.Reason = fmt.Sprintf("avoidance pays: %d accesses clear the break-even of %s (each reuse nets %.3g)",
			k, formatBreakEven(p.BreakEvenAccesses), d)
	case d <= 0:
		p.Status = ProofRefuted
		p.Decision = "always_execute"
		p.Reason = fmt.Sprintf("per-reuse net gain is %.3g <= 0: validation+mutation overhead (v=%.3g, m=%.3g) meets or exceeds the execution a reuse would save — never memoize this class", d, v, m)
	case k <= 1:
		p.Status = ProofRefuted
		p.Decision = "always_execute"
		p.Reason = fmt.Sprintf("single use: one access can never amortize the %.3g capture cost (break-even is %s accesses)", c, formatBreakEven(p.BreakEvenAccesses))
	default:
		p.Status = ProofRefuted
		p.Decision = "always_execute"
		p.Reason = fmt.Sprintf("below break-even: %d accesses do not yet repay the %.3g capture cost (need %s)", k, c, formatBreakEven(p.BreakEvenAccesses))
	}
	return p
}

// memoBreakEven returns the smallest access count k where (k-1)·D > c, i.e. where the
// reuses finally repay the one-time capture. neverBreakEven when D<=0.
func memoBreakEven(d, c float64) int {
	if d <= 0 {
		return neverBreakEven
	}
	// smallest integer (k-1) strictly greater than c/d is floor(c/d)+1, so k = floor(c/d)+2.
	return int(math.Floor(c/d)) + 2
}

// ---------------------------------------------------------------------------
// Account — the effective-productive-turn / amplification headline.
// ---------------------------------------------------------------------------

// Tally is a window of tool-call dispositions. The field names mirror internal/kernel
// Counters so a tier-4 caller maps a real guard session onto it in one obvious line:
// Execute<-EngineCalls, MemoHit<-VDSOHits, Repair<-Transforms, deny<-Denies (split
// into HardDeny vs the productive Redirects the deny reasons carry). Costs default to
// 0 (an idealized vDSO whose validation/capture are near-free, the regime where
// MemoHit paid≈0 is honest); set them to price the cache overhead and stale-miss bet,
// which is where ProveMemo's teeth live.
type Tally struct {
	Execute   int   `json:"execute"`    // real engine dispatches (Counters.EngineCalls).
	MemoHit   int   `json:"memo_hit"`   // calls served from the vDSO without dispatch (Counters.VDSOHits).
	Repair    int   `json:"repair"`     // malformed calls repaired in-syscall, each sparing a retry round-trip (Counters.Transforms).
	StaleMiss int   `json:"stale_miss"` // entries validated, found invalidated, re-dispatched (folds into EngineCalls live; explicit here for analysis).
	HardDeny  int   `json:"hard_deny"`  // fast-rejects with no forward guidance — symmetric, no amplification.
	Redirects []int `json:"redirects"`  // each entry is the bounded futile-variant fan-out a PRODUCTIVE deny pruned.
	// WitnessedRedirects are the productive denies whose pruned fan-out is backed by an
	// enumerated, deduplicated variant set (a non-forgeable witness) rather than an asserted
	// count. Account credits these from their own variants and nets them out of HardDeny; the
	// realized credit comes from the witness, not the count it replaced.
	WitnessedRedirects []WitnessedRedirect `json:"witnessed_redirects,omitempty"`
	ValidateCost       float64             `json:"validate_cost"` // v charged to a MemoHit / StaleMiss (default 0).
	CaptureCost        float64             `json:"capture_cost"`  // c charged to an Execute / StaleMiss (default 0).

	// MaxRedirectFanout caps each Redirect entry; 0 uses DefaultMaxRedirectFanout.
	MaxRedirectFanout int `json:"max_redirect_fanout"`
}

// TurnReport is the amplification scorecard: how far the agent actually got
// (EffectiveTurns, in naive round-trip equivalents) per unit of real work
// (ExecutedTurns), versus the vanity RawTurns count.
type TurnReport struct {
	Schema   string `json:"schema"`
	Status   string `json:"status"` // amplifying | break_even | regressing
	Grade    string `json:"grade"`
	RawTurns int    `json:"raw_turns"` // every proposal/decision in the window — the number people naively cite.

	ExecutedTurns  float64 `json:"executed_turns"`  // execution-equivalents actually paid.
	EffectiveTurns float64 `json:"effective_turns"` // round-trips a naive 1:1 agent must spend to reach the same outcome.
	AvoidedTurns   float64 `json:"avoided_turns"`   // EffectiveTurns - ExecutedTurns.
	Amplification  float64 `json:"amplification"`   // EffectiveTurns / ExecutedTurns (math.Inf if all work was avoided).

	MemoHits         int `json:"memo_hits"`
	Repairs          int `json:"repairs"`
	StaleMisses      int `json:"stale_misses"`
	HardDenies       int `json:"hard_denies"`
	ProductiveDenies int `json:"productive_denies"`
	RedirectPruned   int `json:"redirect_pruned"` // total futile round-trips spared by productive denies (per-deny cap applied).
	RedirectCapped   int `json:"redirect_capped"` // how many fan-outs were clamped to the per-deny cap (surfaced, not silent).

	// SPECULATIVE axis (#816): productive-deny pruning is a COUNTERFACTUAL no kernel
	// counter measures, so it is reported here as a bounded upper bound and NEVER folds
	// into the graded Amplification/Grade/Status (those are realized-only). The
	// aggregate is saturated at DefaultMaxSpeculativePrunedTurns so a 100000×large
	// redirect flood cannot inflate it; SpeculativeAggregateCapped surfaces saturation.
	SpeculativePrunedTurns     int  `json:"speculative_pruned_turns"`     // bounded aggregate futile round-trips a productive deny MAY have spared (excluded from the grade).
	SpeculativeAggregateCapped bool `json:"speculative_aggregate_capped"` // true when the aggregate hit DefaultMaxSpeculativePrunedTurns (the upper bound saturated).

	// REALIZED witnessed axis (#820): unlike the speculative axis above, a WITNESSED
	// productive deny carries the ENUMERATED variants it pruned, so its fan-out is folded
	// INTO the graded EffectiveTurns/Amplification — realized credit backed by a
	// non-forgeable, deduplicated set rather than an asserted count. The counts are bounded
	// per-deny (WitnessedCapped) and in aggregate (WitnessedAggregateCapped); an empty/blank
	// witness credits nothing and is surfaced as WitnessedEmpty so it can never inflate.
	WitnessedDenies          int  `json:"witnessed_denies"`           // productive denies whose enumerated variant set credited a non-zero realized fan-out.
	WitnessedPruned          int  `json:"witnessed_pruned"`           // total realized futile round-trips the witnessed denies pruned (folded into EffectiveTurns).
	WitnessedEmpty           int  `json:"witnessed_empty"`            // witnessed redirects that named no variants — effective hard denies, credited nothing.
	WitnessedCapped          int  `json:"witnessed_capped"`           // how many witnessed fan-outs were clamped to the per-deny cap (surfaced, not silent).
	WitnessedAggregateCapped bool `json:"witnessed_aggregate_capped"` // true when the witnessed total saturated DefaultMaxSpeculativePrunedTurns.

	Actions []string `json:"actions,omitempty"`
	Risks   []string `json:"risks,omitempty"`
}

// Account folds a disposition window into the amplification headline. Each
// disposition contributes what fak PAID (in execution-equivalents) and what a naive
// call-everything agent would have spent to reach the SAME outcome:
//
//	disposition   paid           naive   note
//	Execute       1 + c          1       a real dispatch, plus capturing a reusable fingerprint.
//	MemoHit       v              1       served from the vDSO — the partial avoidance (validate only).
//	Repair        0              1       fixed in-syscall; the naive baseline pays a retry round-trip.
//	StaleMiss     v + 1 + c      1       the cache bet that lost: validate, miss, re-run, re-capture.
//	HardDeny      0              0        both agents propose-and-are-denied once; symmetric.
//	Redirect(f)   0              f       a productive deny prunes f futile variants a naive agent would walk.
//
// Amplification = naive/paid is the user-facing answer to "how much further did we get
// per unit of real work?" — and the regime where one free productive deny stands in
// for a hundred naive round-trips is exactly where an avoiding kernel reaches states a
// naive path cannot, or would reach far slower.
func Account(t Tally) TurnReport {
	// A memo hit / stale validation pays at least ValidateFloor — cheap, never free —
	// so a stale miss is always a strict loss and a pure-cache window is finite (#817).
	v := math.Max(ValidateFloor, nonNeg(t.ValidateCost))
	c := nonNeg(t.CaptureCost)
	fanoutCap := t.MaxRedirectFanout
	if fanoutCap <= 0 {
		fanoutCap = DefaultMaxRedirectFanout
	}

	// Non-negative guards on the scalar count fields (defense-in-depth, symmetric with
	// the Redirects/cost guards below): a negative count can never inflate the ratio.
	execute := nonNegInt(t.Execute)
	memoHit := nonNegInt(t.MemoHit)
	repair := nonNegInt(t.Repair)
	staleMiss := nonNegInt(t.StaleMiss)
	hardDeny := nonNegInt(t.HardDeny)

	executed := float64(execute)*(1+c) +
		float64(memoHit)*v +
		float64(staleMiss)*(v+1+c)
	naive := float64(execute) +
		float64(memoHit) +
		float64(repair) +
		float64(staleMiss)

	// SPECULATIVE axis (#816): the productive-deny fan-out is a counterfactual no kernel
	// counter measures, so it is NOT folded into the graded naive — it is bounded both
	// per-deny (fanoutCap) AND in aggregate (specCap), reported separately, and never
	// moves the grade. The per-deny cap clamps each entry; the aggregate cap saturates
	// the sum so an unbounded NUMBER of entries cannot inflate the upper bound.
	specCap := DefaultMaxSpeculativePrunedTurns
	pruned := 0
	capped := 0
	aggCapped := false
	for _, f := range t.Redirects {
		if f < 0 {
			f = 0
		}
		if f >= fanoutCap { // >= so an at-cap fan-out surfaces too, not just an over-cap one.
			f = fanoutCap
			capped++
		}
		pruned += f
		if pruned >= specCap {
			pruned = specCap
			aggCapped = true
			break
		}
	}
	// HardDeny adds 0 to both sides; it is symmetric and intentionally not credited.

	// REALIZED witnessed axis (#820): a WITNESSED productive deny names the futile variants
	// it pruned, so — unlike the speculative `pruned` above — its deduplicated fan-out IS
	// folded into the graded `naive` (the round-trips a naive agent would have spent walking
	// those variants). The credit is bounded per-deny (fanoutCap) and in aggregate, and an
	// empty/blank witness credits nothing. This is the line that lets enumerated pruning
	// graduate from a speculative upper bound to realized credit.
	witFanout, witDenies, witEmpty, witCapped, witAggCapped := witnessedTotals(t.WitnessedRedirects, fanoutCap)
	naive += float64(witFanout)

	// The graded amplification is built from realized, Counter-backed dispositions
	// (Execute/MemoHit/Repair/StaleMiss) PLUS the witnessed productive-deny fan-out (whose
	// variants are enumerated, hence realized). The speculative `pruned` upper bound stays
	// excluded — "an avoided call is a realized rebate, never a trust claim" (#816).
	raw := execute + memoHit + repair + staleMiss + hardDeny + len(t.Redirects) + witDenies
	amp := safeRatio(naive, executed)

	rep := TurnReport{
		Schema:                     "fak.callavoid.turns.v1",
		RawTurns:                   raw,
		ExecutedTurns:              executed,
		EffectiveTurns:             naive,
		AvoidedTurns:               naive - executed,
		Amplification:              amp,
		MemoHits:                   memoHit,
		Repairs:                    repair,
		StaleMisses:                staleMiss,
		HardDenies:                 hardDeny,
		ProductiveDenies:           len(t.Redirects),
		RedirectPruned:             pruned,
		RedirectCapped:             capped,
		SpeculativePrunedTurns:     pruned,
		SpeculativeAggregateCapped: aggCapped,
		WitnessedDenies:            witDenies,
		WitnessedPruned:            witFanout,
		WitnessedEmpty:             witEmpty,
		WitnessedCapped:            witCapped,
		WitnessedAggregateCapped:   witAggCapped,
	}
	rep.Status = turnStatus(amp)
	rep.Grade = turnGrade(amp)
	rep.Actions, rep.Risks = turnActionsAndRisks(t, rep)
	return rep
}

func turnStatus(amp float64) string {
	switch {
	case amp > 1+1e-9:
		return "amplifying"
	case amp < 1-1e-9:
		return "regressing"
	default:
		return "break_even"
	}
}

func turnGrade(amp float64) string {
	switch {
	case math.IsInf(amp, 1) || amp >= 4:
		return "A"
	case amp >= 2:
		return "B"
	case amp >= 1.5:
		return "C"
	case amp >= 1.05:
		return "D"
	default:
		return "F"
	}
}

func turnActionsAndRisks(t Tally, rep TurnReport) (actions, risks []string) {
	// REALIZED witnessed axis (#820): enumerated productive-deny pruning that DID fold into
	// the grade. Surface the realized credit as an action, and the bounds/empties as risks
	// so a witness can never inflate the headline silently.
	if rep.WitnessedDenies > 0 {
		actions = append(actions, fmt.Sprintf("%d witnessed productive deny(ies) pruned %d enumerated futile round-trip(s) — REALIZED credit folded into the graded amplification (the variants are named, not asserted)",
			rep.WitnessedDenies, rep.WitnessedPruned))
	}
	if rep.WitnessedEmpty > 0 {
		risks = append(risks, fmt.Sprintf("%d witnessed redirect(s) named NO variants — credited nothing and treated as effective hard denies, so an un-enumerated \"productive\" deny cannot inflate the realized headline", rep.WitnessedEmpty))
	}
	if rep.WitnessedCapped > 0 {
		risks = append(risks, fmt.Sprintf("%d witnessed fan-out(s) were clamped to the per-deny cap (%d); the realized witnessed credit is a LOWER bound, never inflated", rep.WitnessedCapped, DefaultMaxRedirectFanout))
	}
	if rep.WitnessedAggregateCapped {
		risks = append(risks, fmt.Sprintf("the witnessed pruned total saturated the aggregate cap (%d); the realized credit is a floor of the true count, never inflated", DefaultMaxSpeculativePrunedTurns))
	}
	if rep.SpeculativePrunedTurns > 0 {
		actions = append(actions, fmt.Sprintf("productive denies MAY have pruned %d futile round-trip(s) across %d deny(ies); keep enriching deny reasons with forward guidance — but this is a speculative rebate, not realized work",
			rep.SpeculativePrunedTurns, rep.ProductiveDenies))
		// Mandatory: the speculative axis is excluded from the grade and is witness-gated.
		// It must always be surfaced so a reader never mistakes the upper bound for the grade.
		risks = append(risks, fmt.Sprintf("SPECULATIVE: %d pruned round-trip(s) are a counterfactual upper bound — EXCLUDED from the graded amplification/grade/status (which are realized, Counter-backed only) and witness-gated until a kernel counter measures the avoided fan-out",
			rep.SpeculativePrunedTurns))
	}
	if rep.SpeculativeAggregateCapped {
		risks = append(risks, fmt.Sprintf("the speculative pruned total saturated at the aggregate cap (%d); the upper bound is a floor of the true count, never inflated", DefaultMaxSpeculativePrunedTurns))
	}
	if rep.RedirectCapped > 0 {
		risks = append(risks, fmt.Sprintf("%d redirect fan-out(s) were clamped to the per-deny cap; the speculative pruned total is a LOWER bound, not inflated", rep.RedirectCapped))
	}
	if rep.StaleMisses > 0 && rep.StaleMisses*2 >= rep.MemoHits {
		risks = append(risks, fmt.Sprintf("stale misses (%d) rival hits (%d): the world mutates faster than the cache amortizes — run ProveMemo for this class; a global world-version may be over-invalidating",
			rep.StaleMisses, rep.MemoHits))
		actions = append(actions, "narrow the vDSO world-version to the written scope so an unrelated write stops invalidating stable read entries")
	}
	switch rep.Status {
	case "regressing":
		risks = append(risks, "avoidance is a net LOSS this window: the cache/redirect overhead exceeds what it saved — fall back to always-execute for the volatile classes")
	case "break_even":
		actions = append(actions, "avoidance is paying for itself but not yet winning; raise reuse (longer-lived entries) or deny productively to amplify")
	}
	if len(actions) == 0 {
		actions = append(actions, "no avoidance signal in this window; nothing to tune")
	}
	return actions, risks
}

// ---------------------------------------------------------------------------
// small numeric helpers (mirroring vcache's local helpers).
// ---------------------------------------------------------------------------

// formatBreakEven renders the sentinel as "never", matching `fak vcache`'s output.
func formatBreakEven(n int) string {
	if n == neverBreakEven {
		return "never"
	}
	return fmt.Sprintf("%d", n)
}

// safeRatio returns num/den, with +Inf when den==0<num (all work avoided) and 1.0
// when both are zero (an empty window neither amplifies nor regresses).
func safeRatio(num, den float64) float64 {
	if den == 0 {
		if num > 0 {
			return math.Inf(1)
		}
		return 1
	}
	return num / den
}

func nonNeg(x float64) float64 {
	if x < 0 || math.IsNaN(x) {
		return 0
	}
	return x
}

func nonNegInt(n int) int {
	if n < 0 {
		return 0
	}
	return n
}

func clamp01(x float64) float64 {
	if x < 0 || math.IsNaN(x) {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
