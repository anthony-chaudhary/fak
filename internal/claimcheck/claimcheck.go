// Package claimcheck grades an efficiency/performance claim against the six
// questions of the net-true-value standard (docs/standards/net-true-value.md) and
// returns one of three verdicts: net-true / strawman / not-yet. It is the named
// follow-on that standard calls out in its own honest fence — "not a single
// `fak claim-check` verb that grades an arbitrary claim against the six questions.
// That verb … is the named follow-on, not built here." This package is that verb's
// pure core (the cross-cutting T14 of epic #1147, issue #1171).
//
// The grader is a lens, not an oracle: it does NOT measure anything or run a
// benchmark. It checks whether a claim, as STATED, answers the six questions a
// real gain must answer. Silence is not a pass — an unstated question fails, so a
// claim grades net-true only when it explicitly answers all six. The rule the
// standard states is mechanized verbatim: "A claim that can't answer one of the
// first five is `not yet`, not a result", and a gain measured against the naive
// floor instead of the tuned alternative is a strawman.
//
// The six questions (net-true-value.md §"The rubric"):
//
//	Q1 Baseline   — measured against the real (tuned) alternative, not a strawman.
//	Q2 Net        — stated after the cost the change itself adds.
//	Q3 Scope      — the conditions it holds under AND the ones it vanishes under.
//	Q4 Provenance — labeled WITNESSED / OBSERVED / MODELED / SIMULATED.
//	Q5 Witness    — a third party can re-derive it (a test, an artifact + command).
//	Q6 Realized   — on by default, or honestly gated with a stated reason.
//
// Pure logic, stdlib-only, off the hot path — a tier-1 leaf. The thin CLI shell is
// cmd/fak/claimcheck.go (`fak claim-check`); the graded self-tests over the honest
// + strawman fixture are claimcheck_test.go (the issue's acceptance witness).
package claimcheck

import (
	"fmt"
	"sort"
	"strings"
)

// Verdict is the closed result of grading a claim. Exactly one of three.
type Verdict string

const (
	// NetTrue: the claim answers all six questions against the real alternative.
	NetTrue Verdict = "net-true"
	// Strawman: the gain is measured against the naive floor, not the tuned
	// alternative a competent operator would actually deploy. This verdict wins
	// over not-yet — a big number against the wrong baseline is noise no matter
	// how many of the other questions it answers.
	Strawman Verdict = "strawman"
	// NotYet: the claim cannot answer one of the six questions, so it is not yet
	// a result. The unanswered questions are named in the Result.
	NotYet Verdict = "not-yet"
)

// BaselineKind names what a claim is measured against (Q1).
type BaselineKind string

const (
	// BaselineNone: no "vs what" at all ("5× faster" with nothing to compare to).
	// Q1 is unanswered ⇒ not-yet.
	BaselineNone BaselineKind = ""
	// BaselineStrawman: measured against the naive floor only (e.g. "60.3× vs
	// naive re-send-everything") with no tuned alternative beside it ⇒ strawman.
	BaselineStrawman BaselineKind = "strawman"
	// BaselineReal: measured against the tuned best-practice alternative an
	// operator would actually run (the headline). Q1 answered.
	BaselineReal BaselineKind = "real"
)

// Provenance is the closed label every reported number must carry (Q4) — the same
// vocabulary net-true-value and the observer-effect standard use. An empty label
// fails Q4 (a number with no provenance is unprovenanced, never a pass).
type Provenance string

const (
	ProvNone  Provenance = ""
	Witnessed Provenance = "WITNESSED" // a fact fak authored and controls
	Observed  Provenance = "OBSERVED"  // a value relayed from an external party
	Modeled   Provenance = "MODELED"   // a deterministic projection
	Simulated Provenance = "SIMULATED" // labeled stand-in data
)

// validProvenance reports whether p is one of the four closed labels (a non-empty
// label that is none of them is a typo, not a pass).
func (p Provenance) valid() bool {
	switch p {
	case Witnessed, Observed, Modeled, Simulated:
		return true
	default:
		return false
	}
}

// Baseline is a claim's answer to Q1: what it is measured against.
type Baseline struct {
	Kind        BaselineKind `json:"kind"`
	Description string       `json:"description,omitempty"` // e.g. "tuned warm-cache stack"
}

// Realized is a claim's answer to Q6: is the gain on by default, or honestly gated?
// A value that exists only behind a flag nobody sets — with no stated reason — is a
// seam, not a realized gain, and fails Q6.
type Realized struct {
	OnByDefault bool `json:"on_by_default"`
	// GateReason is the stated reason a gain ships OFF (e.g. "f16 OOM, structural").
	// A non-empty reason makes an off-by-default gain an HONEST gate (Q6 passes);
	// an empty reason on an off-by-default value is the seam that fails Q6.
	GateReason string `json:"gate_reason,omitempty"`
}

// realized reports whether Q6 is satisfied: on by default, or off with a reason.
func (r Realized) realized() bool {
	return r.OnByDefault || strings.TrimSpace(r.GateReason) != ""
}

// Claim is a structured efficiency/performance claim: the statement plus its
// answers to the six questions. The zero value answers nothing, so it grades
// not-yet — silence is never a pass.
type Claim struct {
	// Statement is the claim itself, e.g. "fleet prompt reuse is 4.1× faster".
	Statement string `json:"statement"`
	// Baseline answers Q1 (vs what).
	Baseline Baseline `json:"baseline"`
	// Net answers Q2: is the gain stated net of the cost the change itself adds
	// (including where the net goes negative)?
	Net bool `json:"net"`
	// Scope answers Q3: the conditions it holds under AND the ones it vanishes
	// under. Non-empty = stated.
	Scope string `json:"scope,omitempty"`
	// Provenance answers Q4: the closed label on the number.
	Provenance Provenance `json:"provenance,omitempty"`
	// Witness answers Q5: how a third party re-derives it (a test name, an
	// artifact + reproduce command). Non-empty = a witness is named.
	Witness string `json:"witness,omitempty"`
	// Realized answers Q6.
	Realized Realized `json:"realized"`
}

// Question is one rubric question's grade for a claim.
type Question struct {
	N      int    `json:"n"`      // 1..6
	Name   string `json:"name"`   // "baseline", "net", …
	Pass   bool   `json:"pass"`   // did the claim answer it?
	Detail string `json:"detail"` // why it passed or failed
}

// Result is the full grade of a claim: the verdict, the six per-question grades,
// and the names of the questions the claim failed to answer.
type Result struct {
	Statement string     `json:"statement"`
	Verdict   Verdict    `json:"verdict"`
	Questions []Question `json:"questions"`
	// Missing names the questions a not-yet claim could not answer (the evidence
	// the standard says to name instead of dressing an unproven claim as shipped).
	Missing []string `json:"missing,omitempty"`
	// Reason is a one-line human summary of the verdict.
	Reason string `json:"reason"`
}

// Grade runs a claim through the six questions and returns the verdict.
//
// The order is the standard's: a strawman baseline short-circuits to Strawman
// (a number against the wrong baseline is noise regardless of the other answers).
// Otherwise every unanswered question is collected; any miss ⇒ NotYet with the
// misses named; all six answered ⇒ NetTrue.
func Grade(c Claim) Result {
	qs := []Question{
		gradeBaseline(c.Baseline),
		gradeNet(c.Net),
		gradeScope(c.Scope),
		gradeProvenance(c.Provenance),
		gradeWitness(c.Witness),
		gradeRealized(c.Realized),
	}

	res := Result{Statement: c.Statement, Questions: qs}

	// A strawman baseline is its own verdict — and it wins over not-yet.
	if c.Baseline.Kind == BaselineStrawman {
		res.Verdict = Strawman
		res.Reason = "measured against the naive floor, not the tuned alternative — strawman baseline (Q1)"
		// Still surface any other gaps so the author sees the whole picture.
		res.Missing = missingNames(qs)
		return res
	}

	missing := missingNames(qs)
	if len(missing) > 0 {
		res.Verdict = NotYet
		res.Missing = missing
		res.Reason = "not yet a result — unanswered: " + strings.Join(missing, ", ")
		return res
	}

	res.Verdict = NetTrue
	res.Reason = "net-true at the stated scope — all six questions answered against the real baseline"
	return res
}

// missingNames returns the names of the failed questions, in question order.
func missingNames(qs []Question) []string {
	var out []string
	for _, q := range qs {
		if !q.Pass {
			out = append(out, q.Name)
		}
	}
	return out
}

func gradeBaseline(b Baseline) Question {
	q := Question{N: 1, Name: "baseline"}
	switch b.Kind {
	case BaselineReal:
		q.Pass = true
		q.Detail = "measured against the real (tuned) alternative"
		if b.Description != "" {
			q.Detail += ": " + b.Description
		}
	case BaselineStrawman:
		q.Pass = false
		q.Detail = "measured against the naive floor (strawman)"
		if b.Description != "" {
			q.Detail += ": " + b.Description
		}
	default:
		q.Pass = false
		q.Detail = `no "vs what" stated`
	}
	return q
}

func gradeNet(net bool) Question {
	q := Question{N: 2, Name: "net"}
	q.Pass = net
	if net {
		q.Detail = "stated net of the cost the change itself adds"
	} else {
		q.Detail = "the cost the change adds is not subtracted"
	}
	return q
}

func gradeScope(scope string) Question {
	q := Question{N: 3, Name: "scope"}
	q.Pass = strings.TrimSpace(scope) != ""
	if q.Pass {
		q.Detail = "scope stated: " + strings.TrimSpace(scope)
	} else {
		q.Detail = "the conditions it holds (and vanishes) under are not stated"
	}
	return q
}

func gradeProvenance(p Provenance) Question {
	q := Question{N: 4, Name: "provenance"}
	switch {
	case p.valid():
		q.Pass = true
		q.Detail = "labeled " + string(p)
	case p == ProvNone:
		q.Pass = false
		q.Detail = "the number carries no provenance label"
	default:
		q.Pass = false
		q.Detail = fmt.Sprintf("provenance %q is not one of WITNESSED/OBSERVED/MODELED/SIMULATED", string(p))
	}
	return q
}

func gradeWitness(w string) Question {
	q := Question{N: 5, Name: "witness"}
	q.Pass = strings.TrimSpace(w) != ""
	if q.Pass {
		q.Detail = "re-derivable: " + strings.TrimSpace(w)
	} else {
		q.Detail = "no witness a third party can re-derive"
	}
	return q
}

func gradeRealized(r Realized) Question {
	q := Question{N: 6, Name: "realized"}
	q.Pass = r.realized()
	switch {
	case r.OnByDefault:
		q.Detail = "on by default"
	case q.Pass:
		q.Detail = "off by default, honestly gated: " + strings.TrimSpace(r.GateReason)
	default:
		q.Detail = "off behind a flag with no stated reason (a seam, not a realized gain)"
	}
	return q
}

// String renders a Result as a compact, operator-readable block.
func (r Result) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "claim-check: %s\n", strings.ToUpper(string(r.Verdict)))
	if r.Statement != "" {
		fmt.Fprintf(&b, "  claim: %s\n", r.Statement)
	}
	for _, q := range r.Questions {
		mark := "PASS"
		if !q.Pass {
			mark = "FAIL"
		}
		fmt.Fprintf(&b, "  Q%d %-10s %s  %s\n", q.N, q.Name, mark, q.Detail)
	}
	fmt.Fprintf(&b, "  => %s\n", r.Reason)
	return b.String()
}

// FixtureCase is one labeled example claim with the verdict it must grade to — the
// graded self-test corpus over honest and strawman claims (issue #1171 acceptance).
type FixtureCase struct {
	Name   string  `json:"name"`
	Claim  Claim   `json:"claim"`
	Expect Verdict `json:"expect"`
}

// Fixture is the built-in corpus of honest and strawman claims, each paired with
// the verdict the grader must return. It is asserted by claimcheck_test.go and is
// runnable from the CLI via `fak claim-check --self-test`, so the grader's own
// behavior is itself a re-derivable witness (Q5, turned on the verb).
func Fixture() []FixtureCase {
	return []FixtureCase{
		{
			Name: "honest-net-true",
			// fak's own fleet-reuse headline, read against the tuned baseline.
			Claim: Claim{
				Statement:  "fleet prompt reuse is 4.1× vs a tuned warm-cache stack",
				Baseline:   Baseline{Kind: BaselineReal, Description: "tuned warm-cache stack"},
				Net:        true,
				Scope:      "cross-worker shared prefix; falls to 1.0–1.1× once each worker is already warm",
				Provenance: Witnessed,
				Witness:    "go test ./internal/gateway -run TestFleetReuse + the committed trace",
				Realized:   Realized{OnByDefault: true},
			},
			Expect: NetTrue,
		},
		{
			Name: "honest-gate-net-true",
			// A real gain shipped OFF for a stated, structural reason is still realized-honest.
			Claim: Claim{
				Statement:  "Q8 GPU-resident decode is 0.99× llama-Metal at 7B",
				Baseline:   Baseline{Kind: BaselineReal, Description: "llama.cpp Metal"},
				Net:        true,
				Scope:      "Apple-silicon, dense Q8; f16 OOMs structurally",
				Provenance: Witnessed,
				Witness:    "cosine 0.999991 vs reference, FAK_METAL_DECODE=1",
				Realized:   Realized{GateReason: "Mac-gated; default OFF until on-device tok/s clears the bar"},
			},
			Expect: NetTrue,
		},
		{
			Name: "strawman-naive-baseline",
			// The 60.3× vs naive re-send-everything number, quoted as the serving win.
			Claim: Claim{
				Statement:  "fleet prompt reuse is 60.3× faster",
				Baseline:   Baseline{Kind: BaselineStrawman, Description: "naive re-send-everything"},
				Net:        true,
				Scope:      "cache-favorable trace",
				Provenance: Witnessed,
				Witness:    "the committed reuse trace",
				Realized:   Realized{OnByDefault: true},
			},
			Expect: Strawman,
		},
		{
			Name: "no-baseline-not-yet",
			// "5× faster" with no "vs what" — cannot answer Q1.
			Claim: Claim{
				Statement:  "5× faster",
				Baseline:   Baseline{Kind: BaselineNone},
				Net:        true,
				Scope:      "the demo",
				Provenance: Modeled,
				Witness:    "the demo run",
				Realized:   Realized{OnByDefault: true},
			},
			Expect: NotYet,
		},
		{
			Name: "no-witness-not-yet",
			// A modeled headline with no artifact a third party can re-derive (Q5).
			Claim: Claim{
				Statement:  "the kernel is 3× faster on H100 decode",
				Baseline:   Baseline{Kind: BaselineReal, Description: "tuned vLLM"},
				Net:        true,
				Scope:      "H100, batch 1",
				Provenance: Modeled,
				Witness:    "",
				Realized:   Realized{OnByDefault: true},
			},
			Expect: NotYet,
		},
		{
			Name: "unlabeled-provenance-not-yet",
			// A number with no WITNESSED/OBSERVED/MODELED/SIMULATED label (Q4).
			Claim: Claim{
				Statement:  "tokens cut by 40%",
				Baseline:   Baseline{Kind: BaselineReal, Description: "tuned compaction"},
				Net:        true,
				Scope:      "long-context sessions",
				Provenance: ProvNone,
				Witness:    "the session ledger",
				Realized:   Realized{OnByDefault: true},
			},
			Expect: NotYet,
		},
		{
			Name: "no-net-not-yet",
			// A cache that is a win on reuse but never states its single-use cost (Q2).
			Claim: Claim{
				Statement:  "the cache saves 90% of prefill",
				Baseline:   Baseline{Kind: BaselineReal, Description: "tuned warm cache"},
				Net:        false,
				Scope:      "high-reuse workloads",
				Provenance: Witnessed,
				Witness:    "the reuse counter",
				Realized:   Realized{OnByDefault: true},
			},
			Expect: NotYet,
		},
		{
			Name: "no-scope-not-yet",
			// A gain with no scope — where does it vanish? (Q3).
			Claim: Claim{
				Statement:  "tool-vDSO saves a redecode",
				Baseline:   Baseline{Kind: BaselineReal, Description: "tuned tool loop"},
				Net:        true,
				Scope:      "",
				Provenance: Witnessed,
				Witness:    "the vDSO hit counter",
				Realized:   Realized{OnByDefault: true},
			},
			Expect: NotYet,
		},
		{
			Name: "unrealized-seam-not-yet",
			// A value that exists only behind a flag nobody sets, no reason (Q6).
			Claim: Claim{
				Statement:  "the speculative path is 2× faster",
				Baseline:   Baseline{Kind: BaselineReal, Description: "tuned single-model decode"},
				Net:        true,
				Scope:      "acceptance-favorable drafts",
				Provenance: Witnessed,
				Witness:    "the acceptance meter",
				Realized:   Realized{OnByDefault: false, GateReason: ""},
			},
			Expect: NotYet,
		},
	}
}

// RunFixture grades every fixture case and returns the per-case outcomes plus how
// many graded as expected. It is the shared core of the Go self-test and the CLI
// `--self-test` mode, so both assert the exact same corpus.
func RunFixture() (cases []FixtureOutcome, passed int) {
	for _, fc := range Fixture() {
		got := Grade(fc.Claim).Verdict
		ok := got == fc.Expect
		if ok {
			passed++
		}
		cases = append(cases, FixtureOutcome{Name: fc.Name, Expect: fc.Expect, Got: got, OK: ok})
	}
	return cases, passed
}

// FixtureOutcome is one fixture case's graded result.
type FixtureOutcome struct {
	Name   string  `json:"name"`
	Expect Verdict `json:"expect"`
	Got    Verdict `json:"got"`
	OK     bool    `json:"ok"`
}

// VerdictCounts tallies a slice of results by verdict (for a batch summary).
func VerdictCounts(results []Result) map[Verdict]int {
	counts := map[Verdict]int{}
	for _, r := range results {
		counts[r.Verdict]++
	}
	return counts
}

// SortedVerdicts returns the verdict keys of a count map in a stable order, so a
// rendered summary is deterministic.
func SortedVerdicts(counts map[Verdict]int) []Verdict {
	out := make([]Verdict, 0, len(counts))
	for v := range counts {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
