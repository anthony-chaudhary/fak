// Package modelroute is fak's model-routing spine: choose WHICH model — or which
// ENSEMBLE of models — serves any ASPECT of a request, under one declarative,
// deterministic, verifiable policy. It is the first-class form of the project's
// "different model per any aspect of a request, ensembles included" thesis.
//
// THE GAP IT FILLS. SOTA model routers (RouteLLM, Martian, NotDiamond,
// OpenRouter, Portkey, LiteLLM's router) route at ONE granularity — the whole
// request — and pick ONE model for it. fak makes model routing first-class at
// EVERY level: the unit of routing is an Aspect — the whole request, OR a single
// tool call, OR a sub-query, OR a planner state, OR a reasoning step — so within
// one request the tool_call can go to a guard model, the retrieval to a small
// model, and the hard reasoning step to a large model, each decided by the same
// policy. And an ENSEMBLE — a SET of models on one item, folded by a Reduction
// (first / vote / best-of / all-reduce / concat) — is a first-class Plan, not a
// bolt-on. Neither generalization is something a request-level router expresses.
//
// THE SHAPE mirrors two existing fak idioms exactly:
//   - internal/gateway/routing.go — pure, data-in / decision-out, no I/O, no
//     goroutines on the hot path, trivially A/B-testable (run two Manifests over
//     one Subject and diff). This package generalizes that file's RequestClass →
//     Tier to Subject → Plan over ANY Aspect, and adds the ensemble half.
//   - internal/policy — the policy is a declarative, version-tagged JSON MANIFEST
//     loaded at runtime (not a compiled-in Go literal), validated fail-loud with
//     DisallowUnknownFields, round-tripping --dump ↔ --check. An operator
//     configures routing by editing a reviewable file, never by forking the kernel.
//
// TWO REAL HALVES, both mechanically witnessed by the test suite:
//   - Route(Subject) → Decision: the first matching Rule's Plan wins, else the
//     fail-closed Default. Pure and deterministic.
//   - Combine(Reduction, []Vote) → Result: the ensemble REDUCE — fold many member
//     outputs into one (first / weighted vote / best-of / concat / numeric
//     all-reduce). Pure and deterministic.
//
// What is deliberately NOT here (the wiring, tracked as GitHub issues): the live
// multi-model DISPATCH that runs a chosen Plan's members on a real engine and
// feeds their outputs into Combine. A single-model Plan's Primary() yields the
// string a wiring layer assigns to the frozen ABI seam abi.ToolCall.Engine
// (already reserved as "optional per-call engine route; empty => kernel default");
// the gateway/kernel consuming that field, and executing an ensemble Plan, is
// additive wiring ON TOP of this spine.
//
// THE WIRING CONTRACT (load-bearing — designed in now so the later wiring cannot
// regress fak's default-deny floor):
//
//   - ROUTE BEFORE ADJUDICATE. The host must write the chosen model to
//     abi.ToolCall.Engine BEFORE Kernel.Submit, never as a post-adjudication /
//     dispatch-time override. The residency PDP (internal/engine residencyGate)
//     reads c.Engine INSIDE the adjudication fold to deny a tenant/sensitive
//     payload bound for a REMOTE engine; if routing instead set the model at
//     dispatch, that gate would have adjudicated an empty route and the sensitive
//     payload would reach a remote model fail-OPEN. Routing is a pre-submit step.
//   - AN ENSEMBLE EXPANDS TO N INDEPENDENTLY-ADJUDICATED CALLS. Executing a Plan
//     with len(Members) > 1 is N separate Kernel.Submit calls, each carrying its
//     member model in Engine, each crossing the syscall boundary and adjudicated
//     on its own — never one fan-out that bypasses the floor for the members.
//   - MEMBER ORDER IS PRESERVED INTO THE FOLD. The dispatcher MUST gather member
//     outputs into the []Vote passed to Combine in Plan.Members order (not engine
//     completion order), or the order-sensitive reductions (concat, vote
//     tie-break) stop being deterministic. See Combine.
//
// SCOPE OF "DETERMINISTIC": the routing DECISION (Route) and the FOLD (Combine,
// given fixed votes) are deterministic and auditable. The member models' OUTPUTS
// are not — they come from non-bit-exact engines — so determinism is pinned to the
// decision and its reduce, never to end-to-end answer reproducibility.
//
// CLOSED vs OPEN sets: Reduction, Latency, and Complexity are CLOSED additive
// vocabularies (a new value is a new constant + validation, never manifest free
// text). Aspect is intentionally OPEN — a deployment may route its own named stage
// — mirroring the ABI's open OpCode/ExtKey ranges; Validate does not restrict it.
//
// The package is pure (stdlib only), so it is reusable by the gateway, the
// adjudicator chain, the agent loop, and the `fak route` CLI alike without
// pulling any of them into each other.
package modelroute

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Version is the current manifest schema tag. A manifest MAY omit it (treated as
// the current version); a manifest naming a different MAJOR is refused. Minor
// bumps (fak-route/v1.x) are forward-accepted so an older binary tolerates a
// newer minor manifest — the additive-only discipline the frozen ABI uses.
const Version = "fak-route/v1"

// ---------------------------------------------------------------------------
// THE SUBJECT — the thing being routed, at ANY granularity.
// ---------------------------------------------------------------------------

// Aspect is the GRANULARITY of the thing being routed — the "every level" axis
// that sets fak apart from a request-only router. It is an OPEN set: the named
// constants below are the well-known aspects, but a deployment may route its own
// named stage by using any string, exactly as the ABI's OpCode/ExtKey ranges are
// open. Validate does not restrict Aspect values.
type Aspect string

const (
	// AspectRequest is the whole inbound request — the unit a SOTA router stops at.
	AspectRequest Aspect = "request"
	// AspectToolCall is one tool call within a request (route the refund tool to a
	// guard model, the search tool to a small one).
	AspectToolCall Aspect = "tool_call"
	// AspectQuery is a retrieval / sub-question issued while answering a request.
	AspectQuery Aspect = "query"
	// AspectState is a planner / agent state transition.
	AspectState Aspect = "state"
	// AspectStep is one reasoning / decode step.
	AspectStep Aspect = "step"
	// AspectScout is the cheap classify-first probe (a scout model labels the
	// subject's complexity/domain, then the real route is taken).
	AspectScout Aspect = "scout"
)

// Latency is the latency requirement a Subject carries (a routing signal, mirrors
// gateway.LatencyClass). "" leaves it unconstrained.
type Latency string

const (
	// LatencyAny imposes no latency constraint.
	LatencyAny Latency = ""
	// LatencyInteractive is a low-latency, user-facing turn.
	LatencyInteractive Latency = "interactive"
	// LatencyBatch is high-throughput, best-effort work.
	LatencyBatch Latency = "batch"
)

// Complexity is the coarse difficulty of a Subject. It is ordered (low < medium <
// high) so a Match can set a floor: route only the HARD aspects to the large model.
type Complexity string

const (
	// ComplexityAny imposes no complexity floor.
	ComplexityAny Complexity = ""
	// ComplexityLow is trivial work.
	ComplexityLow Complexity = "low"
	// ComplexityMedium is moderate work.
	ComplexityMedium Complexity = "medium"
	// ComplexityHigh is hard work.
	ComplexityHigh Complexity = "high"
)

// complexityRank maps a Complexity to a comparable ordinal (unknown/"" == 0).
func complexityRank(c Complexity) int {
	switch c {
	case ComplexityLow:
		return 1
	case ComplexityMedium:
		return 2
	case ComplexityHigh:
		return 3
	}
	return 0
}

// validComplexity reports whether a Complexity is one of the closed set (or "").
func validComplexity(c Complexity) bool {
	switch c {
	case ComplexityAny, ComplexityLow, ComplexityMedium, ComplexityHigh:
		return true
	}
	return false
}

// validLatency reports whether a Latency is one of the closed set (or "").
func validLatency(l Latency) bool {
	switch l {
	case LatencyAny, LatencyInteractive, LatencyBatch:
		return true
	}
	return false
}

// Subject is the classified thing to route — the input to Route. It generalizes
// gateway.RequestClass from "a request" to ANY aspect at any granularity. A Rule's
// Match tests these fields; an unset Match field is a wildcard. Labels carries
// OPEN signals (domain, language, tenant, taint, …) a deployment matches on
// without a code change.
type Subject struct {
	Aspect       Aspect            `json:"aspect,omitempty"`
	Tool         string            `json:"tool,omitempty"`          // the tool name when Aspect == AspectToolCall
	PromptTokens int               `json:"prompt_tokens,omitempty"` // estimated prompt length in tokens
	Latency      Latency           `json:"latency,omitempty"`
	Complexity   Complexity        `json:"complexity,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

// ---------------------------------------------------------------------------
// THE PLAN — a single model PICK or an ENSEMBLE + a Reduction.
// ---------------------------------------------------------------------------

// Member is one model in a Plan. Weight is its vote / aggregation weight (<= 0 is
// treated as 1 at use). Role is an optional label (primary / drafter / verifier /
// judge / …) so an ensemble can carry structure the dispatcher and Combine read.
type Member struct {
	Model  string  `json:"model"`
	Weight float64 `json:"weight,omitempty"`
	Role   string  `json:"role,omitempty"`
}

// Reduction is how an ensemble's member outputs fold into one result. It is a
// CLOSED, additive set — a new reduction is an added constant + a Combine arm,
// never a manifest free-text field.
type Reduction string

const (
	// ReduceFirst takes the first member's output (fastest-wins / fallback chain).
	ReduceFirst Reduction = "first"
	// ReduceVote takes the weighted-majority output over the members' discrete
	// answers (self-consistency / quorum).
	ReduceVote Reduction = "vote"
	// ReduceBestOf takes the highest-scored member's output (a judge/verifier
	// scores each; the best wins).
	ReduceBestOf Reduction = "best_of"
	// ReduceAllReduce numerically aggregates the members' SCALAR outputs into their
	// weighted mean — the map-reduce / all-reduce form for numeric answers (a score,
	// a count, a probability). It is NOT a tensor all-reduce: outputs that do not
	// parse as a float are an error, not a silent guess. (Name borrows the
	// distributed-systems term for the scalar reduce family; the scope is scalars.)
	ReduceAllReduce Reduction = "all_reduce"
	// ReduceConcat concatenates the members' outputs (fan-out gather).
	ReduceConcat Reduction = "concat"
)

// knownReduction reports whether r is one of the closed Reduction set.
func knownReduction(r Reduction) bool {
	switch r {
	case ReduceFirst, ReduceVote, ReduceBestOf, ReduceAllReduce, ReduceConcat:
		return true
	}
	return false
}

// Plan is what a matched Rule selects: a list of Members and, when there is more
// than one, the Reduction that folds their outputs. len(Members) == 1 is a single
// PICK (the SOTA shape); len > 1 is an ENSEMBLE. Scout names an optional cheap
// model that classifies the subject FIRST (the scout-then-route pattern); Reason
// is a free-text note surfaced in the Decision trace.
type Plan struct {
	Members []Member  `json:"members"`
	Reduce  Reduction `json:"reduce,omitempty"`
	Scout   string    `json:"scout,omitempty"`
	Reason  string    `json:"reason,omitempty"`
}

// IsEnsemble reports whether the Plan fans out to more than one model.
func (p Plan) IsEnsemble() bool { return len(p.Members) > 1 }

// Primary returns the model id the host dispatches to for a single-model route:
// the member tagged Role=="primary" if any, else the first member, else "". For a
// single-model Plan this is the value that populates abi.ToolCall.Engine.
func (p Plan) Primary() string {
	for _, m := range p.Members {
		if m.Role == "primary" {
			return m.Model
		}
	}
	if len(p.Members) > 0 {
		return p.Members[0].Model
	}
	return ""
}

// Models returns every member model id, in declared order.
func (p Plan) Models() []string {
	out := make([]string, 0, len(p.Members))
	for _, m := range p.Members {
		out = append(out, m.Model)
	}
	return out
}

// ---------------------------------------------------------------------------
// THE POLICY — ordered Match → Plan rules in a JSON Manifest.
// ---------------------------------------------------------------------------

// Match is a Rule's predicate over a Subject. A rule fires when EVERY set field
// holds (logical AND); an unset field is a wildcard. Tool supports a single
// trailing "*" as a prefix wildcard ("git_*" matches "git_push").
type Match struct {
	Aspect          Aspect            `json:"aspect,omitempty"`
	Tool            string            `json:"tool,omitempty"`
	MinPromptTokens int               `json:"min_prompt_tokens,omitempty"`
	MaxPromptTokens int               `json:"max_prompt_tokens,omitempty"` // 0 == unbounded
	Latency         Latency           `json:"latency,omitempty"`
	MinComplexity   Complexity        `json:"min_complexity,omitempty"`
	Labels          map[string]string `json:"labels,omitempty"` // every pair must equal
}

// Matches reports whether the Subject satisfies the Match predicate.
func (m Match) Matches(s Subject) bool {
	if m.Aspect != "" && m.Aspect != s.Aspect {
		return false
	}
	if m.Tool != "" && !toolMatch(m.Tool, s.Tool) {
		return false
	}
	if s.PromptTokens < m.MinPromptTokens {
		return false
	}
	if m.MaxPromptTokens != 0 && s.PromptTokens > m.MaxPromptTokens {
		return false
	}
	if m.Latency != "" && m.Latency != s.Latency {
		return false
	}
	if m.MinComplexity != "" && complexityRank(s.Complexity) < complexityRank(m.MinComplexity) {
		return false
	}
	for k, v := range m.Labels {
		if s.Labels[k] != v {
			return false
		}
	}
	return true
}

// toolMatch supports an exact tool name or a single trailing-"*" prefix wildcard.
func toolMatch(pat, tool string) bool {
	if pat == "*" {
		return true
	}
	if strings.HasSuffix(pat, "*") {
		return strings.HasPrefix(tool, strings.TrimSuffix(pat, "*"))
	}
	return pat == tool
}

// Rule is one ordered routing rule: a Match predicate and the Plan to apply when
// it fires. Rules are evaluated top-to-bottom; the FIRST match wins (so put the
// most specific rules first), and Name is a unique label surfaced in the Decision.
type Rule struct {
	Name  string `json:"name"`
	Match Match  `json:"match"`
	Plan  Plan   `json:"plan"`
}

// Manifest is the on-disk routing policy: an ordered rule list plus a fail-closed
// Default Plan applied when no rule matches. It is the declarative, version-tagged
// file an operator edits — the model-routing analogue of the capability-floor
// policy manifest.
type Manifest struct {
	Version string `json:"version,omitempty"`
	Default Plan   `json:"default"`
	Rules   []Rule `json:"rules,omitempty"`
}

// Decision is the outcome of Route: the echoed Subject, which Rule fired (empty +
// Matched==false means the Default Plan was used), and the chosen Plan.
type Decision struct {
	Subject  Subject
	RuleName string
	Matched  bool
	Plan     Plan
}

// Route classifies and selects: the first Rule whose Match fires returns its Plan;
// if none fire, the fail-closed Default Plan is returned. Pure and deterministic —
// no I/O, no goroutines — so two Manifests are trivially diffed over one Subject.
func (m Manifest) Route(s Subject) Decision {
	for _, r := range m.Rules {
		if r.Match.Matches(s) {
			return Decision{Subject: s, RuleName: r.Name, Matched: true, Plan: r.Plan}
		}
	}
	return Decision{Subject: s, Matched: false, Plan: m.Default}
}

// ---------------------------------------------------------------------------
// VALIDATION — fail-loud, so a misconfigured route never silently mis-dispatches.
// ---------------------------------------------------------------------------

// Validate checks a Manifest is well-formed: a known major version, a fail-closed
// Default Plan with >= 1 member, unique non-empty rule names, well-formed match
// bounds, a closed-vocabulary latency/complexity, and every Plan (default and per
// rule) valid. A misconfigured routing policy must fail at the boundary, never
// fall through to a silent default model.
func (m Manifest) Validate() error {
	if m.Version != "" && !strings.HasPrefix(m.Version, Version) {
		return fmt.Errorf("modelroute: manifest version %q is not %s.x", m.Version, Version)
	}
	if err := validatePlan("default", m.Default); err != nil {
		return err
	}
	seen := make(map[string]bool, len(m.Rules))
	for i, r := range m.Rules {
		if r.Name == "" {
			return fmt.Errorf("modelroute: rule %d has an empty name", i)
		}
		if seen[r.Name] {
			return fmt.Errorf("modelroute: duplicate rule name %q", r.Name)
		}
		seen[r.Name] = true
		if r.Match.MinPromptTokens < 0 || r.Match.MaxPromptTokens < 0 {
			return fmt.Errorf("modelroute: rule %q has a negative token bound", r.Name)
		}
		if r.Match.MaxPromptTokens != 0 && r.Match.MinPromptTokens > r.Match.MaxPromptTokens {
			return fmt.Errorf("modelroute: rule %q has min_prompt_tokens > max_prompt_tokens", r.Name)
		}
		if !validLatency(r.Match.Latency) {
			return fmt.Errorf("modelroute: rule %q has unknown latency %q", r.Name, r.Match.Latency)
		}
		if !validComplexity(r.Match.MinComplexity) {
			return fmt.Errorf("modelroute: rule %q has unknown min_complexity %q", r.Name, r.Match.MinComplexity)
		}
		if err := validatePlan("rule "+r.Name, r.Plan); err != nil {
			return err
		}
	}
	return nil
}

// validatePlan enforces the fail-closed invariant (>= 1 member) and a well-formed
// ensemble (an ensemble MUST carry a known Reduction; a single-member plan's
// Reduce is ignored but, if set, must still be a known value so a typo fails loud).
func validatePlan(where string, p Plan) error {
	if len(p.Members) == 0 {
		return fmt.Errorf("modelroute: %s plan has no members (a fail-closed route needs >= 1 model)", where)
	}
	for i, mem := range p.Members {
		if mem.Model == "" {
			return fmt.Errorf("modelroute: %s plan member %d has an empty model", where, i)
		}
		if mem.Weight < 0 {
			return fmt.Errorf("modelroute: %s plan member %q has a negative weight", where, mem.Model)
		}
	}
	if len(p.Members) > 1 {
		if !knownReduction(p.Reduce) {
			return fmt.Errorf("modelroute: %s plan is an ensemble (%d members) but its reduce %q is not a known reduction", where, len(p.Members), p.Reduce)
		}
	} else if p.Reduce != "" && !knownReduction(p.Reduce) {
		return fmt.Errorf("modelroute: %s plan has unknown reduce %q", where, p.Reduce)
	}
	return nil
}

// ---------------------------------------------------------------------------
// LOAD / DUMP — the JSON manifest round-trip (mirrors fak policy --dump|--check).
// ---------------------------------------------------------------------------

// JSON renders the Manifest as the canonical indented manifest (stamping the
// current Version when absent), terminated by a newline so --dump > file is clean.
func (m Manifest) JSON() []byte {
	out := m
	if out.Version == "" {
		out.Version = Version
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return append(b, '\n')
}

// ParseManifest decodes and validates a manifest. Unknown JSON fields are REJECTED
// (DisallowUnknownFields) so a typo fails loudly instead of silently changing the
// routing surface.
func ParseManifest(b []byte) (Manifest, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var m Manifest
	if err := dec.Decode(&m); err != nil {
		return Manifest{}, fmt.Errorf("modelroute: parse manifest: %w", err)
	}
	if err := m.Validate(); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

// LoadManifest reads and validates a manifest from a file path.
func LoadManifest(path string) (Manifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, fmt.Errorf("modelroute: read manifest %s: %w", path, err)
	}
	return ParseManifest(b)
}

// DefaultManifest is a small, conservative starting policy `fak route --dump`
// emits for an operator to edit: a fail-closed default, plus three illustrative
// rules — interactive short turns to a small model, hard reasoning to a large
// model, and a write-shaped tool call to a two-model guard ensemble (vote). The
// richer per-aspect example lives in examples/model-routing.example.json.
func DefaultManifest() Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{
			Members: []Member{{Model: "default", Role: "primary"}},
			Reason:  "no rule matched; fail-closed default model",
		},
		Rules: []Rule{
			{
				Name:  "interactive-short",
				Match: Match{Latency: LatencyInteractive, MaxPromptTokens: 4096},
				Plan:  Plan{Members: []Member{{Model: "small"}}, Reason: "short interactive turn -> small/fast model"},
			},
			{
				Name:  "hard-reasoning",
				Match: Match{MinComplexity: ComplexityHigh},
				Plan:  Plan{Members: []Member{{Model: "large"}}, Reason: "high-complexity work -> large model"},
			},
			{
				Name:  "guard-writes",
				Match: Match{Aspect: AspectToolCall, Tool: "write_*"},
				Plan: Plan{
					Members: []Member{{Model: "guard-a", Weight: 1}, {Model: "guard-b", Weight: 1}},
					Reduce:  ReduceVote,
					Reason:  "write-shaped tool call -> two-model guard ensemble (vote)",
				},
			},
		},
	}
}

// ---------------------------------------------------------------------------
// COMBINE — the ensemble REDUCE half (fold many member outputs into one).
// ---------------------------------------------------------------------------

// Vote is one ensemble member's produced output, ready to be folded by Combine.
// Score is consulted only by ReduceBestOf; Member.Weight only by ReduceVote /
// ReduceAllReduce. The dispatcher (the wiring layer) fills these by running the
// Plan's members; Combine is the pure fold over the result.
type Vote struct {
	Member Member  `json:"member"`
	Output string  `json:"output"`
	Score  float64 `json:"score,omitempty"`
}

// Result is the folded ensemble outcome: the rolled-up Output, the winning model
// (for first / vote / best_of), an optional per-distinct-output Tally (vote), and
// the member count. Output is the value the caller returns in place of any single
// model's answer.
type Result struct {
	Reduce  Reduction          `json:"reduce"`
	Output  string             `json:"output"`
	Winner  string             `json:"winner,omitempty"`
	Tally   map[string]float64 `json:"tally,omitempty"`
	Members int                `json:"members"`
}

// Combine folds a set of member outputs into one Result under the given Reduction.
// It is pure and deterministic: every tie is broken by a stable key (output string
// then model id) so the same votes always fold the same way. An empty vote set, or
// non-numeric outputs under ReduceAllReduce, is an error (never a silent guess).
//
// CONTRACT: the caller MUST pass votes in Plan.Members order, not engine-completion
// order. ReduceConcat joins in slice order and ReduceVote's tie-break reads it, so a
// dispatcher that gathers concurrently must re-sort into member order before calling
// Combine — otherwise the "deterministic reduce" property is silently lost.
//
// This is the ensemble half made real: the math that rolls many models' answers
// into one. The live dispatch that PRODUCES the votes (running each member on an
// engine) is the wiring layer above this; the fold itself is witnessed here.
func Combine(reduce Reduction, votes []Vote) (Result, error) {
	if len(votes) == 0 {
		return Result{}, errors.New("modelroute: Combine needs at least one vote")
	}
	if reduce == "" {
		reduce = ReduceFirst
	}
	switch reduce {
	case ReduceFirst:
		return Result{Reduce: ReduceFirst, Output: votes[0].Output, Winner: votes[0].Member.Model, Members: len(votes)}, nil

	case ReduceConcat:
		var sb strings.Builder
		for i, v := range votes {
			if i > 0 {
				sb.WriteByte('\n')
			}
			sb.WriteString(v.Output)
		}
		return Result{Reduce: ReduceConcat, Output: sb.String(), Members: len(votes)}, nil

	case ReduceVote:
		tally := make(map[string]float64, len(votes))
		for _, v := range votes {
			tally[v.Output] += weightOf(v.Member)
		}
		// Deterministic winner: highest weight, tie-break by output string asc.
		outputs := sortedKeys(tally)
		best, bestW := outputs[0], tally[outputs[0]]
		for _, o := range outputs[1:] {
			if tally[o] > bestW {
				best, bestW = o, tally[o]
			}
		}
		return Result{Reduce: ReduceVote, Output: best, Winner: winnerForOutput(votes, best), Tally: tally, Members: len(votes)}, nil

	case ReduceBestOf:
		bi := 0
		for i := range votes {
			if votes[i].Score > votes[bi].Score ||
				(votes[i].Score == votes[bi].Score && votes[i].Member.Model < votes[bi].Member.Model) {
				bi = i
			}
		}
		return Result{Reduce: ReduceBestOf, Output: votes[bi].Output, Winner: votes[bi].Member.Model, Members: len(votes)}, nil

	case ReduceAllReduce:
		var sum, wsum float64
		for _, v := range votes {
			f, err := strconv.ParseFloat(strings.TrimSpace(v.Output), 64)
			if err != nil {
				return Result{}, fmt.Errorf("modelroute: all_reduce needs numeric member outputs, got %q: %w", v.Output, err)
			}
			w := weightOf(v.Member)
			sum += f * w
			wsum += w
		}
		mean := sum / wsum
		return Result{Reduce: ReduceAllReduce, Output: strconv.FormatFloat(mean, 'g', -1, 64), Members: len(votes)}, nil

	default:
		return Result{}, fmt.Errorf("modelroute: unknown reduction %q", reduce)
	}
}

// weightOf returns a member's effective weight (<= 0 means the default weight 1).
func weightOf(m Member) float64 {
	if m.Weight <= 0 {
		return 1
	}
	return m.Weight
}

// winnerForOutput returns the lexicographically smallest member model that voted
// for the winning output, so a vote winner is deterministic.
func winnerForOutput(votes []Vote, output string) string {
	winner := ""
	for _, v := range votes {
		if v.Output != output {
			continue
		}
		if winner == "" || v.Member.Model < winner {
			winner = v.Member.Model
		}
	}
	return winner
}

// sortedKeys returns the map keys in ascending order (determinism helper).
func sortedKeys(m map[string]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
