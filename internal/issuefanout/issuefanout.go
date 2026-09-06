// Package issuefanout expands one shipped (or planned) working spine into the
// follow-on backlog the spine-first default demands: contract-ready QA,
// dogfooding, productization, observability, integration, docs, and release
// candidates. Each candidate carries the full issuecontract scope contract, so
// the fan-out is dispatchable the moment it is filed — not a pile of one-line
// stubs a triage pass has to rescue later.
//
// The planner is pure: stdlib + issuecontract only — no gh, no disk, no clock.
// Filing stays with the caller (gh, a cohort wave, a live producer); this leaf
// only decides WHAT the 3..50+ follow-ons are and proves each one carries a
// complete contract. It refuses to plan without a spine witness: the minimal
// working spine ships first, or the spine itself becomes the issue.
package issuefanout

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// Schema identifies the machine-readable fan-out plan.
const Schema = "fak.issue-fanout-plan.v1"

// MinFanout is the floor of the fan-out default: a shipped spine with fewer
// than this many follow-ons has not been fanned out, it has been abandoned.
const MinFanout = 3

// Input describes the working spine the fan-out hardens.
type Input struct {
	Title              string   `json:"title"`                // human name of the spine, e.g. "issue fanout planner"
	Leaf               string   `json:"leaf"`                 // owning leaf/lane, e.g. "issuefanout"
	SpineRef           string   `json:"spine_ref"`            // spine witness: commit SHA, demo command, or doc path
	ParentRef          string   `json:"parent_ref,omitempty"` // epic/issue the fan-out hangs off (defaults to SpineRef)
	Paths              []string `json:"paths,omitempty"`      // file trees the follow-ons work in (default internal/<leaf>/)
	Areas              []string `json:"areas,omitempty"`      // area filter (empty = full taxonomy)
	Max                int      `json:"max,omitempty"`
	ParentIssue        int      `json:"parent_issue,omitempty"`
	ParentBaseline     float64  `json:"parent_baseline_points,omitempty"`
	CompletionStandard string   `json:"completion_standard,omitempty"`
	TargetEnvelope     string   `json:"target_envelope,omitempty"`
	WitnessedEnvelope  string   `json:"witnessed_envelope,omitempty"`
	// Max caps candidates (0 = full taxonomy; floor MinFanout).
}

// Plan is the fan-out: contract-ready candidates in fixed taxonomy order.
type Plan struct {
	Schema     string                  `json:"schema"`
	Input      Input                   `json:"input"`
	AreaCounts map[string]int          `json:"area_counts"`
	Candidates []issuepolicy.Candidate `json:"candidates"`
}

// template is one follow-on shape. Placeholders {title} {leaf} {spine} {paths}
// are substituted from Input; every scope field the contract requires is here
// or synthesized identically for all templates in expand.
type template struct {
	area         string
	slug         string
	title        string
	spine        string
	inScope      string
	outOfScope   string
	done         string
	witness      string
	gate         string
	confusion    string
	steps        int
	generation   string
	priority     string
	lane         string
	paths        []string
	coordination string
}

// taxonomy is the fixed follow-on order: QA and dogfooding first (gen/now — the
// spine is only as true as its witnesses), then productization and wiring.
var taxonomy = []template{
	{
		area: "qa", slug: "qa-edge-sweep",
		title:      "qa: adversarial + edge-case sweep for {title}",
		spine:      "Every documented input class of {title} keeps working under hostile and edge input; the test table is the map of what is proven.",
		inScope:    "Table-driven tests over {paths} covering empty, oversized, malformed, and hostile inputs and every error path.",
		outOfScope: "New features; refactors beyond what a failing test forces.",
		done:       "Edge/adversarial table tests land in {paths} and fail before / pass after any fix they force.",
		witness:    "go test {go_package} -run 'Edge|Adversarial' -v output captured in the commit or issue.",
		gate:       "go test {go_package} -run 'Edge|Adversarial' -v and make test-fast green.",
		confusion:  "A green test that never drives the real seam is not a witness.",
		steps:      3, generation: "gen/now", priority: "priority/P1",
	},
	{
		area: "qa", slug: "qa-failure-paths",
		title:      "qa: failure-path + refusal coverage for {title}",
		spine:      "Every refusal or error {title} can return names the fix in its message, and a test asserts each one.",
		inScope:    "Tests asserting each error return of {paths}; message-quality fixes those tests force.",
		outOfScope: "Happy-path behavior changes.",
		done:       "Each error return has a test asserting both the refusal and that its message names the recovery.",
		witness:    "go test {go_package} -run 'Refus|Error|Requir' -v output captured in the commit or issue.",
		gate:       "go test {go_package} -run 'Refus|Error|Requir' -v and make test-fast green.",
		confusion:  "Do not weaken a refusal to make a test pass; the refusal is the contract.",
		steps:      3, generation: "gen/now", priority: "priority/P2",
	},
	{
		area: "qa", slug: "qa-determinism",
		title:      "qa: determinism + race witness for {title}",
		spine:      "The same input yields byte-identical output every run, and the package is race-clean, so cohort/replay layers can trust it.",
		inScope:    "A determinism test (two runs, deep-equal) and a -race run over {paths}.",
		outOfScope: "Performance tuning.",
		done:       "Determinism test lands; -race run over the package is clean.",
		witness:    "go test -race {go_package} -run Determinism -v output captured in the commit or issue.",
		gate:       "go test -race {go_package} -run Determinism -v and make test-race green.",
		confusion:  "Map iteration order and clock/rand reads are the usual determinism leaks.",
		steps:      2, generation: "gen/now", priority: "priority/P2",
	},
	{
		area: "dogfood", slug: "dogfood-self-run",
		title:      "dogfood: run {title} on this repo's own live work",
		spine:      "{title} is used for the real work of the repo that ships it, and the readout is committed evidence, not a claim.",
		inScope:    "One captured run of the spine ({spine}) against the repo's real data; defects found get filed; readout committed under docs/notes/ or a ledger.",
		outOfScope: "Fixing every defect the run surfaces (file them).",
		done:       "A committed readout of a real run exists and every defect it surfaced is filed with a marker key.",
		witness:    "A captured readout verified by `git show`, plus the filed issue numbers.",
		gate:       "Readout committed on trunk; issues visible on the tracker.",
		confusion:  "A synthetic fixture run is not dogfood; use the repo's live backlog/data.",
		steps:      3, generation: "gen/now", priority: "priority/P1",
		lane:  "docs",
		paths: []string{"docs/notes/"},
	},
	{
		area: "dogfood", slug: "dogfood-usage-ledger",
		title:      "dogfood: usage ledger so {title} adoption is measured, not claimed",
		spine:      "Every invocation of {title} appends a durable row, so adoption/neglect shows up in a fold instead of an anecdote.",
		inScope:    "A minimal JSONL ledger row (or an existing ledger/conceptusage fold) recording invocation + outcome for the verb.",
		outOfScope: "Dashboards; provider-price math.",
		done:       "Invocations append a row; a fold surfaces counts per week.",
		witness:    "The ledger file with real rows plus the fold output, captured in the issue.",
		gate:       "go vet + make test-fast green with the ledger wired.",
		confusion:  "Do not log private paths or hostnames into a committed ledger (PUBLIC_LEAK).",
		steps:      4, generation: "gen/next", priority: "priority/P2",
	},
	{
		area: "product", slug: "product-cli-reference",
		title:      "product: docs/cli-reference.md entry + usage parity for {title}",
		spine:      "A user who finds the verb in the CLI reference and a user who runs --help read the same truth.",
		inScope:    "The cli-reference section for the verb; usage-text drift fixes.",
		outOfScope: "New flags.",
		done:       "docs/cli-reference.md documents the verb and matches the live usage text.",
		witness:    "The committed docs diff plus a captured --help readout.",
		gate:       "Docs lint/claims-lint green.",
		confusion:  "Document what ships, not what is planned — unshipped flags stay out.",
		steps:      2, generation: "gen/next", priority: "priority/P2",
		lane:  "docs",
		paths: []string{"docs/cli-reference.md"},
	},
	{
		area: "product", slug: "product-lcd-demo",
		title:      "product: LCD demo/example for {title} meeting the run-the-demos bar",
		spine:      "One deterministic command with no key, no network, no GPU shows {title} working end to end for a stranger.",
		inScope:    "A cmd/*demo or examples/ entry with a -selfcheck invariant, registered per docs/run-the-demos.md.",
		outOfScope: "Hosted/heavy tracks.",
		done:       "The demo runs deterministically from a fresh clone and its -selfcheck passes.",
		witness:    "Captured demo output + selfcheck exit 0 in the commit or issue.",
		gate:       "make demo-audit (demo registry) green.",
		confusion:  "The LCD bar is strict: one command, deterministic, zero external dependencies.",
		steps:      5, generation: "gen/next", priority: "priority/P2",
		lane:  "cmd",
		paths: []string{"cmd/*demo", "examples/"},
	},
	{
		area: "product", slug: "product-error-ux",
		title:      "product: refusal/error-message quality pass for {title}",
		spine:      "Every message {title} emits under failure tells the caller the exact next step, in one read.",
		inScope:    "Message wording over {paths}; tests pinning the improved messages.",
		outOfScope: "Behavior changes.",
		done:       "Each failure message names the fix; tests pin the wording.",
		witness:    "Captured before/after messages in the issue plus the pinned tests.",
		gate:       "make test-fast green.",
		confusion:  "Do not bury the fix in prose — lead with the action.",
		steps:      3, generation: "gen/next", priority: "priority/P2",
	},
	{
		area: "observability", slug: "obs-outcome-counters",
		title:      "observability: outcome counters for {title}",
		spine:      "Success/refusal/error counts for {title} are visible in an existing metrics or report surface, so regressions surface without a bug report.",
		inScope:    "Counters (or ledger fold) for invocation outcomes, joined into an existing surface.",
		outOfScope: "New dashboards.",
		done:       "Outcome counts are queryable from an existing surface and covered by a test.",
		witness:    "A captured readout showing real counts.",
		gate:       "make test-fast green.",
		confusion:  "Reuse an existing surface; a new bespoke metrics file is scope creep.",
		steps:      4, generation: "gen/next", priority: "priority/P2",
	},
	{
		area: "observability", slug: "obs-scorecard",
		title:      "observability: scorecard fold for {title} adoption/health",
		spine:      "A deterministic scorecard grades {title} health (usage, failure rate, drift) so 'is it still good' is a command, not an audit.",
		inScope:    "A fold in an existing scorecard (or a small Go scorecard leaf) grading the verb from its ledger/witnesses.",
		outOfScope: "New Python tools (pythongate ratchet).",
		done:       "The scorecard emits a grade with named evidence and runs in CI or a make target.",
		witness:    "Captured scorecard output on real data.",
		gate:       "go vet + make test-fast green.",
		confusion:  "Score from witnesses (ledgers, git), never from self-report.",
		steps:      4, generation: "gen/next", priority: "priority/P2",
	},
	{
		area: "integration", slug: "int-guard-gate",
		title:      "integration: advisory commit-gate nudge for {title}",
		spine:      "The default is enforced below the agent layer: a warn-mode pre-commit gate (PRIOR_ART pattern) nudges when work lands without what {title} provides.",
		inScope:    "An internal/hooks gate with DefaultMode warn, a ModeEnv/EscapeEnv, and a parity test.",
		outOfScope: "Block-mode default (advisory first; tighten later with evidence).",
		done:       "The gate warns on a synthetic violating commit and stays silent on a clean one, proven by tests.",
		witness:    "go test ./internal/hooks -run <Gate> -v output captured in the commit or issue.",
		gate:       "make test-fast green; gate registered in PreCommitGates().",
		confusion:  "Fail open (ErrCouldNotRun) when evidence is unreachable — a broken gate must not wedge the trunk.",
		steps:      5, generation: "gen/next", priority: "priority/P2",
		lane:  "hooks",
		paths: []string{"internal/hooks/"},
	},
	{
		area: "integration", slug: "int-dos-wiring",
		title:      "integration: dos.toml lane + reason wiring for {title}",
		spine:      "The leaf has its own dispatch lane and any refusal it emits resolves via `dos man wedge <TOKEN> --explain`, so fleets can route and recover mechanically.",
		inScope:    "The leaf's lane/tree rows in dos.toml; a [reasons.*] token (refusal=false if advisory) with summary/fix/see_also.",
		outOfScope: "New enforcement paths.",
		done:       "`dos man wedge <TOKEN> --explain` resolves the token; the lane appears in the lane roster.",
		witness:    "Captured `dos man wedge <TOKEN> --explain` output plus the dos.toml diff.",
		gate:       "dos doctor / config parse green.",
		confusion:  "dos.toml is peer-hot: commit only your own hunks, by explicit path, after a fresh diff.",
		steps:      3, generation: "gen/next", priority: "priority/P2",
		lane:  "dos",
		paths: []string{"./dos.toml"},
	},
	{
		area: "integration", slug: "int-superloop",
		title:      "integration: super-loop/dispatch default hookup for {title}",
		spine:      "The loop that should use {title} by default actually calls it — the default lives in the pipeline, not in an agent's memory.",
		inScope:    "The one call site in the super-loop/dispatch path (skill step or Go tick) that invokes the verb at the right moment.",
		outOfScope: "Loop redesign.",
		done:       "A loop turn demonstrably invokes the verb (transcript or ledger row) without being asked.",
		witness:    "A captured loop-turn artifact showing the automatic invocation.",
		gate:       "Existing loop tests/skill lint green.",
		confusion:  "Wire the default at ONE seam; two call sites double-fire on every turn.",
		steps:      5, generation: "gen/next", priority: "priority/P2",
		lane:  "superloop",
		paths: []string{"internal/superloop/"},
	},
	{
		area: "docs", slug: "docs-doctrine-linkage",
		title:      "docs: doctrine + INDEX/llms.txt linkage for {title}",
		spine:      "An agent or human who greps the doc map finds {title}, its doctrine, and its verb in one hop.",
		inScope:    "The doctrine/README section; docs/INDEX.md and llms.txt lines; AGENTS.md pointer if the default is load-bearing.",
		outOfScope: "Marketing copy.",
		done:       "INDEX/llms.txt reference the doc; links resolve; doc names the verb and the floor.",
		witness:    "The committed docs diff verified by `git show`.",
		gate:       "Docs placement gates (DOC_PLACEMENT/INDEX_SYNC) green.",
		confusion:  "Dated notes go under docs/notes/ WITH an INDEX.md line same-commit.",
		steps:      2, generation: "gen/next", priority: "priority/P2",
		lane:  "docs",
		paths: []string{"docs/INDEX.md", "llms.txt", "AGENTS.md", "README.md"},
	},
	{
		area: "release", slug: "release-claims",
		title:      "release: CLAIMS.md tag + release-note line for {title}",
		spine:      "The honesty ledger says exactly what is real: the {title} claim line carries [SHIPPED] with its witness, and the next release note names it.",
		inScope:    "The CLAIMS.md line with the right tag; the release-note bullet.",
		outOfScope: "Cutting the release itself.",
		done:       "claims-lint green with the new tagged line; note drafted under docs/releases/ conventions.",
		witness:    "The committed CLAIMS.md diff plus captured claims-lint output.",
		gate:       "make claims-lint green.",
		confusion:  "Do not upgrade [SIMULATED]/[STUB] to [SHIPPED] without the witness.",
		steps:      2, generation: "gen/next", priority: "priority/P2",
		lane:  "release",
		paths: []string{"CLAIMS.md", "docs/releases/"},
	},
}

// AreaNames returns the distinct taxonomy areas in plan order.
func AreaNames() []string {
	var names []string
	seen := map[string]bool{}
	for _, t := range taxonomy {
		if !seen[t.area] {
			seen[t.area] = true
			names = append(names, t.area)
		}
	}
	return names
}

// Build expands the taxonomy for one spine into contract-ready candidates.
// It refuses inputs that would break the default: no spine witness, or a cap
// below the fan-out floor.
func Build(in Input) (Plan, error) {
	if strings.TrimSpace(in.Title) == "" || strings.TrimSpace(in.Leaf) == "" {
		return Plan{}, refusef("issuefanout: title and leaf are required — set title to the spine's human name and leaf to its owning lane, then re-run")
	}
	if strings.TrimSpace(in.SpineRef) == "" {
		return Plan{}, refusef("issuefanout: spine_ref is required — ship the minimal working spine first (or file the spine issue itself), then fan out from its witness")
	}
	if in.Max != 0 && in.Max < MinFanout {
		return Plan{}, refusef("issuefanout: max %d is below the fan-out floor %d — raise max to at least %d, or leave it 0 for the full taxonomy", in.Max, MinFanout, MinFanout)
	}
	allowed := map[string]bool{}
	known := map[string]bool{}
	for _, name := range AreaNames() {
		known[name] = true
	}
	for _, a := range in.Areas {
		a = strings.ToLower(strings.TrimSpace(a))
		if a == "" {
			continue
		}
		if !known[a] {
			return Plan{}, refusef("issuefanout: unknown area %q (known: %s) — set the area filter to one of those, or drop it for the full taxonomy", a, strings.Join(AreaNames(), ", "))
		}
		allowed[a] = true
	}
	goPackage, err := goPackageForInput(in)
	if err != nil {
		return Plan{}, err
	}

	plan := Plan{Schema: Schema, Input: in, AreaCounts: map[string]int{}}
	for _, t := range taxonomy {
		if len(allowed) != 0 && !allowed[t.area] {
			continue
		}
		if in.Max != 0 && len(plan.Candidates) >= in.Max {
			break
		}
		if in.ParentIssue > 0 && in.ParentBaseline > 0 && float64(t.steps) > in.ParentBaseline {
			return Plan{}, refusef("issuefanout: candidate %s contributes %d points, exceeding the declared parent baseline %g — raise --parent-baseline-points to the parent's audited production scope or fan out from a parent large enough to contain this child", t.slug, t.steps, in.ParentBaseline)
		}
		plan.Candidates = append(plan.Candidates, expand(t, in, goPackage))
		plan.AreaCounts[t.area]++
	}
	if len(plan.Candidates) < MinFanout {
		return Plan{}, refusef("issuefanout: area filter leaves %d candidates, below the fan-out floor %d — widen the area filter (or drop it) to plan at least %d follow-ons", len(plan.Candidates), MinFanout, MinFanout)
	}
	// The leaf's one promise is that every follow-on is dispatchable the moment
	// it is filed. A hostile or malformed input (a leaf outside the marker-key
	// alphabet, a title/spine tripping the private-boundary screen) would
	// otherwise flow silently into candidates the issue contract refuses after
	// filing — so Build reviews its own output and refuses the input instead.
	reviewOptions := issuepolicy.Options{
		StrictModelTier:   true,
		StrictScale:       true,
		StrictWitness:     true,
		StrictBornRouted:  true,
		StrictProjectWork: in.ParentIssue > 0 && in.ParentBaseline > 0,
	}
	for _, c := range plan.Candidates {
		if r := issuepolicy.ReviewCandidate(c, reviewOptions); r.Dispatchability != issuepolicy.Dispatchable {
			detail := strings.Join(r.Reasons, ", ")
			if len(r.MissingFields) > 0 {
				detail += "; missing/invalid: " + strings.Join(r.MissingFields, ", ")
			}
			return Plan{}, refusef("issuefanout: candidate %s fails the issue contract (%s) — fix the input field it names (the leaf must fit the marker-key alphabet; title/spine must not carry private-boundary text)", c.Key, detail)
		}
	}
	return plan, nil
}

// Refusal is a planner CONTRACT refusal: an input that violated the spine-fanout
// default (no spine witness, a cap below the floor, an unknown area). It is a
// deliberate, well-formed rejection — distinct from an unexpected internal error
// — so the outcome fold can bucket it as a refusal rather than a failure. It
// formats to the exact message the bare error carried, so existing substring
// witnesses (and callers that only read Error()) keep matching.
type Refusal struct{ msg string }

func (r *Refusal) Error() string { return r.msg }

// ContractRefusedReason is the closed-vocabulary code this leaf's contract refusals carry
// (dos.toml [reasons.ISSUEFANOUT_CONTRACT_REFUSED], documented in AGENTS.md, which names this
// package as the reason's floor).
//
// The vocabulary declared and documented this code, and no code path produced it (#5608): the
// planner refused correctly through the typed *Refusal, but a refusal never named the reason the
// table promised for it, so a consumer routing on reason codes could not attribute an
// issuefanout refusal at all. A declared reason nothing can emit is a documented capability the
// tree does not have — worse than an undeclared one, because the table reads as total.
const ContractRefusedReason = "ISSUEFANOUT_CONTRACT_REFUSED"

// Reason returns the closed-vocabulary refusal code for a contract refusal. It is deliberately
// NOT folded into Error(): the message format is load-bearing for existing substring witnesses
// and for callers that only read Error(), so the code rides alongside the message rather than
// inside it.
func (r *Refusal) Reason() string { return ContractRefusedReason }

// RefusalReason reports the closed-vocabulary code for err when err is (or wraps) a contract
// refusal, and ok=false otherwise. It is the join a consumer needs: ClassifyOutcome says WHICH
// bucket a result fell into, this says which declared reason to route on, without the caller
// reaching for the concrete type.
func RefusalReason(err error) (string, bool) {
	var r *Refusal
	if err != nil && errors.As(err, &r) {
		return r.Reason(), true
	}
	return "", false
}

// refusef builds a Refusal with a formatted message — the one constructor Build
// uses for every contract rejection, so all of Build's refusals classify as
// OutcomeRefused and any other error is a genuine failure.
func refusef(format string, a ...any) error {
	return &Refusal{msg: fmt.Sprintf(format, a...)}
}

// Outcome buckets one planner invocation for the observability fold (#2519). The
// three buckets mirror the ticket's health axes: a plan that built, a contract
// refusal the planner deliberately returned, and an unexpected error from anything
// else in the invocation path.
type Outcome string

const (
	OutcomeSuccess Outcome = "success"
	OutcomeRefused Outcome = "refused"
	OutcomeError   Outcome = "error"
)

// ClassifyOutcome maps a Build (or a caller's invocation) result to its outcome
// bucket: nil is success, a *Refusal is a deliberate contract refusal, and any
// other error is an unexpected failure. Pure and allocation-light.
func ClassifyOutcome(err error) Outcome {
	if err == nil {
		return OutcomeSuccess
	}
	var r *Refusal
	if errors.As(err, &r) {
		return OutcomeRefused
	}
	return OutcomeError
}

// OutcomeCounts folds a stream of planner invocations into per-bucket counts — the
// observability surface #2519 asks for: success/refusal/error counts visible from
// a report/JSON surface, so a regression in the refusal contract shows up in a fold
// instead of a bug report. It is a plain accumulator, like the rest of this leaf's
// pure folds; the caller (a verb, a wave, a test) records each result with Observe.
type OutcomeCounts struct {
	Success int `json:"success"`
	Refused int `json:"refused"`
	Error   int `json:"error"`
}

// Observe classifies one invocation result, increments the matching bucket, and
// returns the bucket it credited.
func (c *OutcomeCounts) Observe(err error) Outcome {
	o := ClassifyOutcome(err)
	switch o {
	case OutcomeSuccess:
		c.Success++
	case OutcomeRefused:
		c.Refused++
	default:
		c.Error++
	}
	return o
}

// Total is the number of invocations folded.
func (c OutcomeCounts) Total() int { return c.Success + c.Refused + c.Error }

// BuildInto runs Build and folds its outcome into counts — the one-call seam a
// verb or wave uses so every planner invocation is counted, not just narrated. A
// nil counts pointer is a no-op fold, so the seam is drop-in on any existing call.
func BuildInto(in Input, counts *OutcomeCounts) (Plan, error) {
	plan, err := Build(in)
	if counts != nil {
		counts.Observe(err)
	}
	return plan, err
}

// RenderOutcomes prints the outcome fold as one report line, mirroring
// Render/RenderScorecard/RenderAdoption so a `fak issue fanout` health readout
// reads identically to the rest of the leaf.
func RenderOutcomes(c OutcomeCounts) string {
	return fmt.Sprintf("issuefanout invocations: %d (%d success, %d refused, %d error)",
		c.Total(), c.Success, c.Refused, c.Error)
}

func fanoutProblemFrame(t template) issuepolicy.ProblemFrame {
	centrality := issuepolicy.CentralityEnabling
	target := "the shipped spine's observable Core outcome"
	if t.area == "release" {
		centrality = issuepolicy.CentralityStewardship
		target = "honest release claims remain bound to witnessed shipped behavior"
	}
	checks := map[string]issuepolicy.ProblemCheck{
		"p1": {ID: "p1", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "the child captures its own bounded follow-on context while reusing the shipped spine", Valid: true},
		"p2": {ID: "p2", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "the child closes a named operating or evidence gap instead of expanding the spine", Valid: true},
		"p3": {ID: "p3", Status: issuepolicy.ProblemCheckPreserved, Evidence: "the child stays independently dispatchable and can be reprioritized without changing sibling scope", Valid: true},
		"p4": {ID: "p4", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "the child carries its witness through the real follow-on planning and dispatch path", Valid: true},
	}
	return issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: centrality, CentralityTarget: target, Checks: checks}
}

func fanoutClass(area string) string {
	switch area {
	case "qa", "observability", "integration":
		return "class:infra"
	case "dogfood", "product":
		return "class:dev"
	case "docs", "release":
		return "class:frontdoor"
	default:
		return ""
	}
}

func fanoutModelTiers(priority string) (required, optimal string) {
	switch strings.ToLower(strings.TrimSpace(priority)) {
	case "priority/p1":
		return "T1", "T0"
	case "priority/p2":
		return "T2", "T1"
	default:
		return "", ""
	}
}

// expand substitutes one template into a fully-scoped candidate.
func expand(t template, in Input, goPackage string) issuepolicy.Candidate {
	paths := t.paths
	if len(paths) == 0 {
		paths = in.Paths
		if len(paths) == 0 {
			if in.Leaf == "cmd" {
				paths = []string{"cmd/"}
			} else {
				paths = []string{"internal/" + in.Leaf + "/"}
			}
		}
	} else {
		paths = append([]string(nil), paths...)
	}
	lane := t.lane
	if lane == "" {
		lane = in.Leaf
	}
	coordination := t.coordination
	if coordination == "" {
		coordination = "Stay inside {paths}; check the lane lease before dispatch."
	}
	parent := strings.TrimSpace(in.ParentRef)
	if parent == "" {
		parent = in.SpineRef
	}
	r := strings.NewReplacer(
		"{title}", in.Title,
		"{leaf}", in.Leaf,
		"{lane}", lane,
		"{spine}", in.SpineRef,
		"{paths}", strings.Join(paths, ", "),
		"{go_package}", goPackage,
	)
	requiredTier, optimalTier := fanoutModelTiers(t.priority)
	c := issuepolicy.Candidate{
		ProblemFrame:  fanoutProblemFrame(t),
		Key:           fanoutMarkerKey(in.Leaf, in.SpineRef, t.slug),
		Title:         r.Replace(t.title),
		Generation:    t.generation,
		ParentRef:     parent,
		CurrentState:  r.Replace("The working spine of {title} exists ({spine}); this follow-on has not started."),
		WhyNow:        "Spine-first default: the spine just shipped, so its follow-on backlog is filed at creation time while context is hot.",
		WorkingSpine:  r.Replace(t.spine),
		WorkUnit:      "leaf",
		ExpectedSteps: t.steps,
		Assumptions:   []string{r.Replace("The spine at {spine} still builds/runs on trunk.")},
		ConfusionRisks: []string{
			r.Replace(t.confusion),
			"Harden what exists; do not grow the spine's scope inside this follow-on.",
		},
		Coordination:   []string{r.Replace(coordination)},
		Trigger:        "one-shot: filed at spine-ship time by fak issue fanout",
		BatchPolicy:    r.Replace("one batch per spine, capped at --max candidates; marker-key (fanout-{leaf}-*) dedupe against existing issues before filing"),
		InScope:        r.Replace(t.inScope),
		OutOfScope:     r.Replace(t.outOfScope),
		DoneCondition:  r.Replace(t.done),
		Witness:        r.Replace(t.witness),
		AcceptanceGate: r.Replace(t.gate),
		Lane:           lane,
		Paths:          paths,
		Labels:         []string{"fanout", t.area, fanoutClass(t.area), t.priority},
		Priority:       t.priority,
		ClosureBinding: "Close via a commit whose body carries `Closes #<n>` and whose diff lands the witness above.",
		// The body fallback stays explicit because not every tracker exposes tier
		// labels. Priority selects the required capability floor and the stronger
		// recommended tier without conflating priority/P* with tier/T* syntax.
		RequiredModelTier: requiredTier,
		OptimalModelTier:  optimalTier,
	}
	if in.ParentIssue > 0 && in.ParentBaseline > 0 {
		points := float64(t.steps)
		standard := strings.TrimSpace(in.CompletionStandard)
		if standard == "" {
			standard = "production"
		}
		c.ParentRef = fmt.Sprintf("#%d", in.ParentIssue)
		c.WorkEstimate = fmt.Sprintf("Estimate: %g points", points)
		c.ScopeContribution = fmt.Sprintf("Contribution: %g/%g points", points, in.ParentBaseline)
		c.CompletionStandard = standard
		if standard == "production" {
			c.TargetEnvelope = strings.TrimSpace(in.TargetEnvelope)
			c.WitnessedEnvelope = strings.TrimSpace(in.WitnessedEnvelope)
		}
	}
	return c
}

func fanoutMarkerKey(leaf, spineRef, slug string) string {
	return fmt.Sprintf("fanout-%s-spine-%s-%s", leaf, spineMarkerID(spineRef), slug)
}

func spineMarkerID(spineRef string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(spineRef)))
	return fmt.Sprintf("%x", sum[:8])
}

// goPackageForInput selects the one Go package the generic QA templates can
// actually execute. Explicit cmd/ or internal/ paths win over the lane name;
// non-Go companions such as docs are ignored. Multiple package roots or an
// explicit path set with no Go package refuse instead of emitting a plausible
// but unrunnable witness.
func goPackageForInput(in Input) (string, error) {
	if len(in.Paths) == 0 {
		if in.Leaf == "cmd" {
			return "./cmd/fak", nil
		}
		return "./internal/" + in.Leaf, nil
	}

	packages := map[string]bool{}
	for _, raw := range in.Paths {
		path := strings.Trim(strings.TrimPrefix(strings.ReplaceAll(strings.TrimSpace(raw), `\`, "/"), "./"), "/")
		parts := strings.Split(path, "/")
		if len(parts) < 2 {
			continue
		}
		if parts[0] == "cmd" || parts[0] == "internal" {
			packages["./"+parts[0]+"/"+parts[1]] = true
			continue
		}
		if parts[0] == "examples" && examplePathCanIdentifyPackage(parts) && runnableExamplePackage(filepath.Join(parts[0], parts[1])) {
			packages["./"+parts[0]+"/"+parts[1]] = true
		}
	}
	if len(packages) == 0 {
		return "", refusef("issuefanout: paths do not identify a Go package runnable under cmd/, internal/, or examples/ — use one representative package path so QA witnesses are runnable")
	}
	if len(packages) > 1 {
		names := make([]string, 0, len(packages))
		for name := range packages {
			names = append(names, name)
		}
		sort.Strings(names)
		return "", refusef("issuefanout: paths identify multiple Go packages (%s) — pick one representative package and keep any non-Go companion paths", strings.Join(names, ", "))
	}
	for name := range packages {
		return name, nil
	}
	panic("unreachable")
}

// Render prints the plan for a human: one line per candidate plus the next step.
func examplePathCanIdentifyPackage(parts []string) bool {
	if len(parts) == 2 {
		return true
	}
	name := parts[2]
	return name == "**" || strings.HasSuffix(name, ".go")
}

func runnableExamplePackage(path string) bool {
	root, ok := workspaceRoot()
	if !ok {
		return false
	}
	entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(path)))
	if err != nil {
		return false
	}
	hasGo, hasTest := false, false
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasSuffix(name, "_test.go") {
			hasTest = true
		} else if strings.HasSuffix(name, ".go") {
			hasGo = true
		}
	}
	return hasGo && hasTest
}

func workspaceRoot() (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

func Render(p Plan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fanout: %d contract-ready follow-ons for %s (spine: %s)\n",
		len(p.Candidates), p.Input.Title, p.Input.SpineRef)
	for _, c := range p.Candidates {
		area := ""
		if len(c.Labels) > 1 {
			area = c.Labels[1]
		}
		fmt.Fprintf(&b, "  [%-13s] %s  (key %s, ~%d steps, %s)\n", area, c.Title, c.Key, c.ExpectedSteps, c.Generation)
	}
	b.WriteString("next: file with gh (milestone + labels at creation), or wave-plan via `fak issue cohort --from-plan`\n")
	return b.String()
}
