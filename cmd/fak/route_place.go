package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// `fak route --place` — the PLACEMENT oracle (epic #5416).
//
// `fak route` answers WHICH MODEL a subject gets. This answers the orthogonal question
// the three-stratum design is actually about: WHICH RUNG does that work run on — the
// engineer's own machine, the company's servers, or a third-party frontier lab — and who
// therefore pays for it.
//
// It exists BEFORE anything on the dispatch path calls Roster.Place, and that ordering is
// deliberate. An operator must be able to see what the ladder would do with THEIR roster,
// on a subject they describe, before fak starts moving traffic on the strength of it. A
// placement policy nobody can inspect is a policy nobody can be held to.
//
//	fak route --accounts FILE --place --labels work_class=routine
//	fak route --accounts FILE --place --labels work_class=routine \
//	          --capability qwen3.6-4b=t2,glm-5.2=t1
//	fak route --accounts FILE --place --aspect scout --json
//
// The candidate pool is every model the roster BINDS, not just the one the manifest
// routed: the question is which rung can serve this class of work at all.

// capabilityDeclarations parses `--capability model=t0|t1|t2[,…]`.
//
// Declaring a capability here is a MEASURED claim, and it is the only way a model becomes
// eligible for a cheap rung — Roster.Place refuses to let an unmeasured capability descend
// the ladder. That is the point of making this an explicit flag rather than a default:
// today `internal/ablate.StubTierScorer` grades everything UNMEASURED, so absent a
// declaration the honest answer is that only the top rung may serve, and the oracle says
// so rather than quietly inventing a number.
func capabilityDeclarations(spec string) (map[string]modelroute.WorkTier, error) {
	out := map[string]modelroute.WorkTier{}
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return out, nil
	}
	for _, pair := range strings.Split(spec, ",") {
		kv := strings.SplitN(strings.TrimSpace(pair), "=", 2)
		if len(kv) != 2 || strings.TrimSpace(kv[0]) == "" {
			return nil, fmt.Errorf("capability %q: want model=t0|t1|t2", strings.TrimSpace(pair))
		}
		tier, ok := modelroute.ParseWorkTier(strings.TrimSpace(kv[1]))
		if !ok {
			return nil, fmt.Errorf("capability %q: unknown tier %q (want t0|t1|t2)",
				strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1]))
		}
		out[strings.TrimSpace(kv[0])] = tier
	}
	return out, nil
}

// placementCandidates builds the candidate pool from the roster's own bindings, in a
// deterministic order.
//
// Compatibility-only and deprecated-alias bindings are excluded: they are legacy
// SPELLINGS of a model that is already in the pool under its real id, and admitting them
// would let the same hardware appear twice on a rung.
//
// A binding to a MANUAL-ONLY account is excluded too, for a different reason: this pool is
// what Place and the escalation walk choose from on their own, and a reserved account is
// by definition one that may only be spent when a caller NAMES it. Filtering here rather
// than inside Place keeps the reservation true for every consumer of the pool at once —
// placement, escalation, and any later automatic picker — since none of them can select a
// candidate that was never offered.
//
// A model with no declared capability enters the pool UNMEASURED at the strictest tier.
// Both halves matter: unmeasured keeps it off every cheap rung, and T0 means that even on
// the top rung it is admitted on the strength of the rung, not of an invented grade.
func placementCandidates(r modelroute.Roster, declared map[string]modelroute.WorkTier) []modelroute.Candidate {
	reserved := make(map[string]bool, len(r.Accounts))
	for _, a := range r.Accounts {
		if a.ManualOnly {
			reserved[a.ID] = true
		}
	}
	seen := map[string]bool{}
	var out []modelroute.Candidate
	for _, b := range r.Bindings {
		if b.CompatibilityOnly || b.DeprecatedAliasFor != "" || b.Model == "" || seen[b.Model] {
			continue
		}
		if reserved[b.Account] {
			continue
		}
		seen[b.Model] = true
		tier, measured := declared[b.Model]
		out = append(out, modelroute.Candidate{Model: b.Model, Capability: tier, Measured: measured})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })
	return out
}

// placementReport is the --json shape: the classification and the placement side by side,
// because an operator reading a surprising rung needs to know whether the cause was the
// ladder or the fact that nobody declared what the work was.
type placementReport struct {
	Classification modelroute.Classification `json:"classification"`
	Placement      modelroute.Placement      `json:"placement"`
	Candidates     []modelroute.Candidate    `json:"candidates"`
	MeasuredCount  int                       `json:"measured_candidates"`
	Grades         []modelroute.Grade        `json:"grades,omitempty"`
	IgnoredModels  []string                  `json:"ignored_evidence_models,omitempty"`
	Journal        *journalSummary           `json:"journal,omitempty"`
	// Spawn is present only when --spawn-type asked the delegated question. A
	// pointer, so an absent key means "not asked" rather than "a spawn that placed
	// nowhere" (route_place_spawn.go).
	Spawn *spawnPlacementReport `json:"spawn,omitempty"`
	// Serving is present only when --serving supplied a liveness snapshot. A pointer
	// for the same reason: an absent key means no snapshot was given, which must not
	// read as a snapshot that observed nothing (route_place_serving.go).
	Serving *servingSummary `json:"serving,omitempty"`
}

// placeOptions are the two ways a caller may say what a model can do, plus the render
// choice. They are one struct so that adding a third source of capability later is a
// change to this type rather than to every call site's argument order.
type placeOptions struct {
	CapSpec      string // --capability: an operator ASSERTING a grade
	EvidencePath string // --evidence: a summary of outcomes the grader turns into one
	OutcomesPath string // --outcomes: the raw turn journal, counted here
	Since        string // --since: the evidence window ("30d", "720h")
	FloorSpec    string // --grade-floor: the operator's evidentiary bar
	SpawnType    string // --spawn-type: also place a sub-agent of this DECLARED type
	ServingPath  string // --serving: an operator's liveness snapshot of what is answering
	JSON         bool
}

// runRoutePlace walks the zone ladder for one subject against one roster and reports it.
// Exit codes match the rest of `fak route`: 0 ok, 1 a placement/roster failure, 2 usage.
func runRoutePlace(stdout, stderr io.Writer, roster *modelroute.Roster, subj modelroute.Subject, opts placeOptions) int {
	if roster == nil {
		fmt.Fprintln(stderr, "fak route: --place needs --accounts FILE — a rung is a property of the roster (which accounts you have and what kind each is), not of the routing manifest")
		return 2
	}
	declared, err := capabilityDeclarations(opts.CapSpec)
	if err != nil {
		fmt.Fprintln(stderr, "fak route:", err)
		return 2
	}
	evidence, floor, journal, err := placementGradeInputs(opts)
	if err != nil {
		fmt.Fprintln(stderr, "fak route:", err)
		return 2
	}
	// The liveness snapshot is loaded BEFORE anything is placed or printed, so a
	// malformed one refuses outright rather than half-reporting a ladder walk the
	// operator would reasonably read as gated.
	var serving modelroute.ServingReport
	servingPath := strings.TrimSpace(opts.ServingPath)
	if servingPath != "" {
		if serving, err = loadServingReport(servingPath); err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 2
		}
	}
	// One model, two sources of truth about what it can do, is a configuration error.
	// Preferring either one silently would leave the operator reading a grade whose
	// origin they cannot tell — so name the models and refuse.
	if clash := conflictingDeclarations(declared, evidence); len(clash) > 0 {
		fmt.Fprintf(stderr, "fak route: --capability and the graded evidence both claim a capability for %s; "+
			"a measurement and an assertion cannot both be the grade — drop it from one of them\n",
			strings.Join(clash, ", "))
		return 2
	}
	candidates := placementCandidates(*roster, declared)
	if len(candidates) == 0 {
		fmt.Fprintln(stderr, "fak route: --place: the roster binds no models, so there is nothing to place; add a binding or use --accounts-check to inspect it")
		return 1
	}
	var (
		grades  []modelroute.Grade
		ignored []string
	)
	if evidence != nil {
		candidates, grades, ignored = gradedPool(candidates, evidence, floor)
	}

	p, cls, err := roster.PlaceSubjectWithServing(subj, candidates, serving)
	if err != nil {
		fmt.Fprintln(stderr, "fak route:", err)
		return 1
	}

	// The delegated question, asked against the SAME pool the parent walked AND the
	// same snapshot: a child that could only descend because the oracle handed it
	// different candidates — or a rung the parent's own walk was told is dead — would
	// be reporting a rung the fleet cannot actually serve it on.
	var spawn *spawnPlacementReport
	if strings.TrimSpace(opts.SpawnType) != "" {
		rep, err := spawnPlacementFor(*roster, p, opts.SpawnType, candidates, serving)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 1
		}
		spawn = &rep
	}

	// The join between the snapshot and the pool the ladder actually walked — the only
	// place a probe filed under a model nothing binds can be noticed (see
	// route_place_serving.go).
	var servingRep *servingSummary
	if servingPath != "" {
		s := summarizeServing(servingPath, serving, candidates, *roster)
		servingRep = &s
	}

	measured := 0
	for _, c := range candidates {
		if c.Measured {
			measured++
		}
	}
	if opts.JSON {
		b, _ := json.MarshalIndent(placementReport{
			Classification: cls, Placement: p, Candidates: candidates, MeasuredCount: measured,
			Grades: grades, IgnoredModels: ignored, Journal: journal, Spawn: spawn,
			Serving: servingRep,
		}, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	printPlacement(stdout, p, cls, candidates, measured)
	if spawn != nil {
		printSpawnPlacement(stdout, *spawn)
	}
	if servingRep != nil {
		printServingSnapshot(stdout, *servingRep)
	}
	if grades != nil {
		printGrades(stdout, grades, floor, ignored)
	}
	if journal != nil {
		printJournalSummary(stdout, *journal)
	}
	return 0
}

// printPlacement renders a placement for a human, closed reason tokens and all.
func printPlacement(w io.Writer, p modelroute.Placement, cls modelroute.Classification, candidates []modelroute.Candidate, measured int) {
	declaredNote := "UNDECLARED"
	if cls.Declared {
		declaredNote = "declared"
	}
	class := string(cls.Class)
	if class == "" {
		class = "(none)"
	}
	fmt.Fprintf(w, "PLACEMENT  (fak route --place)\n")
	fmt.Fprintf(w, "  work class   %s  [%s]  %s\n", class, declaredNote, strings.Join(cls.Reasons, " "))
	fmt.Fprintf(w, "  placed       zone=%s  model=%s\n", p.Zone, p.Model)
	fmt.Fprintf(w, "  target       kind=%s account=%s upstream=%s\n", p.Target.Kind, p.Target.Account, p.Target.UpstreamModel)
	fmt.Fprintln(w, placementDispositionLine(p))
	fmt.Fprintf(w, "  tier floor   required=%s optimal=%s  chosen-capability=%s\n",
		p.Choice.RequiredTier, p.Choice.OptimalTier, capabilityCell(p))
	fmt.Fprintln(w, "  ladder")
	for _, v := range p.Ladder {
		model := v.Model
		if model == "" {
			model = "-"
		}
		fmt.Fprintf(w, "    %-7s %-24s %s\n", v.Zone, model, strings.Join(tallyReasons(v.Reasons), " "))
	}
	fmt.Fprintf(w, "  candidates   %d bound model(s), %d with a MEASURED capability\n", len(candidates), measured)
	if measured == 0 {
		fmt.Fprintln(w, "  note         no candidate carries a measured capability, so no cheap rung was eligible:")
		fmt.Fprintln(w, "               an asserted grade may not win a device or fleet rung. Give the ladder a")
		fmt.Fprintln(w, "               measurement to descend on: --capability model=t0|t1|t2, or --evidence FILE")
		fmt.Fprintln(w, "               to grade from observed outcomes (see the per-model reasons below).")
	}
	if !cls.Declared {
		fmt.Fprintln(w, "  note         nothing declared what this work IS, so it took the strictest floor. The")
		fmt.Fprintln(w, "               fix is a --labels work_class=… declaration, not more hardware.")
	}
}

// placementDispositionLine renders the three facts about HOW a placement came out, on one
// line and never fewer than three.
//
// failed-over is carried by Placement separately from escalated because they answer
// different questions and want opposite fixes: escalated says a cheaper rung could not do
// this work (a capability fact, stable until the roster changes), while failed-over says a
// rung that COULD have done it was routed around because something was not answering (an
// operations fact that may already be untrue). Printing only escalated attributes vendor
// spend to work that "needed" a vendor, and sends an operator hunting a capability problem
// that does not exist while the dead host stays dead. It is worth printing even when the
// placement stayed cheap: a failover WITHIN a rung kept the money on the fleet and still
// means something is down.
func placementDispositionLine(p modelroute.Placement) string {
	return fmt.Sprintf("  self-hosted  %-3s      escalated  %-3s      failed-over  %s",
		yesNo(p.SelfHosted()), yesNo(p.Escalated), yesNo(p.FailedOver))
}

// capabilityCell renders the winning candidate's capability, and refuses to dress the
// unmeasured default up as a grade.
//
// An unmeasured candidate reaches TierPolicy.Admit carrying the zero-value tier, which is
// the MOST demanding one, so the naive render prints "chosen-capability=T0 (OVER-TIER)"
// about a model nobody has graded — asserting both a capability and a waste finding on no
// evidence, which is exactly the failure this epic exists to stop fak from making.
// Placement.Measured is what separates the two cases, and this is the only place in the
// surface that has to get the distinction right.
func capabilityCell(p modelroute.Placement) string {
	if !p.Measured {
		return "UNMEASURED  (admitted only as the top-rung fallback; no grade was claimed)"
	}
	if p.Choice.OverTier {
		return p.Choice.Capability.String() + "  (OVER-TIER: more model than this class needs)"
	}
	return p.Choice.Capability.String()
}

// tallyReasons collapses a repeated reason token into "token x N".
//
// A rung records one reason per candidate it turned away, so a roster with several ungraded
// models bound on the device rung emits the same token several times over. That
// multiplicity is real information — it says how many candidates the rung actually had —
// so it is counted rather than deduplicated away. Order follows first appearance, which
// keeps the line deterministic for the same reason the reason vocabulary is closed.
func tallyReasons(reasons []string) []string {
	var order []string
	count := map[string]int{}
	for _, r := range reasons {
		if count[r] == 0 {
			order = append(order, r)
		}
		count[r]++
	}
	out := make([]string, 0, len(order))
	for _, r := range order {
		if count[r] > 1 {
			out = append(out, fmt.Sprintf("%s x%d", r, count[r]))
			continue
		}
		out = append(out, r)
	}
	return out
}
