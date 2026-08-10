package issuepolicy

import "strings"

// Scale is the closed ORDER-OF-MAGNITUDE work-size ladder. It is the one field
// that unifies the three ways this package already talks about size —
// ExpectedSteps (a 1..8 int), the WorkUnit shape string (leaf vs epic), and the
// leaf/decompose dispatch split — into a single rung an issue can DECLARE and a
// reviewer can check against its witness.
//
// The problem the ladder exists to kill: "done" is scale-blind. DoneCondition /
// Witness are graded only for PRESENCE, so a worker's scale-1 "done" (one commit
// landed) satisfies an issue whose done was written at scale-100 ("the feature
// works end-to-end"). The word "done" survives the scale jump; the evidence does
// not. Making scale a first-class field, and typing the required witness KIND to
// the scale, is what stops a pile of small commits from masquerading as a large
// "done".
//
// Each tier names its own definition of done and its own witness kind:
//
//	S0 step    ~1 unit      one commit's diff did what its subject claims        (commit/diff witness)
//	S1 leaf    ~2-8 steps   the leaf's done_condition true + gate green          (test/gate witness)
//	S2 feature ~9-100       a capability works end-to-end on live input          (integration/live witness)
//	S3 epic    ~101-1000    epicprogress = 100% + a milestone-level probe        (milestone/rollup witness)
//	S4 program >1000        never "done" — a frontier that advanced              (trend/frontier witness)
//
// The top of the ladder aligns with internal/worktype: S4 is exactly the ONGOING
// program class (worktype.Ongoing()), measured by a frontier + trend and never by
// a completion %. Only S0/S1 are dispatchable; S2+ must decompose first, which is
// the same line MaxDispatchExpectedSteps and isNonDispatchWorkUnit already draw,
// now expressed on one axis.
type Scale string

const (
	// ScaleUnknown is the zero value: the unit did not declare a size and none
	// could be derived from its step budget or shape.
	ScaleUnknown Scale = ""
	// ScaleStep is one atomic commit (~1 unit).
	ScaleStep Scale = "S0"
	// ScaleLeaf is a dispatchable leaf (~2-8 steps).
	ScaleLeaf Scale = "S1"
	// ScaleFeature is a capability that works end-to-end (~9-100 units).
	ScaleFeature Scale = "S2"
	// ScaleEpic is a deliverable epic measured by child completion (~101-1000).
	ScaleEpic Scale = "S3"
	// ScaleProgram is an ongoing program that is never "done" (>1000).
	ScaleProgram Scale = "S4"
)

// scaleFlag values — the closed vocabulary of ways a candidate's declared size,
// its derived size, and its witness can disagree. Mirrors the modelTier flag
// discipline: every disagreement is a named token, never a silent pass.
const (
	flagScaleUndeclared       = "scale_undeclared"
	flagScaleDeclaredInvalid  = "scale_declared_invalid"
	flagScaleContradictsSteps = "scale_contradicts_steps"
	flagScaleContradictsShape = "scale_contradicts_shape"
	flagWitnessUnderScale     = "witness_under_scale"
)

// ScaleFit is the parsed work-scale readout for an issue: what size it declared,
// what size its step budget / shape imply, the effective size the two resolve to,
// whether that size is dispatchable, and whether its witness KIND matches the
// size. Flags name every disagreement from the closed vocabulary above. Advisory
// by default; Options.StrictScale turns a flagged issue triage-only, the same way
// StrictModelTier gates a model-tier flag.
type ScaleFit struct {
	Declared     Scale    `json:"declared,omitempty"`
	Derived      Scale    `json:"derived,omitempty"`
	Effective    Scale    `json:"effective,omitempty"`
	Source       string   `json:"source,omitempty"`
	Dispatchable bool     `json:"dispatchable"`
	NeedsWitness string   `json:"needs_witness,omitempty"`
	WitnessScale Scale    `json:"witness_scale,omitempty"`
	Flags        []string `json:"flags,omitempty"`
}

// scaleFit computes the work-scale readout for a candidate. It always produces a
// readout; whether a flag holds dispatch is the caller's StrictScale choice. The
// effective scale prefers an explicit declaration, then the step budget, then the
// work-unit shape — and it flags a declaration that contradicts either derived
// source so a producer cannot label a 50-step change "S1".
func scaleFit(c Candidate) ScaleFit {
	declared, declaredValid := parseScale(c.Scale)
	fromSteps, okSteps := scaleFromSteps(c.ExpectedSteps)
	fromShape, okShape := scaleFromWorkUnit(c.WorkUnit)

	// Derive conservatively: the effective size is at least as large as the
	// largest available signal. The step budget is the base, but a bigger WorkUnit
	// shape RAISES it — a small step budget must not be able to shrink a "feature"
	// or "epic" below its shape and sneak it past the leaf gate.
	derived := ScaleUnknown
	derivedSource := "none"
	if okSteps {
		derived, derivedSource = fromSteps, "steps"
	}
	if okShape && scaleRank(fromShape) > scaleRank(derived) {
		derived, derivedSource = fromShape, "shape"
	}

	var flags []string
	effective := ScaleUnknown
	source := "none"
	switch {
	case declared != ScaleUnknown:
		effective, source = declared, "declared"
	case derived != ScaleUnknown:
		effective, source = derived, derivedSource
	}

	if strings.TrimSpace(c.Scale) != "" && !declaredValid {
		flags = append(flags, flagScaleDeclaredInvalid)
	}
	if effective == ScaleUnknown {
		flags = append(flags, flagScaleUndeclared)
	}
	// A declaration that fights the step budget or the shape is the "labeled a
	// 50-step change S1" bug: honor the declaration as effective, but surface the
	// contradiction so it is groomed, not buried.
	if declared != ScaleUnknown {
		if okSteps && declared != fromSteps {
			flags = append(flags, flagScaleContradictsSteps)
		}
		if okShape && declared != fromShape {
			flags = append(flags, flagScaleContradictsShape)
		}
	}

	witnessScale := witnessScale(c)
	// Under-witness: the work is a feature/epic/program (S2+) but the strongest
	// witness named is only a commit/test (S1 or below), or none at all. This is
	// the exact "claim a big done with small evidence" leak the ladder targets.
	if scaleRank(effective) >= scaleRank(ScaleFeature) && scaleRank(witnessScale) < scaleRank(ScaleFeature) {
		flags = append(flags, flagWitnessUnderScale)
	}

	return ScaleFit{
		Declared:     declared,
		Derived:      derived,
		Effective:    effective,
		Source:       source,
		Dispatchable: effective == ScaleStep || effective == ScaleLeaf,
		NeedsWitness: needsWitnessKind(effective),
		WitnessScale: witnessScale,
		Flags:        compact(flags),
	}
}

// parseScale reads a declared scale, accepting the canonical S0..S4 tokens and
// the tier names (step/leaf/feature/epic/program), case-insensitively and
// tolerant of surrounding punctuation. It returns ok=false for a non-empty value
// that names no tier — the case that makes a stray "medium" a declared-invalid
// flag rather than a silent unknown.
func parseScale(s string) (Scale, bool) {
	s = strings.ToLower(strings.TrimSpace(strings.Trim(s, "`*_ ")))
	if s == "" {
		return ScaleUnknown, false
	}
	// Take the first token so "S2 (feature)" or "leaf — one deliverable" resolve.
	if fields := strings.FieldsFunc(s, func(r rune) bool {
		return r == ' ' || r == '(' || r == ')' || r == ',' || r == ':' || r == '/' || r == '-'
	}); len(fields) > 0 {
		s = fields[0]
	}
	switch s {
	case "s0", "step", "commit", "atom", "atomic":
		return ScaleStep, true
	case "s1", "leaf", "patch", "task", "worker", "worker-ready":
		return ScaleLeaf, true
	case "s2", "feature", "capability", "component":
		return ScaleFeature, true
	case "s3", "epic", "deliverable", "milestone":
		return ScaleEpic, true
	case "s4", "program", "frontier", "umbrella":
		return ScaleProgram, true
	default:
		return ScaleUnknown, false
	}
}

// scaleFromSteps maps a step budget onto the ladder. The bands follow the human
// round numbers the size ladder is about (1 / 100 / 1000): a single step is S0,
// the dispatchable band tops out at MaxDispatchExpectedSteps (S1), and larger
// budgets climb S2..S4. A zero/negative budget carries no information (ok=false).
func scaleFromSteps(steps int) (Scale, bool) {
	switch {
	case steps <= 0:
		return ScaleUnknown, false
	case steps == 1:
		return ScaleStep, true
	case steps <= MaxDispatchExpectedSteps:
		return ScaleLeaf, true
	case steps <= 100:
		return ScaleFeature, true
	case steps <= 1000:
		return ScaleEpic, true
	default:
		return ScaleProgram, true
	}
}

// scaleFromWorkUnit maps the WorkUnit shape string onto the ladder. The class
// markers that name intent rather than size (research/idea/triage/decompose)
// carry no scale and return ok=false, leaving the step budget or an explicit
// declaration to place them.
func scaleFromWorkUnit(unit string) (Scale, bool) {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "step":
		return ScaleStep, true
	case "leaf", "patch", "task", "work-unit", "work_unit", "worker-ready":
		return ScaleLeaf, true
	case "feature", "capability", "component":
		return ScaleFeature, true
	case "epic":
		return ScaleEpic, true
	case "program", "umbrella":
		return ScaleProgram, true
	default:
		return ScaleUnknown, false
	}
}

// witnessScale reads the STRONGEST witness kind named anywhere in the done /
// witness / acceptance / closure text and returns the scale it corresponds to.
// It answers "how big is the evidence this issue promises to produce?", which is
// then compared against how big the work actually is. Unknown when no recognized
// witness kind is named.
func witnessScale(c Candidate) Scale {
	text := strings.ToLower(strings.Join([]string{
		c.DoneCondition, c.Witness, c.AcceptanceGate, c.ClosureBinding,
	}, "\n"))
	best := ScaleUnknown
	raise := func(s Scale) {
		if scaleRank(s) > scaleRank(best) {
			best = s
		}
	}
	if hasAny(text, "commit", "diff", " sha", "trailer", "commit-audit", "lands") {
		raise(ScaleStep)
	}
	if hasAny(text, "go test", "unit test", " test", "gate", "verified_done", "lint", "compiles", "green", "path exists") {
		raise(ScaleLeaf)
	}
	if hasAny(text, "end-to-end", "end to end", "e2e", "integration", "live", "on real", "dogfood", "smoke", "served turn", "served-turn", "probe", "user path", "acceptance test") {
		raise(ScaleFeature)
	}
	if hasAny(text, "milestone", "all children", "every child", "children closed", "epicprogress", "% complete", "percent complete", "roll-up", "rollup") {
		raise(ScaleEpic)
	}
	if hasAny(text, "frontier", "trend", "benchmark", "sota", "regression budget", "moved the frontier", "metric moved") {
		raise(ScaleProgram)
	}
	return best
}

// needsWitnessKind names, in one human phrase, the minimum witness kind a scale
// requires — the "how it fits into the broader picture" hint a producer reads
// when its witness is flagged as under-scale.
func needsWitnessKind(s Scale) string {
	switch s {
	case ScaleStep:
		return "commit/diff witness"
	case ScaleLeaf:
		return "test/gate witness"
	case ScaleFeature:
		return "integration/live witness"
	case ScaleEpic:
		return "milestone/rollup witness"
	case ScaleProgram:
		return "trend/frontier witness"
	default:
		return ""
	}
}

// scaleRank orders the ladder for comparison. Unknown ranks below S0 so any named
// scale outranks it and an unknown witness never satisfies a scale requirement.
func scaleRank(s Scale) int {
	switch s {
	case ScaleStep:
		return 0
	case ScaleLeaf:
		return 1
	case ScaleFeature:
		return 2
	case ScaleEpic:
		return 3
	case ScaleProgram:
		return 4
	default:
		return -1
	}
}
