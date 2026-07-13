package commitintent

// Quality-case selection maps a commit intent's changed paths (the diff) onto
// the engine surfaces they touch, so a cheap PR check runs only the affected
// cases while an unknown change conservatively expands to full coverage. The
// selection is pure and deterministic — no git, no clock, no I/O — like the
// rest of this package: it turns a path set into a plan a later effectful
// quality run can consume.

import (
	"encoding/json"
	"sort"
	"strings"
)

// SelectionSchema versions the quality-case selection emitted for a changed
// surface set.
const SelectionSchema = "fak.commit-intent.quality-selection.v1"

// Surface is a quality-bearing engine surface a changed path can touch. A diff
// is mapped onto surfaces so the matching quality cases run; a path no rule
// maps is treated as unknown and expands coverage rather than silently passing.
type Surface string

const (
	SurfaceModel        Surface = "model"
	SurfaceTokenizer    Surface = "tokenizer"
	SurfaceEngine       Surface = "engine"
	SurfaceSampler      Surface = "sampler"
	SurfaceCache        Surface = "cache"
	SurfaceReportRubric Surface = "report-rubric"
	SurfaceSentinel     Surface = "sentinel"
)

// Tier is the CI lane a quality case is assigned to.
type Tier string

const (
	TierPR      Tier = "pr"
	TierNightly Tier = "nightly"
	TierRelease Tier = "release"
)

// Provenance records how a quality case is pinned so a failure is replayable
// and a pass is trustworthy: the model/tokenizer/engine under test, the seed or
// deterministic oracle, the code/module revision, and the tolerance/baseline it
// is scored against. Oracle, Revision, and Baseline are mandatory — without
// them a "pass" is not independently verifiable.
type Provenance struct {
	Model     string `json:"model,omitempty"`
	Tokenizer string `json:"tokenizer,omitempty"`
	Engine    string `json:"engine,omitempty"`
	Oracle    string `json:"oracle"`
	Revision  string `json:"revision"`
	Baseline  string `json:"baseline"`
	Tolerance string `json:"tolerance,omitempty"`
}

// QualityCase is a single case the selector picked for a changed surface, with
// the tier it runs in, its runtime/resource cost, the rationale for the choice,
// the changed paths that triggered it, and its provenance.
type QualityCase struct {
	Surface      Surface    `json:"surface"`
	Tier         Tier       `json:"tier"`
	Cost         string     `json:"cost"`
	Rationale    string     `json:"rationale"`
	MatchedPaths []string   `json:"matched_paths,omitempty"`
	Provenance   Provenance `json:"provenance"`
}

// Selection is the deterministic plan of quality cases for one changed path
// set. Expanded is set when an unknown path forced full-surface coverage;
// Inconclusive is set when no path could be classified, so downstream must
// treat the sentinel-only plan as "unproven", never as a silent empty pass.
type Selection struct {
	Schema       string        `json:"schema"`
	Revision     string        `json:"revision"`
	Cases        []QualityCase `json:"cases"`
	UnknownPaths []string      `json:"unknown_paths,omitempty"`
	Expanded     bool          `json:"expanded"`
	Inconclusive bool          `json:"inconclusive"`
}

// SurfaceRule maps changed paths onto one Surface. Match holds lowercase path
// substrings; a changed path whose lowercased form contains any of them selects
// the rule's surface. Provenance carries the case defaults; its Revision is
// filled per selection.
type SurfaceRule struct {
	Surface    Surface
	Match      []string
	Tier       Tier
	Cost       string
	Provenance Provenance
}

// DefaultSurfaceRules is the built-in diff→surface mapping. Matching is coarse
// and explainable — a path substring names the surface — so an operator can
// read why a case was picked. Unmapped paths are handled by expansion, not by
// silently guessing a surface.
func DefaultSurfaceRules() []SurfaceRule {
	return []SurfaceRule{
		{Surface: SurfaceModel, Match: []string{"model", "weights", "checkpoint"}, Tier: TierNightly, Cost: "gpu-min:30",
			Provenance: Provenance{Model: "reference", Tokenizer: "reference", Engine: "fak", Oracle: "seed:0", Baseline: "golden:model", Tolerance: "rel:1e-3"}},
		{Surface: SurfaceTokenizer, Match: []string{"tokeniz"}, Tier: TierPR, Cost: "cpu-fast",
			Provenance: Provenance{Tokenizer: "reference", Oracle: "deterministic:tokenize", Baseline: "golden:tokenizer", Tolerance: "exact"}},
		{Surface: SurfaceEngine, Match: []string{"engine", "backend", "gateway", "serve"}, Tier: TierNightly, Cost: "gpu-min:15",
			Provenance: Provenance{Engine: "fak-gateway", Model: "reference", Oracle: "seed:0", Baseline: "golden:engine", Tolerance: "rel:1e-2"}},
		{Surface: SurfaceSampler, Match: []string{"sampl"}, Tier: TierPR, Cost: "cpu-fast",
			Provenance: Provenance{Engine: "fak-gateway", Oracle: "seed:0", Baseline: "golden:sampler", Tolerance: "exact"}},
		{Surface: SurfaceCache, Match: []string{"cache", "kv"}, Tier: TierPR, Cost: "cpu-fast",
			Provenance: Provenance{Engine: "fak-gateway", Oracle: "deterministic:cache", Baseline: "golden:cache", Tolerance: "exact"}},
		{Surface: SurfaceReportRubric, Match: []string{"report", "rubric", "scorecard"}, Tier: TierPR, Cost: "cpu-fast",
			Provenance: Provenance{Oracle: "deterministic:rubric", Baseline: "golden:report", Tolerance: "exact"}},
	}
}

// sentinelRule is the always-on smoke case appended to every selection so a
// plan is never empty.
func sentinelRule() SurfaceRule {
	return SurfaceRule{Surface: SurfaceSentinel, Tier: TierPR, Cost: "cpu-fast",
		Provenance: Provenance{Oracle: "deterministic:sentinel", Baseline: "golden:sentinel", Tolerance: "exact"}}
}

// SelectCases maps changed onto the quality cases that must run for revision
// rev. It is pure and deterministic: identical inputs yield an identical
// Selection. Passing nil rules uses DefaultSurfaceRules.
//
// A changed path that no rule maps is recorded in UnknownPaths and forces
// Expanded coverage — every surface is selected — because an unclassified
// change could regress anywhere. When no path can be classified at all the
// Selection is Inconclusive and carries only the sentinel case, so an empty
// selection can never be read as "nothing to check".
func SelectCases(changed []string, rev string, rules []SurfaceRule) (Selection, error) {
	rev = strings.TrimSpace(rev)
	if rev == "" {
		return Selection{}, fieldError("revision", ErrMissingField, "code/module revision is required")
	}
	if rules == nil {
		rules = DefaultSurfaceRules()
	}
	if err := validateRules(rules); err != nil {
		return Selection{}, err
	}

	matched := map[Surface][]string{}
	var unknown []string
	seenUnknown := map[string]bool{}
	noteUnknown := func(key string) {
		key = strings.TrimSpace(strings.ReplaceAll(key, "\\", "/"))
		if key != "" && !seenUnknown[key] {
			seenUnknown[key] = true
			unknown = append(unknown, key)
		}
	}

	for _, raw := range changed {
		p, err := NormalizePath(raw)
		if err != nil {
			// An unparseable path cannot be classified: treat it conservatively
			// as unknown so coverage expands instead of dropping the change.
			noteUnknown(raw)
			continue
		}
		low := strings.ToLower(p)
		hit := false
		for _, rule := range rules {
			if ruleMatches(rule, low) {
				matched[rule.Surface] = append(matched[rule.Surface], p)
				hit = true
			}
		}
		if !hit {
			noteUnknown(p)
		}
	}

	sel := Selection{Schema: SelectionSchema, Revision: rev}
	byS := ruleBySurface(rules)

	surfaces := make([]Surface, 0, len(matched))
	for s := range matched {
		surfaces = append(surfaces, s)
	}
	sort.Slice(surfaces, func(i, j int) bool { return surfaces[i] < surfaces[j] })
	for _, s := range surfaces {
		rule := byS[s]
		sel.Cases = append(sel.Cases, caseFromRule(rule, rev, dedupeSorted(matched[s]),
			"changed path(s) map to the "+string(s)+" surface"))
	}

	// Unknown changes expand coverage to every surface a rule can describe.
	if len(unknown) > 0 {
		sel.Expanded = true
		sort.Strings(unknown)
		sel.UnknownPaths = unknown
		for _, rule := range rules {
			if _, ok := matched[rule.Surface]; ok {
				continue
			}
			sel.Cases = append(sel.Cases, caseFromRule(rule, rev, nil,
				"unknown changed path(s) expand coverage to the "+string(rule.Surface)+" surface"))
		}
	}

	// Nothing classified at all: only the sentinel runs, and the plan is flagged
	// inconclusive so a caller never reads it as a proven pass.
	if len(matched) == 0 && len(unknown) == 0 {
		sel.Inconclusive = true
	}

	// Always-on sentinel smoke case, unless the caller's rules already emitted it.
	if !hasSurface(sel.Cases, SurfaceSentinel) {
		sel.Cases = append(sel.Cases, caseFromRule(sentinelRule(), rev, nil, "always-on sentinel smoke"))
	}

	sort.SliceStable(sel.Cases, func(i, j int) bool {
		if sel.Cases[i].Surface != sel.Cases[j].Surface {
			return sel.Cases[i].Surface < sel.Cases[j].Surface
		}
		return sel.Cases[i].Tier < sel.Cases[j].Tier
	})
	return sel, nil
}

// SelectForIntent maps a commit intent's normalized changed paths onto the
// quality cases that must run before the intent drains, using the intent's base
// SHA as the code/module revision.
func SelectForIntent(intent Intent, rules []SurfaceRule) (Selection, error) {
	norm, err := NormalizeIntent(intent)
	if err != nil {
		return Selection{}, err
	}
	return SelectCases(norm.Paths, norm.BaseSHA, rules)
}

// MarshalSelection renders a selection as a stable, newline-terminated JSON
// artifact. The artifact carries only surface labels, changed paths, and
// provenance identifiers — no request content — so it is safe to attach to a
// failure as a replay record.
func MarshalSelection(sel Selection) ([]byte, error) {
	if strings.TrimSpace(sel.Schema) == "" {
		sel.Schema = SelectionSchema
	}
	b, err := json.MarshalIndent(sel, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func caseFromRule(rule SurfaceRule, rev string, paths []string, rationale string) QualityCase {
	prov := rule.Provenance
	prov.Revision = rev
	return QualityCase{
		Surface:      rule.Surface,
		Tier:         rule.Tier,
		Cost:         rule.Cost,
		Rationale:    rationale,
		MatchedPaths: paths,
		Provenance:   prov,
	}
}

func ruleMatches(rule SurfaceRule, lowPath string) bool {
	for _, m := range rule.Match {
		m = strings.ToLower(strings.TrimSpace(m))
		if m != "" && strings.Contains(lowPath, m) {
			return true
		}
	}
	return false
}

func validateRules(rules []SurfaceRule) error {
	if len(rules) == 0 {
		return fieldError("rules", ErrMissingField, "at least one surface rule is required")
	}
	for _, rule := range rules {
		surface := strings.TrimSpace(string(rule.Surface))
		if surface == "" {
			return fieldError("rule.surface", ErrMissingField, "surface is required")
		}
		if err := ValidateTier(rule.Tier); err != nil {
			return err
		}
		if strings.TrimSpace(rule.Cost) == "" {
			return fieldError("rule.cost", ErrMissingField, "runtime/resource cost is required for "+surface)
		}
		if strings.TrimSpace(rule.Provenance.Oracle) == "" {
			return fieldError("rule.provenance.oracle", ErrMissingField, "seed or deterministic oracle is required for "+surface)
		}
		if strings.TrimSpace(rule.Provenance.Baseline) == "" {
			return fieldError("rule.provenance.baseline", ErrMissingField, "tolerance/baseline provenance is required for "+surface)
		}
	}
	return nil
}

// ValidateTier reports whether t is one of the explicit PR, nightly, or release
// tiers a quality case may be assigned to.
func ValidateTier(t Tier) error {
	switch t {
	case TierPR, TierNightly, TierRelease:
		return nil
	default:
		return fieldError("tier", ErrInvalidField, string(t))
	}
}

func ruleBySurface(rules []SurfaceRule) map[Surface]SurfaceRule {
	m := make(map[Surface]SurfaceRule, len(rules))
	for _, rule := range rules {
		if _, ok := m[rule.Surface]; !ok {
			m[rule.Surface] = rule
		}
	}
	return m
}

func hasSurface(cases []QualityCase, s Surface) bool {
	for _, c := range cases {
		if c.Surface == s {
			return true
		}
	}
	return false
}

func dedupeSorted(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}
