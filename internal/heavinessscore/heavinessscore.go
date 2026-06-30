// Package heavinessscore is the operator-heaviness / steering-effort stick.
//
// Every sibling scorecard grades the CODE; the one that comes closest to "steering effort",
// internal/.. (the steerability index behind tools/steerability_scorecard.py), measures the
// SHAPE of the Go package graph -- modularity, coupling, navigability -- i.e. how hard the
// CODE is to change as it grows. None of them answers the question an OPERATOR asks: how heavy
// does the repo feel to DRIVE? -- how many verbs must I discover, how many flags does the front
// door carry, how many structured ways can a green change still be refused, and when a guard
// refuses me wrongly is there an in-product way to appeal? This card scores that surface.
//
// It is deliberately ORTHOGONAL to the steerability index: that one reads the package graph,
// this one reads the OPERATOR-FACING SURFACE (the cmd/fak dispatch table, the front-door verb's
// flag set, the dos.toml refusal vocabulary, and whether the doc map makes the steering surfaces
// discoverable). No KPI here re-derives a code-shape number, so the two never double-count.
//
// The headline an operator tracks is NOT a 0-100 grade: it is heaviness_pressure -- a single,
// unbounded integer that sums how far over the comfortable operator-surface lines the repo sits
// (verbs over the soft line + front-door flags over the soft line + refusal reasons over the
// soft line + meta-verb-share over the soft line). Lower is lighter. It rises when a verb, a
// flag, or a refusal reason is added and falls when the surface is consolidated -- so a clean
// pass holds it flat as the repo grows, which is the whole point. heaviness_debt (the HARD gate
// the control pane folds) stays ~0 on a disciplined tree and fires only for the two friction
// defects whose cheapest fix is genuine work: a steering surface the doc map hides
// (discoverability) and a missing in-product appeal channel for a wrong refusal (recovery).
//
// It is a TREE-READING scorecard (no data dir): the dispatch table, the flag set, the dos.toml
// reasons and the doc map ARE the data, read from the source, so it cannot be gamed by editing a
// JSON file -- only by genuinely slimming the surface. The fold/grade/markdown machinery lives in
// pkg/scorecard; this package holds only the operator-surface extraction and the KPIs.
package heavinessscore

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id.
const Schema = "fak-operator-heaviness-scorecard/1"

// DebtKey is the headline HARD integer the control-pane folds (corpus.heaviness_debt).
const DebtKey = "heaviness_debt"

// CleanFloor is the disciplined tree's expected HARD debt; the live-tree smoke pins it.
const CleanFloor = 0

// The operator-surface source files this card reads. The strings these render ARE the data.
const (
	mainGoRel  = "cmd/fak/main.go"  // the top-level dispatch table (the verb surface)
	guardGoRel = "cmd/fak/guard.go" // the front-door verb's flag set
	dosTomlRel = "dos.toml"         // the structured refusal vocabulary
	docMapRel  = "llms.txt"         // the doc map an operator orients from
)

// Soft lines + hard ceilings for the magnitude KPIs. These are the comfortable operator-surface
// thresholds: at/below the soft line the KPI scores 100 and adds zero pressure; above it the
// score falls linearly and the overage feeds heaviness_pressure; only above the hard ceiling does
// a magnitude become HARD debt (a genuine blowout whose fix is a real consolidation program).
const (
	verbSoftLine    = 90  // top-level subcommands an operator can comfortably hold
	verbHardCeiling = 200 // past this the dispatch table is a navigation wall -> HARD

	flagSoftLine    = 20 // flags on the single front-door verb (`fak guard`)
	flagHardCeiling = 80 // past this the front door is its own steerability tax -> HARD

	reasonSoftLine = 12 // structured refusal reasons an operator must understand
	reasonRef      = 30 // the score floor reference (no hard ceiling -- refusals are SOFT)

	metaShareSoftLine = 0.08 // share of verbs that are meta-scorecards/RSI (clutter the surface)
	metaShareRef      = 0.25 // the score floor reference
)

// requiredDocMapSurface is one steering surface the doc map MUST make discoverable. A surface is
// covered when any of its tokens appears in llms.txt (case-insensitive). The tokens are specific
// on purpose: "steerability-scorecard"/"fak steering" must not be satisfied by the Charter's
// incidental "human-steerable" line.
type requiredDocMapSurface struct {
	name   string
	tokens []string
}

var requiredDocMapSurfaces = []requiredDocMapSurface{
	{name: "steerability surface", tokens: []string{"steerability-scorecard", "fak steering", "steerability index"}},
	{name: "operator-heaviness surface", tokens: []string{"operator-heaviness", "operator heaviness", "heaviness_pressure"}},
}

// Surface is the parsed operator-facing surface every KPI reads. Holding it as plain data is what
// lets the KPIs be pure functions tested against fixtures with no tree.
type Surface struct {
	Verbs          []string // distinct top-level dispatch verbs (sorted)
	MetaVerbs      []string // the subset that are meta-scorecards / RSI verbs (sorted)
	FrontDoorFlags int      // flags defined on the front-door verb (`fak guard`)
	RefusalReasons int      // [reasons.*] blocks declared in dos.toml
	AppealWired    bool     // the dispatch table routes the in-product appeal verb (`complain`)
	DocMap         string   // llms.txt, lowercased (the doc-map coverage oracle)
}

var (
	reCaseVerb     = regexp.MustCompile(`case "([a-z][a-z0-9-]*)":`)
	reGuardFlag    = regexp.MustCompile(`fs\.(?:String|Bool|Int|Int64|Uint|Float64|Duration|Var)\(`)
	reReasonsBlock = regexp.MustCompile(`(?m)^\[reasons\.[A-Z0-9_]+\]`)
)

// isMetaVerb reports whether a dispatch verb is meta-tooling (a scorecard / RSI verb) rather than
// a core operator verb -- the surface clutter an operator does not reach for day to day.
func isMetaVerb(v string) bool {
	return strings.Contains(v, "scorecard") ||
		strings.HasSuffix(v, "-rsi") ||
		strings.HasSuffix(v, "-score") ||
		v == "scorecard"
}

// ParseVerbs returns the distinct top-level dispatch verbs (sorted) from cmd/fak/main.go.
func ParseVerbs(mainGo string) []string {
	seen := map[string]bool{}
	for _, m := range reCaseVerb.FindAllStringSubmatch(mainGo, -1) {
		seen[m[1]] = true
	}
	out := make([]string, 0, len(seen))
	for v := range seen {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// ParseSurface reads the four operator-surface source files under root into a Surface.
func ParseSurface(root string) Surface {
	mainGo := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(mainGoRel)))
	guardGo := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(guardGoRel)))
	dosToml := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(dosTomlRel)))
	docMap := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(docMapRel)))

	verbs := ParseVerbs(mainGo)
	var meta []string
	for _, v := range verbs {
		if isMetaVerb(v) {
			meta = append(meta, v)
		}
	}
	return Surface{
		Verbs:          verbs,
		MetaVerbs:      meta,
		FrontDoorFlags: len(reGuardFlag.FindAllString(guardGo, -1)),
		RefusalReasons: len(reReasonsBlock.FindAllString(dosToml, -1)),
		AppealWired:    strings.Contains(mainGo, `case "complain":`),
		DocMap:         strings.ToLower(docMap),
	}
}

// magnitudeScore maps a measured value to 0-100: 100 at/below the soft line, falling linearly to
// 0 at the ceiling/reference. It is the shared shape behind every magnitude KPI.
func magnitudeScore(value, softLine, ceiling float64) float64 {
	if value <= softLine {
		return 100
	}
	if value >= ceiling {
		return 0
	}
	return 100 * (1 - (value-softLine)/(ceiling-softLine))
}

// overage returns max(0, value-softLine) -- the natural-unit pressure a magnitude KPI contributes.
func overage(value, softLine float64) float64 {
	if value <= softLine {
		return 0
	}
	return value - softLine
}

// --- KPIs --------------------------------------------------------------------------------------

// kpiDocMapCoversSteering (HARD, discoverability): the doc map must make every steering surface
// discoverable. A surface the operator cannot find from llms.txt is a friction defect whose fix is
// a real doc-map entry (not prose spam) -- so it earns HARD debt.
func kpiDocMapCoversSteering(s Surface) scorecard.KPI {
	var defects []string
	covered := 0
	for _, req := range requiredDocMapSurfaces {
		if scorecard.HasAny(s.DocMap, req.tokens) {
			covered++
			continue
		}
		defects = append(defects, fmt.Sprintf("the doc map (llms.txt) does not index the %s -- an operator cannot discover it (expected one of: %s)",
			req.name, strings.Join(req.tokens, ", ")))
	}
	score := 100.0
	if len(requiredDocMapSurfaces) > 0 {
		score = 100.0 * float64(covered) / float64(len(requiredDocMapSurfaces))
	}
	return scorecard.KPI{
		Key: "docmap_covers_steering", Group: "discoverability", Score: score,
		Detail:  fmt.Sprintf("%d/%d steering surface(s) discoverable from the doc map", covered, len(requiredDocMapSurfaces)),
		Defects: defects,
	}
}

// kpiAppealChannelWired (HARD, recovery): the dispatch table must route an in-product channel to
// appeal a WRONG refusal (`fak complain`). Without it, an operator a guard refuses by mistake has
// no low-friction recourse -- the heaviest operator friction of all. Passes on a healthy tree, so
// it is a regression guard that fires only if the channel is ever un-wired.
func kpiAppealChannelWired(s Surface) scorecard.KPI {
	if s.AppealWired {
		return scorecard.KPI{
			Key: "appeal_channel_wired", Group: "recovery", Score: 100,
			Detail: "the in-product appeal channel (`fak complain`) is wired into the dispatch table",
		}
	}
	return scorecard.KPI{
		Key: "appeal_channel_wired", Group: "recovery", Score: 0,
		Detail:  "no in-product appeal channel for a wrong refusal",
		Defects: []string{"the dispatch table routes no `complain` verb -- an operator a guard refuses wrongly has no in-product way to appeal"},
	}
}

// kpiCLIVerbCount (SOFT, surface; HARD only past the ceiling): the top-level verb count is the
// first thing an operator must navigate. Over the soft line it adds pressure and a drift signal;
// only a genuine blowout past the hard ceiling -- where the fix is a real `fak <group> <verb>`
// consolidation program -- becomes HARD debt.
func kpiCLIVerbCount(s Surface) scorecard.KPI {
	n := float64(len(s.Verbs))
	k := scorecard.KPI{
		Key: "cli_verb_count", Group: "surface", Score: magnitudeScore(n, verbSoftLine, verbHardCeiling),
		Detail: fmt.Sprintf("%d top-level subcommands (soft %d, hard ceiling %d)", len(s.Verbs), verbSoftLine, verbHardCeiling),
	}
	if len(s.Verbs) > verbHardCeiling {
		k.Defects = []string{fmt.Sprintf("%d top-level subcommands exceed the %d hard ceiling -- the dispatch table is a navigation wall; consolidate under `fak <group> <verb>`", len(s.Verbs), verbHardCeiling)}
	} else if len(s.Verbs) > verbSoftLine {
		k.Soft = []string{fmt.Sprintf("%d top-level subcommands over the soft line of %d -- the surface an operator must discover is large", len(s.Verbs), verbSoftLine)}
	}
	return k
}

// kpiMetaVerbShare (SOFT, surface): the share of the dispatch table that is meta-tooling
// (scorecards / RSI verbs). A high share clutters the surface an operator scans with verbs they
// never reach for; the real fix is to group them (e.g. `fak score <x>`), which is genuine work.
func kpiMetaVerbShare(s Surface) scorecard.KPI {
	var share float64
	if len(s.Verbs) > 0 {
		share = float64(len(s.MetaVerbs)) / float64(len(s.Verbs))
	}
	k := scorecard.KPI{
		Key: "meta_verb_share", Group: "surface", Score: magnitudeScore(share, metaShareSoftLine, metaShareRef),
		Detail: fmt.Sprintf("%d/%d top-level verbs are meta-scorecards/RSI (%.0f%%, soft %.0f%%)", len(s.MetaVerbs), len(s.Verbs), share*100, metaShareSoftLine*100),
	}
	if share > metaShareSoftLine {
		k.Soft = []string{fmt.Sprintf("%.0f%% of the dispatch table is meta-tooling -- group the scorecard/RSI verbs under one parent so the operator surface is core verbs", share*100)}
	}
	return k
}

// kpiFrontDoorFlagBurden (SOFT, config; HARD past the ceiling): the front-door verb (`fak guard`,
// the one command most operators run) carries the whole config burden. A large flag set is a
// steerability tax on the command least able to afford one.
func kpiFrontDoorFlagBurden(s Surface) scorecard.KPI {
	n := float64(s.FrontDoorFlags)
	k := scorecard.KPI{
		Key: "front_door_flag_burden", Group: "config", Score: magnitudeScore(n, flagSoftLine, flagHardCeiling),
		Detail: fmt.Sprintf("%d flags on the front-door verb `fak guard` (soft %d, hard ceiling %d)", s.FrontDoorFlags, flagSoftLine, flagHardCeiling),
	}
	if s.FrontDoorFlags > flagHardCeiling {
		k.Defects = []string{fmt.Sprintf("%d flags on `fak guard` exceed the %d hard ceiling -- the front door needs sensible defaults / a profile, not a flag per knob", s.FrontDoorFlags, flagHardCeiling)}
	} else if s.FrontDoorFlags > flagSoftLine {
		k.Soft = []string{fmt.Sprintf("%d flags on the one command most operators run -- fold the rarely-touched knobs behind a profile/default", s.FrontDoorFlags)}
	}
	return k
}

// kpiRefusalVocabSize (SOFT, ceremony): the number of structured refusal reasons an operator may
// hit. Each is correct by design, but every reason is one more way a green change is refused at the
// seam and one more token the operator must learn. SOFT -- the only cheap "fix" would be deleting a
// real guard, which is the wrong move; the signal is the trend, watched not gamed.
func kpiRefusalVocabSize(s Surface) scorecard.KPI {
	n := float64(s.RefusalReasons)
	k := scorecard.KPI{
		Key: "refusal_vocab_size", Group: "ceremony", Score: magnitudeScore(n, reasonSoftLine, reasonRef),
		Detail: fmt.Sprintf("%d structured refusal reasons in the vocabulary (soft %d)", s.RefusalReasons, reasonSoftLine),
	}
	if s.RefusalReasons > reasonSoftLine {
		k.Soft = []string{fmt.Sprintf("%d structured refusal reasons -- each is a correct guard, but the trend is the operator's cognitive load; keep new reasons earning their place", s.RefusalReasons)}
	}
	return k
}

// Pressure is the continual, unbounded heaviness number (NOT a 0-100 grade): the summed overage of
// the magnitude KPIs in natural units. It is the headline an operator tracks over time -- lower is
// lighter -- and the number a consolidation program drives down.
func Pressure(s Surface) int {
	var share float64
	if len(s.Verbs) > 0 {
		share = float64(len(s.MetaVerbs)) / float64(len(s.Verbs))
	}
	p := overage(float64(len(s.Verbs)), verbSoftLine) +
		overage(float64(s.FrontDoorFlags), flagSoftLine) +
		overage(float64(s.RefusalReasons), reasonSoftLine) +
		// meta-share is a fraction; price each whole percent over the line as one pressure unit.
		overage(share*100, metaShareSoftLine*100)
	return int(p + 0.5)
}

// Build reads the operator surface, runs the KPIs, and folds them into the control-pane payload.
func Build(root string) scorecard.Payload {
	s := ParseSurface(root)
	kpis := []scorecard.KPI{
		kpiDocMapCoversSteering(s),
		kpiAppealChannelWired(s),
		kpiCLIVerbCount(s),
		kpiMetaVerbShare(s),
		kpiFrontDoorFlagBurden(s),
		kpiRefusalVocabSize(s),
	}
	debt := 0
	for _, k := range kpis {
		debt += len(k.Defects)
	}
	pressure := Pressure(s)

	finding := fmt.Sprintf("operator surface clean of hard friction; heaviness pressure %d (watch the drift signals)", pressure)
	next := "hold -- re-run after a verb, a front-door flag, or a refusal reason is added; drive pressure down by consolidating, not by deleting a real guard"
	if debt > 0 {
		finding = plural(debt, "operator-friction defect") + ": a steering surface is undiscoverable or the appeal channel is unwired"
		next = "fix " + worstHardKPI(kpis) + ": index the steering surface in the doc map / wire the appeal channel"
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"heaviness_pressure": pressure,
			"verbs":              len(s.Verbs),
			"meta_verbs":         len(s.MetaVerbs),
			"front_door_flags":   s.FrontDoorFlags,
			"refusal_reasons":    s.RefusalReasons,
			"appeal_wired":       s.AppealWired,
		},
	})
	p.Workspace = root
	return p
}

// --- small local helpers -----------------------------------------------------------------------

func plural(n int, noun string) string {
	s := fmt.Sprintf("%d %s", n, noun)
	if n != 1 {
		s += "(s)"
	}
	return s
}

// worstHardKPI names the KPI carrying the most HARD defects, for the next-action pointer.
func worstHardKPI(kpis []scorecard.KPI) string {
	worst := kpis[0]
	for _, k := range kpis[1:] {
		if len(k.Defects) > len(worst.Defects) {
			worst = k
		}
	}
	return worst.Key
}
