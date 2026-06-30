// Package heavinessscore is the operator-heaviness / steering-effort stick.
//
// Every sibling scorecard grades the CODE; the one that comes closest to "steering effort",
// the steerability index behind tools/steerability_scorecard.py, measures the SHAPE of the Go
// package graph -- modularity, coupling, navigability -- i.e. how hard the CODE is to change as
// it grows. None of them answers the question an OPERATOR asks: how heavy does the repo feel to
// DRIVE? -- how many verbs must I discover, how many flags does the front door carry, how many
// structured ways can a green change still be refused, and when a guard refuses me wrongly is
// there an in-product way to appeal? This card scores that surface.
//
// It is deliberately ORTHOGONAL to the steerability index: that one reads the package graph,
// this one reads the OPERATOR-FACING SURFACE (the cmd/fak dispatch table, the front-door verb's
// flag set, the dos.toml refusal vocabulary, and whether the doc map makes the steering surfaces
// discoverable). No KPI here re-derives a code-shape number, so the two never double-count.
//
// Shared inputs, NOT shared scores: tools/agent_readiness_scorecard.py also reads dos.toml's
// [reasons.*] blocks (its refusal_recovery_mapped) and the cmd/fak verb table (its
// command_verbs_resolve). The overlap is the SOURCE, not the function -- agent_readiness scores
// whether each refusal maps to a recovery and whether a verb resolves; this card scores the raw
// SIZE of those surfaces as operator load. A tree adding a [reasons.X] block moves both cards,
// but they measure different things about it, so the portfolio is not double-counting a defect.
//
// The headline an operator tracks is NOT a 0-100 grade: it is heaviness_pressure -- an unbounded
// integer that sums, per magnitude KPI, the SHARE OF HEADROOM CONSUMED between the comfortable
// soft line and the blowout ceiling (each term normalized to its own span so a verb and a flag
// are commensurable -- H1 of the self-audit). Lower is lighter. It rises when a verb, a flag, or
// a refusal reason is added and falls when the surface is consolidated, so a clean pass holds it
// flat as the repo grows -- and a real reduction (the consolidation tickets) drives it DOWN,
// which is the whole point. The soft lines sit DELIBERATELY below the current live surface: the
// gauge measures distance-to-comfortable (the repo IS heavy today), not drift-from-today, so a
// reduction is rewarded rather than invisible. heaviness_debt is the HARD gate, ~0 on a
// disciplined tree, firing only for the two friction defects whose cheapest fix is genuine work:
// a steering surface the doc map hides (discoverability) and a missing in-product appeal channel
// for a wrong refusal (recovery).
//
// It is a TREE-READING scorecard (no data dir): the dispatch table, the flag set, the dos.toml
// reasons and the doc map ARE the data, read from the source, so it cannot be gamed by editing a
// JSON file -- only by genuinely slimming the surface. The fold/grade/markdown machinery lives in
// pkg/scorecard; this package holds only the operator-surface extraction and the KPIs.
package heavinessscore

import (
	"fmt"
	"math"
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

// Soft lines + ceilings for the magnitude KPIs. The soft line is the comfortable operator-surface
// threshold; it sits BELOW the current live surface ON PURPOSE so heaviness_pressure measures the
// real distance-to-comfortable and a reduction is rewarded (see the package doc, H2). The ceiling
// (or ref) is the blowout line ~1.5-2x the soft line: only past it does a magnitude become HARD
// debt, where the fix is a genuine consolidation program rather than a tweak. Each (soft, span)
// pair also normalizes that KPI's contribution to heaviness_pressure so the terms are
// commensurable (H1) -- a term is the SHARE of its soft->ceiling headroom consumed.
const (
	verbSoftLine    = 90  // top-level subcommands an operator can comfortably hold
	verbHardCeiling = 200 // past this the dispatch table is a navigation wall -> HARD

	flagSoftLine    = 20 // flags on the single front-door verb (`fak guard`)
	flagHardCeiling = 80 // past this the front door is its own steerability tax -> HARD

	reasonSoftLine = 12 // structured refusal reasons an operator must understand
	reasonRef      = 30 // the headroom reference (no hard ceiling -- refusals are SOFT)

	metaShareSoftLine = 0.08 // share of verbs that are meta-scorecards/RSI (clutter the surface)
	metaShareRef      = 0.25 // the headroom reference
)

// requiredDocMapSurface is one steering surface the doc map MUST make discoverable. A surface is
// covered only when one of its tokens sits on a line that ALSO carries a markdown link (the shape
// of a real doc-map entry) -- so the HARD gate cannot be retired by pasting the token into a bare
// sentence (M3 of the self-audit). The tokens are specific on purpose: "steerability-scorecard"/
// "fak steering" must not be satisfied by the Charter's incidental "human-steerable" line.
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
	reCaseVerb = regexp.MustCompile(`case "([a-z][a-z0-9-]*)":`)
	// reGuardFlag matches every stdlib flag-DEFINING call form, including the binder idiom
	// fs.StringVar/BoolVar/IntVar/... (M1 of the self-audit: the bare-`(` form silently missed
	// every *Var flag, a latent undercount + a cheap game). It deliberately does NOT match the
	// FlagSet METHODS that are not flags (Parse/Args/Usage/Visit/PrintDefaults/NArg/...).
	reGuardFlag    = regexp.MustCompile(`fs\.(?:String|Bool|Int|Int64|Uint|Uint64|Float64|Duration|TextVar|Func|BoolFunc)(?:Var)?\(|fs\.Var\(`)
	reReasonsBlock = regexp.MustCompile(`(?m)^\[reasons\.[A-Z0-9_]+\]`)
	// reMarkdownLink matches a markdown link target `](...)` -- the shape of a real doc-map entry.
	reMarkdownLink = regexp.MustCompile(`\]\([^)]*\)`)
)

// dispatchSwitchHeader anchors the top-level verb switch in cmd/fak/main.go. Bounding the verb
// scan to this block is what keeps `case "x":` lines in main.go's ~10 OTHER switches (session-
// state transitions, a2a principals, sub-arg parsing) from being miscounted as CLI verbs (M2 of
// the self-audit: the whole-file scan over-counted by ~9 and made the headline perturbable by
// non-operator code).
const dispatchSwitchHeader = "switch os.Args[1]"

// dispatchBlock returns the text of the top-level `switch os.Args[1] { ... }` verb table, from the
// switch header to its own `default:` clause (the dispatch's default closes the verb list; it is
// the first 1-tab `default:` after the header, before any inner switch). A defensive fallback
// returns the whole input if the header is absent.
func dispatchBlock(s string) string {
	i := strings.Index(s, dispatchSwitchHeader)
	if i < 0 {
		return s
	}
	rest := s[i:]
	if d := strings.Index(rest, "\n\tdefault:"); d >= 0 {
		return rest[:d]
	}
	return rest
}

// isMetaVerb reports whether a dispatch verb is meta-tooling (a scorecard / RSI verb) rather than
// a core operator verb -- the surface clutter an operator does not reach for day to day.
func isMetaVerb(v string) bool {
	return strings.Contains(v, "scorecard") ||
		strings.HasSuffix(v, "-rsi") ||
		strings.HasSuffix(v, "-score") ||
		v == "scorecard"
}

// ParseVerbs returns the distinct top-level dispatch verbs (sorted) from cmd/fak/main.go, scanning
// ONLY the `switch os.Args[1]` block so inner-switch string cases are not counted as verbs.
func ParseVerbs(mainGo string) []string {
	block := dispatchBlock(mainGo)
	seen := map[string]bool{}
	for _, m := range reCaseVerb.FindAllStringSubmatch(block, -1) {
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
		// scope the appeal check to the same dispatch block as the verbs, so a `case "complain":`
		// buried in an inner switch can't satisfy the recovery gate (self-audit, M2-adjacent).
		AppealWired: strings.Contains(dispatchBlock(mainGo), `case "complain":`),
		DocMap:      strings.ToLower(docMap),
	}
}

// metaShare is the fraction of the dispatch table that is meta-tooling.
func (s Surface) metaShare() float64 {
	if len(s.Verbs) == 0 {
		return 0
	}
	return float64(len(s.MetaVerbs)) / float64(len(s.Verbs))
}

// magnitudeScore maps a measured value to 0-100 for the KPI score: 100 at/below the soft line,
// falling linearly to 0 at the ceiling/reference.
func magnitudeScore(value, softLine, ceiling float64) float64 {
	if value <= softLine {
		return 100
	}
	if value >= ceiling {
		return 0
	}
	return 100 * (1 - (value-softLine)/(ceiling-softLine))
}

// headroomConsumed is the normalized pressure contribution of one magnitude term: the share of the
// soft->ceiling headroom the value has consumed, as an integer 0-100 (clamped). Normalizing every
// term to its own span is what makes a verb, a flag, and a refusal reason commensurable in the
// summed headline (H1 of the self-audit) -- so no single high-count surface dominates the gauge.
func headroomConsumed(value, softLine, ceiling float64) int {
	if ceiling <= softLine {
		return 0
	}
	frac := (value - softLine) / (ceiling - softLine)
	if frac <= 0 {
		return 0
	}
	if frac >= 1 {
		return 100
	}
	return int(math.Round(frac * 100))
}

// --- KPIs --------------------------------------------------------------------------------------

// kpiDocMapCoversSteering (HARD, discoverability): the doc map must make every steering surface
// discoverable VIA A REAL ENTRY -- a token co-located with a markdown link, not a bare prose
// mention. A surface the operator cannot find from a linked llms.txt entry is a friction defect
// whose fix is a genuine doc-map entry, so it earns HARD debt.
func kpiDocMapCoversSteering(s Surface) scorecard.KPI {
	lines := strings.Split(s.DocMap, "\n")
	var defects []string
	covered := 0
	for _, req := range requiredDocMapSurfaces {
		linked := false
		for _, ln := range lines {
			if scorecard.HasAny(ln, req.tokens) && reMarkdownLink.MatchString(ln) {
				linked = true
				break
			}
		}
		if linked {
			covered++
			continue
		}
		defects = append(defects, fmt.Sprintf("the doc map (llms.txt) has no LINKED entry for the %s -- an operator cannot discover it (expected a `](...)` line carrying one of: %s)",
			req.name, strings.Join(req.tokens, ", ")))
	}
	score := 100.0
	if len(requiredDocMapSurfaces) > 0 {
		score = 100.0 * float64(covered) / float64(len(requiredDocMapSurfaces))
	}
	return scorecard.KPI{
		Key: "docmap_covers_steering", Group: "discoverability", Score: score,
		Detail:  fmt.Sprintf("%d/%d steering surface(s) discoverable from a linked doc-map entry", covered, len(requiredDocMapSurfaces)),
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
	share := s.metaShare()
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
// real guard, which is the wrong move; the signal is the trend, watched not gamed. (agent_readiness
// also reads these blocks, for recovery-mapping not size -- see the package doc.)
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

// pressureByTerm returns the per-magnitude-KPI headroom-consumed contributions, keyed for the
// corpus breakdown so a reader sees WHERE the pressure concentrates (H1 of the self-audit).
func pressureByTerm(s Surface) map[string]int {
	return map[string]int{
		"cli_verb_count":         headroomConsumed(float64(len(s.Verbs)), verbSoftLine, verbHardCeiling),
		"front_door_flag_burden": headroomConsumed(float64(s.FrontDoorFlags), flagSoftLine, flagHardCeiling),
		"refusal_vocab_size":     headroomConsumed(float64(s.RefusalReasons), reasonSoftLine, reasonRef),
		"meta_verb_share":        headroomConsumed(s.metaShare(), metaShareSoftLine, metaShareRef),
	}
}

// Pressure is the continual, unbounded heaviness number (NOT a 0-100 grade): the summed
// headroom-consumed of the magnitude KPIs, each normalized to its own soft->ceiling span so the
// terms are commensurable. It is the headline an operator tracks over time -- lower is lighter --
// and the number a consolidation program drives down.
func Pressure(s Surface) int {
	total := 0
	for _, v := range pressureByTerm(s) {
		total += v
	}
	return total
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
	byTerm := pressureByTerm(s)

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
			"pressure_by_term":   byTerm,
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
