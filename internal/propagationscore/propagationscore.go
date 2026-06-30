// Package propagationscore measures CONVENTION PROPAGATION across fak's scorecard
// family -- the degree to which a "scoring concept" improved in ONE card has
// fanned out to its siblings -- and turns each un-propagated gap into a deduped,
// dispatchable GitHub issue (internal/propagationscore/dispatch.go) so the
// operator never has to REMEMBER to extend an improvement by hand.
//
// THE GAP THIS CLOSES
// -------------------
// fak measures itself with a FAMILY of scorecards (conflation, dogfood,
// guard-rsi, token-defaults, ...). They are "the same machine pointed at a
// different surface", so a good idea proven in one -- the shared pkg/scorecard
// kernel that kills grade-table drift, the `--compare` prove-the-drop gate, the
// `--markdown` published snapshot -- SHOULD ride into every sibling. In practice
// it doesn't: the kernel was built to de-duplicate the skeleton yet rides only a
// minority of the cards, and the shared `scorecardCmdSetup` CLI helper never
// propagated `--compare`. Nothing notices, so the operator carries the debt in
// their head ("remember to add --compare to the new card"). That memory-tax IS
// the heaviness this card exists to remove: a deterministic measure of which
// improvements have fanned out, and an auto-filed issue per laggard so the fleet
// extends them without anyone remembering.
//
// It is a TREE-READING scorecard (no data dir): the family roster is cross-checked
// against the real tree (each member's cmd shell + package must exist), and every
// convention is probed from the SOURCE (does this package import pkg/scorecard, does
// this shell define a `--compare` flag), so a defect is retired only by doing the
// real extension -- never by editing a JSON file. The fold/grade/markdown machinery
// lives in pkg/scorecard; this package holds only the family roster, the convention
// probes, and the KPIs -- and it rides that same kernel, so it holds itself to the
// discipline it measures.
package propagationscore

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id.
const Schema = "fak-propagation-scorecard/1"

// DebtKey is the headline HARD integer the control-pane folds (corpus.propagation_debt).
const DebtKey = "propagation_debt"

// QuorumFraction is the adoption share at/above which a NON-declared convention's laggards
// become HARD propagation debt: once a majority of the family has adopted a convention the
// improvement has PROVEN itself, so the stragglers are debt worth fanning out. Below it (but
// above zero) the convention is "emerging" -- a SOFT candidate, not yet debt. A DECLARED
// convention (an explicit documented standard) is HARD for every laggard regardless of count.
const QuorumFraction = 0.5

// Member is one scorecard in the family: an operator-facing `fak <Verb>` whose pure logic lives
// in PkgDir and whose CLI shell is CmdFile. The row is cross-checked against the tree (CmdFile +
// PkgDir must exist), so the roster cannot claim a member that isn't there.
type Member struct {
	Verb    string // the dispatch verb, e.g. "conflation-scorecard"
	CmdFile string // repo-relative cmd shell, e.g. "cmd/fak/conflationscore.go"
	PkgDir  string // repo-relative internal package dir, e.g. "internal/conflationscore"
	DebtKey string // the corpus debt integer (for output + control-pane match)
}

// Family is the curated roster of debt-scorecards held to a uniform propagation discipline. A
// new debt-scorecard is added here so its convention adoption is measured; the roster_complete
// KPI flags (SOFT) any `-score`/`-scorecard` verb left off, so a laggard cannot hide by omission.
var Family = []Member{
	{Verb: "conflation-scorecard", CmdFile: "cmd/fak/conflationscore.go", PkgDir: "internal/conflationscore", DebtKey: "conflation_debt"},
	{Verb: "dogfood-score", CmdFile: "cmd/fak/dogfoodscore.go", PkgDir: "internal/dogfoodscore", DebtKey: "dogfood_debt"},
	{Verb: "concept-usage-score", CmdFile: "cmd/fak/conceptusagescore.go", PkgDir: "internal/conceptusage", DebtKey: "concept_usage_debt"},
	{Verb: "support-maturity-scorecard", CmdFile: "cmd/fak/supportmaturityscore.go", PkgDir: "internal/supportmaturityscore", DebtKey: "support_maturity_debt"},
	{Verb: "token-defaults-scorecard", CmdFile: "cmd/fak/token_defaults.go", PkgDir: "internal/defaultvaluescore", DebtKey: "token_defaults_debt"},
	{Verb: "guard-rsi-scorecard", CmdFile: "cmd/fak/guardrsi.go", PkgDir: "internal/guardrsi", DebtKey: "guard_rsi_debt"},
	{Verb: "loop-index-scorecard", CmdFile: "cmd/fak/loopscore.go", PkgDir: "internal/loopscore", DebtKey: "loopindex_debt"},
	{Verb: "ui-quality-scorecard", CmdFile: "cmd/fak/uiqualityscore.go", PkgDir: "internal/uiquality", DebtKey: "ui_quality_debt"},
	{Verb: "propagation-scorecard", CmdFile: "cmd/fak/propagationscore.go", PkgDir: "internal/propagationscore", DebtKey: DebtKey},
}

// Convention is one "scoring concept" that, once improved in one card, SHOULD fan out to the
// rest of the family. Declared marks an explicit documented standard (its laggards are HARD
// regardless of count); a non-declared convention uses the quorum rule.
type Convention struct {
	Key      string // stable id, also the cmd flag name for the surface conventions
	Short    string // short name for a title ("the shared scorecard kernel")
	Label    string // the full scoring concept, for the work-list + issue body
	Group    string
	Declared bool
	Source   string // where the standard is declared, cited in the issue body
}

// Conventions is the set of cross-pollinatable scoring concepts this card tracks. The order is
// the work-list order. Each is probed from source by adopts() below.
var Conventions = []Convention{
	{
		Key: "kernel", Short: "the shared pkg/scorecard kernel",
		Label: "ride the shared pkg/scorecard kernel (Fold/grade/render) instead of a copy-pasted skeleton",
		Group: "reuse", Declared: true, Source: "the pkg/scorecard package doc",
	},
	{
		Key: "json", Short: "the --json control-pane payload",
		Label: "expose --json (the control-pane payload every card must emit)",
		Group: "surface", Declared: true, Source: "the scorecard skill, law 5",
	},
	{
		Key: "compare", Short: "the --compare regression gate",
		Label: "expose --compare (the prove-the-debt-drop regression gate)",
		Group: "surface",
	},
	{
		Key: "markdown", Short: "the --markdown snapshot",
		Label: "expose --markdown (the committed snapshot for the published page)",
		Group: "surface",
	},
	{
		Key: "test", Short: "a package regression test",
		Label: "carry a regression test in its package (the live-floor sentinel)",
		Group: "rigor",
	},
	{
		Key: "controlpane", Short: "control-pane registration",
		Label: "be registered in the scorecard control-pane ratchet so its debt folds into the portfolio",
		Group: "rollup",
	},
}

// Probe is one member's measured convention adoption, plus whether its files exist on the tree.
type Probe struct {
	Member  Member
	Exists  bool
	Adopted map[string]bool // convention key -> adopted
}

var (
	reKernelImport = "anthony-chaudhary/fak/pkg/scorecard"
	reSetupHelper  = "scorecardCmdSetup("
)

// flagRE matches a flag DEFINITION for the named flag in a cmd shell, e.g. fs.Bool("compare",
// ...). Anchoring on the flag-method call (not a bare occurrence of the word) keeps a comment or
// a help string from reading as adoption.
func flagRE(name string) *regexp.Regexp {
	return regexp.MustCompile(`\.(?:String|Bool|Int|Int64|Uint|Float64|Duration|Var)\(\s*"` + regexp.QuoteMeta(name) + `"`)
}

// ProbeMembers cross-checks each rostered member against the tree and probes its convention
// adoption from source. It is pure over the filesystem at root (no clock, no network).
func ProbeMembers(root string, family []Member) []Probe {
	controlPane := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash("tools/scorecard_control_pane.py")))
	probes := make([]Probe, 0, len(family))
	for _, m := range family {
		cmdText := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(m.CmdFile)))
		pkgAbs := filepath.Join(root, filepath.FromSlash(m.PkgDir))
		usesSetup := strings.Contains(cmdText, reSetupHelper)
		ad := map[string]bool{
			// The shared scorecardCmdSetup helper provides --json + --markdown (but NOT
			// --compare), so a member that calls it adopts those two without an inline flag.
			"json":        usesSetup || flagRE("json").MatchString(cmdText),
			"markdown":    usesSetup || flagRE("markdown").MatchString(cmdText),
			"compare":     flagRE("compare").MatchString(cmdText),
			"kernel":      pkgImportsKernel(pkgAbs),
			"test":        pkgHasTest(pkgAbs),
			"controlpane": controlPaneMentions(controlPane, m),
		}
		probes = append(probes, Probe{
			Member:  m,
			Exists:  cmdText != "" && dirExists(pkgAbs),
			Adopted: ad,
		})
	}
	return probes
}

// kpiForConvention scores one convention's propagation across the family and returns the HARD
// laggard gaps the dispatcher fans out. score = adoption percent (scale-free). A laggard is HARD
// debt when the convention is Declared OR adoption has cleared the quorum (and is not yet 100%);
// below quorum it is an emerging SOFT signal, never debt.
func kpiForConvention(c Convention, probes []Probe) (scorecard.KPI, []Gap) {
	adopters, total := 0, 0
	var laggards []Member
	for _, p := range probes {
		if !p.Exists {
			continue
		}
		total++
		if p.Adopted[c.Key] {
			adopters++
		} else {
			laggards = append(laggards, p.Member)
		}
	}
	adoption := 0.0
	if total > 0 {
		adoption = float64(adopters) / float64(total)
	}
	k := scorecard.KPI{
		Key:    "propagate_" + c.Key,
		Group:  c.Group,
		Score:  adoption * 100,
		Detail: fmt.Sprintf("%d/%d family cards %s", adopters, total, c.Short),
	}
	hard := c.Declared || (adoption >= QuorumFraction && adoption < 1.0)
	var gaps []Gap
	for _, m := range laggards {
		msg := fmt.Sprintf("fak %s (%s) does not %s -- %d/%d cards already do%s",
			m.Verb, m.PkgDir, c.Label, adopters, total, declaredNote(c))
		if hard {
			k.Defects = append(k.Defects, msg)
			gaps = append(gaps, Gap{Member: m, Convention: c, Adopters: adopters, Total: total})
		} else if adoption > 0 {
			k.Soft = append(k.Soft, msg+" [emerging -- below the propagation quorum]")
		}
	}
	return k, gaps
}

func declaredNote(c Convention) string {
	if c.Declared {
		return " (a declared standard: " + c.Source + ")"
	}
	return ""
}

// kpiMemberIntegrity (HARD, roster): every rostered member's cmd shell + package must exist. A
// missing one means the roster drifted from the tree (a card renamed/moved) -- the one defect
// that would silently corrupt every adoption rate, so it earns HARD debt. It is the live-floor
// the smoke test pins (zero on a fresh roster).
func kpiMemberIntegrity(probes []Probe) scorecard.KPI {
	present := 0
	var defects []string
	for _, p := range probes {
		if p.Exists {
			present++
			continue
		}
		defects = append(defects, fmt.Sprintf("rostered family member fak %s is missing its files (%s / %s) -- the roster drifted from the tree; fix Family",
			p.Member.Verb, p.Member.CmdFile, p.Member.PkgDir))
	}
	score := 100.0
	if len(probes) > 0 {
		score = 100.0 * float64(present) / float64(len(probes))
	}
	return scorecard.KPI{
		Key: "member_integrity", Group: "roster", Score: score,
		Detail:  fmt.Sprintf("%d/%d rostered members resolve to a real cmd shell + package", present, len(probes)),
		Defects: defects,
	}
}

var reScoreVerb = regexp.MustCompile(`case "([a-z][a-z0-9-]*-(?:score|scorecard))":`)

// excludedVerbs are dispatch verbs that match the -score/-scorecard shape but are NOT debt
// scorecards (the portfolio pane, the Slack publisher), so roster_complete does not nag for them.
var excludedVerbs = map[string]bool{
	"scorecard":  true, // the control-pane pane, not a card
	"scoreboard": true, // a Slack publisher, not a card
	"loop-score": true, // alias surface of loop-index-scorecard
}

// scorecardVerbs lists the dispatch verbs in cmd/fak/main.go that look like scorecards.
func scorecardVerbs(root string) []string {
	main := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash("cmd/fak/main.go")))
	seen := map[string]bool{}
	for _, m := range reScoreVerb.FindAllStringSubmatch(main, -1) {
		seen[m[1]] = true
	}
	out := make([]string, 0, len(seen))
	for v := range seen {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// kpiRosterComplete (SOFT, roster): every scorecard verb in the dispatch table should be in the
// propagation family so its adoption is measured. A verb left off is a SOFT nudge (an
// unrostered card is not itself a code defect -- the fix is to add the row), so it never gates
// an unrelated scorecard from landing, but it can never be hidden either.
func kpiRosterComplete(root string, family []Member) scorecard.KPI {
	rostered := map[string]bool{}
	for _, m := range family {
		rostered[m.Verb] = true
	}
	verbs := scorecardVerbs(root)
	covered := 0
	var soft []string
	considered := 0
	for _, v := range verbs {
		if excludedVerbs[v] {
			continue
		}
		considered++
		if rostered[v] {
			covered++
			continue
		}
		soft = append(soft, fmt.Sprintf("fak %s is a scorecard verb but not in the propagation family -- add it to Family so its convention adoption is measured", v))
	}
	score := 100.0
	if considered > 0 {
		score = 100.0 * float64(covered) / float64(considered)
	}
	return scorecard.KPI{
		Key: "roster_complete", Group: "roster", Score: score,
		Detail: fmt.Sprintf("%d/%d dispatch scorecard verbs are in the propagation family", covered, considered),
		Soft:   soft,
	}
}

// Build reads the family, probes each convention, and folds the KPIs into the control-pane
// payload via the shared kernel. root is the repo root.
func Build(root string) scorecard.Payload {
	probes := ProbeMembers(root, Family)
	kpis := make([]scorecard.KPI, 0, len(Conventions)+2)
	gapCount := 0
	for _, c := range Conventions {
		k, gaps := kpiForConvention(c, probes)
		kpis = append(kpis, k)
		gapCount += len(gaps)
	}
	kpis = append(kpis, kpiMemberIntegrity(probes), kpiRosterComplete(root, Family))

	debt := 0
	for _, k := range kpis {
		debt += len(k.Defects)
	}

	finding := "every proven scorecard convention has fanned out to the whole family"
	next := "hold -- re-run after a new card or convention lands; a regression means an improvement stopped propagating"
	if debt > 0 {
		finding = fmt.Sprintf("%s: a proven scorecard convention has not fanned out to every sibling", plural(debt, "un-propagated convention gap"))
		next = "run `fak propagation-debt-dispatch` to fan out one tracked issue per gap, then extend the laggards worst-first"
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"members":     len(Family),
			"conventions": len(Conventions),
			"fanout_gaps": gapCount,
		},
	})
	p.Workspace = root
	return p
}

// --- convention probes (pure over the filesystem) --------------------------------------------

func dirExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

// pkgImportsKernel reports whether any non-test .go file in pkgDir imports pkg/scorecard.
func pkgImportsKernel(pkgDir string) bool {
	entries, err := os.ReadDir(pkgDir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		if strings.Contains(scorecard.SafeRead(filepath.Join(pkgDir, name)), reKernelImport) {
			return true
		}
	}
	return false
}

// pkgHasTest reports whether pkgDir carries any _test.go regression sentinel.
func pkgHasTest(pkgDir string) bool {
	entries, err := os.ReadDir(pkgDir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), "_test.go") {
			return true
		}
	}
	return false
}

// controlPaneMentions reports whether the control-pane source registers this member, matching on
// any stable token (the debt key the pane reads, the dispatch verb, or the package base name).
func controlPaneMentions(text string, m Member) bool {
	if text == "" {
		return false
	}
	needles := []string{m.Verb}
	if m.DebtKey != "" {
		needles = append(needles, m.DebtKey)
	}
	needles = append(needles, baseName(m.PkgDir))
	return scorecard.HasAny(text, needles)
}

func baseName(p string) string {
	if i := strings.LastIndexAny(p, "/\\"); i >= 0 {
		return p[i+1:]
	}
	return p
}

func plural(n int, noun string) string {
	s := fmt.Sprintf("%d %s", n, noun)
	if n != 1 {
		s += "(s)"
	}
	return s
}
