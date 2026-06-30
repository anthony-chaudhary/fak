package scorecardpane

// hygiene.go — the native port of tools/repo_hygiene_scorecard.py: the deterministic
// repo-hygiene fold over the git-tracked tree. Twelve mechanical KPIs in four groups
// (verbosity, organization, indexing, accessibility), folded into a composite score,
// an A-F grade, and the headline hygiene-debt integer. The JSON payload shape
// (schema/corpus/kpis with corpus.hygiene_debt etc.) is byte-compatible with the
// Python --json so the control-pane fold reads it identically.
//
// The pure KPI checks (KPIRedundancy … KPIPlainLanguage, BuildHygienePayload) are
// the tested surface. Gather (disk + git) is the impure shell in hygienegather.go.

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// HygieneSchema mirrors the Python SCHEMA constant.
const HygieneSchema = "fak-repo-hygiene-scorecard/1"

// Calibration thresholds — each a deliberate value carried over byte-for-byte from
// the Python tool so the two implementations score a tree identically.
const (
	docHardLines        = 1000
	docSoftLines        = 600
	dupHardJaccard      = 0.80
	dupSoftJaccard      = 0.55
	shingleK            = 8
	rootOtherSample     = 40
	emdashPer100WBudget = 0.6
	fleschFloor         = 30.0
	aitellPerDocCap     = 8
	accessSample        = 30
)

// notesDir is the sanctioned home for dated/research notes (check_doc_placement).
const notesDir = "docs/notes"

// generatedSnapshot is this tool's own published snapshot — never scored, never an
// orphan (you don't lint your linter's output).
const generatedSnapshot = "docs/REPO-HYGIENE-SCORECARD.md"

// hygieneGroups is the canonical KPI group order.
var hygieneGroups = []string{"verbosity", "organization", "indexing", "accessibility"}

// kpiWeights is the composite weighting; sums to 1.0 so the score tops out at 100.
var kpiWeights = map[string]float64{
	"redundancy":      0.10,
	"bloat":           0.06,
	"root_hygiene":    0.12,
	"placement":       0.10,
	"dir_discipline":  0.06,
	"index_presence":  0.10,
	"index_integrity": 0.12,
	"orphans":         0.12,
	"alt_text":        0.06,
	"ai_tells":        0.06,
	"jargon":          0.04,
	"plain_language":  0.06,
}

// allowedRootMD is the DOC_PLACEMENT root .md allowlist (the inline fallback the
// Python tool uses when check_doc_placement is unimportable — the same set the
// gate enforces).
var allowedRootMD = map[string]bool{
	"README.md": true, "START-HERE.md": true, "INSTALL.md": true, "INDEX.md": true,
	"CONTRIBUTING.md": true, "CLA.md": true, "AGENTS.md": true, "CLAUDE.md": true,
	"SECURITY.md": true, "ARCHITECTURE.md": true, "EXTENDING.md": true,
	"GETTING-STARTED.md": true, "POLICY.md": true, "STATUS.md": true, "CLAIMS.md": true,
}

// archiveDirs are the archival/evidence/journal subtrees excluded from the
// reader-facing discovery surface.
var archiveDirs = map[string]bool{
	"releases": true, "stable-releases": true, "notes": true, "proofs": true,
	"launch": true, "benchmarks": true, "benchmark": true, "benchmarking": true,
	"testing": true, "serving": true,
}

// rootMetaExempt are the contract/platform meta docs GitHub surfaces directly —
// exempt from the orphan check (reached via the platform, not a docs hub).
var rootMetaExempt = map[string]bool{
	"README.md": true, "START-HERE.md": true, "INDEX.md": true, "INSTALL.md": true,
	"CONTRIBUTING.md": true, "CLA.md": true, "AGENTS.md": true, "CLAUDE.md": true,
	"SECURITY.md": true, "GOVERNANCE.md": true, "CODE_OF_CONDUCT.md": true,
	"MAINTAINERS.md": true, "AUTHORS.md": true, "NOTICE.md": true, "SUPPORT.md": true,
	"HISTORY.md": true, "CHANGELOG.md": true, "ROADMAP.md": true, "TRADEMARK.md": true,
	"LICENSING.md": true, "PUBLIC-SCRUB-POLICY.md": true,
}

// expectedIndexes / frontDoors / indexManifests mirror the Python lists.
var expectedIndexes = []string{"INDEX.md", "llms.txt", "docs/index.md"}
var frontDoors = []string{"README.md", "llms.txt", "INDEX.md", "docs/index.md", "START-HERE.md", "AGENTS.md"}
var indexManifests = []string{"llms.txt", "INDEX.md", "docs/index.md"}

// rootAllowedOther are the non-.md files legitimately tracked at the repo root.
var rootAllowedOther = map[string]bool{
	"go.mod": true, "go.sum": true, "Makefile": true, "LICENSE": true, "VERSION": true,
	"Dockerfile": true, ".dockerignore": true, ".gitignore": true, ".gitattributes": true,
	".editorconfig": true, "install.sh": true, "test.sh": true, "test.ps1": true,
	"dos.toml": true, "opencode.json": true, "CITATION.cff": true, "llms.txt": true,
	"llms-full.txt": true, ".golangci.yml": true, ".golangci.yaml": true,
	".cursorrules": true, ".mcp.json": true, ".gitmodules": true, ".markdownlint.json": true,
	"server.json": true, "glama.json": true, "smithery.yaml": true,
	"fak-mac.local.ps1.example": true,
}

// HygieneKPI is one KPI's result. defects = HARD units of hygiene-debt; soft =
// score-only judgment nudges. Field tags match the Python dict keys.
type HygieneKPI struct {
	KPI     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   int      `json:"score"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

// HygienePayload is the folded repo-hygiene payload. Shape matches the Python
// build_payload return dict.
type HygienePayload struct {
	Schema     string        `json:"schema"`
	OK         bool          `json:"ok"`
	Verdict    string        `json:"verdict"`
	Finding    string        `json:"finding"`
	Reason     string        `json:"reason"`
	NextAction string        `json:"next_action"`
	Workspace  string        `json:"workspace"`
	Corpus     HygieneCorpus `json:"corpus"`
	KPIs       []HygieneKPI  `json:"kpis"`
}

// HygieneCorpus is the corpus-level summary (the control-pane reads corpus.grade +
// corpus.hygiene_debt from here).
type HygieneCorpus struct {
	Score           float64               `json:"score"`
	Grade           string                `json:"grade"`
	HygieneDebt     int                   `json:"hygiene_debt"`
	A11yDebt        int                   `json:"a11y_debt"`
	SoftSignals     int                   `json:"soft_signals"`
	DebtByGroup     map[string]int        `json:"debt_by_group"`
	KPIScores       map[string]int        `json:"kpi_scores"`
	DebtByKPI       map[string]int        `json:"debt_by_kpi"`
	Breakdown       []HygieneBreakdownRow `json:"breakdown"`
	WorktreeClutter []string              `json:"worktree_clutter"`
}

// HygieneBreakdownRow is one row of the worst-first KPI breakdown.
type HygieneBreakdownRow struct {
	KPI    string `json:"kpi"`
	Group  string `json:"group"`
	Score  int    `json:"score"`
	Debt   int    `json:"debt"`
	Detail string `json:"detail"`
}

func clampScore(score float64) int {
	v := int(math.Round(score))
	if v < 0 {
		return 0
	}
	if v > 100 {
		return 100
	}
	return v
}

// HygieneGrade is the A-F ladder for a composite score (the same thresholds the
// family shares).
func HygieneGrade(score float64) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

// kpiResult builds a HygieneKPI, normalizing nil slices to empty (the Python lists).
func kpiResult(kpi, group string, score int, detail string, defects, soft []string) HygieneKPI {
	if defects == nil {
		defects = []string{}
	}
	if soft == nil {
		soft = []string{}
	}
	return HygieneKPI{KPI: kpi, Group: group, Score: score, Detail: detail, Defects: defects, Soft: soft}
}

// --- pure per-KPI checks ---------------------------------------------------

// DupDoc is one reader-facing doc's similarity inputs.
type DupDoc struct {
	Path     string
	Shingles map[uint64]struct{}
	Words    int
}

// KPIRedundancy flags near-duplicate reader-facing docs (one concept in two files).
func KPIRedundancy(docs []DupDoc) HygieneKPI {
	var defects, soft []string
	n := len(docs)
	for i := 0; i < n; i++ {
		di := docs[i]
		for j := i + 1; j < n; j++ {
			dj := docs[j]
			wi, wj := di.Words, dj.Words
			if wi == 0 || wj == 0 {
				continue
			}
			ratio := float64(wi) / float64(wj)
			if ratio < 0.5 || ratio > 2.0 {
				continue
			}
			sim := jaccard(di.Shingles, dj.Shingles)
			if sim >= dupHardJaccard {
				defects = append(defects, fmt.Sprintf("near-duplicate (%s): %s ≈ %s — consolidate to one",
					pct0(sim), di.Path, dj.Path))
			} else if sim >= dupSoftJaccard {
				soft = append(soft, fmt.Sprintf("consolidation candidate (%s): %s ~ %s", pct0(sim), di.Path, dj.Path))
			}
		}
	}
	detail := "no near-duplicate docs"
	if len(defects) > 0 || len(soft) > 0 {
		detail = fmt.Sprintf("%d near-duplicate pair(s), %d candidate(s)", len(defects), len(soft))
	}
	score := clampScore(100 - 14*float64(len(defects)) - math.Min(20, 3*float64(len(soft))))
	return kpiResult("redundancy", "verbosity", score, detail, defects, soft)
}

// BloatDoc is one reader-facing doc's line count.
type BloatDoc struct {
	Path   string
	NLines int
}

// KPIBloat flags reader-facing docs past the hard line ceiling.
func KPIBloat(docs []BloatDoc) HygieneKPI {
	var defects, soft []string
	for _, d := range docs {
		if d.NLines > docHardLines {
			defects = append(defects, fmt.Sprintf("oversized doc %s (%d lines > %d) — split into sections or trim",
				d.Path, d.NLines, docHardLines))
		} else if d.NLines > docSoftLines {
			soft = append(soft, fmt.Sprintf("long doc %s (%d lines)", d.Path, d.NLines))
		}
	}
	detail := "no oversized docs"
	if len(defects) > 0 || len(soft) > 0 {
		detail = fmt.Sprintf("%d oversized, %d long", len(defects), len(soft))
	}
	score := clampScore(100 - 12*float64(len(defects)) - math.Min(20, 2*float64(len(soft))))
	return kpiResult("bloat", "verbosity", score, detail, defects, soft)
}

// KPIRootHygiene flags stray root docs + clutter. rootMD/rootOther are the root
// tracked file basenames.
func KPIRootHygiene(rootMD, rootOther []string) HygieneKPI {
	var defects []string
	badMD := rootMDViolations(rootMD)
	badOther := rootOtherViolations(rootOther)
	for _, n := range badMD {
		defects = append(defects, fmt.Sprintf("non-front-door doc at root: %s → move to %s/ "+
			"(or add to the allowlist if genuinely a root doc)", n, notesDir))
	}
	limit := badOther
	if len(limit) > rootOtherSample {
		limit = limit[:rootOtherSample]
	}
	for _, n := range limit {
		why := classifyJunk(n)
		if why == "" {
			why = "not a front-door / build-essential file"
		}
		defects = append(defects, fmt.Sprintf("clutter at root: %s — %s", n, why))
	}
	var soft []string
	if extra := len(badOther) - rootOtherSample; extra > 0 {
		soft = append(soft, fmt.Sprintf("... and %d more root non-doc file(s)", extra))
	}
	detail := "root holds only front-door / meta files"
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d stray root doc(s), %d stray root file(s)", len(badMD), len(badOther))
	}
	return kpiResult("root_hygiene", "organization", clampScore(100-10*float64(len(defects))), detail, defects, soft)
}

// KPIPlacement flags dated/research docs outside docs/notes/.
func KPIPlacement(datedMisplaced []string) HygieneKPI {
	sorted := append([]string(nil), datedMisplaced...)
	sort.Strings(sorted)
	var defects []string
	for _, p := range sorted {
		defects = append(defects, fmt.Sprintf("dated/research doc outside %s/: %s → move it and index it", notesDir, p))
	}
	detail := fmt.Sprintf("dated docs live under %s/", notesDir)
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d misplaced dated doc(s)", len(defects))
	}
	return kpiResult("placement", "organization", clampScore(100-10*float64(len(defects))), detail, defects, nil)
}

// KPIDirDiscipline flags near-duplicate sibling directory names (benchmark vs
// benchmarks). dirs is the list of tracked directory rels.
func KPIDirDiscipline(dirs []string) HygieneKPI {
	var defects []string
	byParent := map[string][]string{}
	for _, d := range dirs {
		parent := ""
		if idx := strings.LastIndex(d, "/"); idx >= 0 {
			parent = d[:idx]
		}
		byParent[parent] = append(byParent[parent], d)
	}
	// iterate parents deterministically
	parents := make([]string, 0, len(byParent))
	for p := range byParent {
		parents = append(parents, p)
	}
	sort.Strings(parents)
	for _, parent := range parents {
		kids := byParent[parent]
		stems := map[string][]string{}
		for _, k := range kids {
			base := strings.ToLower(k)
			if idx := strings.LastIndex(base, "/"); idx >= 0 {
				base = base[idx+1:]
			}
			stem := dirStem(base)
			stems[stem] = append(stems[stem], k)
		}
		stemKeys := make([]string, 0, len(stems))
		for s := range stems {
			stemKeys = append(stemKeys, s)
		}
		sort.Strings(stemKeys)
		for _, stem := range stemKeys {
			group := stems[stem]
			if len(group) > 1 && stem != "" {
				sg := append([]string(nil), group...)
				sort.Strings(sg)
				defects = append(defects, fmt.Sprintf("near-duplicate sibling dirs: [%s] — merge into one",
					strings.Join(quoteAll(sg), ", ")))
			}
		}
	}
	detail := "no near-duplicate sibling directories"
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d near-duplicate dir group(s)", len(defects))
	}
	return kpiResult("dir_discipline", "organization", clampScore(100-12*float64(len(defects))), detail, defects, nil)
}

// KPIIndexPresence flags missing expected index surfaces. present maps each
// EXPECTED_INDEXES entry to existence.
func KPIIndexPresence(present map[string]bool) HygieneKPI {
	var defects []string
	for _, name := range expectedIndexes {
		if !present[name] {
			defects = append(defects, fmt.Sprintf("missing index surface: %s (readers and guards are pointed here)", name))
		}
	}
	detail := "all expected index surfaces present"
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d missing index surface(s)", len(defects))
	}
	return kpiResult("index_presence", "indexing", clampScore(100-25*float64(len(defects))), detail, defects, nil)
}

// KPIIndexIntegrity flags dead local links inside curated index manifests.
// deadByIndex maps each manifest to its dead local targets. order is the manifest
// iteration order (indexManifests) for determinism.
func KPIIndexIntegrity(deadByIndex map[string][]string) HygieneKPI {
	var defects []string
	for _, idx := range indexManifests {
		dead := append([]string(nil), deadByIndex[idx]...)
		sort.Strings(dead)
		for _, t := range dead {
			defects = append(defects, fmt.Sprintf("dead index entry in %s: %s", idx, t))
		}
	}
	detail := "every index entry resolves"
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d dead index entr(y/ies)", len(defects))
	}
	return kpiResult("index_integrity", "indexing", clampScore(100-15*float64(len(defects))), detail, defects, nil)
}

// KPIOrphans flags reader-facing docs reachable from no index/hub.
func KPIOrphans(orphans []string, nReader int) HygieneKPI {
	sorted := append([]string(nil), orphans...)
	sort.Strings(sorted)
	var defects []string
	for _, p := range sorted {
		defects = append(defects, fmt.Sprintf("orphan (reachable from no index/hub): %s — index it or delete it", p))
	}
	indexed := nReader - len(orphans)
	if indexed < 0 {
		indexed = 0
	}
	denom := nReader
	if denom < 1 {
		denom = 1
	}
	pct := round1(100 * float64(indexed) / float64(denom))
	detail := fmt.Sprintf("%d/%d reader-facing docs reachable from an index (%s%%)", indexed, nReader, fmtFloat(pct))
	return kpiResult("orphans", "indexing", clampScore(pct), detail, defects, nil)
}

// AltDoc is one doc's images-missing-alt list.
type AltDoc struct {
	Path    string
	Missing []string
}

// KPIAltText flags doc images with empty/missing alt-text (a11y-debt).
func KPIAltText(perDoc []AltDoc) HygieneKPI {
	var defects []string
	for _, d := range perDoc {
		for _, src := range d.Missing {
			defects = append(defects, fmt.Sprintf("image without alt-text in %s: %s — add descriptive alt-text", d.Path, src))
		}
	}
	detail := "every doc image carries alt-text"
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d image(s) missing alt-text", len(defects))
	}
	return kpiResult("alt_text", "accessibility", clampScore(100-6*float64(len(defects))), detail, defects, nil)
}

// AITellDoc is one doc's AI-tell hits + em-dash overage.
type AITellDoc struct {
	Path       string
	Hits       []string
	EmdashOver int
}

// KPIAITells flags cliché / LLM-scaffolding phrases (capped per doc).
func KPIAITells(perDoc []AITellDoc) HygieneKPI {
	var defects, soft []string
	for _, d := range perDoc {
		hits := d.Hits
		if len(hits) > aitellPerDocCap {
			hits = hits[:aitellPerDocCap]
		}
		for _, ph := range hits {
			defects = append(defects, fmt.Sprintf("AI-tell phrase in %s: “%s” — say it plainly", d.Path, ph))
		}
		if len(d.Hits) > aitellPerDocCap {
			soft = append(soft, fmt.Sprintf("%d more AI-tells in %s (capped)", len(d.Hits)-aitellPerDocCap, d.Path))
		}
		if d.EmdashOver > 0 {
			soft = append(soft, fmt.Sprintf("em-dash flood in %s (%d past budget)", d.Path, d.EmdashOver))
		}
	}
	detail := "no AI-tell phrases"
	if len(defects) > 0 {
		detail = fmt.Sprintf("%d AI-tell phrase(s) across %d doc(s)", len(defects), len(perDoc))
	}
	score := clampScore(100 - 3*float64(len(defects)) - math.Min(15, float64(len(soft))))
	return kpiResult("ai_tells", "accessibility", score, detail, defects, soft)
}

// KPIJargon (SOFT) scores first-screen jargon density. naked is '<path>: <term>'.
func KPIJargon(naked []string, nReader int) HygieneKPI {
	var soft []string
	limit := naked
	if len(limit) > accessSample {
		limit = limit[:accessSample]
	}
	for _, n := range limit {
		soft = append(soft, fmt.Sprintf("first-screen jargon, no gloss: %s", n))
	}
	if len(naked) > accessSample {
		soft = append(soft, fmt.Sprintf("... and %d more naked jargon term(s)", len(naked)-accessSample))
	}
	denom := nReader
	if denom < 1 {
		denom = 1
	}
	rate := float64(len(naked)) / float64(denom)
	detail := "first-screen terms carry plain glosses"
	if len(naked) > 0 {
		detail = fmt.Sprintf("%d naked first-screen jargon term(s) (%s/doc)", len(naked), fmtFloat(round1(rate)))
	}
	score := clampScore(100 - math.Min(60, float64(roundInt(45*rate))))
	return kpiResult("jargon", "accessibility", score, detail, nil, soft)
}

// KPIPlainLanguage (SOFT) scores reading-ease, undefined acronyms, literal idioms.
func KPIPlainLanguage(signals []string, nDense, nAcroDocs, nIdiom, nReader int) HygieneKPI {
	var soft []string
	limit := signals
	if len(limit) > accessSample {
		limit = limit[:accessSample]
	}
	soft = append(soft, limit...)
	if len(signals) > accessSample {
		soft = append(soft, fmt.Sprintf("... and %d more accessibility signal(s)", len(signals)-accessSample))
	}
	total := nDense + nAcroDocs + nIdiom
	denom := nReader
	if denom < 1 {
		denom = 1
	}
	rate := float64(total) / float64(denom)
	detail := "reads plainly (ease, acronyms, idioms)"
	if total > 0 {
		detail = fmt.Sprintf("%d dense doc(s), %d doc(s) with undefined acronyms, %d literal-reader idiom(s)",
			nDense, nAcroDocs, nIdiom)
	}
	score := clampScore(100 - math.Min(60, float64(roundInt(45*rate))))
	return kpiResult("plain_language", "accessibility", score, detail, nil, soft)
}

// BuildHygienePayload folds the KPIs into the composite score + control-pane payload.
// Ported from the Python build_payload (the non-error branch). worktreeClutter is the
// advisory concurrency canary (never debt).
func BuildHygienePayload(workspace string, kpis []HygieneKPI, worktreeClutter []string) HygienePayload {
	byName := map[string]HygieneKPI{}
	for _, k := range kpis {
		byName[k.KPI] = k
	}
	score := 0.0
	for n, w := range kpiWeights {
		if k, ok := byName[n]; ok {
			score += w * float64(k.Score)
		}
	}
	score = round1(score)
	hygieneDebt := 0
	nSoft := 0
	for _, k := range kpis {
		hygieneDebt += len(k.Defects)
		nSoft += len(k.Soft)
	}
	grade := HygieneGrade(score)
	debtByGroup := map[string]int{}
	for _, g := range hygieneGroups {
		debtByGroup[g] = 0
	}
	for _, k := range kpis {
		debtByGroup[k.Group] += len(k.Defects)
	}
	kpiScores := map[string]int{}
	debtByKPI := map[string]int{}
	for _, k := range kpis {
		kpiScores[k.KPI] = k.Score
		debtByKPI[k.KPI] = len(k.Defects)
	}
	breakdown := make([]HygieneBreakdownRow, 0, len(kpis))
	for _, k := range kpis {
		breakdown = append(breakdown, HygieneBreakdownRow{
			KPI: k.KPI, Group: k.Group, Score: k.Score, Debt: len(k.Defects), Detail: k.Detail,
		})
	}
	// sort by (-debt, score) — worst-first, ties by ascending score (Python key).
	sort.SliceStable(breakdown, func(i, j int) bool {
		if breakdown[i].Debt != breakdown[j].Debt {
			return breakdown[i].Debt > breakdown[j].Debt
		}
		return breakdown[i].Score < breakdown[j].Score
	})

	if worktreeClutter == nil {
		worktreeClutter = []string{}
	}
	corpus := HygieneCorpus{
		Score: score, Grade: grade, HygieneDebt: hygieneDebt,
		A11yDebt: debtByGroup["accessibility"], SoftSignals: nSoft,
		DebtByGroup: debtByGroup, KPIScores: kpiScores, DebtByKPI: debtByKPI,
		Breakdown: breakdown, WorktreeClutter: worktreeClutter,
	}

	var ok bool
	var verdict, finding, reason, nextAction string
	if hygieneDebt == 0 {
		ok, verdict, finding = true, "OK", "repo_clean"
		reason = fmt.Sprintf("repo clean: score %s/100 (grade %s), zero hygiene-debt across %d KPIs (%d advisory signal(s))",
			fmtFloat(score), grade, len(kpis), nSoft)
		nextAction = "no required edit; re-run after the next structural change"
	} else {
		ok, verdict, finding = false, "ACTION", "hygiene_debt"
		worst := breakdown[0]
		reason = fmt.Sprintf("%d unit(s) of hygiene-debt; score %s/100 (grade %s); heaviest: %s (%d defect(s))",
			hygieneDebt, fmtFloat(score), grade, worst.KPI, worst.Debt)
		nextAction = "retire hygiene-debt worst-first (see corpus.breakdown + per-KPI defects): " +
			"consolidate duplicates, split/trim oversized docs, clear root clutter, " +
			"move dated docs to docs/notes/, index orphans, cut AI-tell phrases; " +
			"re-run to prove the drop"
	}

	return HygienePayload{
		Schema: HygieneSchema, OK: ok, Verdict: verdict, Finding: finding,
		Reason: reason, NextAction: nextAction, Workspace: workspace,
		Corpus: corpus, KPIs: kpis,
	}
}

// HygieneError builds the AUDIT_ERROR payload (not a git repo / read failure).
func HygieneError(workspace, errMsg string) HygienePayload {
	return HygienePayload{
		Schema: HygieneSchema, OK: false, Verdict: "AUDIT_ERROR", Finding: "tooling_error",
		Reason: errMsg, NextAction: "fix the read (run from repo ROOT, with git), then re-run",
		Workspace: workspace, Corpus: HygieneCorpus{}, KPIs: []HygieneKPI{},
	}
}

// --- small pure helpers ----------------------------------------------------

func rootMDViolations(rootMD []string) []string {
	var out []string
	for _, n := range rootMD {
		if !allowedRootMD[n] {
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

func rootOtherViolations(rootOther []string) []string {
	var out []string
	for _, n := range rootOther {
		if !rootAllowedOther[n] {
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

func jaccard(a, b map[uint64]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}
	inter := 0
	small, large := a, b
	if len(b) < len(a) {
		small, large = b, a
	}
	for k := range small {
		if _, ok := large[k]; ok {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	return float64(inter) / float64(union)
}

func dirStem(base string) string {
	for _, suf := range []string{"ing", "es", "s"} {
		if strings.HasSuffix(base, suf) {
			return base[:len(base)-len(suf)]
		}
	}
	return base
}

func quoteAll(in []string) []string {
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = "'" + s + "'"
	}
	return out
}

func pct0(f float64) string { return fmt.Sprintf("%d%%", roundInt(f*100)) }

func round1(f float64) float64 { return math.Round(f*10) / 10 }

func roundInt(f float64) int { return int(math.Round(f)) }

// fmtFloat renders a float like Python: an integral value as "N.0"? Python prints a
// trailing .0 for floats (e.g. 96.9, 100.0). Go's %g drops the .0; use a 1-decimal
// format when fractional and a plain decimal otherwise to match the JSON numbers
// Python emits — JSON encoders normalize 96.0 and 96 identically for consumers, so
// this only affects the human render. We keep one-decimal for human strings.
func fmtFloat(f float64) string {
	if f == math.Trunc(f) {
		return fmt.Sprintf("%.1f", f)
	}
	return fmt.Sprintf("%.1f", f)
}
