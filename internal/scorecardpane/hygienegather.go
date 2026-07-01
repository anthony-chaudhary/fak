package scorecardpane

// hygienegather.go — the impure shell around the pure hygiene KPIs: read the
// git-tracked tree (so two clones of one commit score identically), run every KPI,
// and fold. Ported from the Python gather() / collect() / _git_lines /
// _local_md_targets / _all_local_links / _worktree_clutter. CollectHygiene is the
// public entry the cmd wiring calls.

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// CollectHygiene reads the tracked tree at root and returns the folded payload. When
// root is not a git repo it returns the AUDIT_ERROR payload (matching the Python
// collect()).
func CollectHygiene(root string) HygienePayload {
	abs, err := filepath.Abs(root)
	if err != nil {
		abs = root
	}
	if !isGitRepo(abs) {
		return HygieneError(abs, "not a git repo at "+abs+" — run from the repo ROOT")
	}
	// Single-source the ENFORCEMENT vocab from the gates that own it (exactly as the
	// Python tool imports check_doc_placement / check_committed_files): the live
	// allowlist + exempt dirs are authoritative; the inline package defaults are the
	// stand-alone fallback. Loading them keeps the measurement from drifting from the
	// refusal at commit time.
	loadEnforcementVocab(abs)
	kpis, clutter := gatherHygiene(abs)
	return BuildHygienePayload(abs, kpis, clutter)
}

var pyStrLitRE = regexp.MustCompile(`"([^"]+)"`)

// pyAnyStrLitRE matches a double- OR single-quoted Python string literal (group 1 =
// double-quoted, group 2 = single-quoted). The phrase lists mix quote styles
// (e.g. 'delve' and "in today's"), so the set/tuple double-quote-only matcher would
// drop the single-quoted members.
var pyAnyStrLitRE = regexp.MustCompile(`"([^"]*)"|'([^']*)'`)

// loadEnforcementVocab refreshes allowedRootMD + exemptDataDirs from the gate source
// files (tools/check_doc_placement.py, tools/check_committed_files.py) — the same
// single-source the Python tool imports. On a parse miss the package's inline
// defaults stand (the stand-alone fallback). This makes the Go fold's allowlist
// track the gate's allowlist exactly, so root_hygiene/placement match the Python.
func loadEnforcementVocab(root string) {
	if set := parsePySet(safeRead(root, "tools/check_doc_placement.py"), "ALLOWED_ROOT_MD"); len(set) > 0 {
		allowedRootMD = set
	}
	if dirs := parsePyTuple(safeRead(root, "tools/check_committed_files.py"), "EXEMPT_DATA_DIRS"); len(dirs) > 0 {
		exemptDataDirs = dirs
	}
	// The corpus-wide AI-tell + jargon vocab is single-sourced from doc_appeal /
	// docs scorecards (as the Python tool imports it). Rebuild the derived matchers
	// so a phrase added to those lists is measured here too.
	appeal := safeRead(root, "tools/doc_appeal_scorecard.py")
	cliche := parsePyList(appeal, "CLICHE_PHRASES")
	scaffold := parsePyList(appeal, "LLM_SCAFFOLD_PHRASES")
	if len(cliche) > 0 || len(scaffold) > 0 {
		if len(cliche) > 0 {
			clichePhrases = cliche
		}
		if len(scaffold) > 0 {
			llmScaffoldPhrases = scaffold
		}
		aiTellPhrases = buildAITellPhrases()
		aiTellMatchers = buildTellMatchers()
	}
	if jt := parsePyList(safeRead(root, "tools/docs_scorecard.py"), "JARGON_TERMS"); len(jt) > 0 {
		jargonTerms = jt
	}
}

// parsePyList extracts the quoted string members of a Python list literal
// `NAME = [ "a", "b", ... ]`, preserving order.
func parsePyList(src, name string) []string {
	body := pyAssignBody(src, name, '[', ']')
	if body == "" {
		return nil
	}
	var out []string
	for _, m := range pyAnyStrLitRE.FindAllStringSubmatch(body, -1) {
		s := m[1]
		if s == "" {
			s = m[2]
		}
		out = append(out, s)
	}
	return out
}

// parsePySet extracts the quoted string members of a Python set literal
// `NAME = { "a", "b", ... }` (multi-line). Comments are ignored because they carry
// no double-quoted token that is also a set member here.
func parsePySet(src, name string) map[string]bool {
	body := pyAssignBody(src, name, '{', '}')
	if body == "" {
		return nil
	}
	out := map[string]bool{}
	for _, m := range pyStrLitRE.FindAllStringSubmatch(body, -1) {
		out[m[1]] = true
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// parsePyTuple extracts the quoted string members of a Python tuple literal
// `NAME = ( "a", "b", ... )`, preserving order.
func parsePyTuple(src, name string) []string {
	body := pyAssignBody(src, name, '(', ')')
	if body == "" {
		return nil
	}
	var out []string
	for _, m := range pyStrLitRE.FindAllStringSubmatch(body, -1) {
		out = append(out, m[1])
	}
	return out
}

// pyAssignBody returns the text between the opening and matching closing bracket of
// `NAME = <open> ... <close>`, scanning from the assignment. Returns "" if absent.
func pyAssignBody(src, name string, open, close byte) string {
	idx := strings.Index(src, name)
	for idx >= 0 {
		// require the match to begin a line (a top-level assignment), not a mention.
		if idx == 0 || src[idx-1] == '\n' {
			rest := src[idx+len(name):]
			eq := strings.IndexByte(rest, '=')
			// The open bracket must be the first one AFTER the '=' (a type annotation
			// like `: list[str] =` puts a '[' before the '=' that is not the literal).
			ob := -1
			if eq >= 0 {
				if rel := strings.IndexByte(rest[eq:], open); rel >= 0 {
					ob = eq + rel
				}
			}
			if eq >= 0 && ob >= 0 && ob > eq {
				depth := 0
				for i := ob; i < len(rest); i++ {
					switch rest[i] {
					case open:
						depth++
					case close:
						depth--
						if depth == 0 {
							return rest[ob+1 : i]
						}
					}
				}
			}
		}
		next := strings.Index(src[idx+1:], name)
		if next < 0 {
			break
		}
		idx = idx + 1 + next
	}
	return ""
}

func isGitRepo(root string) bool {
	if _, err := os.Stat(filepath.Join(root, ".git")); err == nil {
		return true
	}
	return len(gitLines([]string{"rev-parse", "--git-dir"}, root)) > 0
}

func gitLines(args []string, root string) []string {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var lines []string
	for _, ln := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(ln) != "" {
			lines = append(lines, ln)
		}
	}
	return lines
}

func safeRead(root, rel string) string {
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		return ""
	}
	return string(b)
}

func pathExists(root, rel string) bool {
	_, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	return err == nil
}

// gatherHygiene reads the tracked tree and runs every pure KPI. Returns (kpis, clutter).
func gatherHygiene(root string) ([]HygieneKPI, []string) {
	tracked := gitLines([]string{"ls-files"}, root)
	trackedSet := map[string]bool{}
	for _, f := range tracked {
		trackedSet[f] = true
	}
	var mdFiles []string
	for _, f := range tracked {
		if strings.HasSuffix(f, ".md") {
			mdFiles = append(mdFiles, f)
		}
	}
	var reader []string
	for _, f := range mdFiles {
		if isReaderFacing(f) {
			reader = append(reader, f)
		}
	}

	texts := map[string]string{}
	for _, f := range reader {
		texts[f] = safeRead(root, f)
	}

	// verbosity
	var dupDocs []DupDoc
	var bloatDocs []BloatDoc
	for _, f := range reader {
		dupDocs = append(dupDocs, DupDoc{
			Path: f, Shingles: shingles(texts[f]), Words: wordCount(proseOnly(texts[f])),
		})
		bloatDocs = append(bloatDocs, BloatDoc{Path: f, NLines: strings.Count(texts[f], "\n") + 1})
	}

	// organization
	var rootMD, rootOther []string
	for _, f := range mdFiles {
		if !strings.Contains(f, "/") {
			rootMD = append(rootMD, f)
		}
	}
	for _, f := range tracked {
		if !strings.Contains(f, "/") && !strings.HasSuffix(f, ".md") {
			rootOther = append(rootOther, f)
		}
	}
	var datedMisplaced []string
	for _, f := range mdFiles {
		if !strings.Contains(f, "/") {
			continue
		}
		base := f[strings.LastIndex(f, "/")+1:]
		if !isDatedDoc(base) {
			continue
		}
		if strings.HasPrefix(f, notesDir+"/") || strings.HasPrefix(f, "docs/releases/") ||
			strings.HasPrefix(f, "docs/stable-releases/") || strings.HasPrefix(f, "blog/") ||
			strings.HasPrefix(f, "experiments/") {
			continue
		}
		datedMisplaced = append(datedMisplaced, f)
	}
	sort.Strings(datedMisplaced)
	dirSet := map[string]bool{}
	for _, f := range tracked {
		if idx := strings.LastIndex(f, "/"); idx >= 0 {
			dirSet[f[:idx]] = true
		}
	}
	var trackedDirs []string
	for d := range dirSet {
		if d != "" {
			trackedDirs = append(trackedDirs, d)
		}
	}
	sort.Strings(trackedDirs)

	// indexing: presence + integrity + orphans
	present := map[string]bool{}
	for _, name := range expectedIndexes {
		present[name] = pathExists(root, name)
	}
	deadByIndex := map[string][]string{}
	for _, idx := range indexManifests {
		if pathExists(root, idx) {
			var dead []string
			for _, lk := range allLocalLinks(safeRead(root, idx), idx, root) {
				if !lk.exists {
					dead = append(dead, lk.target)
				}
			}
			if len(dead) > 0 {
				deadByIndex[idx] = dead
			}
		}
	}
	// reachability BFS seeds: existing front doors + every README hub.
	var seeds []string
	for _, d := range frontDoors {
		if pathExists(root, d) {
			seeds = append(seeds, d)
		}
	}
	for _, f := range mdFiles {
		if f[strings.LastIndex(f, "/")+1:] == "README.md" {
			seeds = append(seeds, f)
		}
	}
	linksByDoc := map[string][]string{}
	for _, f := range mdFiles {
		linksByDoc[f] = filterTracked(localMDTargets(safeRead(root, f), f, root), trackedSet)
	}
	for _, s := range seeds {
		if _, ok := linksByDoc[s]; !ok {
			linksByDoc[s] = filterTracked(localMDTargets(safeRead(root, s), s, root), trackedSet)
		}
	}
	reachable := reachableMD(seeds, linksByDoc)
	seedSet := map[string]bool{}
	for _, s := range seeds {
		seedSet[s] = true
	}
	var orphanPool []string
	for _, f := range reader {
		base := f[strings.LastIndex(f, "/")+1:]
		if !rootMetaExempt[base] {
			orphanPool = append(orphanPool, f)
		}
	}
	var orphans []string
	for _, f := range orphanPool {
		if !reachable[f] && !seedSet[f] {
			orphans = append(orphans, f)
		}
	}

	// accessibility
	acc := gatherAccessibility(reader, texts)

	kpis := []HygieneKPI{
		KPIRedundancy(dupDocs),
		KPIBloat(bloatDocs),
		KPIRootHygiene(rootMD, rootOther),
		KPIPlacement(datedMisplaced),
		KPIDirDiscipline(trackedDirs),
		KPIIndexPresence(present),
		KPIIndexIntegrity(deadByIndex),
		KPIOrphans(orphans, len(orphanPool)),
		KPIAltText(acc.altPerDoc),
		KPIAITells(acc.aiPerDoc),
		KPIJargon(acc.nakedJargon, len(reader)),
		KPIPlainLanguage(acc.plainSignals, acc.nDense, acc.nAcroDocs, acc.nIdiom, len(reader)),
	}
	return kpis, worktreeClutter(root)
}

// accessibilitySignals are the per-reader-doc accessibility findings gatherHygiene
// folds into the alt-text, AI-tell, jargon, and plain-language KPIs.
type accessibilitySignals struct {
	altPerDoc    []AltDoc
	aiPerDoc     []AITellDoc
	nakedJargon  []string
	plainSignals []string
	nDense       int
	nAcroDocs    int
	nIdiom       int
}

// gatherAccessibility scans each reader-facing doc for the accessibility signals:
// missing image alt text, AI-tell phrasing + em-dash overuse, undefined jargon on
// the first screen, and plain-language reading-ease / undefined-acronym / literal-idiom
// defects. Split out of gatherHygiene so each scan phase reads as its own named step.
func gatherAccessibility(reader []string, texts map[string]string) accessibilitySignals {
	var s accessibilitySignals
	for _, f := range reader {
		prose := proseOnly(texts[f])
		low := strings.ToLower(prose)
		if missingAlt := imageAltDefects(prose); len(missingAlt) > 0 {
			s.altPerDoc = append(s.altPerDoc, AltDoc{Path: f, Missing: missingAlt})
		}
		hits := tellHits(low)
		words := wordCount(prose)
		nDash := strings.Count(prose, "—")
		budget := int(float64(words) * emdashPer100WBudget / 100)
		if budget < 2 {
			budget = 2
		}
		if len(hits) > 0 || nDash > budget {
			over := nDash - budget
			if over < 0 {
				over = 0
			}
			s.aiPerDoc = append(s.aiPerDoc, AITellDoc{Path: f, Hits: hits, EmdashOver: over})
		}
		// jargon on the first screen (top 60 lines)
		headLines := strings.SplitN(texts[f], "\n", -1)
		if len(headLines) > 60 {
			headLines = headLines[:60]
		}
		for _, term := range jargonTerms {
			termLow := strings.ToLower(term)
			for _, line := range headLines {
				if strings.Contains(strings.ToLower(line), termLow) {
					if !(strings.Contains(line, "(") || strings.Contains(line, "—") || strings.Contains(line, " - ")) {
						s.nakedJargon = append(s.nakedJargon, f+": "+term)
					}
					break
				}
			}
		}
		// plain-language: reading ease, undefined acronyms, literal idioms
		if words > 120 {
			ease := flesch(prose)
			if ease < fleschFloor {
				s.nDense++
				s.plainSignals = append(s.plainSignals, fmtDenseSignal(ease, f))
			}
		}
		undefined := undefinedAcronyms(prose)
		if len(undefined) > 0 {
			s.nAcroDocs++
			lim := undefined
			if len(lim) > 6 {
				lim = lim[:6]
			}
			s.plainSignals = append(s.plainSignals, "acronym(s) used before definition in "+f+": "+strings.Join(lim, ", "))
		}
		for _, idiom := range literalIdioms {
			if strings.Contains(low, idiom) {
				s.nIdiom++
				s.plainSignals = append(s.plainSignals, "literal-reader idiom “"+idiom+"” in "+f)
			}
		}
	}
	return s
}

func fmtDenseSignal(ease float64, f string) string {
	return "dense reading-ease " + fmtRound0(ease) + " (< " + fmtRound0(fleschFloor) + "): " + f
}

// localMDTargets returns the repo-relative .md targets a doc links to (local links).
func localMDTargets(text, docRel, root string) []string {
	base := docDir(docRel)
	var out []string
	for _, m := range linkRE.FindAllStringSubmatch(text, -1) {
		t := strings.TrimSpace(m[2])
		if hasExternalPrefix(t) {
			continue
		}
		pathPart := strings.TrimSpace(stripAfter(stripAfter(t, "#"), "?"))
		if !strings.HasSuffix(pathPart, ".md") {
			continue
		}
		rel := resolveRel(base, pathPart, root)
		if rel != "" {
			out = append(out, rel)
		}
	}
	return out
}

type localLink struct {
	target string
	exists bool
}

// allLocalLinks returns every local link in a doc as (target_display, exists).
// Non-.md included (an index manifest can point at any file).
func allLocalLinks(text, docRel, root string) []localLink {
	base := docDir(docRel)
	var out []localLink
	seen := map[string]bool{}
	for _, m := range linkRE.FindAllStringSubmatch(text, -1) {
		t := strings.TrimSpace(m[2])
		if hasExternalPrefix(t) {
			continue
		}
		pathPart := strings.TrimSpace(stripAfter(stripAfter(t, "#"), "?"))
		if pathPart == "" || seen[pathPart] {
			continue
		}
		seen[pathPart] = true
		var resolved string
		if strings.HasPrefix(pathPart, "/") {
			resolved = filepath.Join(root, filepath.FromSlash(strings.TrimLeft(pathPart, "/")))
		} else {
			resolved = filepath.Join(root, filepath.FromSlash(base), filepath.FromSlash(pathPart))
		}
		_, err := os.Stat(resolved)
		out = append(out, localLink{target: pathPart, exists: err == nil})
	}
	return out
}

// resolveRel resolves a link target to a repo-relative posix path, or "" if it
// escapes the root.
func resolveRel(base, pathPart, root string) string {
	var resolved string
	if strings.HasPrefix(pathPart, "/") {
		resolved = filepath.Join(root, filepath.FromSlash(strings.TrimLeft(pathPart, "/")))
	} else {
		resolved = filepath.Join(root, filepath.FromSlash(base), filepath.FromSlash(pathPart))
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || strings.HasPrefix(rel, "..") {
		return ""
	}
	return filepath.ToSlash(rel)
}

func filterTracked(targets []string, trackedSet map[string]bool) []string {
	var out []string
	for _, t := range targets {
		if trackedSet[t] {
			out = append(out, t)
		}
	}
	return out
}

// reachableMD is the BFS over local .md links from the seeds; returns the reachable set.
func reachableMD(seeds []string, linksByDoc map[string][]string) map[string]bool {
	visited := map[string]bool{}
	var queue []string
	for _, s := range seeds {
		if _, ok := linksByDoc[s]; ok {
			queue = append(queue, s)
		}
	}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if visited[cur] {
			continue
		}
		visited[cur] = true
		for _, nxt := range linksByDoc[cur] {
			if !visited[nxt] {
				queue = append(queue, nxt)
			}
		}
	}
	return visited
}

var scratchRE = regexp.MustCompile(`\.(txt|csv|log|tmp|out|err)$|(^|/)(report|agent-report)\.json$`)

// exemptDataDirs mirrors the Python EXEMPT_DATA_DIRS prefixes for clutter exemption.
var exemptDataDirs = []string{"experiments/", "testdata/", "internal/"}

// worktreeClutter is the advisory concurrency canary: untracked, non-ignored scratch.
func worktreeClutter(root string) []string {
	others := gitLines([]string{"ls-files", "--others", "--exclude-standard"}, root)
	var out []string
	for _, f := range others {
		exempt := false
		for _, d := range exemptDataDirs {
			if strings.HasPrefix(f, d) {
				exempt = true
				break
			}
		}
		if exempt {
			continue
		}
		isRootData := false
		if !strings.Contains(f, "/") {
			ext := f[strings.LastIndex(f, ".")+1:]
			switch ext {
			case "csv", "json", "txt", "log", "out", "err":
				isRootData = true
			}
		}
		if isRootData || scratchRE.MatchString(f) {
			out = append(out, f)
		}
	}
	sort.Strings(out)
	return out
}

// --- small string helpers --------------------------------------------------

func hasExternalPrefix(t string) bool {
	for _, p := range []string{"http://", "https://", "mailto:", "#", "tel:"} {
		if strings.HasPrefix(t, p) {
			return true
		}
	}
	return false
}

func stripAfter(s, sep string) string {
	if idx := strings.Index(s, sep); idx >= 0 {
		return s[:idx]
	}
	return s
}

// docDir returns the directory portion of a repo-relative doc path ("" when the doc
// sits at the repo root).
func docDir(docRel string) string {
	if idx := strings.LastIndex(docRel, "/"); idx >= 0 {
		return docRel[:idx]
	}
	return ""
}
