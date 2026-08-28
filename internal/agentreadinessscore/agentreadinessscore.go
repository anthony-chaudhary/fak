// Package agentreadinessscore grades the ONE thing the sibling scorecards do not: can an
// autonomous coding agent — Claude Code, OpenAI Codex, Cursor, an MCP client — (1) DISCOVER
// fak, (2) WANT to adopt and build on it, and (3) do so effectively and easily? The other
// inward sticks grade a surface a human reviewer cares about (the tree's shape, the Go
// module, a doc's prose); this one grades agent attractiveness, the number an agent-first
// project lives or dies on.
//
// It scores the git-tracked tree on twenty-three mechanical KPIs in three groups — the exact
// three steps an agent walks (DISCOVER / ADOPT / BUILD) — folds them into a weighted score
// and an A-F grade, and reports two headline numbers, because agent-readiness is two
// questions:
//
//   - friction_debt (the HARD gate, lower = better, floor 0): the count of concrete,
//     re-derivable defects that make the BASELINE affordances missing or broken — a dead
//     link, an un-runnable first command, an un-tagged claim. It saturates: drive it to zero
//     and there is nothing left to fix. The 0-100 score / A-F grade is the same
//     baseline-coverage question in percent form.
//   - experience_frontier (the headline, higher = better, UNBOUNDED): the weighted count of
//     real, WORKING agent affordances the tree provides — each integration recipe, each
//     zero-setup harness config, each kernel refusal an agent can recover from, each tool an
//     agent drives via --json. Agent experience is a never-done program, so it has no ceiling
//     and is tracked as a frontier + a trend, the deliberate mirror of the operator-heaviness
//     gauge. You move it by ADDING a real affordance, never by gaming a substring.
//
// This is the Go port of the former tools/agent_readiness_scorecard.py. It is a faithful,
// self-contained re-implementation (Idiom B, like internal/focusscore): the default / --json
// / --markdown / --stamp / --compare surfaces are preserved so the committed snapshot and the
// control-pane fold are unchanged. Deterministic + read-only: it reads the git-tracked tree
// (two clones of the same commit score identically) and edits nothing.
package agentreadinessscore

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"

	"github.com/anthony-chaudhary/fak/internal/devhandoff"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func buildPayload(workspace string, kpis []KPI, facts map[string]int, errText string) Payload {
	if errText != "" {
		return Payload{Schema: Schema, OK: false, Verdict: "AUDIT_ERROR", Finding: "tooling_error",
			Reason: errText, NextAction: "fix the read (run from repo ROOT, with git), then re-run",
			Workspace: workspace, Corpus: map[string]any{}, KPIs: []KPI{}}
	}
	byName := make(map[string]KPI, len(kpis))
	for _, k := range kpis {
		byName[k.Kpi] = k
	}
	var scoreSum float64
	for _, n := range kpiWeightOrder {
		if k, ok := byName[n]; ok {
			scoreSum += kpiWeights[n] * float64(k.Score)
		}
	}
	score := round1(scoreSum)
	frictionDebt := 0
	nSoft := 0
	for _, k := range kpis {
		frictionDebt += len(k.Defects)
		nSoft += len(k.Soft)
	}
	grade := scorecard.GradeStd(score)
	frontier, frontierByTerm := experienceFrontier(facts)

	debtByGroup := map[string]int{"discover": 0, "adopt": 0, "build": 0}
	for _, k := range kpis {
		debtByGroup[k.Group] += len(k.Defects)
	}
	scoreByGroup := map[string]float64{"discover": 0, "adopt": 0, "build": 0}
	wsumByGroup := map[string]float64{"discover": 0, "adopt": 0, "build": 0}
	for _, k := range kpis {
		w := kpiWeights[k.Kpi]
		scoreByGroup[k.Group] += w * float64(k.Score)
		wsumByGroup[k.Group] += w
	}
	groupScores := map[string]float64{}
	for _, g := range groups {
		if wsumByGroup[g] != 0 {
			groupScores[g] = round1(scoreByGroup[g] / wsumByGroup[g])
		} else {
			groupScores[g] = 0.0
		}
	}

	breakdown := make([]breakdownRow, 0, len(kpis))
	for _, k := range kpis {
		breakdown = append(breakdown, breakdownRow{k.Kpi, k.Group, k.Score, len(k.Defects), k.Detail})
	}
	sort.SliceStable(breakdown, func(i, j int) bool {
		if breakdown[i].Debt != breakdown[j].Debt {
			return breakdown[i].Debt > breakdown[j].Debt
		}
		return breakdown[i].Score < breakdown[j].Score
	})

	kpiScores := map[string]int{}
	debtByKpi := map[string]int{}
	for _, k := range kpis {
		kpiScores[k.Kpi] = k.Score
		debtByKpi[k.Kpi] = len(k.Defects)
	}

	frontierUnitsCopy := map[string]int{}
	for k, v := range frontierUnits {
		frontierUnitsCopy[k] = v
	}
	corpus := map[string]any{
		"experience_frontier": frontier,
		"frontier_by_term":    frontierByTerm,
		"frontier_units":      frontierUnitsCopy,
		"score":               score,
		"grade":               grade,
		"friction_debt":       frictionDebt,
		"soft_signals":        nSoft,
		"group_scores":        groupScores,
		"debt_by_group":       debtByGroup,
		"kpi_scores":          kpiScores,
		"debt_by_kpi":         debtByKpi,
		"breakdown":           breakdown,
	}

	standing := "discover " + fmt0(groupScores["discover"]) + " · adopt " + fmt0(groupScores["adopt"]) + " · build " + fmt0(groupScores["build"])
	var ok bool
	var verdict, finding, reason, nextAction string
	if frictionDebt == 0 {
		ok, verdict, finding = true, "OK", "agent_ready"
		reason = "experience-frontier " + strconv.Itoa(frontier) + " (unbounded; higher = better) · baseline score " +
			ftoa(score) + "/100 (grade " + grade + "), zero friction-debt across " + strconv.Itoa(len(kpis)) +
			" KPIs (" + standing + "; " + strconv.Itoa(nSoft) + " advisory). An agent can discover, adopt, and build on fak with no missing affordance — and the frontier still has headroom: onboard a harness, map a refusal, expose a --json surface"
		nextAction = "climb the frontier — add the next real affordance (a new integration recipe / harness config, a refusal mapped to a recovery, a tool given --json); hold friction-debt at 0; re-run to prove the climb"
	} else {
		ok, verdict, finding = false, "ACTION", "friction_debt"
		worst := breakdown[0]
		reason = "experience-frontier " + strconv.Itoa(frontier) + " (unbounded) · " + strconv.Itoa(frictionDebt) + " unit(s) of friction-debt; baseline score " +
			ftoa(score) + "/100 (grade " + grade + "); heaviest: " + worst.Kpi + " (" + strconv.Itoa(worst.Debt) + " defect(s)); standing " + standing
		nextAction = "retire friction-debt worst-first (see corpus.breakdown + per-KPI defects): fix the agents.md entry point, the doc-map, the quotable identity, dead orientation links, the first command, the install one-liner, the tagged ledger, the per-agent recipes, the leaf scaffold, the surfaced guardrails, the contributor contract; re-run to prove the drop"
	}

	return Payload{Schema: Schema, OK: ok, Verdict: verdict, Finding: finding, Reason: reason,
		NextAction: nextAction, Workspace: workspace, Corpus: corpus, KPIs: kpis}
}

// ---------------------------------------------------------------------------
// Disk + git gathering (the impure shell around the pure KPIs).
// ---------------------------------------------------------------------------

func gitLines(args []string, root string) []string {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
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

func fileExists(root, rel string) bool {
	_, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	return err == nil
}

// localLink is one local link in a doc: the target as written + whether it resolves on disk.
type localLink struct {
	target string
	exists bool
}

func localLinks(text, docRel, root string) []localLink {
	base := filepath.Dir(filepath.Join(root, filepath.FromSlash(docRel)))
	var out []localLink
	seen := map[string]bool{}
	for _, m := range linkRe.FindAllStringSubmatch(text, -1) {
		t := strings.TrimSpace(m[2])
		if strings.HasPrefix(t, "http://") || strings.HasPrefix(t, "https://") ||
			strings.HasPrefix(t, "mailto:") || strings.HasPrefix(t, "#") || strings.HasPrefix(t, "tel:") {
			continue
		}
		pathPart := strings.TrimSpace(strings.SplitN(strings.SplitN(t, "#", 2)[0], "?", 2)[0])
		if pathPart == "" || seen[pathPart] {
			continue
		}
		seen[pathPart] = true
		var resolved string
		if strings.HasPrefix(pathPart, "/") {
			resolved = filepath.Join(root, filepath.FromSlash(strings.TrimLeft(pathPart, "/")))
		} else {
			resolved = filepath.Join(base, filepath.FromSlash(pathPart))
		}
		_, err := os.Stat(resolved)
		out = append(out, localLink{pathPart, err == nil})
	}
	return out
}

func toolHasJSON(text string) bool {
	return strings.Contains(text, `"--json"`) || strings.Contains(text, "'--json'")
}

func isSubstantiveRecipe(text string) bool {
	t := strings.TrimSpace(text)
	if len(t) < recipeMinChars {
		return false
	}
	return strings.Contains(t, "```") || strings.Contains(t, "](")
}

// badFencedPaths flags every fenced path operand that won't resolve in a clean clone.
func badFencedPaths(docTexts map[string]string, root string) []string {
	var bad []string
	for _, doc := range sortedKeys(docTexts) {
		text := docTexts[doc]
		if text == "" {
			continue
		}
		for _, block := range fencedBlocks(text) {
			for _, line := range strings.Split(block, "\n") {
				low := strings.ToLower(line)
				for _, lit := range placeholderLiteral {
					if strings.Contains(low, strings.ToLower(lit)) {
						bad = append(bad, doc+": `"+lit+"…` placeholder in a runnable command — make it a real path or an <angle-bracket> slot")
						break
					}
				}
			}
			for _, op := range pathOperands(block) {
				low := strings.ToLower(op)
				stale := false
				for _, p := range stalePathPrefixes {
					if strings.HasPrefix(low, strings.ToLower(p)) {
						stale = true
						break
					}
				}
				if stale {
					bad = append(bad, doc+": `"+op+"` — stale private-monorepo path (a clean clone has no fleet/ parent; the module IS the repo root)")
					continue
				}
				clean := op
				if strings.HasPrefix(op, "./") {
					clean = op[2:]
				}
				isRepoRel := false
				for _, d := range repoTopDirs {
					if strings.HasPrefix(strings.ToLower(clean), d) {
						isRepoRel = true
						break
					}
				}
				if isRepoRel && !fileExists(root, clean) {
					bad = append(bad, doc+": `"+op+"` — repo-relative path does not exist on disk")
				}
			}
		}
	}
	return dedup(bad)
}

// firstCommandFacts resolves the runnability of the no-setup first command.
func firstCommandFacts(texts map[string]string, root string) (found, policyOK bool, policyRef string, needsKey bool) {
	for _, doc := range firstCommandDocs {
		for _, block := range fencedBlocks(texts[doc]) {
			if !has(block, firstCommandTokens...) {
				continue
			}
			policyRef = ""
			if m := proofPolicyRe.FindStringSubmatch(block); m != nil {
				policyRef = strings.Trim(m[1], "\"'`")
			}
			policyOK = true
			if policyRef != "" && !isTemplateSlot(policyRef) {
				policyOK = fileExists(root, policyRef)
			}
			needsKey = has(block, proofNeedsKeyTokens...)
			return true, policyOK, policyRef, needsKey
		}
	}
	return false, true, "", false
}

func unknownCommandVerbs(docTexts map[string]string, verbs map[string]bool) []string {
	if len(verbs) == 0 {
		return []string{}
	}
	var out []string
	seen := map[string]bool{}
	for _, verb := range commandFamilyVerbs {
		verbs[verb] = true
	}
	for _, command := range devhandoff.Commands {
		verbs[command.Name] = true
	}
	for _, doc := range sortedKeys(docTexts) {
		text := docTexts[doc]
		if text == "" {
			continue
		}
		for _, v := range commandVerbs(text) {
			if verbs[v] {
				continue
			}
			key := doc + ": fak " + v
			if !seen[key] {
				seen[key] = true
				out = append(out, key)
			}
		}
	}
	return out
}

func deadRecipeLinks(root string, recipeTexts map[string]string) []string {
	var dead []string
	for _, rel := range sortedKeys(recipeTexts) {
		text := recipeTexts[rel]
		if text == "" {
			continue
		}
		for _, l := range localLinks(text, rel, root) {
			if !l.exists {
				dead = append(dead, rel+" -> "+l.target)
			}
		}
	}
	return dead
}

func agentConfigIntegrity(root string) []string {
	var bad []string
	if !fileExists(root, mcpConfigFile) {
		return bad
	}
	raw := safeRead(root, mcpConfigFile)
	var data any
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		return []string{mcpConfigFile + " is present but does not parse as JSON: " + err.Error()}
	}
	obj, isObj := data.(map[string]any)
	if !isObj {
		bad = append(bad, mcpConfigFile+" has no mcpServers map — the harness finds no server to start")
		return bad
	}
	servers, ok := obj["mcpServers"].(map[string]any)
	if !ok || len(servers) == 0 {
		bad = append(bad, mcpConfigFile+" has no mcpServers map — the harness finds no server to start")
		return bad
	}
	for _, name := range sortedKeys(anyMapToStr(servers)) {
		spec := servers[name]
		if strings.HasPrefix(strings.TrimSpace(name), "//") {
			continue
		}
		specMap, isMap := spec.(map[string]any)
		if !isMap {
			bad = append(bad, mcpConfigFile+" server '"+name+"' is not an object — the harness can't read it")
			continue
		}
		if strings.TrimSpace(anyToStr(specMap["command"])) == "" && strings.TrimSpace(anyToStr(specMap["url"])) == "" {
			bad = append(bad, mcpConfigFile+" server '"+name+"' names neither a launch command nor a url — the harness can't start it")
		}
	}
	return bad
}

// Build scores whether an agent can discover, adopt, and build on fak, re-derived from the
// git-tracked tree at root. It is the collect() entry point.
func Build(root string) Payload {
	abs, err := filepath.Abs(root)
	if err != nil || abs == "" {
		abs = root
	}
	if !fileExists(abs, ".git") && gitLines([]string{"rev-parse", "--git-dir"}, abs) == nil {
		return buildPayload(abs, nil, nil, "not a git repo at "+abs+" — run from the repo ROOT")
	}
	kpis, facts := gather(abs)
	return buildPayload(abs, kpis, facts, "")
}
