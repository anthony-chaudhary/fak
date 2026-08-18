package agentreadinessscore

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/numbermap"
	"sort"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// Typed accessors into the (Go-native) corpus map. Within a process the corpus holds native
// types (int / float64 / nested maps / []breakdownRow); these read them back without churn.
// ---------------------------------------------------------------------------

func corpusInt(c map[string]any, key string) int {
	return asInt(c[key])
}

func corpusFloat(c map[string]any, key string) float64 {
	return asFloat(c[key])
}

func corpusStr(c map[string]any, key string) string {
	if s, ok := c[key].(string); ok {
		return s
	}
	return ""
}

// asInt coerces the numeric shapes a corpus value can take (native int, or float64 when the
// value came back through a JSON baseline) to int.
func asInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		if n < 0 {
			return int(n - 0.5)
		}
		return int(n + 0.5)
	}
	return 0
}

func asFloat(v any) float64 {
	switch n := v.(type) {
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case float64:
		return n
	}
	return 0
}

// floatMap reads a map-of-float corpus value tolerant of a JSON baseline.
func floatMap(v any) map[string]float64 {
	out := map[string]float64{}
	switch m := v.(type) {
	case map[string]float64:
		for k, x := range m {
			out[k] = x
		}
	case map[string]any:
		for k, x := range m {
			out[k] = asFloat(x)
		}
	}
	return out
}

// breakdownRows reads corpus.breakdown as typed rows tolerant of a JSON baseline.
func breakdownRows(v any) []breakdownRow {
	switch rows := v.(type) {
	case []breakdownRow:
		return rows
	case []any:
		out := make([]breakdownRow, 0, len(rows))
		for _, r := range rows {
			if m, ok := r.(map[string]any); ok {
				out = append(out, breakdownRow{
					Kpi: anyToStr(m["kpi"]), Group: anyToStr(m["group"]),
					Score: asInt(m["score"]), Debt: asInt(m["debt"]), Detail: anyToStr(m["detail"]),
				})
			}
		}
		return out
	}
	return nil
}

// ---------------------------------------------------------------------------
// Render — the default human view.
// ---------------------------------------------------------------------------

// Render returns the human scorecard (default view; no trailing newline — the caller's
// Fprintln supplies it, matching Python's print()).
func Render(p Payload) string {
	c := p.Corpus
	gs := floatMap(c["group_scores"])
	fbt := numbermap.Ints(c["frontier_by_term"], asInt)
	debtByGroup := numbermap.Ints(c["debt_by_group"], asInt)

	var terms []string
	for _, dim := range frontierDims {
		if _, ok := fbt[dim]; ok {
			terms = append(terms, dim+":"+strconv.Itoa(fbt[dim]))
		}
	}
	frontierTerms := strings.Join(terms, "  ")

	lines := []string{
		"agent-readiness-scorecard: " + p.Verdict + " (" + p.Finding + ")",
		"  " + p.Reason,
		"",
		"EXPERIENCE-FRONTIER " + strconv.Itoa(corpusInt(c, "experience_frontier")) +
			"  (unbounded · higher = better · a frontier, not a % — there is always one more harness to serve)",
	}
	if frontierTerms != "" {
		lines = append(lines, "  by affordance: "+frontierTerms)
	} else {
		lines = append(lines, "")
	}
	lines = append(lines,
		"baseline: score "+ftoa(corpusFloat(c, "score"))+"/100 (grade "+corpusStr(c, "grade")+") · FRICTION-DEBT "+
			strconv.Itoa(corpusInt(c, "friction_debt"))+" · "+strconv.Itoa(corpusInt(c, "soft_signals"))+" advisory",
		"agent journey:  discover "+fmt0(gs["discover"])+"  ·  adopt "+fmt0(gs["adopt"])+"  ·  build "+fmt0(gs["build"]),
	)
	var debtParts []string
	for _, g := range groups {
		debtParts = append(debtParts, g+":"+strconv.Itoa(debtByGroup[g]))
	}
	lines = append(lines,
		"debt by step: "+strings.Join(debtParts, "  "),
		"",
		"per-KPI (worst first):",
		fmt.Sprintf("  %5s %4s  %-10s %-22s detail", "score", "debt", "step", "kpi"),
	)
	for _, b := range breakdownRows(c["breakdown"]) {
		lines = append(lines, fmt.Sprintf("  %5d %4d  %-10s %-22s %s", b.Score, b.Debt, b.Group, b.Kpi, b.Detail))
	}
	lines = append(lines, "", "friction-debt work-list:")

	worklist := append([]KPI{}, p.KPIs...)
	sort.SliceStable(worklist, func(i, j int) bool {
		return len(worklist[i].Defects) > len(worklist[j].Defects)
	})
	anyDefect := false
	for _, k := range worklist {
		if len(k.Defects) == 0 {
			continue
		}
		anyDefect = true
		lines = append(lines, "  "+k.Kpi+" ("+strconv.Itoa(len(k.Defects))+"):")
		lim := k.Defects
		if len(lim) > 12 {
			lim = lim[:12]
		}
		for _, it := range lim {
			lines = append(lines, "      - "+it)
		}
		if len(k.Defects) > 12 {
			lines = append(lines, "      ... and "+strconv.Itoa(len(k.Defects)-12)+" more")
		}
	}
	if !anyDefect {
		lines = append(lines, "  (none — zero friction-debt)")
	}
	lines = append(lines, "", "next: "+p.NextAction)
	return strings.Join(lines, "\n")
}

// ---------------------------------------------------------------------------
// Markdown — the committed snapshot body.
// ---------------------------------------------------------------------------

// Markdown returns the scorecard as the committed snapshot's Markdown body. The stamp (a date
// string) rides an HTML process comment under the H1. The result ends with a trailing newline
// so `fak score agent-readiness --markdown` (which Fprints it) matches the Python tool's
// print(). It is a port of render_markdown; the former tool's self-references (the process
// comment, the "re-derived by" prose, the Regenerate line) now name the Go tool that owns it.
func Markdown(p Payload, stamp string) string {
	c := p.Corpus
	gs := floatMap(c["group_scores"])

	var out []string
	out = append(out,
		"---",
		"title: \"fak agent-readiness scorecard — the experience-frontier + friction-debt measuring stick\"",
		"description: \"fak's deterministic agent-readiness scorecard: KPIs across the three steps "+
			"an AI agent walks — discover, adopt, build — folded into an unbounded experience-frontier "+
			"(higher = better, the surface to grow) and a baseline friction-debt gate, re-derived from "+
			"the git-tracked tree. Presence KPIs ask does-the-affordance-exist; the paste-and-run and "+
			"Codex-currentness KPIs ask whether the docs work for current agents.\"",
		"---",
		"",
		"# Agent-readiness scorecard — can an agent discover, adopt, and build on fak",
		"",
	)
	if stamp != "" {
		out = append(out, "<!-- agent-readiness-scorecard: "+stamp+" · process: internal/agentreadinessscore -->", "")
	}
	out = append(out,
		"This is the measuring stick for fak's **agent attractiveness** — the question an "+
			"agent-first project lives or dies on: can an autonomous coding agent (Claude Code, "+
			"OpenAI Codex, Cursor, an MCP client) **discover** fak, **want** to adopt it, and "+
			"**build** on it effectively? Every number below is re-derived from the git-tracked "+
			"tree by `fak score agent-readiness` (internal/agentreadinessscore) — no hand-entry. There are two "+
			"headline numbers. **Experience-frontier** (unbounded, higher = better) is the one "+
			"to grow: the weighted count of real, working agent affordances the tree provides — "+
			"a never-done program with no ceiling, the mirror of the operator-heaviness gauge. "+
			"**Friction-debt** (lower = better, floor 0) is the baseline gate: the count of "+
			"concrete, mechanical defects that make fak harder for an agent to find, trust, and "+
			"build on — a missing entry point, a dead orientation link, no copy-pasteable first "+
			"command, an un-tagged claim, a guard that ambushes instead of teaches. Driving "+
			"friction-debt to zero makes fak the path of least resistance for the agent that "+
			"lands in it cold; climbing the frontier keeps widening the set of agents it serves.",
		"",
		"> Regenerate: `go run ./cmd/fak score agent-readiness --markdown --stamp DATE > docs/AGENT-READINESS-SCORECARD.md`",
		"",
		"## Headline",
		"",
	)

	fbt := numbermap.Ints(c["frontier_by_term"], asInt)
	funits := numbermap.Ints(c["frontier_units"], asInt)
	var frontierTermParts []string
	for _, dim := range frontierDims {
		if _, ok := funits[dim]; !ok {
			continue
		}
		if _, ok := fbt[dim]; ok {
			frontierTermParts = append(frontierTermParts, dim+" "+strconv.Itoa(fbt[dim]))
		}
	}
	frontierTerms := strings.Join(frontierTermParts, " · ")

	out = append(out,
		"| Metric | Value |",
		"|---|---|",
		"| **Experience-frontier (unbounded · higher = better)** | **"+strconv.Itoa(corpusInt(c, "experience_frontier"))+"** |",
	)
	if frontierTerms != "" {
		out = append(out, "| Frontier by affordance (weight×count) | "+frontierTerms+" |")
	}
	out = append(out,
		"| Friction-debt (total HARD defects) | "+strconv.Itoa(corpusInt(c, "friction_debt"))+" |",
		"| Baseline coverage score | "+ftoa(corpusFloat(c, "score"))+"/100 (grade "+corpusStr(c, "grade")+") |",
		"| Agent journey | discover "+fmt0(gs["discover"])+" · adopt "+fmt0(gs["adopt"])+" · build "+fmt0(gs["build"])+" |",
		"| Advisory (soft) signals | "+strconv.Itoa(corpusInt(c, "soft_signals"))+" |",
		"| Debt by step | discover:"+strconv.Itoa(numbermap.Ints(c["debt_by_group"], asInt)["discover"])+" · adopt:"+strconv.Itoa(numbermap.Ints(c["debt_by_group"], asInt)["adopt"])+" · build:"+strconv.Itoa(numbermap.Ints(c["debt_by_group"], asInt)["build"])+" |",
		"",
		"### The two questions (why two headline numbers)",
		"",
	)

	var weightParts []string
	for _, d := range frontierDims {
		weightParts = append(weightParts, "`"+d+"`×"+strconv.Itoa(funits[d]))
	}
	out = append(out,
		"**Friction-debt** (lower = better, floor 0) is the BASELINE gate — are the "+
			"expected affordances present and working? It saturates: drive it to zero and "+
			"there is nothing left to fix. **Experience-frontier** (higher = better, "+
			"*unbounded*) is the never-done program — the weighted count of real, working "+
			"agent affordances the tree actually provides (an integration recipe an agent "+
			"follows, a zero-setup harness config, a kernel refusal an agent can recover "+
			"from, a tool an agent drives via `--json`). It has **no ceiling**: there is "+
			"always one more agent harness to make fak first-class for, one more refusal to "+
			"make recoverable — so a \"100% / done\" line would be a category error. It is "+
			"the deliberate mirror of `internal/heavinessscore`'s unbounded "+
			"`heaviness_pressure` (the load an operator carries); this is the surface an "+
			"agent gains. You climb it by ADDING a real affordance, never by gaming a "+
			"substring. Per-unit weights: "+strings.Join(weightParts, ", ")+".",
		"",
		"## The three steps an agent walks",
		"",
		strconv.Itoa(len(p.KPIs))+" KPIs, each 0–100, grouped by the step they gate. "+
			"`debt` = units of HARD friction-debt. The presence KPIs ask "+
			"does-the-affordance-exist; the paste-and-run / executable-truth KPIs ask the "+
			"question presence can't reach — does an agent who pastes the docs actually "+
			"succeed: `fenced_paths_resolve` (the path resolves), `command_verbs_resolve` "+
			"(the `fak <verb>` is a real dispatched verb, parsed live from cmd/fak/main.go), "+
			"`first_command_runs` (the proof runs cold), `recipe_links_resolve` (the link "+
			"inside the recipe is alive), `agent_config_valid` (the auto-loaded config "+
			"parses), `platform_guidance_consistent` (the gate names its Windows bridge). "+
			"`codex_recipe_current` asks whether the Codex guide still matches the current "+
			"Codex MCP / AGENTS.md / exec JSON / Responses-vs-Chat-Completions shape. "+
			"`machine_consumable` is "+
			"advisory (it scores but emits no hard debt — a token is cheap to game).",
		"",
		"| Step | KPI | Score | Debt | Detail |",
		"|---|---|---:|:--:|---|",
	)
	for _, b := range breakdownRows(c["breakdown"]) {
		out = append(out, "| "+b.Group+" | `"+b.Kpi+"` | "+strconv.Itoa(b.Score)+" | "+strconv.Itoa(b.Debt)+" | "+b.Detail+" |")
	}
	out = append(out, "", "## Friction-debt work-list", "")

	worklist := append([]KPI{}, p.KPIs...)
	sort.SliceStable(worklist, func(i, j int) bool {
		return len(worklist[i].Defects) > len(worklist[j].Defects)
	})
	anyDefect := false
	for _, k := range worklist {
		if len(k.Defects) == 0 {
			continue
		}
		anyDefect = true
		out = append(out, "### `"+k.Kpi+"` ("+k.Group+") — "+strconv.Itoa(len(k.Defects))+" defect(s), score "+strconv.Itoa(k.Score))
		for _, it := range k.Defects {
			out = append(out, "- "+it)
		}
		out = append(out, "")
	}
	if !anyDefect {
		out = append(out, "No friction-debt: an agent can discover, adopt, and build on fak with no "+
			"missing affordance. 🎉", "")
	}
	return strings.Join(out, "\n") + "\n"
}

// ---------------------------------------------------------------------------
// Compare — the frontier/debt trend vs a prior --json baseline.
// ---------------------------------------------------------------------------

// Compare reports the experience-frontier (+35% goal) and friction-debt (>=2x reduction) deltas
// of the current payload against a prior --json baseline. No trailing newline (the caller's
// Fprintln supplies it).
func Compare(cur Payload, baseline map[string]any) string {
	b := mapOf(baseline["corpus"])
	c := cur.Corpus

	bf := asInt(b["experience_frontier"])
	cf := corpusInt(c, "experience_frontier")
	fdelta := cf - bf
	var fpct string
	if bf != 0 {
		fpct = fmt.Sprintf("%+.0f%%", 100.0*float64(fdelta)/float64(bf))
	} else if cf != 0 {
		fpct = "new"
	} else {
		fpct = "+0%"
	}
	bd := asInt(b["friction_debt"])
	cd := corpusInt(c, "friction_debt")
	bo := asFloat(b["score"])
	co := corpusFloat(c, "score")
	ratio := "∞ (zero)"
	if cd != 0 {
		ratio = fmt.Sprintf("%.1f×", float64(bd)/float64(cd))
	}

	lines := []string{
		"experience-frontier: " + strconv.Itoa(bf) + " -> " + strconv.Itoa(cf) + "   (" + fmt.Sprintf("%+d", fdelta) + ", " + fpct + ")   [unbounded — the headline]",
		"friction-debt:       " + strconv.Itoa(bd) + " -> " + strconv.Itoa(cd) + "   (" + ratio + " fewer defects)",
		"baseline score:      " + ftoa(bo) + "/100 -> " + ftoa(co) + "/100   (" + fmt.Sprintf("%+.1f", co-bo) + ")",
	}
	bbt := numbermap.Ints(b["frontier_by_term"], asInt)
	cbt := numbermap.Ints(c["frontier_by_term"], asInt)
	for _, dim := range frontierDims {
		ov, nv := bbt[dim], cbt[dim]
		if ov != 0 || nv != 0 {
			lines = append(lines, fmt.Sprintf("  frontier:%-22s %d -> %d", dim, ov, nv))
		}
	}
	gb := numbermap.Ints(b["debt_by_group"], asInt)
	gc := numbermap.Ints(c["debt_by_group"], asInt)
	for _, gp := range groups {
		lines = append(lines, fmt.Sprintf("  debt:%-10s %d -> %d", gp, gb[gp], gc[gp]))
	}

	goal := float64(bf) * 1.35
	if bf != 0 && float64(cf) >= goal {
		lines = append(lines, "VERDICT: experience-frontier +35% achieved ("+strconv.Itoa(bf)+" -> "+strconv.Itoa(cf)+"; goal >= "+fmt0(goal)+").")
	} else if bf != 0 {
		lines = append(lines, "VERDICT: not yet +35% on the frontier — at "+strconv.Itoa(cf)+", need >= "+fmt0(goal)+
			" (add ~"+fmt0(goal-float64(cf))+" more points: a recipe ×"+strconv.Itoa(frontierUnits["integration_recipes"])+
			", a config ×"+strconv.Itoa(frontierUnits["harness_configs"])+", a refusal ×"+strconv.Itoa(frontierUnits["refusal_recoveries"])+").")
	} else {
		lines = append(lines, "VERDICT: frontier baseline "+strconv.Itoa(cf)+" (no prior frontier to compare).")
	}
	target := bd / 2
	if target < 0 {
		target = 0
	}
	if cd <= target {
		lines = append(lines, "  (gate) >=2x friction-debt reduction held ("+strconv.Itoa(bd)+" -> "+strconv.Itoa(cd)+").")
	} else {
		lines = append(lines, "  (gate) friction-debt not yet 2x — need <= "+strconv.Itoa(target)+" (now "+strconv.Itoa(cd)+").")
	}
	return strings.Join(lines, "\n")
}

func mapOf(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}
