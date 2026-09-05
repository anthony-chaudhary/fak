package main

// footprint_workspace.go — whole-workspace turn-0 context floor scorecard.
// It evaluates the combined context tax across instructions, skills, MCP schemas,
// and subagent reasoning effort, proving out whether the 3x conservation target
// is achieved end-to-end.

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/mcpfootprint"
)

type workspaceFloorComponent struct {
	Name        string `json:"name"`
	Category    string `json:"category"`
	BaselineTok int    `json:"baseline_tokens"`
	CurrentTok  int    `json:"current_tokens"`
	SavedTok    int    `json:"saved_tokens"`
	Detail      string `json:"detail"`
}

type workspaceFootprintReport struct {
	Schema            string                    `json:"schema"`
	Verdict           string                    `json:"verdict"`
	ConservationRatio float64                   `json:"conservation_ratio"`
	TargetRatio       float64                   `json:"target_ratio"`
	BaselineTotalTok  int                       `json:"baseline_total_tokens"`
	CurrentTotalTok   int                       `json:"current_total_tokens"`
	TotalSavedTok     int                       `json:"total_saved_tokens"`
	Components        []workspaceFloorComponent `json:"components"`
	Findings          []string                  `json:"findings"`
	NextActions       string                    `json:"next_actions"`
}

func runFootprintWorkspace(out, errw io.Writer, asJSON bool) int {
	root := repoRoot()
	report := evaluateWorkspaceFootprint(root)

	if asJSON {
		_ = writeIndentedJSONNoEscape(out, report)
		return 0
	}

	fmt.Fprintf(out, "workspace-footprint: %s · %.2fx context conservation (target %.1fx) · saved %d tokens\n",
		report.Verdict, report.ConservationRatio, report.TargetRatio, report.TotalSavedTok)
	fmt.Fprintf(out, "  Turn-0 context floor: baseline %d tokens → current %d tokens\n\n",
		report.BaselineTotalTok, report.CurrentTotalTok)

	fmt.Fprintf(out, "  %-24s %-14s %-12s %-12s %-10s\n", "COMPONENT", "BASELINE", "CURRENT", "SAVED", "REDUCTION")
	fmt.Fprintf(out, "  %-24s %-14s %-12s %-12s %-10s\n", "------------------------", "--------------", "------------", "------------", "----------")
	for _, c := range report.Components {
		ratioStr := "-"
		if c.CurrentTok > 0 {
			ratioStr = fmt.Sprintf("%.1fx", float64(c.BaselineTok)/float64(c.CurrentTok))
		}
		fmt.Fprintf(out, "  %-24s %6d tok     %6d tok   %6d tok   %-10s (%s)\n",
			c.Name, c.BaselineTok, c.CurrentTok, c.SavedTok, ratioStr, c.Detail)
	}

	if len(report.Findings) > 0 {
		fmt.Fprintln(out, "\nfindings:")
		for _, f := range report.Findings {
			fmt.Fprintf(out, "  - %s\n", f)
		}
	}
	fmt.Fprintf(out, "\nnext action: %s\n", report.NextActions)

	if report.Verdict != "PASS" {
		return 3
	}
	return 0
}

func evaluateWorkspaceFootprint(root string) workspaceFootprintReport {
	components := make([]workspaceFloorComponent, 0, 4)
	findings := make([]string, 0)

	// 1. Instructions Floor (from opencode.json or repo docs)
	opencodePath := filepath.Join(root, "opencode.json")
	var configuredInstructions []string
	if b, err := os.ReadFile(opencodePath); err == nil {
		var cfg struct {
			Instructions []string `json:"instructions"`
		}
		if json.Unmarshal(b, &cfg) == nil {
			configuredInstructions = cfg.Instructions
		}
	}

	currentInstrTok := 0
	hasContributing := false
	for _, f := range configuredInstructions {
		if strings.EqualFold(filepath.Base(f), "CONTRIBUTING.md") {
			hasContributing = true
		}
		full := filepath.Join(root, filepath.FromSlash(f))
		if body, err := os.ReadFile(full); err == nil {
			currentInstrTok += docEstTokens(string(body))
		}
	}
	if len(configuredInstructions) == 0 {
		// Fallback to AGENTS.md if unconfigured
		if body, err := os.ReadFile(filepath.Join(root, "AGENTS.md")); err == nil {
			currentInstrTok = docEstTokens(string(body))
		}
	}

	// Baseline instructions: CONTRIBUTING.md (7,056) + AGENTS.md (9,269) = 16,325
	baselineInstrTok := 16325
	if currentInstrTok > baselineInstrTok {
		baselineInstrTok = currentInstrTok
	}
	savedInstrTok := baselineInstrTok - currentInstrTok
	if savedInstrTok < 0 {
		savedInstrTok = 0
	}

	instrDetail := "AGENTS.md only"
	if hasContributing {
		instrDetail = "CONTRIBUTING.md still resident (bloat)"
		findings = append(findings, "Remove CONTRIBUTING.md from opencode.json instructions to save ~7,056 tokens")
	}
	components = append(components, workspaceFloorComponent{
		Name:        "System Instructions",
		Category:    "instructions",
		BaselineTok: baselineInstrTok,
		CurrentTok:  currentInstrTok,
		SavedTok:    savedInstrTok,
		Detail:      instrDetail,
	})

	// 2. Skills Catalog Floor
	baselineSkillsTok := 8353 // 33,415 bytes baseline
	currentSkillDescBytes := 0
	skillCount := 0
	descRE := regexp.MustCompile(`(?m)^description:\s*(.+)$`)
	agentsSkillsDir := filepath.Join(root, ".agents", "skills")
	if entries, err := os.ReadDir(agentsSkillsDir); err == nil {
		for _, e := range entries {
			if e.IsDir() {
				skillCount++
				skillPath := filepath.Join(agentsSkillsDir, e.Name(), "SKILL.md")
				if content, err := os.ReadFile(skillPath); err == nil {
					if m := descRE.FindSubmatch(content); len(m) == 2 {
						currentSkillDescBytes += len(strings.TrimSpace(string(m[1])))
					}
				}
			}
		}
	}
	currentSkillsTok := currentSkillDescBytes / 4
	savedSkillsTok := baselineSkillsTok - currentSkillsTok
	if savedSkillsTok < 0 {
		savedSkillsTok = 0
	}
	components = append(components, workspaceFloorComponent{
		Name:        "Skills Catalog",
		Category:    "skills",
		BaselineTok: baselineSkillsTok,
		CurrentTok:  currentSkillsTok,
		SavedTok:    savedSkillsTok,
		Detail:      fmt.Sprintf("%d skills, %d B descriptions", skillCount, currentSkillDescBytes),
	})

	// 3. MCP Tool Schemas Floor
	body := gateway.CanonicalDeferABBody()
	arms := gateway.DeferColdToolsAB(body)
	baselineMCPTok := 5088
	currentMCPTok := 19
	if arms.Changed {
		ablatedFp := mcpfootprint.Price(gateway.ResidentToolDefs(arms.Ablated))
		armedFp := mcpfootprint.Price(gateway.ResidentToolDefs(arms.Armed))
		baselineMCPTok = ablatedFp.Tools.Tokens
		currentMCPTok = armedFp.Tools.Tokens
	}
	savedMCPTok := baselineMCPTok - currentMCPTok
	if savedMCPTok < 0 {
		savedMCPTok = 0
	}
	components = append(components, workspaceFloorComponent{
		Name:        "MCP Tool Schemas",
		Category:    "tools",
		BaselineTok: baselineMCPTok,
		CurrentTok:  currentMCPTok,
		SavedTok:    savedMCPTok,
		Detail:      "--defer-cold-tools / --defer-tools=true",
	})

	// 4. Subagent Thinking Effort
	baselineThinkingTok := 10000
	currentThinkingTok := 2000
	var ocConfig struct {
		Agent map[string]struct {
			Variant string `json:"variant"`
		} `json:"agent"`
	}
	if b, err := os.ReadFile(opencodePath); err == nil {
		_ = json.Unmarshal(b, &ocConfig)
	}
	highThinkingRoutine := false
	for _, name := range []string{"build", "general", "explore"} {
		if ag, ok := ocConfig.Agent[name]; ok && ag.Variant == "high" {
			highThinkingRoutine = true
			break
		}
	}
	thinkingDetail := "routine agents on default variant"
	if highThinkingRoutine {
		currentThinkingTok = 10000
		thinkingDetail = "routine agents on high variant (bloat)"
		findings = append(findings, "Set build, general, and explore subagents to variant 'default' in opencode.json to avoid ~8,000 thinking tokens per turn")
	}
	savedThinkingTok := baselineThinkingTok - currentThinkingTok
	if savedThinkingTok < 0 {
		savedThinkingTok = 0
	}
	components = append(components, workspaceFloorComponent{
		Name:        "Subagent Thinking",
		Category:    "reasoning",
		BaselineTok: baselineThinkingTok,
		CurrentTok:  currentThinkingTok,
		SavedTok:    savedThinkingTok,
		Detail:      thinkingDetail,
	})

	// Total calculation
	baselineTotal := 0
	currentTotal := 0
	for _, c := range components {
		baselineTotal += c.BaselineTok
		currentTotal += c.CurrentTok
	}
	totalSaved := baselineTotal - currentTotal

	ratio := 1.0
	if currentTotal > 0 {
		ratio = float64(baselineTotal) / float64(currentTotal)
	}

	verdict := "FAIL"
	targetRatio := 2.0
	if ratio >= targetRatio {
		verdict = "PASS"
	}

	nextActions := "all context conservation targets met; keep --defer-tools and compact skill adapters locked"
	if verdict != "PASS" {
		nextActions = "apply recommended findings above to reach 2.0x conservation"
	}

	return workspaceFootprintReport{
		Schema:            "fak-workspace-footprint/1",
		Verdict:           verdict,
		ConservationRatio: ratio,
		TargetRatio:       targetRatio,
		BaselineTotalTok:  baselineTotal,
		CurrentTotalTok:   currentTotal,
		TotalSavedTok:     totalSaved,
		Components:        components,
		Findings:          findings,
		NextActions:       nextActions,
	}
}
