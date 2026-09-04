package debtlane

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
)

// WavePlanSchema is the canonical schema identifier for concurrent safe wave plans.
const WavePlanSchema = "fak.debt-orchestrator-wave-plan.v1"

// WaveSafety describes the concurrency safety classification of a wave.
type WaveSafety string

const (
	// WaveSafetyDisjointLeaf indicates all units of work in this wave are pairwise tree-disjoint
	// with verified zero cross-package import contention.
	WaveSafetyDisjointLeaf WaveSafety = "pairwise_tree_disjoint"

	// WaveSafetySerialSingleton indicates a critical/core leaf with high blast radius that MUST
	// execute alone as a dedicated single-worker wave without concurrent siblings.
	WaveSafetySerialSingleton WaveSafety = "serial_singleton"
)

// Wave represents one execution wave of concurrent-safe debt lanes.
type Wave struct {
	Index             int        `json:"index"`              // 1-based wave sequence number
	ID                string     `json:"id"`                 // e.g. "wave-1"
	Safety            WaveSafety `json:"safety"`             // pairwise_tree_disjoint or serial_singleton
	Lanes             []DebtLane `json:"lanes"`              // Debt lanes allocated to this wave
	LaneNames         []string   `json:"lane_names"`         // Slice of lane identifiers
	Paths             []string   `json:"paths"`              // Package paths (e.g. "internal/faultlab")
	WaveSize          int        `json:"wave_size"`          // Number of concurrent workers in this wave
	TotalDebt         float64    `json:"total_debt"`         // Sum of TotalDebt across lanes in this wave
	DebtPrincipal     float64    `json:"debt_principal"`     // Sum of DebtPrincipal across lanes
	CarryingCost      float64    `json:"carrying_cost"`      // Sum of CarryingCost across lanes
	PotentialRealized float64    `json:"potential_realized"` // Realized points gain if target reached
}

// WavePlan is the full multi-wave campaign dispatch plan.
type WavePlan struct {
	Schema           string   `json:"schema"`
	Workspace        string   `json:"workspace"`
	TargetGrade      string   `json:"target_grade,omitempty"`
	TargetPoints     float64  `json:"target_points,omitempty"`
	WaveSizeCap      int      `json:"wave_size_cap"`
	TotalWaves       int      `json:"total_waves"`
	PlannedLanes     int      `json:"planned_lanes"`
	ExcludedLanes    []string `json:"excluded_lanes,omitempty"`
	HeldLanes        []string `json:"held_lanes,omitempty"`
	TotalDebtInPlan  float64  `json:"total_debt_in_plan"`
	PotentialPoints  float64  `json:"potential_points"`
	StartingGrade    string   `json:"starting_grade"`
	ProjectedGrade   string   `json:"projected_grade"`
	StartingPercent  float64  `json:"starting_percent"`
	ProjectedPercent float64  `json:"projected_percent"`
	Waves            []Wave   `json:"waves"`
}

// WavePlanOptions configures wave generation.
type WavePlanOptions struct {
	WaveSize       int
	MaxWaves       int
	TargetGrade    string
	TargetPoints   float64
	ExcludedLanes  []string
	AutoDetectHeld bool
	// Graph override allows injecting an import dependency graph in tests.
	Graph map[string]map[string]struct{}
}

// PlanWaves partitions candidate debt lanes into ordered, collision-free, concurrent-safe waves.
func PlanWaves(report Report, opts WavePlanOptions) WavePlan {
	waveSizeCap := opts.WaveSize
	if waveSizeCap <= 0 {
		waveSizeCap = 4
	}

	excludedMap := make(map[string]bool)
	for _, l := range opts.ExcludedLanes {
		trimmed := strings.ToLower(strings.TrimSpace(l))
		if trimmed != "" {
			excludedMap[trimmed] = true
		}
	}

	var heldLanes []string
	if opts.AutoDetectHeld && report.Workspace != "" {
		discovered, _ := DiscoverHeldLanes(report.Workspace)
		for _, l := range discovered {
			lower := strings.ToLower(strings.TrimSpace(l))
			if lower != "" {
				heldLanes = append(heldLanes, l)
				excludedMap[lower] = true
			}
		}
	}

	graph := opts.Graph
	if graph == nil && report.Workspace != "" {
		graph, _ = buildInternalImportGraph(report.Workspace)
	}

	// Gather candidates: lanes with active maturity debt not excluded or held.
	candidatesPool := report.Lanes
	if len(candidatesPool) == 0 {
		candidatesPool = report.Hotspots
	}

	var candidates []DebtLane
	for _, l := range candidatesPool {
		if l.MaturityGap <= 0.05 && l.TotalDebt <= 0.05 {
			continue
		}
		if excludedMap[strings.ToLower(l.Lane)] {
			continue
		}
		candidates = append(candidates, l)
	}

	// Sort candidates: worst-first total debt, carrying cost, maturity gap.
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].TotalDebt != candidates[j].TotalDebt {
			return candidates[i].TotalDebt > candidates[j].TotalDebt
		}
		if candidates[i].CarryingCost != candidates[j].CarryingCost {
			return candidates[i].CarryingCost > candidates[j].CarryingCost
		}
		if candidates[i].MaturityGap != candidates[j].MaturityGap {
			return candidates[i].MaturityGap > candidates[j].MaturityGap
		}
		return candidates[i].Lane < candidates[j].Lane
	})

	var serialWaves []Wave
	var parallelWaves []Wave

	for _, c := range candidates {
		if isSerialSingleton(c) {
			sw := Wave{
				Safety: WaveSafetySerialSingleton,
				Lanes:  []DebtLane{c},
			}
			serialWaves = append(serialWaves, sw)
			continue
		}

		// Find first open parallel wave that can safely accommodate candidate c.
		placed := false
		for i := range parallelWaves {
			w := &parallelWaves[i]
			if len(w.Lanes) >= waveSizeCap {
				continue
			}
			conflict := false
			for _, existing := range w.Lanes {
				if TreesOverlap(c.UnitOfWork, existing.UnitOfWork) {
					conflict = true
					break
				}
				if ImportsContend(c.Lane, existing.Lane, graph) {
					conflict = true
					break
				}
			}
			if !conflict {
				w.Lanes = append(w.Lanes, c)
				placed = true
				break
			}
		}

		if !placed {
			newW := Wave{
				Safety: WaveSafetyDisjointLeaf,
				Lanes:  []DebtLane{c},
			}
			parallelWaves = append(parallelWaves, newW)
		}
	}

	// Disjoint parallel cohorts first (high volume / rapid lift), then serial core singletons.
	allWaves := append(parallelWaves, serialWaves...)

	var targetPct float64
	var hasTargetGrade bool
	if opts.TargetGrade != "" {
		targetPct, hasTargetGrade = ParseTargetGrade(opts.TargetGrade)
	}

	targetPoints := opts.TargetPoints
	startingPct := report.ProductionGrade.GradePercent
	startingGrade := report.ProductionGrade.GradeLetter

	// If already at or above target grade and no explicit target points, 0 waves needed.
	if hasTargetGrade && targetPoints <= 0 && startingPct >= targetPct {
		return WavePlan{
			Schema:           WavePlanSchema,
			Workspace:        report.Workspace,
			TargetGrade:      opts.TargetGrade,
			TargetPoints:     opts.TargetPoints,
			WaveSizeCap:      waveSizeCap,
			TotalWaves:       0,
			PlannedLanes:     0,
			ExcludedLanes:    opts.ExcludedLanes,
			HeldLanes:        heldLanes,
			TotalDebtInPlan:  0,
			PotentialPoints:  0,
			StartingGrade:    startingGrade,
			ProjectedGrade:   startingGrade,
			StartingPercent:  startingPct,
			ProjectedPercent: startingPct,
			Waves:            nil,
		}
	}

	var selectedWaves []Wave
	var totalDebtInPlan, potentialPoints float64
	var plannedCount int

	for i := range allWaves {
		w := allWaves[i]
		w.Index = len(selectedWaves) + 1
		w.ID = fmt.Sprintf("wave-%d", w.Index)
		w.WaveSize = len(w.Lanes)
		w.LaneNames = make([]string, 0, len(w.Lanes))
		w.Paths = make([]string, 0, len(w.Lanes))

		var waveDebt, wavePrincipal, waveCarrying, wavePotential float64
		for _, l := range w.Lanes {
			w.LaneNames = append(w.LaneNames, l.Lane)
			w.Paths = append(w.Paths, strings.ReplaceAll(l.UnitOfWork, "\\", "/"))
			waveDebt += l.TotalDebt
			wavePrincipal += l.DebtPrincipal
			waveCarrying += l.CarryingCost
			wavePotential += (l.TargetMaturity - l.Maturity) * l.Weight
		}

		w.TotalDebt = math.Round(waveDebt*10) / 10
		w.DebtPrincipal = math.Round(wavePrincipal*10) / 10
		w.CarryingCost = math.Round(waveCarrying*10) / 10
		w.PotentialRealized = math.Round(wavePotential*10) / 10

		selectedWaves = append(selectedWaves, w)
		totalDebtInPlan += w.TotalDebt
		potentialPoints += w.PotentialRealized
		plannedCount += w.WaveSize

		if opts.MaxWaves > 0 && len(selectedWaves) >= opts.MaxWaves {
			break
		}

		if hasTargetGrade || targetPoints > 0 {
			gradeMet := true
			if hasTargetGrade && report.ProductionGrade.DenominatorPoints > 0 {
				curRealized := report.ProductionGrade.RealizedPoints + potentialPoints
				curPct := math.Round((curRealized/report.ProductionGrade.DenominatorPoints)*1000) / 10
				gradeMet = curPct >= targetPct
			}
			pointsMet := true
			if targetPoints > 0 {
				pointsMet = totalDebtInPlan >= targetPoints || potentialPoints >= targetPoints
			}
			if gradeMet && pointsMet {
				break
			}
		}
	}

	totalDebtInPlan = math.Round(totalDebtInPlan*10) / 10
	potentialPoints = math.Round(potentialPoints*10) / 10

	projectedPct := startingPct
	projectedGrade := startingGrade

	if report.ProductionGrade.DenominatorPoints > 0 {
		newRealized := report.ProductionGrade.RealizedPoints + potentialPoints
		projectedPct = math.Round((newRealized/report.ProductionGrade.DenominatorPoints)*1000) / 10
		projectedGrade = GradeLetter(projectedPct)
	}

	return WavePlan{
		Schema:           WavePlanSchema,
		Workspace:        report.Workspace,
		TargetGrade:      opts.TargetGrade,
		TargetPoints:     opts.TargetPoints,
		WaveSizeCap:      waveSizeCap,
		TotalWaves:       len(selectedWaves),
		PlannedLanes:     plannedCount,
		ExcludedLanes:    opts.ExcludedLanes,
		HeldLanes:        heldLanes,
		TotalDebtInPlan:  totalDebtInPlan,
		PotentialPoints:  potentialPoints,
		StartingGrade:    startingGrade,
		ProjectedGrade:   projectedGrade,
		StartingPercent:  startingPct,
		ProjectedPercent: projectedPct,
		Waves:            selectedWaves,
	}
}

// ParseTargetGrade parses target grade specifications into a target percentage.
// Accepts: "80%", "85%", "90", "A", "B", "C", "D", "Grade A", "Grade B", etc.
func ParseTargetGrade(s string) (float64, bool) {
	raw := strings.ToUpper(strings.TrimSpace(s))
	if raw == "" {
		return 0, false
	}
	raw = strings.TrimPrefix(raw, "GRADE ")
	raw = strings.TrimPrefix(raw, "GRADE")
	raw = strings.TrimSpace(raw)
	switch raw {
	case "A", "A+":
		return 90.0, true
	case "B", "B+":
		return 80.0, true
	case "C", "C+":
		return 70.0, true
	case "D", "D+":
		return 60.0, true
	}
	raw = strings.TrimSuffix(raw, "%")
	raw = strings.TrimSpace(raw)
	val, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, false
	}
	if val > 0 && val <= 1.0 {
		val = val * 100.0
	}
	if val > 0 && val <= 100.0 {
		return val, true
	}
	return 0, false
}

func isSerialSingleton(l DebtLane) bool {
	if l.Criticality == CriticalityCore {
		if l.Interest.Band == InterestCritical || l.Evidence.DependentsCount >= 10 {
			return true
		}
		lane := strings.ToLower(l.Lane)
		switch lane {
		case "abi", "kernel", "adjudicator", "policy", "gateway", "vdso", "shipgate", "architest":
			return true
		}
	}
	return false
}

// TreesOverlap reports whether two relative directory paths geometrically overlap.
func TreesOverlap(p1, p2 string) bool {
	s1 := strings.ReplaceAll(strings.TrimSpace(p1), "\\", "/")
	s2 := strings.ReplaceAll(strings.TrimSpace(p2), "\\", "/")
	s1 = strings.TrimSuffix(s1, "/**")
	s1 = strings.TrimSuffix(s1, "/*")
	s2 = strings.TrimSuffix(s2, "/**")
	s2 = strings.TrimSuffix(s2, "/*")
	if s1 == "" || s2 == "" {
		return true // conservative on empty / unknown scope
	}
	c1 := strings.ReplaceAll(filepath.Clean(s1), "\\", "/")
	c2 := strings.ReplaceAll(filepath.Clean(s2), "\\", "/")
	if c1 == "." || c2 == "." {
		return true
	}
	if strings.EqualFold(c1, c2) {
		return true
	}
	l1 := strings.ToLower(c1)
	l2 := strings.ToLower(c2)
	if strings.HasPrefix(l1, l2+"/") || strings.HasPrefix(l2, l1+"/") {
		return true
	}
	return false
}

// ImportsContend reports whether either lane directly imports the other from the internal package graph.
func ImportsContend(lane1, lane2 string, graph map[string]map[string]struct{}) bool {
	if graph == nil {
		return false
	}
	if deps1 := getDeps(graph, lane1); hasDep(deps1, lane2) {
		return true
	}
	if deps2 := getDeps(graph, lane2); hasDep(deps2, lane1) {
		return true
	}
	return false
}

func getDeps(graph map[string]map[string]struct{}, lane string) map[string]struct{} {
	if deps, ok := graph[lane]; ok {
		return deps
	}
	lower := strings.ToLower(strings.TrimSpace(lane))
	if deps, ok := graph[lower]; ok {
		return deps
	}
	for k, v := range graph {
		if strings.ToLower(k) == lower {
			return v
		}
	}
	return nil
}

func hasDep(deps map[string]struct{}, lane string) bool {
	if deps == nil {
		return false
	}
	if _, ok := deps[lane]; ok {
		return true
	}
	lower := strings.ToLower(strings.TrimSpace(lane))
	if _, ok := deps[lower]; ok {
		return true
	}
	for k := range deps {
		if strings.ToLower(k) == lower {
			return true
		}
	}
	return false
}

// DiscoverHeldLanes inspects the workspace lane journal (.dos/lane-journal.jsonl)
// and returns all currently held lane leases.
func DiscoverHeldLanes(workspace string) ([]string, error) {
	journalPath := filepath.Join(workspace, ".dos", "lane-journal.jsonl")
	f, err := os.Open(journalPath)
	if err != nil {
		return nil, nil // No journal or cannot open; fail-open with nil
	}
	defer f.Close()

	held := make(map[string]bool)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "{") {
			continue
		}
		var row struct {
			Op   string `json:"op"`
			Lane string `json:"lane"`
		}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		lane := strings.TrimSpace(row.Lane)
		if lane == "" {
			continue
		}
		switch row.Op {
		case "ACQUIRE":
			held[lane] = true
		case "RELEASE":
			delete(held, lane)
		}
	}

	result := make([]string, 0, len(held))
	for lane := range held {
		result = append(result, lane)
	}
	sort.Strings(result)
	return result, nil
}

// RenderWaves formats a WavePlan as human-readable terminal text.
func RenderWaves(plan WavePlan, pg ProductionGrade) string {
	var b strings.Builder
	b.WriteString("=== FAK DEBT ORCHESTRATOR: CONCURRENT SAFE WAVE PLAN ===\n")
	b.WriteString(fmt.Sprintf("Campaign Projection: Grade %s (%.1f%%) → Grade %s (%.1f%%) [Potential Lift: +%.1f pts]\n",
		plan.StartingGrade, plan.StartingPercent,
		plan.ProjectedGrade, plan.ProjectedPercent,
		plan.PotentialPoints,
	))
	b.WriteString(fmt.Sprintf("Plan Summary:        %d wave(s) planned · %d total worker slot(s) · %.1f total debt pts\n",
		plan.TotalWaves, plan.PlannedLanes, plan.TotalDebtInPlan,
	))
	b.WriteString(fmt.Sprintf("Concurrency Cap:     Max %d concurrent workers per wave · Concurrency safety: pairwise tree-disjoint\n",
		plan.WaveSizeCap,
	))

	if len(plan.HeldLanes) > 0 {
		b.WriteString(fmt.Sprintf("Held Lanes Excluded: %d active lease(s) held in workspace (%s)\n",
			len(plan.HeldLanes), strings.Join(plan.HeldLanes, ", "),
		))
	}
	b.WriteString("\n")

	for _, w := range plan.Waves {
		safetyBadge := "DISJOINT PARALLEL"
		if w.Safety == WaveSafetySerialSingleton {
			safetyBadge = "SERIAL CORE SINGLETON"
		}

		b.WriteString(fmt.Sprintf("--- %s: %s (%d worker(s) · %.1f debt pts · +%.1f potential realized) ---\n",
			strings.ToUpper(w.ID), safetyBadge, w.WaveSize, w.TotalDebt, w.PotentialRealized,
		))

		tw := tabwriter.NewWriter(&b, 2, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "  SLOT\tLANE\tDIRECTORY\tCRITICALITY\tGAP\tDEBT\tNEXT ACTION")
		for j, l := range w.Lanes {
			fmt.Fprintf(tw, "  Worker %d\t%s\t%s\t%s\t%.1f\t%.1f\t%s\n",
				j+1,
				l.Lane,
				strings.ReplaceAll(l.UnitOfWork, "\\", "/"),
				l.Criticality,
				l.MaturityGap,
				l.TotalDebt,
				truncProse(l.NextAction, 55),
			)
		}
		tw.Flush()
		b.WriteString("\n")
	}

	b.WriteString("Dispatch Guide: Launch each wave's workers concurrently via `task` tool with strict package boundaries.\n")
	b.WriteString("Verification:   Arbitrate lane lease (`dos arbitrate`) before dispatch; commit by explicit path on harvest.\n")
	return b.String()
}

// MarkdownWaves formats a WavePlan as Markdown documentation.
func MarkdownWaves(plan WavePlan, pg ProductionGrade) string {
	var b bytes.Buffer
	b.WriteString("## Concurrent Safe Debt Retirement Wave Plan\n\n")
	b.WriteString(fmt.Sprintf("> **Campaign Lift:** Grade %s (%.1f%%) → Grade %s (%.1f%%) · **Potential Realized:** `+%.1f` pts · **Waves:** `%d`\n\n",
		plan.StartingGrade, plan.StartingPercent,
		plan.ProjectedGrade, plan.ProjectedPercent,
		plan.PotentialPoints, plan.TotalWaves,
	))

	b.WriteString(fmt.Sprintf("- **Planned waves:** `%d` waves across `%d` targeted units\n", plan.TotalWaves, plan.PlannedLanes))
	b.WriteString(fmt.Sprintf("- **Debt in plan:** `%.1f` points (principal + carrying cost)\n", plan.TotalDebtInPlan))
	b.WriteString(fmt.Sprintf("- **Concurrency safety:** verified pairwise tree-disjoint with serial core isolation\n\n"))

	for _, w := range plan.Waves {
		safetyBadge := "Pairwise Tree-Disjoint"
		if w.Safety == WaveSafetySerialSingleton {
			safetyBadge = "Serial Core Singleton (Runs Solo)"
		}

		b.WriteString(fmt.Sprintf("### %s (%s · %d Workers · %.1f Debt Pts · +%.1f Realized)\n\n",
			strings.Title(w.ID), safetyBadge, w.WaveSize, w.TotalDebt, w.PotentialRealized,
		))

		b.WriteString("| Worker | Lane | Path | Criticality | Maturity Gap | Total Debt | Next Action |\n")
		b.WriteString("|---|---|---|---|---:|---:|---|\n")
		for j, l := range w.Lanes {
			b.WriteString(fmt.Sprintf("| Worker %d | `%s` | `%s` | %s | %.1f | **%.1f** | %s |\n",
				j+1, l.Lane, strings.ReplaceAll(l.UnitOfWork, "\\", "/"), l.Criticality, l.MaturityGap, l.TotalDebt, l.NextAction,
			))
		}
		b.WriteString("\n")
	}

	return b.String()
}
