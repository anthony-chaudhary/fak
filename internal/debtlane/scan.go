package debtlane

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
)

var (
	laneTreeRe = regexp.MustCompile(`^(?:")?([A-Za-z0-9_-]+)(?:")?\s*=\s*(?:\[\s*)?"(?:internal|pkg|platform|tools|cmd)/([A-Za-z0-9_-]+)`)
	importRe   = regexp.MustCompile(`github\.com/anthony-chaudhary/(?:fak|fak-private)/(?:internal|pkg|platform|tools|cmd)/([A-Za-z0-9_-]+)`)
)

func resolvePrivateRoot(fakRoot, explicit string) string {
	if explicit != "" {
		if info, err := os.Stat(explicit); err == nil && info.IsDir() {
			return explicit
		}
	}
	if env := os.Getenv("FAK_PRIVATE_ROOT"); env != "" {
		if info, err := os.Stat(env); err == nil && info.IsDir() {
			return env
		}
	}
	candidate := filepath.Join(filepath.Dir(fakRoot), "fak-private")
	if info, err := os.Stat(candidate); err == nil && info.IsDir() {
		return candidate
	}
	return ""
}

func resolveFakRoot(privRoot, explicit string) string {
	if explicit != "" {
		if info, err := os.Stat(explicit); err == nil && info.IsDir() {
			return explicit
		}
	}
	if env := os.Getenv("FAK_ROOT"); env != "" {
		if info, err := os.Stat(env); err == nil && info.IsDir() {
			return env
		}
	}
	candidate := filepath.Join(filepath.Dir(privRoot), "fak")
	if info, err := os.Stat(candidate); err == nil && info.IsDir() {
		return candidate
	}
	return ""
}

// Scan evaluates debt lanes across the workspace or against provided facts.
func Scan(opts Options) (Report, error) {
	root := opts.WorkspaceRoot
	if root == "" {
		root = "."
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return Report{}, err
	}

	targetRepo := strings.ToLower(strings.TrimSpace(opts.TargetRepo))
	if targetRepo == "" {
		targetRepo = "fak"
		if filepath.Base(absRoot) == "fak-private" {
			targetRepo = "fak-private"
		}
	}

	evalTime := time.Now().UTC()
	if opts.Clock != nil {
		evalTime = opts.Clock().UTC()
	}

	var allLanes []DebtLane
	if len(opts.Facts) > 0 {
		allLanes = make([]DebtLane, len(opts.Facts))
		copy(allLanes, opts.Facts)
		for i := range allLanes {
			recomputeLane(&allLanes[i])
		}
	} else if targetRepo == "both" {
		privRoot := resolvePrivateRoot(absRoot, opts.PrivateRoot)
		if privRoot == "" {
			return Report{}, fmt.Errorf("fak-private repository not found: specify private root or set FAK_PRIVATE_ROOT")
		}
		fakLanes, err := discoverLanesFromDisk(absRoot, privRoot)
		if err != nil {
			return Report{}, fmt.Errorf("scan fak: %w", err)
		}
		for i := range fakLanes {
			fakLanes[i].Repo = "fak"
		}
		privLanes, err := discoverLanesFromDisk(privRoot, absRoot)
		if err != nil {
			return Report{}, fmt.Errorf("scan fak-private: %w", err)
		}
		for i := range privLanes {
			privLanes[i].Repo = "fak-private"
		}

		// Cross-index dual-repo companions
		privMap := make(map[string]DebtLane)
		for _, pl := range privLanes {
			privMap[pl.Lane] = pl
		}
		for i := range fakLanes {
			if pl, ok := privMap[fakLanes[i].Lane]; ok {
				fakLanes[i].Related.CompanionRepo = "fak-private"
				fakLanes[i].Related.CompanionLane = pl.Lane
				fakLanes[i].Related.CompanionUnitOfWork = pl.UnitOfWork
			}
		}
		fakMap := make(map[string]DebtLane)
		for _, fl := range fakLanes {
			fakMap[fl.Lane] = fl
		}
		for i := range privLanes {
			if fl, ok := fakMap[privLanes[i].Lane]; ok {
				privLanes[i].Related.CompanionRepo = "fak"
				privLanes[i].Related.CompanionLane = fl.Lane
				privLanes[i].Related.CompanionUnitOfWork = fl.UnitOfWork
			}
		}

		allLanes = append(fakLanes, privLanes...)
	} else if targetRepo == "fak-private" {
		privRoot := absRoot
		if filepath.Base(absRoot) != "fak-private" {
			privRoot = resolvePrivateRoot(absRoot, opts.PrivateRoot)
			if privRoot == "" {
				return Report{}, fmt.Errorf("fak-private repository not found: specify private root or set FAK_PRIVATE_ROOT")
			}
		}
		fakRoot := resolveFakRoot(privRoot, "")
		allLanes, err = discoverLanesFromDisk(privRoot, fakRoot)
		if err != nil {
			return Report{}, err
		}
		for i := range allLanes {
			allLanes[i].Repo = "fak-private"
		}
		absRoot = privRoot
	} else {
		privRoot := resolvePrivateRoot(absRoot, opts.PrivateRoot)
		allLanes, err = discoverLanesFromDisk(absRoot, privRoot)
		if err != nil {
			return Report{}, err
		}
		for i := range allLanes {
			allLanes[i].Repo = "fak"
		}
	}

	// Calculate overall production grade over ALL discovered units of work
	// BEFORE applying optional display filters, so the denominator accurately
	// reflects the whole system.
	productionGrade := CalculateProductionGrade(allLanes)

	// Filter lanes for display if requested.
	filtered := make([]DebtLane, 0, len(allLanes))
	for _, l := range allLanes {
		if opts.LaneFilter != "" && !strings.EqualFold(l.Lane, opts.LaneFilter) {
			continue
		}
		if opts.QueryFilter != "" && !matchesQuery(l, opts.QueryFilter) {
			continue
		}
		if opts.HealthFilter != "" && !matchesHealth(l, opts.HealthFilter) {
			continue
		}
		if opts.CriticalityFilter != "" && !strings.EqualFold(string(l.Criticality), opts.CriticalityFilter) {
			continue
		}
		if opts.MinGap > 0 && l.MaturityGap < opts.MinGap {
			continue
		}
		filtered = append(filtered, l)
	}

	// Calculate interest summary.
	bands := map[string]int{
		string(InterestLow):      0,
		string(InterestModerate): 0,
		string(InterestHigh):     0,
		string(InterestCritical): 0,
	}
	var rateSum, maxRate float64
	var activeDebtCount int
	var totalPrincipal, totalCarryingCost, totalDebt float64

	for _, l := range allLanes {
		bands[string(l.Interest.Band)]++
		totalPrincipal += l.DebtPrincipal
		totalCarryingCost += l.CarryingCost
		totalDebt += l.TotalDebt
		if l.MaturityGap > 0 {
			rateSum += l.Interest.Rate
			activeDebtCount++
			if l.Interest.Rate > maxRate {
				maxRate = l.Interest.Rate
			}
		}
	}

	avgRate := 0.0
	if activeDebtCount > 0 {
		avgRate = math.Round((rateSum/float64(activeDebtCount))*1000) / 1000
	}

	interestSummary := InterestSummary{
		Bands:       bands,
		AverageRate: avgRate,
		MaxRate:     math.Round(maxRate*1000) / 1000,
	}

	// Calculate health summary.
	healthSummary := HealthSummary{}
	var healthScoreSum float64
	for _, l := range allLanes {
		switch l.Health.Status {
		case HealthHealthy:
			healthSummary.HealthyCount++
		case HealthDegraded:
			healthSummary.DegradedCount++
		case HealthCritical:
			healthSummary.CriticalCount++
		}
		healthScoreSum += l.Health.Score
	}
	if len(allLanes) > 0 {
		healthSummary.AverageScore = math.Round((healthScoreSum/float64(len(allLanes)))*100) / 100
	}

	// Rank hotspots worst-first.
	hotspots := make([]DebtLane, len(filtered))
	copy(hotspots, filtered)
	sort.SliceStable(hotspots, func(i, j int) bool {
		if hotspots[i].TotalDebt != hotspots[j].TotalDebt {
			return hotspots[i].TotalDebt > hotspots[j].TotalDebt
		}
		if hotspots[i].CarryingCost != hotspots[j].CarryingCost {
			return hotspots[i].CarryingCost > hotspots[j].CarryingCost
		}
		if hotspots[i].MaturityGap != hotspots[j].MaturityGap {
			return hotspots[i].MaturityGap > hotspots[j].MaturityGap
		}
		return hotspots[i].Lane < hotspots[j].Lane
	})

	topN := opts.TopN
	if topN <= 0 {
		topN = 10
	}
	if len(hotspots) > topN {
		hotspots = hotspots[:topN]
	}

	ok := totalDebt <= 0.05
	verdict := "OK"
	finding := "production_grade_achieved"
	reason := "all units of work meet target maturity; production grade denominator is 100% satisfied with 0 debt."
	nextAction := "maintain green baseline and test coverage for all declared lanes"

	if !ok {
		verdict = "ACTION"
		finding = "maturity_debt_present"
		firstHotspot := "advance lowest maturity unit"
		if len(hotspots) > 0 {
			firstHotspot = hotspots[0].NextAction
		}
		reason = fmt.Sprintf(
			"maturity debt: %d active debt lane(s) carrying %.1f total debt points (%.1f principal + %.1f carrying cost); production grade %s (%.1f%%, denominator %.1f pts, %.1f%% dilution from WIP)",
			productionGrade.WIPUnits,
			totalDebt,
			totalPrincipal,
			totalCarryingCost,
			productionGrade.GradeLetter,
			productionGrade.GradePercent,
			productionGrade.DenominatorPoints,
			productionGrade.DilutionFromWIP,
		)
		nextAction = "retire top maturity debt: " + firstHotspot
	}

	score := int(productionGrade.GradePercent + 0.5)

	corpus := map[string]any{
		"debt":                         math.Round(totalDebt*10) / 10,
		"maturity_debt_points":         math.Round(totalDebt*10) / 10,
		"debt_principal":               math.Round(totalPrincipal*10) / 10,
		"carrying_cost":                math.Round(totalCarryingCost*10) / 10,
		"production_grade_percent":     productionGrade.GradePercent,
		"production_grade_denominator": productionGrade.DenominatorPoints,
		"production_grade_realized":    productionGrade.RealizedPoints,
		"dilution_from_wip":            productionGrade.DilutionFromWIP,
		"grade":                        productionGrade.GradeLetter,
		"score":                        score,
		"total_units":                  productionGrade.TotalUnits,
		"wip_units":                    productionGrade.WIPUnits,
		"ready_units":                  productionGrade.ProductionReadyUnits,
		"average_interest_rate":        interestSummary.AverageRate,
		"critical_interest_lanes":      interestSummary.Bands[string(InterestCritical)],
		"high_interest_lanes":          interestSummary.Bands[string(InterestHigh)],
		"health_healthy_lanes":         healthSummary.HealthyCount,
		"health_degraded_lanes":        healthSummary.DegradedCount,
		"health_critical_lanes":        healthSummary.CriticalCount,
		"health_average_score":         healthSummary.AverageScore,
	}

	return Report{
		Schema:          Schema,
		OK:              ok,
		Verdict:         verdict,
		Finding:         finding,
		Reason:          reason,
		NextAction:      nextAction,
		Workspace:       absRoot,
		TargetRepo:      targetRepo,
		EvaluatedAt:     evalTime.Format(time.RFC3339),
		Corpus:          corpus,
		ProductionGrade: productionGrade,
		InterestSummary: interestSummary,
		HealthSummary:   healthSummary,
		Lanes:           filtered,
		Hotspots:        hotspots,
	}, nil
}

// EvaluateLaneHealth computes the multi-dimensional health verdict, composite score, and issues.
func EvaluateLaneHealth(l DebtLane) LaneHealth {
	testStatus := "passing"
	if !l.Evidence.HasTests || l.Evidence.TestFilesCount == 0 {
		testStatus = "missing"
	}

	commentHygiene := "clean"
	if l.Evidence.ExcessComments {
		commentHygiene = "bloat"
	}

	wiringStatus := "integrated"
	if !l.Evidence.Integrated {
		wiringStatus = "disconnected"
	}

	proofStatus := "dogfooded"
	if !l.Evidence.Dogfooded {
		proofStatus = "unproven"
	}

	benchStatus := "benchmarked"
	if !l.Evidence.Benchmarked {
		benchStatus = "unmeasured"
	}

	var issues []string
	if !l.Evidence.HasTests {
		issues = append(issues, "missing_tests")
	}
	if l.Evidence.ExcessComments {
		issues = append(issues, "excess_comments")
	}
	if !l.Evidence.Integrated {
		issues = append(issues, "disconnected_wiring")
	}
	if l.Interest.Band == InterestCritical {
		issues = append(issues, "critical_interest")
	}
	if l.MaturityGap >= 4.0 {
		issues = append(issues, "high_maturity_gap")
	}
	if !l.Evidence.Dogfooded && (l.Criticality == CriticalityCore || l.Criticality == CriticalityEnabling) {
		issues = append(issues, "unproven_runtime")
	}
	if !l.Evidence.Benchmarked && (l.Criticality == CriticalityCore || l.Criticality == CriticalityEnabling) {
		issues = append(issues, "unbenchmarked")
	}

	score := 1.0
	if !l.Evidence.HasTests {
		score -= 0.30
	}
	if l.Evidence.ExcessComments {
		score -= 0.10
	}
	if !l.Evidence.Integrated {
		score -= 0.15
	}
	if !l.Evidence.Dogfooded && (l.Criticality == CriticalityCore || l.Criticality == CriticalityEnabling) {
		score -= 0.05
	}
	if !l.Evidence.Benchmarked && (l.Criticality == CriticalityCore || l.Criticality == CriticalityEnabling) {
		score -= 0.05
	}
	if l.Interest.Band == InterestCritical {
		score -= 0.20
	} else if l.Interest.Band == InterestHigh {
		score -= 0.10
	}
	if score < 0 {
		score = 0
	}
	score = math.Round(score*100) / 100

	status := HealthHealthy
	if l.Interest.Band == InterestCritical || (!l.Evidence.HasTests && l.Criticality == CriticalityCore) || score < 0.30 {
		status = HealthCritical
	} else if len(issues) > 0 || score < 0.85 {
		status = HealthDegraded
	}

	return LaneHealth{
		Status:          status,
		Score:           score,
		TestStatus:      testStatus,
		CommentHygiene:  commentHygiene,
		WiringStatus:    wiringStatus,
		ProofStatus:     proofStatus,
		BenchmarkStatus: benchStatus,
		Issues:          issues,
	}
}

func matchesQuery(l DebtLane, q string) bool {
	q = strings.ToLower(strings.TrimSpace(q))
	if q == "" {
		return true
	}
	if strings.Contains(strings.ToLower(l.Lane), q) {
		return true
	}
	if strings.Contains(strings.ToLower(l.UnitOfWork), q) {
		return true
	}
	if strings.Contains(strings.ToLower(string(l.Criticality)), q) {
		return true
	}
	if strings.Contains(strings.ToLower(l.MaturityRung), q) {
		return true
	}
	if strings.Contains(strings.ToLower(string(l.Health.Status)), q) {
		return true
	}
	if strings.Contains(strings.ToLower(l.NextAction), q) {
		return true
	}
	for _, issue := range l.Health.Issues {
		if strings.Contains(strings.ToLower(issue), q) {
			return true
		}
	}
	for _, d := range l.Interest.Drivers {
		if strings.Contains(strings.ToLower(d), q) {
			return true
		}
	}
	if strings.Contains(strings.ToLower(l.Related.CompanionLane), q) || strings.Contains(strings.ToLower(l.Related.CompanionUnitOfWork), q) {
		return true
	}
	return false
}

func matchesHealth(l DebtLane, h string) bool {
	h = strings.ToLower(strings.TrimSpace(h))
	if h == "" {
		return true
	}
	parts := strings.Split(h, ",")
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if strings.EqualFold(string(l.Health.Status), trimmed) {
			return true
		}
	}
	return false
}

func recomputeLane(l *DebtLane) {
	if l.Weight == 0 {
		l.Weight = CriticalityWeight(l.Criticality)
	}
	if l.Bounds.TargetCeiling == 0 {
		l.Bounds = DefaultBoundsAndLimits(l.Criticality)
	}
	l.TargetMaturity = l.Bounds.TargetCeiling
	if l.Bounds.Pacing == PacingFrozen {
		l.TargetMaturity = l.Maturity
	}
	gap := l.TargetMaturity - l.Maturity
	if gap < 0 {
		gap = 0
	}
	l.MaturityGap = math.Round(gap*10) / 10
	if l.Evidence.CodeLines > 0 && l.Evidence.CommentRatio == 0 && l.Evidence.CommentLines > 0 {
		l.Evidence.CommentRatio = math.Round((float64(l.Evidence.CommentLines)/float64(l.Evidence.CodeLines))*1000) / 1000
	}
	if l.Evidence.CodeLines > 30 && l.Evidence.CommentRatio > 0.35 {
		l.Evidence.ExcessComments = true
	}
	l.Interest = CalculateInterest(l.Criticality, l.Bounds, l.Evidence, l.MaturityGap)
	l.DebtPrincipal, l.CarryingCost, l.TotalDebt = CalculateDebt(l.Maturity, l.TargetMaturity, l.Weight, l.Interest, l.Bounds)
	l.DenominatorContribution = math.Round(l.TargetMaturity*l.Weight*10) / 10
	l.RealizedContribution = math.Round(l.Maturity*l.Weight*10) / 10
	if l.NextAction == "" {
		l.NextAction = NextActionForGap(l.Lane, l.UnitOfWork, l.Maturity, l.TargetMaturity, l.Evidence)
	}
	l.Health = EvaluateLaneHealth(*l)
}

func discoverLanesFromDisk(root string, companionRoots ...string) ([]DebtLane, error) {
	var compRoot string
	if len(companionRoots) > 0 {
		compRoot = companionRoots[0]
	}
	dosPath := filepath.Join(root, "dos.toml")
	lanes, laneTrees := parseLaneTrees(dosPath)

	// Build dependency graph and reachability.
	graph, internalPkgs := BuildInternalImportGraph(root)
	reachable := scanReachableFromCmd(root, graph)

	inboundDependents := make(map[string][]string)
	for pkg, edges := range graph {
		for dep := range edges {
			inboundDependents[dep] = append(inboundDependents[dep], pkg)
		}
	}
	for k := range inboundDependents {
		sort.Strings(inboundDependents[k])
	}

	// Read benchmarks authority mentions.
	benchDocs := readBenchmarkAuthorityLanes(filepath.Join(root, "BENCHMARK-AUTHORITY.md"))
	runtimeProofs := readRuntimeProofs(root)

	// Collect candidate lanes
	discovered := make(map[string]bool)
	for _, lane := range lanes {
		discovered[lane] = true
	}
	for pkg := range internalPkgs {
		discovered[pkg] = true
	}

	for _, sub := range []string{"pkg", "platform"} {
		subDir := filepath.Join(root, sub)
		if entries, err := os.ReadDir(subDir); err == nil {
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				if dirContainsGoFiles(filepath.Join(subDir, e.Name())) {
					discovered[e.Name()] = true
				}
			}
		}
	}

	laneNames := make([]string, 0, len(discovered))
	for lane := range discovered {
		laneNames = append(laneNames, lane)
	}
	sort.Strings(laneNames)

	var result []DebtLane
	for _, lane := range laneNames {
		if lane == "cmd" || lane == "tools" {
			continue
		}

		trees := laneTrees[lane]

		// Filter out non-Go asset directories or non-Go paths if declared.
		if len(trees) > 0 {
			allNonGo := true
			for _, t := range trees {
				if !isNonGoPath(t) {
					allNonGo = false
					break
				}
			}
			if allNonGo {
				continue
			}
		}

		// Resolve unitDir
		var unitDir string

		// First, check conventional Go source directories matching the lane name.
		// If internal/<lane> exists on disk and contains Go files, prioritize it.
		if info, err := os.Stat(filepath.Join(root, "internal", lane)); err == nil && info.IsDir() {
			unitDir = filepath.Join("internal", lane)
		} else if info, err := os.Stat(filepath.Join(root, "pkg", lane)); err == nil && info.IsDir() {
			unitDir = filepath.Join("pkg", lane)
		} else if info, err := os.Stat(filepath.Join(root, "platform", lane)); err == nil && info.IsDir() {
			unitDir = filepath.Join("platform", lane)
		}

		// a) If not resolved yet and a declared tree exists: extract the root directory.
		// If that directory exists on disk, use it! (e.g. "platform/dispatch/**" -> "platform/dispatch")
		if unitDir == "" && len(trees) > 0 {
			// Prioritize trees that point to Go source prefixes
			for _, t := range trees {
				rootGlob := extractGlobRoot(t)
				if rootGlob != "" && hasGoSourcePrefix(rootGlob) {
					if info, err := os.Stat(filepath.Join(root, rootGlob)); err == nil && info.IsDir() {
						unitDir = filepath.FromSlash(rootGlob)
						break
					}
				}
			}
			if unitDir == "" {
				for _, t := range trees {
					rootGlob := extractGlobRoot(t)
					if rootGlob != "" {
						if info, err := os.Stat(filepath.Join(root, rootGlob)); err == nil && info.IsDir() {
							unitDir = filepath.FromSlash(rootGlob)
							break
						}
					}
				}
			}
		}

		// If unitDir not found yet, check other conventional source directories.
		if unitDir == "" {
			if info, err := os.Stat(filepath.Join(root, "tools", lane)); err == nil && info.IsDir() {
				unitDir = filepath.Join("tools", lane)
			} else if info, err := os.Stat(filepath.Join(root, "cmd", lane)); err == nil && info.IsDir() {
				unitDir = filepath.Join("cmd", lane)
			} else if info, err := os.Stat(filepath.Join(root, "cmd", "fak-"+lane)); err == nil && info.IsDir() {
				unitDir = filepath.Join("cmd", "fak-"+lane)
			}
		}

		// c) If a lane does not exist on disk, check whether its declared tree explicitly starts with a Go source root.
		if unitDir == "" {
			hasDeclaredGoPrefix := false
			for _, t := range trees {
				if hasGoSourcePrefix(t) {
					hasDeclaredGoPrefix = true
					break
				}
			}
			if !hasDeclaredGoPrefix {
				// Neither exists on disk nor declares a Go source tree; do NOT synthesize phantom internal/<lane>.
				continue
			}
			unitDir = filepath.Join("internal", lane)
		}

		absUnitDir := filepath.Join(root, unitDir)

		evidence := inspectUnitEvidence(absUnitDir, lane, graph, reachable, benchDocs, runtimeProofs)
		crit := ClassifyCriticality(lane)
		bounds := DefaultBoundsAndLimits(crit)
		weight := CriticalityWeight(crit)

		maturityScore, rung := EvaluateMaturityCurve(evidence)
		target := bounds.TargetCeiling
		gap := target - maturityScore
		if gap < 0 {
			gap = 0
		}
		gap = math.Round(gap*10) / 10

		interest := CalculateInterest(crit, bounds, evidence, gap)
		principal, carrying, total := CalculateDebt(maturityScore, target, weight, interest, bounds)

		repoName := "fak"
		cleanUnit := filepath.ToSlash(unitDir)
		if strings.HasPrefix(cleanUnit, "platform/") || strings.HasPrefix(cleanUnit, "tools/") || filepath.Base(root) == "fak-private" {
			repoName = "fak-private"
		}

		// Build related cross-indexed artifacts
		var proofWitnesses []string
		if runtimeProofs[lane] {
			proofWitnesses = []string{lane}
		}
		var benchWitnesses []string
		if benchDocs[lane] {
			benchWitnesses = []string{lane}
		}

		var deps []string
		if edges, ok := graph[lane]; ok {
			for dep := range edges {
				deps = append(deps, dep)
			}
			sort.Strings(deps)
		}
		dependents := append([]string(nil), inboundDependents[lane]...)

		var compRepo, compLane, compUnit string
		if compRoot != "" {
			isPriv := filepath.Base(root) == "fak-private" || strings.HasPrefix(filepath.ToSlash(unitDir), "platform/")
			if isPriv {
				compRepo = "fak"
				if info, err := os.Stat(filepath.Join(compRoot, "internal", lane)); err == nil && info.IsDir() {
					compLane = lane
					compUnit = filepath.Join("internal", lane)
				} else if info, err := os.Stat(filepath.Join(compRoot, "pkg", lane)); err == nil && info.IsDir() {
					compLane = lane
					compUnit = filepath.Join("pkg", lane)
				}
			} else {
				compRepo = "fak-private"
				if info, err := os.Stat(filepath.Join(compRoot, "platform", lane)); err == nil && info.IsDir() {
					compLane = lane
					compUnit = filepath.Join("platform", lane)
				} else if info, err := os.Stat(filepath.Join(compRoot, "tools", lane)); err == nil && info.IsDir() {
					compLane = lane
					compUnit = filepath.Join("tools", lane)
				} else if info, err := os.Stat(filepath.Join(compRoot, "cmd", lane)); err == nil && info.IsDir() {
					compLane = lane
					compUnit = filepath.Join("cmd", lane)
				} else if info, err := os.Stat(filepath.Join(compRoot, "cmd", "fak-"+lane)); err == nil && info.IsDir() {
					compLane = lane
					compUnit = filepath.Join("cmd", "fak-"+lane)
				}
			}
		}

		related := RelatedThings{
			CompanionRepo:       compRepo,
			CompanionLane:       compLane,
			CompanionUnitOfWork: compUnit,
			Dependents:          dependents,
			Dependencies:        deps,
			DosTrees:            append([]string(nil), trees...),
			ProofWitnesses:      proofWitnesses,
			BenchmarkWitnesses:  benchWitnesses,
		}

		dl := DebtLane{
			Lane:                    lane,
			Repo:                    repoName,
			UnitOfWork:              unitDir,
			Criticality:             crit,
			Weight:                  weight,
			Maturity:                maturityScore,
			MaturityRung:            rung,
			TargetMaturity:          target,
			MaturityGap:             gap,
			DebtPrincipal:           principal,
			Interest:                interest,
			CarryingCost:            carrying,
			TotalDebt:               total,
			DenominatorContribution: math.Round(target*weight*10) / 10,
			RealizedContribution:    math.Round(maturityScore*weight*10) / 10,
			Bounds:                  bounds,
			Evidence:                evidence,
			Related:                 related,
			NextAction:              NextActionForGap(lane, unitDir, maturityScore, target, evidence),
		}
		dl.Health = EvaluateLaneHealth(dl)
		result = append(result, dl)
	}

	return result, nil
}

// ClassifyCriticality maps a lane name to its architectural criticality.
func ClassifyCriticality(lane string) Criticality {
	lower := strings.ToLower(lane)
	switch {
	case lower == "kernel" || lower == "gateway" || lower == "vdso" || lower == "adjudicator" ||
		lower == "policy" || lower == "ctxmmu" || lower == "engine" || lower == "kvmmu" ||
		lower == "model" || lower == "preflight" || lower == "shipgate" || lower == "architest" ||
		lower == "abi" || lower == "procguard":
		return CriticalityCore

	case lower == "promptmmu" || lower == "radixkv" || lower == "taskmgr" || lower == "modelengine" ||
		lower == "tokenizer" || lower == "metalgemm" || lower == "grammar" || lower == "contextq" ||
		lower == "journal" || lower == "dropin" || lower == "comm" || lower == "vcachegov" ||
		lower == "vcachechain" || lower == "vcachescore" || lower == "dispatchorder" || lower == "agent":
		return CriticalityEnabling

	case strings.Contains(lower, "lint") || strings.Contains(lower, "test") || strings.Contains(lower, "score") ||
		strings.Contains(lower, "doctor") || strings.Contains(lower, "metrics") || strings.Contains(lower, "steward") ||
		strings.Contains(lower, "leak") || strings.Contains(lower, "audit") || strings.Contains(lower, "hygiene") ||
		strings.Contains(lower, "witness") || lower == "debtlane":
		return CriticalityStewardship

	case strings.Contains(lower, "bench") || strings.Contains(lower, "demo") || strings.Contains(lower, "visual") ||
		strings.Contains(lower, "toon") || strings.Contains(lower, "experiment"):
		return CriticalityPeripheral

	default:
		return CriticalityEnabling
	}
}

func inspectUnitEvidence(dir, lane string, graph map[string]map[string]struct{}, reachable map[string]struct{}, benchDocs map[string]bool, runtimeProofs map[string]bool) Evidence {
	var ev Evidence
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return ev
	}

	// Dependents count from internal import graph.
	dependents := 0
	for src, edges := range graph {
		if src == lane {
			continue
		}
		if _, ok := edges[lane]; ok {
			dependents++
		}
	}
	ev.DependentsCount = dependents

	if edges, ok := graph[lane]; ok {
		ev.TransitiveDependencies = len(edges)
	}

	if _, ok := reachable[lane]; ok {
		ev.Integrated = true
	}

	if runtimeProofs[lane] {
		ev.Dogfooded = true
	}

	if benchDocs[lane] {
		ev.Benchmarked = true
	}

	fset := token.NewFileSet()
	formulaicCount := 0
	hasFormulaicFiller := false

	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			if path != dir {
				name := d.Name()
				if name == "testdata" || name == "vendor" || name == ".git" || name == "_scratch" {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".go") {
			return nil
		}
		fullPath := path
		isTest := strings.HasSuffix(d.Name(), "_test.go")

		if isTest {
			content, err := os.ReadFile(fullPath)
			if err != nil {
				return nil
			}
			testNode, err := parser.ParseFile(fset, fullPath, content, 0)
			if err != nil {
				return nil
			}

			hasRealTests := false
			for _, decl := range testNode.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok {
					continue
				}
				if strings.HasPrefix(fn.Name.Name, "Test") && fn.Body != nil && len(fn.Body.List) > 0 {
					hasRealTests = true
				}
				if strings.HasPrefix(fn.Name.Name, "Benchmark") && !ev.Benchmarked {
					if isSubstantiveBenchmark(fn) {
						ev.Benchmarked = true
					}
				}
			}

			if hasRealTests {
				ev.HasTests = true
				ev.TestFilesCount++
			}
			return nil
		}

		ev.FilesCount++
		ev.HasCode = true

		// Parse non-test files for exported symbols and comment metrics.
		content, err := os.ReadFile(fullPath)
		if err != nil {
			return nil
		}
		ev.CodeLines += countNonEmptyLines(content)

		node, err := parser.ParseFile(fset, fullPath, content, parser.ParseComments)
		if err != nil {
			return nil
		}

		for _, cg := range node.Comments {
			for _, c := range cg.List {
				ev.CommentLines += strings.Count(c.Text, "\n") + 1
			}
			isForm, isFiller := isFormulaicGamingComment(cg)
			if isForm {
				formulaicCount++
			}
			if isFiller {
				hasFormulaicFiller = true
			}
		}

		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(d.Name.Name) {
					ev.ExportedSymbols++
					if isSubstantiveDoc(d.Name.Name, d.Doc) {
						ev.DocumentedExports++
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							ev.ExportedSymbols++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if isSubstantiveDoc(s.Name.Name, doc) {
								ev.DocumentedExports++
							}
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								ev.ExportedSymbols++
								doc := s.Doc
								if doc == nil {
									doc = d.Doc
								}
								if isSubstantiveDoc(name.Name, doc) {
									ev.DocumentedExports++
								}
							}
						}
					}
				}
			}
		}
		return nil
	})

	if ev.CodeLines > 0 {
		ev.CommentRatio = math.Round((float64(ev.CommentLines)/float64(ev.CodeLines))*1000) / 1000
	}
	if (ev.CodeLines > 30 && ev.CommentRatio > 0.35) || (hasFormulaicFiller && formulaicCount >= 2) || formulaicCount >= 3 {
		ev.ExcessComments = true
	}

	if ev.ExportedSymbols > 0 && float64(ev.DocumentedExports)/float64(ev.ExportedSymbols) >= 0.75 {
		ev.Documented = true
	}

	return ev
}

func countNonEmptyLines(b []byte) int {
	scanner := bufio.NewScanner(strings.NewReader(string(b)))
	lines := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) != "" {
			lines++
		}
	}
	return lines
}

func dirContainsGoFiles(dir string) bool {
	hasGo := false
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || hasGo {
			return nil
		}
		if d.IsDir() {
			if path != dir {
				name := d.Name()
				if name == "testdata" || name == "vendor" || name == ".git" || name == "_scratch" {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if strings.HasSuffix(d.Name(), ".go") {
			hasGo = true
			return filepath.SkipDir
		}
		return nil
	})
	return hasGo
}

func extractGlobRoot(glob string) string {
	clean := filepath.ToSlash(strings.TrimSpace(glob))
	clean = strings.Trim(clean, `"`+"'"+` `)
	if idx := strings.Index(clean, "*"); idx >= 0 {
		clean = clean[:idx]
	}
	clean = strings.TrimRight(clean, "/")
	return clean
}

func isNonGoPath(path string) bool {
	clean := filepath.ToSlash(strings.TrimSpace(path))
	clean = strings.TrimPrefix(clean, "./")
	if strings.HasPrefix(clean, "docs/") || clean == "docs" ||
		strings.HasPrefix(clean, "scripts/") || clean == "scripts" ||
		strings.HasPrefix(clean, ".claude/") || clean == ".claude" ||
		strings.HasPrefix(clean, ".agents/") || clean == ".agents" ||
		strings.HasPrefix(clean, "examples/") || clean == "examples" ||
		strings.HasPrefix(clean, "visuals/") || clean == "visuals" ||
		strings.HasPrefix(clean, "agent-memory/") || clean == "agent-memory" {
		return true
	}
	if strings.HasSuffix(clean, ".py") || strings.HasSuffix(clean, ".sh") ||
		strings.HasSuffix(clean, ".md") || strings.HasSuffix(clean, ".txt") ||
		strings.HasSuffix(clean, ".json") || strings.HasSuffix(clean, ".toml") ||
		strings.HasSuffix(clean, ".yaml") || strings.HasSuffix(clean, ".yml") {
		return true
	}
	return false
}

func hasGoSourcePrefix(path string) bool {
	clean := filepath.ToSlash(strings.TrimSpace(path))
	clean = strings.TrimPrefix(clean, "./")
	return strings.HasPrefix(clean, "internal/") || clean == "internal" ||
		strings.HasPrefix(clean, "pkg/") || clean == "pkg" ||
		strings.HasPrefix(clean, "platform/") || clean == "platform" ||
		strings.HasPrefix(clean, "tools/") || clean == "tools" ||
		strings.HasPrefix(clean, "cmd/") || clean == "cmd"
}

func parseLaneTrees(path string) ([]string, map[string][]string) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil
	}
	defer f.Close()

	var lanes []string
	laneTrees := make(map[string][]string)
	seen := make(map[string]bool)
	scanner := bufio.NewScanner(f)
	currentSection := ""
	inConcurrent := false

	for scanner.Scan() {
		rawLine := scanner.Text()
		line := strings.TrimSpace(rawLine)
		if strings.HasPrefix(line, "#") {
			continue
		}
		if idx := strings.Index(line, "#"); idx >= 0 {
			line = strings.TrimSpace(line[:idx])
		}
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			currentSection = strings.ToLower(strings.TrimSpace(line[1 : len(line)-1]))
			inConcurrent = false
			continue
		}

		if currentSection == "lanes" {
			if strings.HasPrefix(line, "concurrent = [") {
				inConcurrent = true
				line = strings.TrimPrefix(line, "concurrent = [")
			}
			if inConcurrent {
				if strings.Contains(line, "]") {
					inConcurrent = false
					line = strings.TrimSuffix(line, "]")
				}
				parts := strings.Split(line, ",")
				for _, part := range parts {
					trimmed := strings.Trim(strings.TrimSpace(part), `", `)
					if trimmed != "" && isPotentialLaneName(trimmed) && !seen[trimmed] {
						seen[trimmed] = true
						lanes = append(lanes, trimmed)
					}
				}
				continue
			}
		}

		if currentSection == "lanes.trees" || currentSection == "paths" {
			eqIdx := strings.Index(line, "=")
			if eqIdx >= 0 {
				rawKey := strings.TrimSpace(line[:eqIdx])
				lane := strings.Trim(rawKey, `"' `)
				valPart := strings.TrimSpace(line[eqIdx+1:])
				tokens := quotedTokens(valPart)
				if len(tokens) > 0 {
					laneTrees[lane] = append(laneTrees[lane], tokens...)
				}
			}

			m := laneTreeRe.FindStringSubmatch(line)
			if len(m) >= 2 && isPotentialLaneName(m[1]) && !seen[m[1]] {
				seen[m[1]] = true
				lanes = append(lanes, m[1])
			}
		}
	}
	return lanes, laneTrees
}

func quotedTokens(s string) []string {
	var out []string
	for {
		i := strings.IndexByte(s, '"')
		if i < 0 {
			return out
		}
		j := strings.IndexByte(s[i+1:], '"')
		if j < 0 {
			return out
		}
		if tok := s[i+1 : i+1+j]; tok != "" {
			out = append(out, tok)
		}
		s = s[i+j+2:]
	}
}

// BuildInternalImportGraph parses Go files in internal/, pkg/, and platform/ to construct a package import dependency graph.
func BuildInternalImportGraph(root string) (map[string]map[string]struct{}, map[string]struct{}) {
	graph := make(map[string]map[string]struct{})
	internalPkgs := make(map[string]struct{})

	for _, base := range []string{"internal", "pkg", "platform"} {
		baseDir := filepath.Join(root, base)
		entries, err := os.ReadDir(baseDir)
		if err != nil {
			continue
		}

		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			pkgName := e.Name()
			dirPath := filepath.Join(baseDir, pkgName)
			edges := make(map[string]struct{})
			hasGo := false

			_ = filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, walkErr error) error {
				if walkErr != nil {
					return nil
				}
				if d.IsDir() {
					if path != dirPath {
						name := d.Name()
						if name == "testdata" || name == "vendor" || name == ".git" || name == "_scratch" {
							return filepath.SkipDir
						}
					}
					return nil
				}
				if !strings.HasSuffix(path, ".go") {
					return nil
				}
				hasGo = true
				content, readErr := os.ReadFile(path)
				if readErr != nil {
					return nil
				}
				matches := importRe.FindAllStringSubmatch(string(content), -1)
				for _, m := range matches {
					if len(m) == 2 && m[1] != pkgName {
						edges[m[1]] = struct{}{}
					}
				}
				return nil
			})

			if hasGo {
				internalPkgs[pkgName] = struct{}{}
				if existing, ok := graph[pkgName]; ok {
					for edge := range edges {
						existing[edge] = struct{}{}
					}
				} else {
					graph[pkgName] = edges
				}
			}
		}
	}

	return graph, internalPkgs
}

func scanReachableFromCmd(root string, graph map[string]map[string]struct{}) map[string]struct{} {
	reachable := make(map[string]struct{})
	queue := make([]string, 0)

	cmdDir := filepath.Join(root, "cmd")
	_ = filepath.WalkDir(cmdDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || d.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		content, readErr := os.ReadFile(path)
		if readErr != nil {
			return nil
		}
		matches := importRe.FindAllStringSubmatch(string(content), -1)
		for _, m := range matches {
			if len(m) == 2 {
				if _, ok := reachable[m[1]]; !ok {
					reachable[m[1]] = struct{}{}
					queue = append(queue, m[1])
				}
			}
		}
		return nil
	})

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		if deps, ok := graph[curr]; ok {
			for dep := range deps {
				if _, ok := reachable[dep]; !ok {
					reachable[dep] = struct{}{}
					queue = append(queue, dep)
				}
			}
		}
	}

	return reachable
}

func isSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
	if doc == nil || len(doc.List) == 0 {
		return false
	}
	text := strings.TrimSpace(doc.Text())
	if len(text) < 12 {
		return false
	}
	return !isTautologicalDoc(name, text)
}

func splitIdentifierWords(name string) map[string]bool {
	set := make(map[string]bool)
	set[strings.ToLower(name)] = true
	var curr strings.Builder
	for i, r := range name {
		if r == '_' || r == '-' {
			if curr.Len() > 0 {
				set[strings.ToLower(curr.String())] = true
				curr.Reset()
			}
			continue
		}
		if unicode.IsUpper(r) && i > 0 && curr.Len() > 0 {
			set[strings.ToLower(curr.String())] = true
			curr.Reset()
		}
		curr.WriteRune(r)
	}
	if curr.Len() > 0 {
		set[strings.ToLower(curr.String())] = true
	}
	return set
}

func isTautologicalDoc(name string, text string) bool {
	nameLower := strings.ToLower(name)
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return true
	}
	firstWord := strings.Trim(strings.ToLower(fields[0]), ":,.-()")
	if firstWord != nameLower && !strings.HasPrefix(strings.ToLower(text), nameLower) {
		return false
	}
	remainder := strings.TrimSpace(text[len(firstWord):])
	words := strings.FieldsFunc(remainder, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})

	fillers := map[string]bool{
		"is": true, "are": true, "does": true, "do": true, "returns": true, "return": true,
		"represents": true, "represent": true, "holds": true, "hold": true, "the": true,
		"a": true, "an": true, "of": true, "for": true, "to": true, "that": true, "which": true,
		"will": true, "can": true, "provides": true, "provide": true, "specifies": true,
		"specify": true, "defines": true, "define": true, "indicates": true, "indicate": true,
		"details": true, "detail": true, "records": true, "record": true, "encapsulates": true,
		"encapsulate": true, "captures": true, "capture": true, "contains": true, "contain": true,
	}

	nameParts := splitIdentifierWords(name)
	meaningfulWords := 0
	for _, w := range words {
		wl := strings.ToLower(w)
		if fillers[wl] || nameParts[wl] {
			continue
		}
		meaningfulWords++
	}
	return meaningfulWords < 2
}

func isSubstantiveBenchmark(fn *ast.FuncDecl) bool {
	if fn.Body == nil || len(fn.Body.List) == 0 {
		return false
	}
	hasLoopOrRun := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if hasLoopOrRun {
			return false
		}
		switch stmt := n.(type) {
		case *ast.ForStmt:
			if stmt.Cond != nil && referencesName(stmt.Cond, "N") {
				hasLoopOrRun = true
			}
		case *ast.RangeStmt:
			if referencesName(stmt.X, "N") {
				hasLoopOrRun = true
			}
		case *ast.CallExpr:
			if sel, ok := stmt.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Run" {
				hasLoopOrRun = true
			}
		}
		return true
	})
	return hasLoopOrRun
}

func referencesName(expr ast.Expr, name string) bool {
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if found {
			return false
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return true
	})
	return found
}

// isFormulaicGamingComment detects formulaic comments like "Contract:", "Invariant:", "Fail-closed:"
// that are used as filler (keyword stuffing, short stubs) or formulaic bloat.
func isFormulaicGamingComment(cg *ast.CommentGroup) (isFormulaic bool, isFiller bool) {
	if cg == nil {
		return false, false
	}
	text := strings.TrimSpace(cg.Text())
	lower := strings.ToLower(text)

	hasMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.HasPrefix(lower, "invariant") ||
		strings.HasPrefix(lower, "guard") ||
		strings.HasPrefix(lower, "contract") ||
		strings.HasPrefix(lower, "fail-closed")

	if !hasMarker {
		return false, false
	}

	words := strings.Fields(lower)
	if len(words) <= 3 {
		return true, true
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.25 || keywordCount >= 3 {
		return true, true
	}

	return true, false
}

var nonLaneTokens = map[string]bool{
	"claim": true, "number": true, "model": true, "baseline": true,
	"commit": true, "artifact": true, "metric": true, "speedup": true,
	"regime": true, "measures": true, "example": true, "status": true,
	"target": true, "notes": true, "result": true, "date": true,
	"scope": true, "action": true, "verdict": true, "value": true,
	"lane": true, "summary": true, "piece": true, "syscall": true,
	"route": true, "context": true, "description": true, "owner": true,
	"purpose": true, "visual": true, "relevance": true, "diff": true,
	"file": true, "path": true, "type": true, "name": true,
	"pass": true, "fail": true, "none": true, "same": true,
	"true": true, "false": true, "stale": true, "todo": true,
	"item": true, "field": true, "layer": true, "gate": true,
	"fak": true, "raw": true, "allow": true, "deny": true,
	"all": true, "yes": true, "no": true, "turns": true,
	"agents": true, "state": true, "pending": true, "optimized": true,
	"replacement": true, "arm": true, "caught": true, "quarantine": true,
	"provenance": true, "prefix": true, "radix": true, "hardware": true,
}

func isPotentialLaneName(s string) bool {
	if len(s) < 3 || len(s) > 30 {
		return false
	}
	if s[0] < 'a' || s[0] > 'z' {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

func readBenchmarkAuthorityLanes(path string) map[string]bool {
	set := make(map[string]bool)
	content, err := os.ReadFile(path)
	if err != nil {
		return set
	}
	scanner := bufio.NewScanner(strings.NewReader(string(content)))
	backtickRe := regexp.MustCompile("`([a-zA-Z0-9_/-]+)`")
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "|") {
			for _, m := range backtickRe.FindAllStringSubmatch(line, -1) {
				if len(m) == 2 {
					token := strings.ToLower(m[1])
					token = strings.TrimPrefix(token, "internal/")
					token = strings.TrimPrefix(token, "cmd/")
					token = strings.TrimPrefix(token, "pkg/")
					if token != "" {
						set[token] = true
					}
				}
			}
			if strings.Contains(line, "---") {
				continue
			}
			parts := strings.Split(line, "|")
			for _, part := range parts {
				cell := strings.TrimSpace(strings.ToLower(part))
				cell = strings.Trim(cell, "`*_-")
				if !isPotentialLaneName(cell) || nonLaneTokens[cell] {
					continue
				}
				set[cell] = true
			}
		} else if strings.HasPrefix(line, "#") {
			for _, m := range backtickRe.FindAllStringSubmatch(line, -1) {
				if len(m) == 2 {
					token := strings.ToLower(m[1])
					token = strings.TrimPrefix(token, "internal/")
					token = strings.TrimPrefix(token, "pkg/")
					token = strings.TrimPrefix(token, "platform/")
					token = strings.TrimPrefix(token, "tools/")
					if token != "" {
						set[token] = true
					}
				}
			}
		}
	}
	return set
}

func readRuntimeProofs(root string) map[string]bool {
	set := make(map[string]bool)
	for _, sub := range []string{"internal", "platform"} {
		path := filepath.Join(root, sub, "maturity", "runtime-proofs.json")
		content, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		// Extract lanes quickly using regex across the witness registry.
		re := regexp.MustCompile(`"lane":\s*"([A-Za-z0-9_]+)"`)
		matches := re.FindAllStringSubmatch(string(content), -1)
		for _, m := range matches {
			if len(m) == 2 {
				set[m[1]] = true
			}
		}
	}
	return set
}
