package debtlane

import (
	"fmt"
	"sort"
	"strings"
)

// CoverageReceiptSchema is the canonical schema identifier for debt coverage receipts.
const CoverageReceiptSchema = "fak.debt-coverage-receipt.v1"

// SurfaceClass identifies the structural type of a scanned repository surface.
type SurfaceClass string

const (
	SurfaceInternal    SurfaceClass = "internal"    // Core Go packages
	SurfacePkg         SurfaceClass = "pkg"         // Public shared Go packages
	SurfacePlatform    SurfaceClass = "platform"    // Platform-specific Go packages
	SurfaceCmd         SurfaceClass = "cmd"         // Go CLI binary commands
	SurfaceTools       SurfaceClass = "tools"       // Tooling programs and submodules
	SurfaceSkills      SurfaceClass = "skills"      // Agent skills (.claude/skills, .agents/skills)
	SurfaceWorkflows   SurfaceClass = "workflows"   // CI/CD workflows (.github/workflows)
	SurfaceExamples    SurfaceClass = "examples"    // Examples and runnable demos
	SurfaceDocs        SurfaceClass = "docs"        // Technical documentation and guides
	SurfaceUnknown     SurfaceClass = "unknown"     // Unknown surface class
	SurfaceUnsupported SurfaceClass = "unsupported" // Unsupported repository surface root
)

// Standard 9 surface classes for full-breadth coverage.
var StandardSurfaceClasses = []SurfaceClass{
	SurfaceInternal,
	SurfacePkg,
	SurfacePlatform,
	SurfaceCmd,
	SurfaceTools,
	SurfaceSkills,
	SurfaceWorkflows,
	SurfaceExamples,
	SurfaceDocs,
}

// DetectorDimension defines an independent debt evaluation aspect.
type DetectorDimension string

const (
	DimTestStatus                  DetectorDimension = "test_status"                   // Passing, missing, or failing tests
	DimCommentHygiene              DetectorDimension = "comment_hygiene"               // Clean comments vs bloat/gaming
	DimWiringStatus                DetectorDimension = "wiring_status"                 // Integrated in production graph vs disconnected
	DimProofStatus                 DetectorDimension = "proof_status"                  // Dogfooded runtime proof vs unproven
	DimBenchmarkStatus             DetectorDimension = "benchmark_status"              // Substantive benchmark vs unmeasured
	DimModularityDeficit           DetectorDimension = "modularity_deficit"            // God files / god functions
	DimModelCoupling               DetectorDimension = "model_coupling"                // Model family hardcoding outside model/
	DimBlastRadius                 DetectorDimension = "blast_radius"                  // Inbound dependents > 25
	DimThinTests                   DetectorDimension = "thin_tests"                    // Vacuous/assertion-free test files
	DimStubDebt                    DetectorDimension = "stub_debt"                     // TODO/FIXME/unimplemented panics
	DimUndocumentedExports         DetectorDimension = "undocumented_exports"          // Exported symbols without godoc
	DimUnsafeUsage                 DetectorDimension = "unsafe_usage"                  // Unsafe pointer usage in non-core
	DimSubprocessExec              DetectorDimension = "subprocess_exec"               // Subprocess spawning in library packages
	DimRaceFuzzStatus              DetectorDimension = "race_fuzz_status"              // Concurrency/parser lacking race/fuzz tests
	DimStalePerfProof              DetectorDimension = "stale_perf_proof"              // Missing from benchmark authority or proof registry
	DimCoverageDebt                DetectorDimension = "coverage_debt"                 // Unknown, unsupported, or unreadable content
	DimUngatedPerformanceBenchmark DetectorDimension = "ungated_performance_benchmark" // Benchmarked lanes without executable regression gates
	DimMockHazard                  DetectorDimension = "mock_hazard"                   // Mock/fake struct or stub panic hazard
)

// Aliases for dimensional lookup compatibility.
const (
	DimUngatedBenchmark = DimUngatedPerformanceBenchmark
)

// Standard 16 detector dimensions for full-depth evaluation.
var StandardDetectorDimensions = []DetectorDimension{
	DimTestStatus,
	DimCommentHygiene,
	DimWiringStatus,
	DimProofStatus,
	DimBenchmarkStatus,
	DimModularityDeficit,
	DimModelCoupling,
	DimBlastRadius,
	DimThinTests,
	DimStubDebt,
	DimUndocumentedExports,
	DimUnsafeUsage,
	DimSubprocessExec,
	DimRaceFuzzStatus,
	DimStalePerfProof,
	DimUngatedPerformanceBenchmark,
}

// AllDetectorDimensions returns all detector dimensions supported by debtlane.
func AllDetectorDimensions() []DetectorDimension {
	dims := make([]DetectorDimension, 0, len(StandardDetectorDimensions)+2)
	dims = append(dims, StandardDetectorDimensions...)
	dims = append(dims, DimCoverageDebt, DimMockHazard)
	return dims
}

// FindingProvenance records typed provenance for a specific debt finding.
type FindingProvenance struct {
	Dimension string `json:"dimension"` // e.g. "thin_tests", "stub_debt", "unbenchmarked"
	Surface   string `json:"surface"`   // e.g. "internal", "cmd", "skills"
	Lane      string `json:"lane"`      // lane or unit identifier
	Path      string `json:"path"`      // file or directory path
	Severity  string `json:"severity"`  // "critical", "warning", "info"
	Message   string `json:"message"`   // specific finding detail
}

// CoverageReceipt provides a machine-readable report on discovery breadth and detector depth.
type CoverageReceipt struct {
	Schema             string              `json:"schema"`                        // "fak.debt-coverage-receipt.v1"
	Workspace          string              `json:"workspace"`                     // workspace root path
	TargetRepo         string              `json:"target_repo,omitempty"`         // "fak", "fak-private", "both"
	BaselineBreadth    int                 `json:"baseline_breadth"`              // 3 (internal, pkg, platform)
	ObservedBreadth    int                 `json:"observed_breadth"`              // count of distinct surface classes observed
	TargetBreadth      int                 `json:"target_breadth"`                // 9
	BreadthRatio       float64             `json:"breadth_ratio"`                 // ObservedBreadth / BaselineBreadth
	BaselineDepth      int                 `json:"baseline_depth"`                // 5 headline health dimensions
	ObservedDepth      int                 `json:"observed_depth"`                // count of distinct detector dimensions evaluated
	TargetDepth        int                 `json:"target_depth"`                  // 15
	DepthRatio         float64             `json:"depth_ratio"`                   // ObservedDepth / BaselineDepth
	ScannedUnits       int                 `json:"scanned_units"`                 // total units of work scanned across all surfaces
	ScannedFiles       int                 `json:"scanned_files"`                 // total files inspected
	SurfaceClasses     []string            `json:"surface_classes"`               // list of observed surface class names
	Dimensions         []string            `json:"dimensions"`                    // list of evaluated detector dimension names
	DeclaredDimensions []string            `json:"declared_dimensions,omitempty"` // list of declared detector dimension names
	ExecutedDimensions []string            `json:"executed_dimensions,omitempty"` // list of executed detector dimension names
	SkippedDimensions  []string            `json:"skipped_dimensions,omitempty"`  // list of skipped detector dimension names
	ErroredDimensions  []string            `json:"errored_dimensions,omitempty"`  // list of detector dimension names that errored
	Gaps               []string            `json:"gaps"`                          // list of unexecuted (skipped or errored) detector dimension names
	CoverageDebt       int                 `json:"coverage_debt"`                 // count of coverage debt items (errored detectors)
	FindingsCount      int                 `json:"findings_count"`                // total actionable findings detected
	Findings           []FindingProvenance `json:"findings,omitempty"`
}

// DetectorStatus indicates the execution outcome of a detector dimension.
type DetectorStatus string

const (
	DetectorStatusExecuted DetectorStatus = "executed"
	DetectorStatusSkipped  DetectorStatus = "skipped"
	DetectorStatusErrored  DetectorStatus = "errored"
)

// DetectorExecution records the execution status of a single detector dimension.
type DetectorExecution struct {
	Dimension DetectorDimension `json:"dimension"`
	Status    DetectorStatus    `json:"status"`
	Error     string            `json:"error,omitempty"`
}

// CoverageOption configures coverage receipt construction.
type CoverageOption func(*coverageConfig)

type coverageConfig struct {
	hasExplicit bool
	executed    map[string]bool
	skipped     map[string]bool
	errored     map[string]string // dimension -> error detail
	declared    []string
}

// WithExecutedDimensions specifies the detector dimensions that were actually evaluated.
func WithExecutedDimensions(dims ...DetectorDimension) CoverageOption {
	return func(c *coverageConfig) {
		c.hasExplicit = true
		for _, d := range dims {
			c.executed[string(d)] = true
		}
	}
}

// WithExecutedDimensionNames specifies evaluated dimension names as strings.
func WithExecutedDimensionNames(dims ...string) CoverageOption {
	return func(c *coverageConfig) {
		c.hasExplicit = true
		for _, d := range dims {
			c.executed[d] = true
		}
	}
}

// WithSkippedDimensions specifies detector dimensions that were declared but skipped.
func WithSkippedDimensions(dims ...DetectorDimension) CoverageOption {
	return func(c *coverageConfig) {
		c.hasExplicit = true
		for _, d := range dims {
			c.skipped[string(d)] = true
		}
	}
}

// WithSkippedDimensionNames specifies skipped dimension names as strings.
func WithSkippedDimensionNames(dims ...string) CoverageOption {
	return func(c *coverageConfig) {
		c.hasExplicit = true
		for _, d := range dims {
			c.skipped[d] = true
		}
	}
}

// WithErroredDimension records a detector dimension that encountered an execution error.
func WithErroredDimension(dim DetectorDimension, err error) CoverageOption {
	return func(c *coverageConfig) {
		c.hasExplicit = true
		msg := "execution error"
		if err != nil {
			msg = err.Error()
		}
		c.errored[string(dim)] = msg
	}
}

// WithErroredDimensionName records a detector dimension name that encountered an execution error.
func WithErroredDimensionName(dim string, err error) CoverageOption {
	return func(c *coverageConfig) {
		c.hasExplicit = true
		msg := "execution error"
		if err != nil {
			msg = err.Error()
		}
		c.errored[dim] = msg
	}
}

// WithDetectorExecutions applies multiple detector execution records.
func WithDetectorExecutions(execs ...DetectorExecution) CoverageOption {
	return func(c *coverageConfig) {
		c.hasExplicit = true
		for _, e := range execs {
			dim := string(e.Dimension)
			switch e.Status {
			case DetectorStatusExecuted:
				c.executed[dim] = true
			case DetectorStatusSkipped:
				c.skipped[dim] = true
			case DetectorStatusErrored:
				msg := e.Error
				if msg == "" {
					msg = "execution error"
				}
				c.errored[dim] = msg
			}
		}
	}
}

// WithDeclaredDimensions overrides the declared detector dimensions.
func WithDeclaredDimensions(dims ...DetectorDimension) CoverageOption {
	return func(c *coverageConfig) {
		c.declared = make([]string, len(dims))
		for i, d := range dims {
			c.declared[i] = string(d)
		}
	}
}

// BuildCoverageReceipt constructs the coverage receipt from scanned lanes, findings, and execution provenance.
func BuildCoverageReceipt(workspace, targetRepo string, lanes []DebtLane, findings []FindingProvenance, scannedFiles int, opts ...CoverageOption) CoverageReceipt {
	cfg := coverageConfig{
		executed: make(map[string]bool),
		skipped:  make(map[string]bool),
		errored:  make(map[string]string),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	observedSurfaces := make(map[string]bool)
	for _, l := range lanes {
		surface := classifySurface(l.UnitOfWork)
		observedSurfaces[string(surface)] = true
	}
	for _, f := range findings {
		if f.Surface != "" {
			observedSurfaces[f.Surface] = true
		}
	}

	var declaredList []string
	if len(cfg.declared) > 0 {
		declaredList = append([]string(nil), cfg.declared...)
	} else {
		declaredList = make([]string, len(StandardDetectorDimensions))
		for i, d := range StandardDetectorDimensions {
			declaredList[i] = string(d)
		}
	}
	sort.Strings(declaredList)

	declaredMap := make(map[string]bool, len(declaredList))
	for _, d := range declaredList {
		declaredMap[d] = true
	}

	observedDimensions := make(map[string]bool)
	if cfg.hasExplicit {
		for d := range cfg.executed {
			if cfg.errored[d] == "" && !cfg.skipped[d] {
				observedDimensions[d] = true
			}
		}
	} else {
		for _, f := range findings {
			if f.Dimension != "" {
				observedDimensions[f.Dimension] = true
			}
		}
	}

	for d := range cfg.errored {
		delete(observedDimensions, d)
	}
	for d := range cfg.skipped {
		delete(observedDimensions, d)
	}

	erroredList := make([]string, 0, len(cfg.errored))
	for d := range cfg.errored {
		erroredList = append(erroredList, d)
	}
	sort.Strings(erroredList)

	combinedFindings := make([]FindingProvenance, 0, len(findings)+len(erroredList))
	combinedFindings = append(combinedFindings, findings...)

	for _, d := range erroredList {
		errMsg := cfg.errored[d]
		combinedFindings = append(combinedFindings, FindingProvenance{
			Dimension: d,
			Surface:   string(SurfaceInternal),
			Lane:      "coverage",
			Path:      workspace,
			Severity:  "critical",
			Message:   fmt.Sprintf("coverage debt: detector %s execution error: %s", d, errMsg),
		})
	}

	coverageDebt := len(erroredList)

	gapList := make([]string, 0, len(declaredList))
	skippedList := make([]string, 0, len(declaredList))

	for _, d := range declaredList {
		if !observedDimensions[d] {
			gapList = append(gapList, d)
			if cfg.errored[d] == "" {
				skippedList = append(skippedList, d)
			}
		}
	}
	sort.Strings(gapList)
	sort.Strings(skippedList)

	surfaceList := make([]string, 0, len(observedSurfaces))
	for s := range observedSurfaces {
		surfaceList = append(surfaceList, s)
	}
	sort.Strings(surfaceList)

	dimList := make([]string, 0, len(observedDimensions))
	for d := range observedDimensions {
		dimList = append(dimList, d)
	}
	sort.Strings(dimList)

	executedList := make([]string, len(dimList))
	copy(executedList, dimList)

	baselineBreadth := 3
	baselineDepth := 5
	observedBreadth := len(surfaceList)
	observedDepth := len(dimList)

	breadthRatio := 0.0
	if baselineBreadth > 0 && observedBreadth > 0 {
		breadthRatio = float64(observedBreadth) / float64(baselineBreadth)
	}

	depthRatio := 0.0
	if baselineDepth > 0 && observedDepth > 0 {
		depthRatio = float64(observedDepth) / float64(baselineDepth)
	}

	return CoverageReceipt{
		Schema:             CoverageReceiptSchema,
		Workspace:          workspace,
		TargetRepo:         targetRepo,
		BaselineBreadth:    baselineBreadth,
		ObservedBreadth:    observedBreadth,
		TargetBreadth:      len(StandardSurfaceClasses),
		BreadthRatio:       float64(int(breadthRatio*100)) / 100.0,
		BaselineDepth:      baselineDepth,
		ObservedDepth:      observedDepth,
		TargetDepth:        len(declaredList),
		DepthRatio:         float64(int(depthRatio*100)) / 100.0,
		ScannedUnits:       len(lanes),
		ScannedFiles:       scannedFiles,
		SurfaceClasses:     surfaceList,
		Dimensions:         dimList,
		DeclaredDimensions: declaredList,
		ExecutedDimensions: executedList,
		SkippedDimensions:  skippedList,
		ErroredDimensions:  erroredList,
		Gaps:               gapList,
		CoverageDebt:       coverageDebt,
		FindingsCount:      len(combinedFindings),
		Findings:           combinedFindings,
	}
}

// classifySurface maps a unit of work directory path to its SurfaceClass.
func classifySurface(path string) SurfaceClass {
	norm := filepathToSlash(path)
	norm = strings.TrimPrefix(norm, "./")
	norm = strings.TrimPrefix(norm, "/")
	switch {
	case strings.HasPrefix(norm, "internal/") || norm == "internal":
		return SurfaceInternal
	case strings.HasPrefix(norm, "pkg/") || norm == "pkg":
		return SurfacePkg
	case strings.HasPrefix(norm, "platform/") || norm == "platform":
		return SurfacePlatform
	case strings.HasPrefix(norm, "cmd/") || norm == "cmd":
		return SurfaceCmd
	case strings.HasPrefix(norm, "tools/") || norm == "tools":
		return SurfaceTools
	case strings.HasPrefix(norm, ".claude/skills/") || strings.HasPrefix(norm, ".agents/skills/") || strings.Contains(norm, "skills/"):
		return SurfaceSkills
	case strings.HasPrefix(norm, ".github/workflows") || strings.Contains(norm, "workflows/"):
		return SurfaceWorkflows
	case strings.HasPrefix(norm, "examples/") || norm == "examples":
		return SurfaceExamples
	case strings.HasPrefix(norm, "docs/") || norm == "docs":
		return SurfaceDocs
	case strings.HasPrefix(norm, "unsupported/") || norm == "unsupported" || strings.Contains(norm, "unsupported"):
		return SurfaceUnsupported
	default:
		return SurfaceUnknown
	}
}

func filepathToSlash(p string) string {
	var b []byte
	for i := 0; i < len(p); i++ {
		if p[i] == '\\' {
			b = append(b, '/')
		} else {
			b = append(b, p[i])
		}
	}
	return string(b)
}
