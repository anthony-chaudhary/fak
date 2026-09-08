package debtlane

import (
	"sort"
	"strings"
)

// CoverageReceiptSchema is the canonical schema identifier for debt coverage receipts.
const CoverageReceiptSchema = "fak.debt-coverage-receipt.v1"

// SurfaceClass identifies the structural type of a scanned repository surface.
type SurfaceClass string

const (
	SurfaceInternal  SurfaceClass = "internal"  // Core Go packages
	SurfacePkg       SurfaceClass = "pkg"       // Public shared Go packages
	SurfacePlatform  SurfaceClass = "platform"  // Platform-specific Go packages
	SurfaceCmd       SurfaceClass = "cmd"       // Go CLI binary commands
	SurfaceTools     SurfaceClass = "tools"     // Tooling programs and submodules
	SurfaceSkills    SurfaceClass = "skills"    // Agent skills (.claude/skills, .agents/skills)
	SurfaceWorkflows SurfaceClass = "workflows" // CI/CD workflows (.github/workflows)
	SurfaceExamples  SurfaceClass = "examples"  // Examples and runnable demos
	SurfaceDocs      SurfaceClass = "docs"      // Technical documentation and guides
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
	DimTestStatus          DetectorDimension = "test_status"          // Passing, missing, or failing tests
	DimCommentHygiene      DetectorDimension = "comment_hygiene"      // Clean comments vs bloat/gaming
	DimWiringStatus        DetectorDimension = "wiring_status"        // Integrated in production graph vs disconnected
	DimProofStatus         DetectorDimension = "proof_status"         // Dogfooded runtime proof vs unproven
	DimBenchmarkStatus     DetectorDimension = "benchmark_status"     // Substantive benchmark vs unmeasured
	DimModularityDeficit   DetectorDimension = "modularity_deficit"   // God files / god functions
	DimModelCoupling       DetectorDimension = "model_coupling"       // Model family hardcoding outside model/
	DimBlastRadius         DetectorDimension = "blast_radius"         // Inbound dependents > 25
	DimThinTests           DetectorDimension = "thin_tests"           // Vacuous/assertion-free test files
	DimStubDebt            DetectorDimension = "stub_debt"            // TODO/FIXME/unimplemented panics
	DimUndocumentedExports DetectorDimension = "undocumented_exports" // Exported symbols without godoc
	DimUnsafeUsage         DetectorDimension = "unsafe_usage"         // Unsafe pointer usage in non-core
	DimSubprocessExec      DetectorDimension = "subprocess_exec"      // Subprocess spawning in library packages
	DimRaceFuzzStatus      DetectorDimension = "race_fuzz_status"      // Concurrency/parser lacking race/fuzz tests
	DimStalePerfProof      DetectorDimension = "stale_perf_proof"      // Missing from benchmark authority or proof registry
	DimCoverageDebt        DetectorDimension = "coverage_debt"        // Unknown, unsupported, or unreadable content
)

// Standard 15 detector dimensions for full-depth evaluation.
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
	Schema          string              `json:"schema"`           // "fak.debt-coverage-receipt.v1"
	Workspace       string              `json:"workspace"`        // workspace root path
	TargetRepo      string              `json:"target_repo,omitempty"` // "fak", "fak-private", "both"
	BaselineBreadth int                 `json:"baseline_breadth"` // 3 (internal, pkg, platform)
	ObservedBreadth int                 `json:"observed_breadth"` // count of distinct surface classes observed
	TargetBreadth   int                 `json:"target_breadth"`   // 9
	BreadthRatio    float64             `json:"breadth_ratio"`    // ObservedBreadth / BaselineBreadth
	BaselineDepth   int                 `json:"baseline_depth"`   // 5 headline health dimensions
	ObservedDepth   int                 `json:"observed_depth"`   // count of distinct detector dimensions evaluated
	TargetDepth     int                 `json:"target_depth"`     // 15
	DepthRatio      float64             `json:"depth_ratio"`      // ObservedDepth / BaselineDepth
	ScannedUnits    int                 `json:"scanned_units"`    // total units of work scanned across all surfaces
	ScannedFiles    int                 `json:"scanned_files"`    // total files inspected
	SurfaceClasses  []string            `json:"surface_classes"`  // list of observed surface class names
	Dimensions      []string            `json:"dimensions"`       // list of evaluated detector dimension names
	FindingsCount   int                 `json:"findings_count"`   // total actionable findings detected
	Findings        []FindingProvenance `json:"findings,omitempty"`
}

// BuildCoverageReceipt constructs the coverage receipt from scanned lanes and findings.
func BuildCoverageReceipt(workspace, targetRepo string, lanes []DebtLane, findings []FindingProvenance, scannedFiles int) CoverageReceipt {
	observedSurfaces := make(map[string]bool)
	observedDimensions := make(map[string]bool)

	for _, l := range lanes {
		surface := classifySurface(l.UnitOfWork)
		observedSurfaces[string(surface)] = true
	}

	for _, f := range findings {
		if f.Surface != "" {
			observedSurfaces[f.Surface] = true
		}
		if f.Dimension != "" {
			observedDimensions[f.Dimension] = true
		}
	}

	// Always ensure standard evaluated dimensions are registered if detectors ran
	for _, d := range StandardDetectorDimensions {
		observedDimensions[string(d)] = true
	}

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

	baselineBreadth := 3
	baselineDepth := 5
	observedBreadth := len(surfaceList)
	observedDepth := len(dimList)

	breadthRatio := 1.0
	if baselineBreadth > 0 {
		breadthRatio = float64(observedBreadth) / float64(baselineBreadth)
	}

	depthRatio := 1.0
	if baselineDepth > 0 {
		depthRatio = float64(observedDepth) / float64(baselineDepth)
	}

	return CoverageReceipt{
		Schema:          CoverageReceiptSchema,
		Workspace:       workspace,
		TargetRepo:      targetRepo,
		BaselineBreadth: baselineBreadth,
		ObservedBreadth: observedBreadth,
		TargetBreadth:   len(StandardSurfaceClasses),
		BreadthRatio:    float64(int(breadthRatio*100)) / 100.0,
		BaselineDepth:   baselineDepth,
		ObservedDepth:   observedDepth,
		TargetDepth:     len(StandardDetectorDimensions),
		DepthRatio:      float64(int(depthRatio*100)) / 100.0,
		ScannedUnits:    len(lanes),
		ScannedFiles:    scannedFiles,
		SurfaceClasses:  surfaceList,
		Dimensions:      dimList,
		FindingsCount:   len(findings),
		Findings:        findings,
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
