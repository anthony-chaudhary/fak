package debtlane

import (
	"time"
)

// Schema is the canonical schema identifier for the maturity debt lane scorecard and ledger.
const Schema = "fak.maturity-debt-lane.v1"

// Performance proof freshness failure reasons.
const (
	ReasonMismatchedRevision        = "mismatched_revision"
	ReasonMismatchedWorkload        = "mismatched_workload"
	ReasonMismatchedQualityEnvelope = "mismatched_quality_envelope"
	ReasonReferenceEngine           = "reference_engine"
	ReasonMissingPerfProof          = "missing_performance_proof"
	ReasonStaleMeasurementAge       = "stale_measurement_age"
	ReasonIncompatibleScope         = "incompatible_scope"
)

// PerformanceProof captures verified, workload-bound, and quality-constrained performance authority records.
type PerformanceProof struct {
	Lane              string `json:"lane,omitempty"`
	Revision          string `json:"revision,omitempty"`           // Source revision (e.g. module@rev "internal/gateway@r10+g33144e097" or commit SHA)
	Engine            string `json:"engine,omitempty"`             // Execution engine ("fak-native", "fak", or external reference like "llama.cpp")
	Workload          string `json:"workload,omitempty"`           // Bound workload description (e.g. "T=50 A=5 P=2048", "batch=1,ctx=2048")
	QualityEnvelope   string `json:"quality_envelope,omitempty"`   // Bound quality constraints (e.g. "Q8_0,lossless", "matched-weights,cosine=1.0")
	ObservedAt        string `json:"observed_at,omitempty"`        // RFC3339 timestamp of the measurement
	Artifact          string `json:"artifact,omitempty"`           // Path to committed benchmark artifact
	IncompatibleScope string `json:"incompatible_scope,omitempty"` // Explicit incompatible scope for historical preservation (e.g. "qwen3.6-historical")
	IsHistorical      bool   `json:"is_historical,omitempty"`      // Preserved as historical evidence
	Reason            string `json:"reason,omitempty"`             // Evaluated freshness verdict or failure reason
}

// Criticality describes a unit of work's architectural role and blast radius.
type Criticality string

const (
	CriticalityCore        Criticality = "core"        // Core runtime, mediation, security, or data plane.
	CriticalityEnabling    Criticality = "enabling"    // Infrastructure, execution engines, schedulers, adapters.
	CriticalityStewardship Criticality = "stewardship" // Governance, linting, testing, release, hygiene.
	CriticalityPeripheral  Criticality = "peripheral"  // Demos, experiments, visualization, optional tools.
)

// PacingTier defines how urgently a unit of work is expected to advance on its maturity curve.
type PacingTier string

const (
	PacingUrgent   PacingTier = "urgent"   // Active bottleneck; must reach production grade promptly.
	PacingStandard PacingTier = "standard" // Regular delivery rhythm; standard interest rate.
	PacingRelaxed  PacingTier = "relaxed"  // Experimental, exploratory; lower interest carrying cost.
	PacingFrozen   PacingTier = "frozen"   // Quiescent / reference artifact; debt gap clamped to zero.
)

// InterestBand categorizes the carrying cost risk.
type InterestBand string

const (
	InterestLow      InterestBand = "low"      // Baseline carrying cost (0 - 5%).
	InterestModerate InterestBand = "moderate" // Elevated carrying cost (6 - 15%).
	InterestHigh     InterestBand = "high"     // Accelerating carrying cost (16 - 25%).
	InterestCritical InterestBand = "critical" // Compounding carrying cost (> 25%).
)

// BoundsAndLimits specifies constraints on a debt lane's target ceiling, interest, and pacing.
type BoundsAndLimits struct {
	TargetCeiling   float64    `json:"target_ceiling"`    // Maximum maturity required (e.g. 10.0 for core, 4.0 for peripheral).
	Pacing          PacingTier `json:"pacing"`            // Urgent, standard, relaxed, or frozen.
	MaxInterestCap  float64    `json:"max_interest_cap"`  // Upper bound on effective interest rate (e.g. 0.35).
	CarryingCostCap float64    `json:"carrying_cost_cap"` // Upper bound on carrying cost amount.
}

// Interest details the carrying cost rate, band, and causal drivers.
type Interest struct {
	Band      InterestBand `json:"band"`       // low, moderate, high, critical.
	Rate      float64      `json:"rate"`       // Effective interest rate as a fraction (e.g. 0.18 for 18%).
	RateLabel string       `json:"rate_label"` // baseline, elevated, accelerating, compounding.
	Drivers   []string     `json:"drivers"`    // Structural reasons determining this rate.
}

// HealthStatus categorizes the operational and maturity health of a lane.
type HealthStatus string

const (
	HealthHealthy  HealthStatus = "healthy"  // High maturity, passing tests, integrated, clean comments.
	HealthDegraded HealthStatus = "degraded" // Untested stubs, excess comments, disconnected, or unbenchmarked.
	HealthCritical HealthStatus = "critical" // Critical compounding carrying cost (>25%) or untested core paths.
)

// LaneHealth captures multi-dimensional health metrics and specific findings for a debt lane.
type LaneHealth struct {
	Status          HealthStatus `json:"status"`           // healthy, degraded, critical.
	Score           float64      `json:"score"`            // 0.0 - 1.0 composite health score.
	TestStatus      string       `json:"test_status"`      // passing, missing, failing.
	CommentHygiene  string       `json:"comment_hygiene"`  // clean, bloat.
	WiringStatus    string       `json:"wiring_status"`    // integrated, disconnected.
	ProofStatus     string       `json:"proof_status"`     // dogfooded, unproven.
	BenchmarkStatus string       `json:"benchmark_status"` // benchmarked, unmeasured.
	Issues          []string     `json:"issues,omitempty"` // specific health issue tokens.
}

// RelatedThings cross-indexes related items, companions, dependencies, and artifacts for a debt lane.
type RelatedThings struct {
	CompanionRepo       string   `json:"companion_repo,omitempty"`         // Opposing repo (fak <-> fak-private).
	CompanionLane       string   `json:"companion_lane,omitempty"`         // Corresponding companion lane name.
	CompanionUnitOfWork string   `json:"companion_unit_of_work,omitempty"` // Directory path in companion repo.
	Dependents          []string `json:"dependents,omitempty"`             // Internal packages importing this lane.
	Dependencies        []string `json:"dependencies,omitempty"`           // Internal packages imported by this lane.
	DosTrees            []string `json:"dos_trees,omitempty"`              // Tree globs declared in dos.toml.
	ProofWitnesses      []string `json:"proof_witnesses,omitempty"`        // Runtime proof entries / IDs.
	BenchmarkWitnesses  []string `json:"benchmark_witnesses,omitempty"`    // Benchmark authority mentions.
}

// Evidence captures verified facts discovered from disk for a unit of work.
type Evidence struct {
	FilesCount             int                `json:"files_count"`
	TestFilesCount         int                `json:"test_files_count"`
	CodeLines              int                `json:"code_lines"`
	CommentLines           int                `json:"comment_lines"`
	CommentRatio           float64            `json:"comment_ratio"`
	ExcessComments         bool               `json:"excess_comments"`
	HasCode                bool               `json:"has_code"`
	HasTests               bool               `json:"has_tests"`
	Integrated             bool               `json:"integrated"`
	Dogfooded              bool               `json:"dogfooded"`
	Benchmarked            bool               `json:"benchmarked"`
	Documented             bool               `json:"documented"`
	ExportedSymbols        int                `json:"exported_symbols"`
	DocumentedExports      int                `json:"documented_exports"`
	DependentsCount        int                `json:"dependents_count"` // Inbound internal imports (blast radius).
	TransitiveDependencies int                `json:"transitive_dependencies"`
	HasContractComments    bool               `json:"has_contract_comments,omitempty"` // Deprecated: formulaic comments do not award maturity points.
	GodFilesCount          int                `json:"god_files_count,omitempty"`
	GodFuncsCount          int                `json:"god_funcs_count,omitempty"`
	MaxFileLines           int                `json:"max_file_lines,omitempty"`
	MaxFuncLines           int                `json:"max_func_lines,omitempty"`
	ModelHardcodingCount   int                `json:"model_hardcoding_count,omitempty"`
	HasModelHardcoding     bool               `json:"has_model_hardcoding,omitempty"`
	HighCoupling           bool               `json:"high_coupling,omitempty"`
	ModularityDeficit      bool               `json:"modularity_deficit,omitempty"`
	ModularityIssues       []string           `json:"modularity_issues,omitempty"`
	PerformanceProof       *PerformanceProof  `json:"performance_proof,omitempty"`
	HistoricalProofs       []PerformanceProof `json:"historical_proofs,omitempty"`
	CurrentRevision        string             `json:"current_revision,omitempty"`
	RequiredWorkload       string             `json:"required_workload,omitempty"`
	RequiredQuality        string             `json:"required_quality,omitempty"`
	PerfProofReason        string             `json:"perf_proof_reason,omitempty"`
}

// DebtLane represents a dedicated maturity debt lane for one single unit of work.
type DebtLane struct {
	Lane                    string              `json:"lane"`                       // Unique lane identifier (leaf package or subsystem name).
	Repo                    string              `json:"repo,omitempty"`             // Repository name (e.g. "fak", "fak-private").
	UnitOfWork              string              `json:"unit_of_work"`               // Primary directory path (e.g. "internal/gateway" or "platform/dispatch").
	Criticality             Criticality         `json:"criticality"`                // core, enabling, stewardship, peripheral.
	Weight                  float64             `json:"weight"`                     // Relative weight in production denominator (e.g. 3.0 for core).
	Maturity                float64             `json:"maturity"`                   // Current maturity on 0.0 - 10.0 curve.
	MaturityRung            string              `json:"maturity_rung"`              // Name of closest lifecycle rung.
	TargetMaturity          float64             `json:"target_maturity"`            // Target maturity ceiling under declared bounds.
	MaturityGap             float64             `json:"maturity_gap"`               // max(0, TargetMaturity - Maturity).
	DebtPrincipal           float64             `json:"debt_principal"`             // MaturityGap * Weight.
	Interest                Interest            `json:"interest"`                   // Relative carrying cost rate & drivers.
	CarryingCost            float64             `json:"carrying_cost"`              // DebtPrincipal * Interest.Rate (capped).
	TotalDebt               float64             `json:"total_debt"`                 // DebtPrincipal + CarryingCost.
	DenominatorContribution float64             `json:"denominator_contribution"`   // TargetMaturity * Weight (adds to production denominator).
	RealizedContribution    float64             `json:"realized_contribution"`      // Maturity * Weight.
	Bounds                  BoundsAndLimits     `json:"bounds"`                     // Declared or derived constraints.
	Evidence                Evidence            `json:"evidence"`                   // Ground-truth facts.
	Related                 RelatedThings       `json:"related"`                    // Cross-indexed related items and companions.
	Health                  LaneHealth          `json:"health"`                     // Multi-dimensional health verdict and score.
	NextAction              string              `json:"next_action"`                // Concrete action to retire debt.
	OpencodeCommand         []string            `json:"opencode_command,omitempty"` // Ready-to-run OpenCode worker command.
	PerformanceProof        *PerformanceProof   `json:"performance_proof,omitempty"`
	HistoricalProofs        []PerformanceProof  `json:"historical_proofs,omitempty"`
	CurrentRevision         string              `json:"current_revision,omitempty"`
	RequiredWorkload        string              `json:"required_workload,omitempty"`
	RequiredQuality         string              `json:"required_quality,omitempty"`
	PerfProofReason         string              `json:"perf_proof_reason,omitempty"`
	IsPerformance           bool                `json:"is_performance,omitempty"`  // Explicit performance lane override
	NonPerformance          bool                `json:"non_performance,omitempty"` // Explicit non-performance lane override
	CriticalPath            *CriticalPathInfo   `json:"critical_path,omitempty"`   // Production critical-path reachability and provenance (#12361).
	Findings                []string            `json:"findings,omitempty"`        // Summary messages for actionable findings on this lane.
	FindingProvs            []FindingProvenance `json:"finding_provs,omitempty"`   // Provenance-rich typed debt findings.
}

// IsPerformanceLane returns true if the lane represents a performance-critical path that requires proof freshness.
func (l DebtLane) IsPerformanceLane() bool {
	if l.NonPerformance {
		return false
	}
	if l.IsPerformance {
		return true
	}
	if l.Criticality == CriticalityStewardship || l.Criticality == CriticalityPeripheral {
		return false
	}
	return l.Criticality == CriticalityCore || l.Criticality == CriticalityEnabling
}

// ProductionGrade holds the system-wide denominator and realized production-readiness metrics.
type ProductionGrade struct {
	DenominatorPoints    float64 `json:"denominator_points"`     // Total production-grade denominator (all units of work).
	RealizedPoints       float64 `json:"realized_points"`        // Total matured points currently realized.
	GradePercent         float64 `json:"grade_percent"`          // 100 * RealizedPoints / DenominatorPoints.
	GradeLetter          string  `json:"grade_letter"`           // A, B, C, D, F.
	DilutionFromWIP      float64 `json:"dilution_from_wip"`      // Percentage points lost to immature WIP in the denominator.
	TotalUnits           int     `json:"total_units"`            // Total units of work tracked.
	ProductionReadyUnits int     `json:"production_ready_units"` // Units meeting or exceeding target maturity.
	WIPUnits             int     `json:"wip_units"`              // Units with active maturity debt.
}

// InterestSummary summarizes interest distribution across the system.
type InterestSummary struct {
	Bands       map[string]int `json:"bands"`        // Count of lanes in low, moderate, high, critical.
	AverageRate float64        `json:"average_rate"` // Mean carrying cost rate across active debt lanes.
	MaxRate     float64        `json:"max_rate"`     // Peak interest rate in the system.
}

// HealthSummary summarizes health distribution and average health score across the system.
type HealthSummary struct {
	HealthyCount  int     `json:"healthy_count"`
	DegradedCount int     `json:"degraded_count"`
	CriticalCount int     `json:"critical_count"`
	AverageScore  float64 `json:"average_score"`
}

// Report is the top-level scorecard payload for dedicated maturity debt lanes.
type Report struct {
	Schema          string           `json:"schema"`
	OK              bool             `json:"ok"`
	Verdict         string           `json:"verdict"` // OK or ACTION.
	Finding         string           `json:"finding"`
	Reason          string           `json:"reason"`
	NextAction      string           `json:"next_action"`
	Workspace       string           `json:"workspace"`
	TargetRepo      string           `json:"target_repo,omitempty"` // "fak", "fak-private", "both"
	EvaluatedAt     string           `json:"evaluated_at"`
	Corpus          map[string]any   `json:"corpus"`
	ProductionGrade ProductionGrade  `json:"production_grade"`
	InterestSummary InterestSummary  `json:"interest_summary"`
	HealthSummary   HealthSummary    `json:"health_summary"`
	Lanes           []DebtLane       `json:"lanes"`
	Hotspots        []DebtLane       `json:"hotspots"` // Top debt lanes ranked worst-first.
	WavePlan        *WavePlan        `json:"wave_plan,omitempty"`
	Coverage        *CoverageReceipt `json:"coverage,omitempty"`       // 3x discovery breadth & detector depth receipt (#12318).
	CriticalPaths   CriticalPathMap  `json:"critical_paths,omitempty"` // Production critical path map (#12361).
}

// Options parameters for scanning and evaluating debt lanes.
type Options struct {
	WorkspaceRoot     string
	TargetRepo        string // "fak", "fak-private", "both" (default: "fak")
	PrivateRoot       string // optional explicit path to fak-private
	LaneFilter        string
	QueryFilter       string // substring/regex query over lane name, unit, drivers, issues, related
	HealthFilter      string // filter by health status: "healthy", "degraded", "critical"
	SurfaceFilter     string // filter by surface class: "internal", "pkg", "cmd", "tools", "skills", etc.
	NoExpandedBreadth bool   // opt-out to disable expanded 9-surface discovery
	ExpandedBreadth   bool   // inventory at least 9 surface classes (#12318)
	DeepDetectors     bool   // evaluate at least 15 typed detector dimensions (#12318)
	CrossIndex        bool   // output rich cross-indexed related items
	MinGap            float64
	CriticalityFilter string
	TopN              int
	// Facts override allows tests to inject hermetic unit of work facts without disk I/O.
	Facts []DebtLane
	// Graph override allows tests to inject an import dependency graph in tests.
	Graph map[string]map[string]struct{}
	// Clock allows deterministic timestamp injection in tests.
	Clock func() time.Time
}
