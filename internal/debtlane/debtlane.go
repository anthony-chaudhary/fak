package debtlane

import (
	"time"
)

// Schema is the canonical schema identifier for the maturity debt lane scorecard and ledger.
const Schema = "fak.maturity-debt-lane.v1"

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

// Evidence captures verified facts discovered from disk for a unit of work.
type Evidence struct {
	FilesCount             int  `json:"files_count"`
	TestFilesCount         int  `json:"test_files_count"`
	CodeLines              int  `json:"code_lines"`
	HasCode                bool `json:"has_code"`
	HasTests               bool `json:"has_tests"`
	Integrated             bool `json:"integrated"`
	Dogfooded              bool `json:"dogfooded"`
	Benchmarked            bool `json:"benchmarked"`
	Documented             bool `json:"documented"`
	ExportedSymbols        int  `json:"exported_symbols"`
	DocumentedExports      int  `json:"documented_exports"`
	DependentsCount        int  `json:"dependents_count"` // Inbound internal imports (blast radius).
	TransitiveDependencies int  `json:"transitive_dependencies"`
	HasContractComments    bool `json:"has_contract_comments"`
}

// DebtLane represents a dedicated maturity debt lane for one single unit of work.
type DebtLane struct {
	Lane                    string          `json:"lane"`                     // Unique lane identifier (leaf package or subsystem name).
	UnitOfWork              string          `json:"unit_of_work"`             // Primary directory path (e.g. "internal/gateway").
	Criticality             Criticality     `json:"criticality"`              // core, enabling, stewardship, peripheral.
	Weight                  float64         `json:"weight"`                   // Relative weight in production denominator (e.g. 3.0 for core).
	Maturity                float64         `json:"maturity"`                 // Current maturity on 0.0 - 10.0 curve.
	MaturityRung            string          `json:"maturity_rung"`            // Name of closest lifecycle rung.
	TargetMaturity          float64         `json:"target_maturity"`          // Target maturity ceiling under declared bounds.
	MaturityGap             float64         `json:"maturity_gap"`             // max(0, TargetMaturity - Maturity).
	DebtPrincipal           float64         `json:"debt_principal"`           // MaturityGap * Weight.
	Interest                Interest        `json:"interest"`                 // Relative carrying cost rate & drivers.
	CarryingCost            float64         `json:"carrying_cost"`            // DebtPrincipal * Interest.Rate (capped).
	TotalDebt               float64         `json:"total_debt"`               // DebtPrincipal + CarryingCost.
	DenominatorContribution float64         `json:"denominator_contribution"` // TargetMaturity * Weight (adds to production denominator).
	RealizedContribution    float64         `json:"realized_contribution"`    // Maturity * Weight.
	Bounds                  BoundsAndLimits `json:"bounds"`                   // Declared or derived constraints.
	Evidence                Evidence        `json:"evidence"`                 // Ground-truth facts.
	NextAction              string          `json:"next_action"`              // Concrete action to retire debt.
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

// Report is the top-level scorecard payload for dedicated maturity debt lanes.
type Report struct {
	Schema          string          `json:"schema"`
	OK              bool            `json:"ok"`
	Verdict         string          `json:"verdict"` // OK or ACTION.
	Finding         string          `json:"finding"`
	Reason          string          `json:"reason"`
	NextAction      string          `json:"next_action"`
	Workspace       string          `json:"workspace"`
	EvaluatedAt     string          `json:"evaluated_at"`
	Corpus          map[string]any  `json:"corpus"`
	ProductionGrade ProductionGrade `json:"production_grade"`
	InterestSummary InterestSummary `json:"interest_summary"`
	Lanes           []DebtLane      `json:"lanes"`
	Hotspots        []DebtLane      `json:"hotspots"` // Top debt lanes ranked worst-first.
}

// Options parameters for scanning and evaluating debt lanes.
type Options struct {
	WorkspaceRoot     string
	LaneFilter        string
	MinGap            float64
	CriticalityFilter string
	TopN              int
	// Facts override allows tests to inject hermetic unit of work facts without disk I/O.
	Facts []DebtLane
	// Clock allows deterministic timestamp injection in tests.
	Clock func() time.Time
}
