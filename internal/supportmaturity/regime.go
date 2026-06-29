package supportmaturity

// Regime is the dev-regime / time-horizon a cell sits in, derived from its rung.
type Regime uint8

const (
	R0Explore Regime = iota
	R1Prototype
	R2Optimize
	R3Production
)

// RegimePlaybook is the fixed answer to "how do I work on a cell in this regime?"
type RegimePlaybook struct {
	ID          string
	Name        string
	Expectation string
	Tooling     []string
	ReportStyle string
	WhoOperates string
	StepBudget  StepBudget
}

// StepBudget is the work/effort horizon for a regime. It is the step-count dual
// of dormancy horizons (#1178): dormancy budgets time away from work; this budgets
// steps to advance a rung. Continuous regimes have no finite maximum.
type StepBudget struct {
	MinSteps   int
	MaxSteps   int
	Continuous bool
}

// ScopeDecision is the closed result of checking an observed step count against a
// regime budget.
type ScopeDecision string

const (
	ScopeWithinBudget ScopeDecision = "within-budget"
	ScopeMisScoped    ScopeDecision = "re-regime-or-escalate"
)

// BudgetCheck is the read-back for "is this cell scoped to the right regime?"
type BudgetCheck struct {
	Regime   Regime
	Steps    int
	Budget   StepBudget
	Decision ScopeDecision
}

// Over reports whether steps exceeds this finite budget. Continuous budgets never
// trip the over-budget flag.
func (b StepBudget) Over(steps int) bool {
	return !b.Continuous && b.MaxSteps > 0 && steps > b.MaxSteps
}

var regimePlaybooks = []RegimePlaybook{
	R0Explore: {
		ID:          "R0",
		Name:        "explore",
		Expectation: "can this even work? -- find out cheaply, abandon cheaply",
		Tooling:     []string{"idea-scout", "research notes", "scouts"},
		ReportStyle: "a feasibility note -- is there a path worth a fence at all?",
		WhoOperates: "human + scout",
		StepBudget:  StepBudget{MinSteps: 1, MaxSteps: 9},
	},
	R1Prototype: {
		ID:          "R1",
		Name:        "prototype",
		Expectation: "make it run end-to-end / get a CI oracle",
		Tooling:     []string{"dispatch worker", "get-to-green", "tests"},
		ReportStyle: "green-or-red -- does it run end-to-end and pass a CI oracle?",
		WhoOperates: "dispatch fleet",
		StepBudget:  StepBudget{MinSteps: 10, MaxSteps: 100},
	},
	R2Optimize: {
		ID:          "R2",
		Name:        "optimize",
		Expectation: "make it fast / better",
		Tooling:     []string{"rsiloop", "shipgate", "kernel/compiler tooling", "benches"},
		ReportStyle: "a witnessed bench delta vs the reference baseline (x-speedup)",
		WhoOperates: "autonomous RSI loop",
		StepBudget:  StepBudget{MinSteps: 1000, MaxSteps: 10000},
	},
	R3Production: {
		ID:          "R3",
		Name:        "production",
		Expectation: "keep it that way for users",
		Tooling:     []string{"self-tax #1147 gate", "SLOs", "bgloop", "UX"},
		ReportStyle: "a continuous SLO + regression-gate readout -- is parity held?",
		WhoOperates: "gate + on-call",
		StepBudget:  StepBudget{Continuous: true},
	},
}

// Regimes is the closed roster of dev-regimes in ascending order.
var Regimes = []Regime{R0Explore, R1Prototype, R2Optimize, R3Production}

var regimeForRung = []Regime{
	M0None:       R0Explore,
	M1Fenced:     R1Prototype,
	M2Loads:      R1Prototype,
	M3Runs:       R1Prototype,
	M4Correct:    R2Optimize,
	M5Optimized:  R2Optimize,
	M6Parity:     R3Production,
	M7BeyondSOTA: R3Production,
}

// RegimeFor routes a Rung to its dev-regime. Out-of-range rungs floor to R0.
func RegimeFor(r Rung) Regime {
	if int(r) < len(regimeForRung) {
		return regimeForRung[r]
	}
	return R0Explore
}

// Valid reports whether g is one of the closed R0-R3 regimes.
func (g Regime) Valid() bool { return int(g) < len(regimePlaybooks) }

// String renders the regime's short id.
func (g Regime) String() string {
	if int(g) < len(regimePlaybooks) {
		return regimePlaybooks[g].ID
	}
	return "R?"
}

// Name renders the regime's one-word name.
func (g Regime) Name() string {
	if int(g) < len(regimePlaybooks) {
		return regimePlaybooks[g].Name
	}
	return "unknown"
}

// Label is the short human label for the regime.
func (g Regime) Label() string { return g.Name() }

// Playbook returns the regime's fixed playbook. Out-of-range regimes floor to R0.
func (g Regime) Playbook() RegimePlaybook {
	if int(g) < len(regimePlaybooks) {
		return regimePlaybooks[g]
	}
	return regimePlaybooks[R0Explore]
}

// StepBudget returns the regime's work/effort horizon.
func (g Regime) StepBudget() StepBudget { return g.Playbook().StepBudget }

// CheckStepBudget flags work that has outgrown its regime's horizon. A mis-scoped
// cell should either escalate to a human or be re-regimed to a longer-horizon loop.
func (g Regime) CheckStepBudget(steps int) BudgetCheck {
	b := g.StepBudget()
	decision := ScopeWithinBudget
	if b.Over(steps) {
		decision = ScopeMisScoped
	}
	return BudgetCheck{Regime: g, Steps: steps, Budget: b, Decision: decision}
}

// PlaybookFor routes a Rung to its regime and returns that regime's playbook.
func PlaybookFor(r Rung) (Regime, RegimePlaybook) {
	g := RegimeFor(r)
	return g, g.Playbook()
}
