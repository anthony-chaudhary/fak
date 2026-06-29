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
}

var regimePlaybooks = []RegimePlaybook{
	R0Explore: {
		ID:          "R0",
		Name:        "explore",
		Expectation: "can this even work? -- find out cheaply, abandon cheaply",
		Tooling:     []string{"idea-scout", "research notes", "scouts"},
		ReportStyle: "a feasibility note -- is there a path worth a fence at all?",
		WhoOperates: "human + scout",
	},
	R1Prototype: {
		ID:          "R1",
		Name:        "prototype",
		Expectation: "make it run end-to-end / get a CI oracle",
		Tooling:     []string{"dispatch worker", "get-to-green", "tests"},
		ReportStyle: "green-or-red -- does it run end-to-end and pass a CI oracle?",
		WhoOperates: "dispatch fleet",
	},
	R2Optimize: {
		ID:          "R2",
		Name:        "optimize",
		Expectation: "make it fast / better",
		Tooling:     []string{"rsiloop", "shipgate", "kernel/compiler tooling", "benches"},
		ReportStyle: "a witnessed bench delta vs the reference baseline (x-speedup)",
		WhoOperates: "autonomous RSI loop",
	},
	R3Production: {
		ID:          "R3",
		Name:        "production",
		Expectation: "keep it that way for users",
		Tooling:     []string{"self-tax #1147 gate", "SLOs", "bgloop", "UX"},
		ReportStyle: "a continuous SLO + regression-gate readout -- is parity held?",
		WhoOperates: "gate + on-call",
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

// PlaybookFor routes a Rung to its regime and returns that regime's playbook.
func PlaybookFor(r Rung) (Regime, RegimePlaybook) {
	g := RegimeFor(r)
	return g, g.Playbook()
}
