package turnbench

import (
	"runtime"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionreset"
)

const ManagedContextSoakVersion = "fak.managed-context-soak.v1"

// ManagedContextSoakConfig sizes the deterministic reset soak. The default runs
// enough cycles to catch reset drift without growing runtime state.
type ManagedContextSoakConfig struct {
	ResetCycles          int                `json:"reset_cycles"`
	ContextBudgetTokens  int                `json:"context_budget_tokens"`
	ResidentBudgetTokens int                `json:"resident_budget_tokens"`
	Goal                 string             `json:"goal"`
	AssumptionNeedles    []string           `json:"assumption_needles,omitempty"`
	QueryEvery           int                `json:"query_every,omitempty"`
	InitialMessages      []sessionreset.Msg `json:"initial_messages,omitempty"`
}

// DefaultManagedContextSoakConfig gives the issue #1623 witness shape: many
// resets, bounded resident seed, and a couple of context facts that must survive.
func DefaultManagedContextSoakConfig() ManagedContextSoakConfig {
	return ManagedContextSoakConfig{
		ResetCycles:          1000,
		ContextBudgetTokens:  256,
		ResidentBudgetTokens: 512,
		Goal:                 "ship managed context across hidden resets",
		AssumptionNeedles: []string{
			"budget stays bounded",
			"ask when context is uncertain",
		},
		QueryEvery: 50,
	}
}

// ManagedContextFailure is one deterministic failure observed during the soak.
type ManagedContextFailure struct {
	Cycle  int    `json:"cycle"`
	Kind   string `json:"kind"`
	Detail string `json:"detail"`
}

// ManagedContextSoakReport is the artifact #1623 asks for: reset count, resident
// bounds, and concrete failure rows for continuity, budget, and query behavior.
type ManagedContextSoakReport struct {
	Schema              string                   `json:"schema"`
	Provenance          Provenance               `json:"provenance"`
	Config              ManagedContextSoakConfig `json:"config"`
	ResetCount          int                      `json:"reset_count"`
	PeakResidentTokens  int                      `json:"peak_resident_tokens"`
	ResidentBoundTokens int                      `json:"resident_bound_tokens"`
	WarmPrefixHits      int                      `json:"warm_prefix_hits"`
	QueryChecks         int                      `json:"query_checks"`
	ContinuityFailures  []ManagedContextFailure  `json:"continuity_failures,omitempty"`
	BudgetViolations    []ManagedContextFailure  `json:"budget_violations,omitempty"`
	QueryFailures       []ManagedContextFailure  `json:"query_failures,omitempty"`
}

// OK reports whether the soak preserved continuity and its resident bound.
func (r ManagedContextSoakReport) OK() bool {
	return len(r.ContinuityFailures) == 0 && len(r.BudgetViolations) == 0 && len(r.QueryFailures) == 0
}

// JSON renders the report.
func (r *ManagedContextSoakReport) JSON() []byte { return marshalArtifact(r) }

// RunManagedContextSoak drives a long goal through repeated budget resets using
// the real session reset seam: DebitUsage drains the current trace, BuildSeed
// creates the carryover, and Recontinue mints the next trace.
func RunManagedContextSoak(cfg ManagedContextSoakConfig) ManagedContextSoakReport {
	cfg = normalizeManagedContextSoakConfig(cfg)
	trace := "managed-context-0000"
	budget := session.Budget{
		TurnsLeft:         session.Unbounded,
		TokensLeft:        session.Unbounded,
		ContextTokensLeft: cfg.ContextBudgetTokens,
		ContextTokensCap:  cfg.ContextBudgetTokens,
	}
	tbl := session.NewTable()
	tbl.SetBudget(trace, budget)
	messages := initialManagedContextMessages(cfg)
	rep := ManagedContextSoakReport{
		Schema:              ManagedContextSoakVersion,
		Provenance:          managedContextProvenance(),
		Config:              cfg,
		ResidentBoundTokens: cfg.ResidentBudgetTokens,
	}
	for cycle := 1; cycle <= cfg.ResetCycles; cycle++ {
		drained := tbl.DebitUsage(trace, session.Usage{ContextTokens: cfg.ContextBudgetTokens})
		if drained.ContinuationID == "" {
			rep.BudgetViolations = append(rep.BudgetViolations, ManagedContextFailure{
				Cycle: cycle, Kind: "reset", Detail: "session drain did not mint a continuation id",
			})
			drained.ContinuationID = trace + "-continuation"
		}
		seed := sessionreset.BuildSeed(sessionreset.Input{
			Trace:          trace,
			Messages:       messages,
			FreshBudgetTok: cfg.ContextBudgetTokens,
		})
		resident := approxManagedContextTokens(seed.Recap)
		if resident > rep.PeakResidentTokens {
			rep.PeakResidentTokens = resident
		}
		if resident > cfg.ResidentBudgetTokens {
			rep.BudgetViolations = append(rep.BudgetViolations, ManagedContextFailure{
				Cycle: cycle, Kind: "resident_bound", Detail: "carryover seed exceeded resident token bound",
			})
		}
		checkContinuity(&rep, cfg, cycle, seed.Recap)
		if hasPart(seed, "warm_prefix") {
			rep.WarmPrefixHits++
		}
		if cfg.QueryEvery > 0 && cycle%cfg.QueryEvery == 0 {
			rep.QueryChecks++
			if !containsFold(seed.Recap, "clarification answer") {
				rep.QueryFailures = append(rep.QueryFailures, ManagedContextFailure{
					Cycle: cycle, Kind: "query_answer", Detail: "carryover seed lost the clarification answer marker",
				})
			}
		}
		child := drained.ContinuationID
		fresh := tbl.Recontinue(trace, child, budget)
		if fresh.Generation != cycle {
			rep.BudgetViolations = append(rep.BudgetViolations, ManagedContextFailure{
				Cycle: cycle, Kind: "generation", Detail: "recontinued generation did not match reset count",
			})
		}
		rep.ResetCount++
		trace = child
		messages = nextManagedContextMessages(cfg, seed.Recap, cycle)
	}
	return rep
}

func normalizeManagedContextSoakConfig(cfg ManagedContextSoakConfig) ManagedContextSoakConfig {
	def := DefaultManagedContextSoakConfig()
	if cfg.ResetCycles <= 0 {
		cfg.ResetCycles = def.ResetCycles
	}
	if cfg.ContextBudgetTokens <= 0 {
		cfg.ContextBudgetTokens = def.ContextBudgetTokens
	}
	if cfg.ResidentBudgetTokens <= 0 {
		cfg.ResidentBudgetTokens = def.ResidentBudgetTokens
	}
	if strings.TrimSpace(cfg.Goal) == "" {
		cfg.Goal = def.Goal
	}
	cfg.Goal = strings.TrimSpace(cfg.Goal)
	if cfg.AssumptionNeedles == nil {
		cfg.AssumptionNeedles = append([]string(nil), def.AssumptionNeedles...)
	} else {
		cfg.AssumptionNeedles = cleanNeedles(cfg.AssumptionNeedles)
	}
	if cfg.QueryEvery < 0 {
		cfg.QueryEvery = 0
	}
	return cfg
}

func cleanNeedles(in []string) []string {
	out := make([]string, 0, len(in))
	for _, s := range in {
		if trimmed := strings.TrimSpace(s); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func managedContextProvenance() Provenance {
	return Provenance{
		AppVersion:  appversion.Current(),
		Command:     "turnbench managed-context-soak",
		SliceID:     "managed-context-soak",
		GoVersion:   runtime.Version(),
		OS:          runtime.GOOS + "/" + runtime.GOARCH,
		GeneratedBy: "fak/internal/turnbench (managed-context reset soak)",
	}
}

func initialManagedContextMessages(cfg ManagedContextSoakConfig) []sessionreset.Msg {
	if len(cfg.InitialMessages) > 0 {
		return append([]sessionreset.Msg(nil), cfg.InitialMessages...)
	}
	return []sessionreset.Msg{
		{Role: "system", Content: "fak managed-context stable prefix."},
		{Role: "user", Content: "Goal: " + cfg.Goal + ". Assumptions: " + strings.Join(cfg.AssumptionNeedles, "; ") + "."},
		{Role: "assistant", Content: "I will preserve objective, assumptions, and budget state across resets."},
		{Role: "user", Content: "Clarification answer: " + strings.Join(cfg.AssumptionNeedles, "; ") + "."},
	}
}

func nextManagedContextMessages(cfg ManagedContextSoakConfig, seed string, cycle int) []sessionreset.Msg {
	ledger := "Goal: " + cfg.Goal + ". Assumptions: " + strings.Join(cfg.AssumptionNeedles, "; ") +
		". Clarification answer: " + strings.Join(cfg.AssumptionNeedles, "; ") + "."
	return []sessionreset.Msg{
		{Role: "system", Content: "fak managed-context stable prefix."},
		{Role: "user", Content: ledger},
		{Role: "assistant", Content: "Managed-context reset cycle " + strconv.Itoa(cycle) + " completed."},
		{Role: "user", Content: seed},
	}
}

func checkContinuity(rep *ManagedContextSoakReport, cfg ManagedContextSoakConfig, cycle int, recap string) {
	if !containsFold(recap, cfg.Goal) {
		rep.ContinuityFailures = append(rep.ContinuityFailures, ManagedContextFailure{
			Cycle: cycle, Kind: "goal", Detail: "carryover seed lost the goal",
		})
	}
	for _, needle := range cfg.AssumptionNeedles {
		if !containsFold(recap, needle) {
			rep.ContinuityFailures = append(rep.ContinuityFailures, ManagedContextFailure{
				Cycle: cycle, Kind: "assumption", Detail: "carryover seed lost assumption: " + needle,
			})
		}
	}
}

func hasPart(seed sessionreset.Seed, name string) bool {
	for _, p := range seed.Parts {
		if p.Name == name {
			return true
		}
	}
	return false
}

func containsFold(s, sub string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(strings.TrimSpace(sub)))
}

func approxManagedContextTokens(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}
