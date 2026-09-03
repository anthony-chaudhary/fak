package orchestration

import "strings"

// SOLMode names the execution strategy FAK selects for a guarded issue worker.
type SOLMode string

const (
	SOLStandard SOLMode = "standard"
	SOLMax      SOLMode = "max"
	SOLUltra    SOLMode = "ultra"
	SOLPro      SOLMode = "pro"
)

// SOLRoute is a provider-honest execution decision. Pro remains consult-only
// until Codex can put reasoning.mode on the Responses request; Ultra is FAK's
// guarded multi-worker workflow, not a reasoning-effort alias.
type SOLRoute struct {
	Mode            SOLMode `json:"mode"`
	Model           string  `json:"model"`
	ReasoningEffort string  `json:"reasoning_effort"`
	ReasoningMode   string  `json:"reasoning_mode,omitempty"`
	MultiAgent      bool    `json:"multi_agent"`
	ConsultOnly     bool    `json:"consult_only,omitempty"`
	Decision        string  `json:"decision"`
}

var (
	rigorSignals = []string{
		"architect", "adversarial", "ambiguous", "audit", "benchmark", "correctness",
		"critical", "design", "diagnos", "hard root", "invariant", "mathemat", "migrat",
		"race", "research", "root cause", "security", "trade-off", "uncertain", "unknown",
		"verify", "witness",
	}
	parallelSignals = []string{
		"all night", "backlog", "bulk", "fan out", "fan-out", "fleet", "independent",
		"many issues", "multi-agent", "multi agent", "parallel", "several issues", "super loop",
		"super-loop", "unattended", "wave",
	}
	consultSignals = []string{
		"consult pro", "pro review", "pro mode", "reasoning.mode=pro", "reasoning mode pro",
	}
)

// SelectSOLRoute chooses the smallest execution mode justified by the issue.
// Profile-off is the escape hatch; explicit Pro language is recorded as a
// consult request but never misrepresented as enabled on a Codex worker.
func SelectSOLRoute(task string, profile Profile, workClass WorkClass, model string) SOLRoute {
	lower := strings.ToLower(strings.TrimSpace(task))
	if model == "" {
		model = "gpt-5.6-sol"
	}
	route := SOLRoute{
		Mode:            SOLStandard,
		Model:           model,
		ReasoningEffort: "high",
		Decision:        "single worker at high effort is the least-cost adequate route",
	}
	if profile == ProfileOff {
		route.ReasoningEffort = "medium"
		route.Decision = "orchestration is disabled, so retain the provider default-sized route"
		return route
	}
	if containsAny(lower, consultSignals...) {
		route.Mode = SOLPro
		route.ReasoningEffort = "xhigh"
		route.ReasoningMode = "pro"
		route.ConsultOnly = true
		route.Decision = "explicit Pro consultation requested; keep it separate because Codex cannot yet transmit reasoning.mode"
		return route
	}
	rigor := workClass == WorkRigor || containsAny(lower, rigorSignals...)
	if profile == ProfileUltracode || containsAny(lower, parallelSignals...) || (workClass == WorkGrind && profile == ProfileAuto) {
		route.Mode = SOLUltra
		route.MultiAgent = true
		if rigor {
			route.ReasoningEffort = "xhigh"
			route.Decision = "independent work can be delegated, but correctness or uncertainty keeps each worker at maximum effort"
		} else {
			route.Decision = "independent issue work can be delegated behind leases and reconciled by witnesses"
		}
		return route
	}
	if rigor {
		route.Mode = SOLMax
		route.ReasoningEffort = "xhigh"
		route.Decision = "correctness or uncertainty dominates, so spend the maximum supported reasoning effort"
	}
	return route
}

// RouteResolution applies task-text evidence that is intentionally absent from
// the stable fixture schema. Callers use it before rendering or launching.
func RouteResolution(res *Resolution, taskText string, model string) {
	if res == nil {
		return
	}
	profile := res.Resolved.Profile
	if res.Requested.Name == ProfileAuto {
		profile = ProfileAuto
	}
	pinnedEffort := ""
	if strings.Contains(res.Resolved.SOLRoute.Decision, "; effort pinned by operator to ") {
		pinnedEffort = res.Resolved.SOLRoute.ReasoningEffort
	}
	res.Resolved.SOLRoute = SelectSOLRoute(taskText, profile, res.Resolved.WorkClass, model)
	if pinnedEffort != "" {
		res.Resolved.SOLRoute.ReasoningEffort = pinnedEffort
		res.Resolved.SOLRoute.Decision += "; effort pinned by operator to " + pinnedEffort
	}
	res.Resolved.Explanation = append(res.Resolved.Explanation, "SOL route "+string(res.Resolved.SOLRoute.Mode)+": "+res.Resolved.SOLRoute.Decision)
}
