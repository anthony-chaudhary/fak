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
	Mode                  SOLMode `json:"mode"`
	Model                 string  `json:"model"`
	ReasoningEffort       string  `json:"reasoning_effort"`
	ReasoningMode         string  `json:"reasoning_mode,omitempty"`
	MultiAgent            bool    `json:"multi_agent"`
	ConsultOnly           bool    `json:"consult_only,omitempty"`
	Decision              string  `json:"decision"`
	WorkerModel           string  `json:"worker_model,omitempty"`
	WorkerReasoningEffort string  `json:"worker_reasoning_effort,omitempty"`
}

const (
	DefaultAstraChildWorkerModel  = "gemini-3.8-flash"
	DefaultAstraChildWorkerEffort = "medium"
)

// IsAstraModel reports whether model refers to an Astra model family.
func IsAstraModel(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	switch m {
	case "gpt-6-astra", "gpt 6 astra", "gpt6astra", "gpt-6", "gpt6", "astra",
		"astra-gpt-6", "astra gpt 6", "astra gpt-6", "astra-gpt6", "astragpt6",
		"gpt-6 astra", "gpt 6-astra", "gpt6-astra",
		"openai/gpt-6-astra", "openai/gpt-6", "openai/astra",
		"openai/astra-gpt-6", "openai/astra gpt 6":
		return true
	default:
		return strings.Contains(m, "astra")
	}
}

// ChildWorkerRoute returns the effective model and reasoning effort for child workers.
// Under an Astra manager, workers default to gemini-3.8-flash with medium effort
// unless explicitly overridden.
func ChildWorkerRoute(managerModel, overrideModel, overrideEffort string) (string, string) {
	model := strings.TrimSpace(overrideModel)
	effort := strings.ToLower(strings.TrimSpace(overrideEffort))
	if IsAstraModel(managerModel) {
		if model == "" {
			model = DefaultAstraChildWorkerModel
		}
		if effort == "" {
			effort = DefaultAstraChildWorkerEffort
		}
		return model, effort
	}
	if model == "" {
		model = managerModel
	}
	if effort == "" {
		effort = "high"
	}
	return model, effort
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
		"many issues", "multi-agent", "multi agent", "parallel", "several issues",
		"subagent", "subagents", "sub agent", "sub agents", "super loop",
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
	if IsAstraModel(model) {
		route.WorkerModel = DefaultAstraChildWorkerModel
		route.WorkerReasoningEffort = DefaultAstraChildWorkerEffort
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
	pinnedWorkerModel := res.Resolved.SOLRoute.WorkerModel
	pinnedWorkerEffort := res.Resolved.SOLRoute.WorkerReasoningEffort
	res.Resolved.SOLRoute = SelectSOLRoute(taskText, profile, res.Resolved.WorkClass, model)
	if pinnedEffort != "" {
		res.Resolved.SOLRoute.ReasoningEffort = pinnedEffort
		res.Resolved.SOLRoute.Decision += "; effort pinned by operator to " + pinnedEffort
		if pinnedWorkerEffort == "" {
			res.Resolved.SOLRoute.WorkerReasoningEffort = pinnedEffort
		}
	}
	if pinnedWorkerModel != "" {
		res.Resolved.SOLRoute.WorkerModel = pinnedWorkerModel
	}
	if pinnedWorkerEffort != "" {
		res.Resolved.SOLRoute.WorkerReasoningEffort = pinnedWorkerEffort
	}
	res.Resolved.Explanation = append(res.Resolved.Explanation, "SOL route "+string(res.Resolved.SOLRoute.Mode)+": "+res.Resolved.SOLRoute.Decision)
}
