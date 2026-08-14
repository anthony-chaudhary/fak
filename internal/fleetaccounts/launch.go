package fleetaccounts

import (
	"fmt"
	"path/filepath"
	"strings"
)

// LaunchRequest is the typed fleet launch boundary. Callers must present the
// resolved account and the task tier; raw environment-only OpenCode fallbacks do
// not cross this seam.
type LaunchRequest struct {
	Account       Resolved
	TaskTier      int
	InvokedModel  string
	Prompt        string
	Tier3Override bool
	Speed         string
}

// LaunchDecision contains only non-secret launch metadata plus argv/environment.
type LaunchDecision struct {
	OK               bool              `json:"ok"`
	Reason           string            `json:"reason,omitempty"`
	Account          string            `json:"account,omitempty"`
	Product          string            `json:"product,omitempty"`
	ConfiguredModel  string            `json:"configured_model,omitempty"`
	InvokedModel     string            `json:"invoked_model,omitempty"`
	EndpointClass    string            `json:"endpoint_class,omitempty"`
	Speed            string            `json:"speed,omitempty"`
	TaskTier         int               `json:"task_tier,omitempty"`
	Argv             []string          `json:"argv,omitempty"`
	Env              map[string]string `json:"env,omitempty"`
	OperatorOverride bool              `json:"operator_override,omitempty"`
}

// DecideLaunch binds an invocation to a resolved roster record. Tier 3 seats are
// narrow-work seats: an override may authorize a tier-3 task, never downgrade a
// hard tier-1 engineering task onto that seat.
func DecideLaunch(req LaunchRequest) LaunchDecision {
	a := req.Account
	d := LaunchDecision{Account: a.Account, Product: strings.ToLower(a.Product), ConfiguredModel: a.Model, TaskTier: req.TaskTier, OperatorOverride: req.Tier3Override}
	if !a.OK || strings.TrimSpace(a.Account) == "" || strings.TrimSpace(a.ConfigDir) == "" {
		d.Reason = "resolved account record required"
		return d
	}
	if req.TaskTier < 1 || req.TaskTier > 3 {
		d.Reason = "task tier must be 1, 2, or 3"
		return d
	}
	if a.ModelTier != nil && *a.ModelTier == 3 && req.TaskTier < 3 {
		d.Reason = "restricted tier-3 account cannot serve tier-1/2 work"
		return d
	}
	if a.ModelTier != nil && *a.ModelTier == 3 && !req.Tier3Override {
		d.Reason = "tier-3 account requires explicit narrow-work operator override"
		return d
	}
	model := strings.TrimSpace(req.InvokedModel)
	if model == "" {
		model = strings.TrimSpace(a.Model)
	}
	d.InvokedModel = model
	d.EndpointClass = endpointClass(a)
	switch d.Product {
	case "claude":
		d.Argv = []string{"claude", "-p", req.Prompt}
		if model != "" {
			d.Argv = append(d.Argv, "--model", model)
		}
		speed := strings.ToLower(strings.TrimSpace(req.Speed))
		if speed == "" || speed == "auto" {
			speed = "fast"
		}
		if speed != "fast" && speed != "standard" {
			d.Reason = fmt.Sprintf("invalid Claude speed %q (want auto, fast, or standard)", req.Speed)
			return d
		}
		d.Speed = speed
		if speed == "fast" {
			d.Argv = append(d.Argv, "--settings", `{"fastMode":true}`)
		}
		d.Env = map[string]string{"CLAUDE_CONFIG_DIR": a.ConfigDir}
	case "codex":
		d.Argv = []string{"codex", "exec"}
		if model != "" {
			d.Argv = append(d.Argv, "--model", model)
		}
		d.Argv = append(d.Argv, req.Prompt)
		d.Env = map[string]string{"CODEX_HOME": a.ConfigDir}
	case "opencode":
		d.Argv = []string{"opencode", "run"}
		if model != "" {
			d.Argv = append(d.Argv, "-m", model)
		}
		d.Argv = append(d.Argv, req.Prompt)
		d.Env = map[string]string{"XDG_CONFIG_HOME": opencodeConfigHome(a.ConfigDir)}
	default:
		d.Reason = fmt.Sprintf("unsupported worker product %q", a.Product)
		return d
	}
	d.OK = true
	return d
}

func endpointClass(a Resolved) string {
	switch strings.ToLower(strings.TrimSpace(a.Product)) {
	case "claude", "codex":
		return "subscription"
	case "opencode":
		return "api"
	default:
		return "unknown"
	}
}

func opencodeConfigHome(configDir string) string {
	clean := filepath.Clean(configDir)
	if filepath.Base(clean) == "opencode" {
		return filepath.Dir(clean)
	}
	return clean
}
