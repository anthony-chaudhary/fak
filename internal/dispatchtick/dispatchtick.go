// Package dispatchtick holds the pure contract for one issue-resolution dispatch tick.
//
// The cmd/fak shell owns the I/O: Python helper calls, process spawn, leases, and JSON
// records. This leaf holds the stable parts that must not drift between the old Python
// tick and the first-class `fak dispatch tick` verb: backend command shapes, guard wrapping,
// issue picking, and wave/account sidecar records.
package dispatchtick

import (
	"fmt"
	"path/filepath"
	"strings"
)

const (
	Schema               = "fleet-issue-resolve-dispatch/1"
	RunsDirName          = ".dispatch-runs"
	WaveSidecarSuffix    = ".wave"
	AccountSidecarSuffix = ".account"
	BaseSHASidecarSuffix = ".basesha"
	// DefaultMaxWorkers is the operator's *aspirational* outer ceiling on live
	// dispatch workers, not the safety bound. The real DoS proof is the preflight's
	// adaptive cap = min(this, host_cap, seats): host_cap (#1337) auto-throttles to
	// the box's current cores/RAM/thread headroom, and the seat pool (#1336) hard-
	// bounds at one worker per routable account so a spawn can never double-book a
	// rate limit. Doubled 2->4 once those two gates landed: the static 2 sat below
	// both and was the artificial bottleneck; raising it lets the adaptive gates --
	// which can only LOWER the effective cap -- govern, so concurrency rises to what
	// the box and the account pool can actually carry and no further.
	DefaultMaxWorkers      = 4
	DefaultCooldownMinutes = 120
	DefaultWorkerTimeoutS  = 1800
	DefaultSpawnProbeS     = 5.0
	LeaseTTLMarginS        = 600
)

var validBackends = map[string]bool{
	"claude":   true,
	"opencode": true,
	"codex":    true,
}

// Account is the switcher account stamped into a worker's sidecar.
type Account struct {
	Tag   string `json:"tag,omitempty"`
	Tier  any    `json:"tier,omitempty"`
	Model string `json:"model,omitempty"`
	Dir   string `json:"dir,omitempty"`
}

// Membership is the per-worker wave identity stamped into env and a .wave sidecar.
type Membership struct {
	Rank      int    `json:"rank"`
	WaveID    string `json:"wave_id"`
	Size      int    `json:"size"`
	Shortfall int    `json:"shortfall"`
}

// NormalizeBackend validates the worker backend token.
func NormalizeBackend(raw string) (string, error) {
	backend := strings.ToLower(strings.TrimSpace(raw))
	if backend == "" {
		backend = "claude"
	}
	if !validBackends[backend] {
		return "", fmt.Errorf("unknown backend %q; expected claude, opencode, or codex", raw)
	}
	return backend, nil
}

// ProductForBackend is the preflight/account-switcher product name.
func ProductForBackend(backend string) string {
	if backend == "opencode" {
		return "opencode"
	}
	if backend == "codex" {
		return "codex"
	}
	return "claude"
}

// DefaultWorkKind mirrors the Python dispatcher's backend-aware default.
func DefaultWorkKind(backend string) string {
	if backend == "opencode" {
		return "gardening"
	}
	return "engineering"
}

// PickTargetIssue returns the first lane issue not currently live or cooling.
func PickTargetIssue(numbers []int, skip map[int]bool) (int, bool) {
	for _, n := range numbers {
		if !skip[n] {
			return n, true
		}
	}
	return 0, false
}

// PreviewPrompt is the prompt placeholder stored in a dry-run command.
func PreviewPrompt(issue, chars int) string {
	return fmt.Sprintf("<resolve #%d prompt, %d chars>", issue, chars)
}

// BuildWorkerCommand returns the backend-specific issue-resolution worker argv.
func BuildWorkerCommand(backend, prompt, model string) ([]string, error) {
	switch backend {
	case "claude":
		return []string{"claude", "-p", "--permission-mode", "bypassPermissions", prompt}, nil
	case "opencode":
		// --print-logs is required for unattended workers: opencode writes run-level
		// failures such as GLM quota walls to its logger, and without this flag #1275
		// degrades into a banner-only no-op log.
		cmd := []string{"opencode", "run", "--print-logs", "--dangerously-skip-permissions"}
		if strings.TrimSpace(model) != "" {
			cmd = append(cmd, "-m", model)
		}
		return append(cmd, prompt), nil
	case "codex":
		cmd := []string{"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"}
		if strings.TrimSpace(model) != "" {
			cmd = append(cmd, "-m", model)
		}
		return append(cmd, prompt), nil
	default:
		return nil, fmt.Errorf("unknown backend %q; expected claude, opencode, or codex", backend)
	}
}

// WaveMembershipEnv stamps a detached worker's place in a wave.
func WaveMembershipEnv(m Membership) map[string]string {
	return map[string]string{
		"FLEET_WAVE_ID":        m.WaveID,
		"FLEET_WAVE_RANK":      fmt.Sprintf("%d", m.Rank),
		"FLEET_WAVE_SIZE":      fmt.Sprintf("%d", m.Size),
		"FLEET_WAVE_SHORTFALL": fmt.Sprintf("%d", m.Shortfall),
	}
}

// AccountSidecar returns the non-empty account fields that should be written beside a log.
func AccountSidecar(a Account) map[string]any {
	out := map[string]any{}
	if a.Tag != "" {
		out["tag"] = a.Tag
	}
	if a.Tier != nil {
		out["tier"] = a.Tier
	}
	if a.Model != "" {
		out["model"] = a.Model
	}
	if a.Dir != "" {
		out["dir"] = a.Dir
	}
	return out
}

// GuardProvider is the upstream provider wire fak guard should proxy for a backend.
func GuardProvider(backend string) string {
	if backend == "claude" {
		return "anthropic"
	}
	return "openai"
}

// GuardAuditPath is the per-worker decision journal path used by fak guard.
func GuardAuditPath(workspace, lane, backend string) string {
	name := fmt.Sprintf("guard-%s-%s.audit.jsonl", cleanPathToken(backend), cleanPathToken(lane))
	return filepath.Join(workspace, RunsDirName, name)
}

// GuardedLaunchCommand returns command fronted by `fak guard` when a fak binary is available.
func GuardedLaunchCommand(command []string, fakBin, lane, backend, workspace, baseURL string) ([]string, bool) {
	if len(command) == 0 || strings.TrimSpace(fakBin) == "" {
		return append([]string(nil), command...), false
	}
	args := []string{fakBin, "guard", "--provider", GuardProvider(backend)}
	if backend != "claude" {
		if strings.TrimSpace(baseURL) == "" {
			return append([]string(nil), command...), false
		}
		args = append(args, "--base-url", baseURL)
	}
	args = append(args, "--audit", GuardAuditPath(workspace, lane, backend), "--")
	args = append(args, command...)
	return args, true
}

func cleanPathToken(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return b.String()
}
