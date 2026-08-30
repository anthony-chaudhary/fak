// Package dispatchtick holds the pure contract for one issue-resolution dispatch tick.
//
// The cmd/fak shell owns the I/O: Python helper calls, process spawn, leases, and JSON
// records. This leaf holds the stable parts that must not drift between the old Python
// tick and the first-class `fak dispatch tick` verb: backend command shapes, guard wrapping,
// issue picking, and wave/account sidecar records.
package dispatchtick

import (
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const (
	Schema               = "fleet-issue-resolve-dispatch/1"
	RunsDirName          = ".dispatch-runs"
	WaveSidecarSuffix    = ".wave"
	AccountSidecarSuffix = ".account"
	BaseSHASidecarSuffix = ".basesha"
	// ModelSidecarSuffix records the primary model a worker slot was pinned to (Layer
	// 5b), written at spawn only when a non-default model is resolved and scraped back
	// by the witness sweep into WitnessRecord.Model. Absent for a floor (seat-default)
	// worker, so an unconfigured fleet writes no extra sidecar.
	ModelSidecarSuffix   = ".model"
	SpeedSidecarSuffix   = ".speed"
	OpencodePromptNotice = "Resolve GitHub issue # from the attached dispatch prompt."
	// FallbackMaxWorkers is the built-in aspirational ceiling used when the
	// operator sets no FAK_MAX_WORKERS; see DefaultMaxWorkers for the contract.
	FallbackMaxWorkers     = 20
	DarwinMaxWorkers       = 30
	DefaultCooldownMinutes = 120
	DefaultWorkerTimeoutS  = 1800
	DefaultSpawnProbeS     = 5.0
	LeaseTTLMarginS        = 600
)

// DefaultMaxWorkers is the operator's *aspirational* outer ceiling on live
// dispatch workers, not the safety bound. The real DoS proof is the preflight's
// adaptive cap = min(this, host_cap, account_slots): host_cap (#1337) auto-throttles
// to the box's current cores/RAM/thread headroom, and the account session-slot pool
// (#1336) hard-bounds launches so a spawn can never exceed the live account budget.
// Raised 8->20 when Claude worker accounts were verified to carry
// four sessions each: the static ceiling's only job is to sit ABOVE the adaptive
// gates -- which can only LOWER the effective cap -- so concurrency rises to what
// the box and the account pool can actually carry and no further. Resolved once
// at startup from FAK_MAX_WORKERS so the fleet-wide ceiling is an env knob shared
// with the Python launchers, not a rebuild.
var DefaultMaxWorkers = defaultMaxWorkers(runtime.GOOS)

// WaveHint is the dispatch-computed workflow signal consumed by downstream schedulers.
// It carries a ready-set decision, not graph edges, so consumers cannot rederive a DAG.
type WaveHint struct {
	Agent            string
	Node             string
	Wave             string
	StepsToExecution int
	Worker           int
}

// NewWaveHint stamps a workflow decision at the dispatch boundary.
func NewWaveHint(agent, node, wave string, stepsToExecution, worker int) WaveHint {
	return WaveHint{Agent: agent, Node: node, Wave: wave, StepsToExecution: stepsToExecution, Worker: worker}
}

// defaultMaxWorkers returns the platform ceiling unless FAK_MAX_WORKERS sets a
// positive explicit override. Lower host, resource, account, seat, and target gates
// remain independently binding during preflight.
func defaultMaxWorkers(goos string) int {
	fallback := FallbackMaxWorkers
	if goos == "darwin" {
		fallback = DarwinMaxWorkers
	}
	return envPosInt("FAK_MAX_WORKERS", fallback)
}

// envPosInt returns the positive-int value of the named env var, or fallback on
// unset/garbage -- the same tolerant contract as dispatch_preflight._env_pos_int,
// so the Go and Python halves of the dispatch stack read one knob one way.
func envPosInt(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return fallback
	}
	return n
}

// MicroBackend is the #2030 gen/second-next dispatch backend OPTION: instead of
// exec-spawning one detached guarded CLI process per lane, it enrolls the routed
// issue as a single microagent into a running in-process host (internal/microagent,
// M2). It is opt-in only — selected by config (`--backend micro` or
// FLEET_WORKER_BACKEND=micro) — and is NEVER the default: the detached-CLI path stays
// the default until the M29 quality bar clears (#2030 scope 3, "never default
// exposure" for the gen/second-next stream).
const MicroBackend = "micro"

var validBackends = map[string]bool{
	"claude":     true,
	"opencode":   true,
	"codex":      true,
	MicroBackend: true,
}

// IsMicroBackend reports whether the backend token selects the in-process microagent
// host-enroll path (#2030) rather than a detached CLI spawn. It is case- and
// whitespace-insensitive, matching NormalizeBackend's tolerance.
func IsMicroBackend(backend string) bool {
	return strings.TrimSpace(strings.ToLower(backend)) == MicroBackend
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
		return "", fmt.Errorf("unknown backend %q; expected claude, opencode, codex, or micro", raw)
	}
	return backend, nil
}

// ProductForBackend is the preflight/account-switcher product name.
func ProductForBackend(backend string) string {
	if IsMicroBackend(backend) {
		return MicroBackend
	}
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

// WorkerModelPolicy is the RESOLVED per-worker model decision for one headless
// dispatch worker. Primary is the model to pin (via --model for claude, -m for
// opencode/codex); "" means "use the seat/agent default" — the historical claude
// floor where no model flag is emitted at all. Chain is the ordered Claude
// --fallback-model list tried when the primary is overloaded/unavailable. The
// cmd/fak shell resolves this from flags/env/policy; this leaf stays pure (no I/O)
// so the resolved shape is unit-tested without spawning anything.
type WorkerModelPolicy struct {
	Primary string
	Chain   []string
}

// Model is the -m/--model value for the worker command; "" leaves the seat/agent
// default in place (no model flag), which is the historical claude behavior.
func (p WorkerModelPolicy) Model() string { return strings.TrimSpace(p.Primary) }

// FallbackModel is the comma-joined Claude --fallback-model chain, deduped against
// the primary and blanks. It is Claude-specific and print-mode scoped, so a
// non-claude backend gets "" — opencode/codex pin their own model with -m only.
func (p WorkerModelPolicy) FallbackModel(backend string) string {
	if backend != "claude" {
		return ""
	}
	return joinDedupedChain(p.Primary, p.Chain)
}

// joinDedupedChain returns the ordered, comma-joined fallback chain with blanks, the
// primary itself, and any duplicate dropped — the same dedup the interactive launcher
// applies (cmd/fak/accounts_launch.go modelFallbackChain), so both fronts build the
// chain identically. Each element may itself be comma-separated (an env value stuffed
// whole into one slot), so it is re-split. A chain that collapses to empty returns "".
func joinDedupedChain(primary string, chain []string) string {
	seen := map[string]bool{}
	if p := strings.ToLower(strings.TrimSpace(primary)); p != "" {
		seen[p] = true
	}
	var out []string
	for _, part := range chain {
		for _, m := range strings.Split(part, ",") {
			m = strings.TrimSpace(m)
			if m == "" {
				continue
			}
			key := strings.ToLower(m)
			if seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, m)
		}
	}
	return strings.Join(out, ",")
}

// WorkerLaunch is the resolved launch configuration BuildWorkerCommand renders into a
// backend argv. Model/Fallback are the historical knobs; Effort/Ultracode carry the
// per-issue tier uplift. All four default to the zero value, so a WorkerLaunch{} yields
// the exact pre-tier argv (byte-identical to the old model+fallback-only path).
type WorkerLaunch struct {
	Model               string // primary model to pin (Claude --model / opencode|codex -m)
	Fallback            string // Claude-only comma-separated --fallback-model chain
	Effort              string // Claude-only reasoning effort (--effort); ignored when Ultracode
	Ultracode           bool
	Speed               string // Claude-only: auto|fast|standard posture; fast emits per-session fastMode settings   // Claude-only: emit --settings ultracode (implies xhigh + workflow)
	AccountTag          string // resolved fleet account; required for OpenCode fleet launches
	AccountDir          string // resolved product config directory; never written to the ledger as a secret
	TaskTier            int    // requested task tier (1 hard .. 3 narrow); required for OpenCode fleet launches
	Override            bool   // explicit operator authorization for a restricted tier-3 OpenCode launch
	RequireAccountBound bool   // fleet/super-loop seam; legacy non-fleet callers remain compatible
}

// BuildWorkerCommand returns the backend-specific issue-resolution worker argv.
//
// Fallback is the comma-separated Claude fallback CHAIN handed to `claude -p`
// via --fallback-model: when the primary (seat-default) model is overloaded or
// unavailable, Claude Code retries the same headless turn on the next model in the
// list instead of failing the worker outright. It is Claude-specific and print-mode
// scoped — the flag only takes effect under -p — so it is emitted ONLY for the claude
// backend and only when non-empty; opencode/codex pin their own model with -m and
// ignore it. Empty disables the flag (historical behavior). The cmd/fak shell resolves
// the default and the env override, keeping this leaf pure. This is the background/
// headless counterpart of the interactive `fak accounts launch` fallback chain
// (accounts_launch.go): unattended fleet work degrades gracefully to the fallback model
// through a transient overload window instead of dying and re-dispatching the same model.
//
// Effort and Ultracode are the Claude-only per-issue tier uplift (both ignored by
// opencode/codex). They are mutually exclusive on emit: ultracode already runs at xhigh
// PLUS multi-agent workflow orchestration, so it supersedes a bare --effort. Both default
// off, so an unconfigured fleet stays byte-identical to today.
func BuildWorkerCommand(backend, prompt string, launch WorkerLaunch) ([]string, error) {
	switch backend {
	case "claude":
		cmd := []string{"claude", "-p", "--permission-mode", "bypassPermissions"}
		// Un-blank the primary model (Layer 4): claude takes --model (not -m). Gated on
		// non-empty so an unconfigured fleet is byte-identical to today (model==""), and
		// emitted BEFORE the effort/ultracode and --fallback-model flags to match the
		// interactive launcher's ordering (accounts_launch.go).
		cmd = appendModelFlag(cmd, "--model", launch.Model)
		switch {
		case launch.Ultracode:
			cmd = append(cmd, "--settings", UltracodeSettingsArg)
		case strings.EqualFold(strings.TrimSpace(launch.Speed), "fast"):
			cmd = append(cmd, "--settings", `{"fastMode":true}`)
		case strings.TrimSpace(launch.Effort) != "":
			cmd = append(cmd, "--effort", launch.Effort)
		}
		if strings.TrimSpace(launch.Fallback) != "" {
			cmd = append(cmd, "--fallback-model", launch.Fallback)
		}
		return append(cmd, prompt), nil
	case "opencode":
		if launch.RequireAccountBound && (strings.TrimSpace(launch.AccountTag) == "" || strings.TrimSpace(launch.AccountDir) == "" || launch.TaskTier == 0) {
			return nil, fmt.Errorf("account-bound opencode launch requires resolved account record and task tier")
		}
		// --print-logs is required for unattended workers: opencode writes run-level
		// failures such as GLM quota walls to its logger, and without this flag #1275
		// degrades into a banner-only no-op log.
		cmd := []string{"opencode", "run", "--print-logs", "--dangerously-skip-permissions"}
		return append(appendModelFlag(cmd, "-m", launch.Model), OpencodePromptNotice), nil
	case "codex":
		cmd := []string{"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"}
		return append(appendModelFlag(cmd, "-m", launch.Model), "-"), nil
	default:
		return nil, fmt.Errorf("unknown backend %q; expected claude, opencode, or codex", backend)
	}
}

// appendModelFlag appends the backend's model selector only when a model is actually
// configured, so an unconfigured fleet emits argv byte-identical to the pre-Layer-4
// shape (model==""). flag is the backend's own spelling of the selector — claude takes
// `--model`, opencode and codex take `-m` — which is exactly why this is a parameter
// rather than a constant.
func appendModelFlag(cmd []string, flag, model string) []string {
	if strings.TrimSpace(model) == "" {
		return cmd
	}
	return append(cmd, flag, model)
}

// WorkerStdinPayload returns the prompt bytes that must be piped to, or staged for,
// the worker outside argv.
func WorkerStdinPayload(backend, prompt string) string {
	if backend == "codex" || backend == "opencode" {
		return prompt
	}
	return ""
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
	if backend == "codex" && runtime.GOOS == "windows" && filepath.Ext(command[0]) == "" {
		if resolved, err := exec.LookPath(command[0] + ".cmd"); err == nil {
			command = append([]string(nil), command...)
			command[0] = resolved
		}
	}
	args := []string{fakBin, "guard"}
	// Dispatch already audits Codex loop posture before spawn. Carry that decision into
	// the child guard so it does not re-audit unrelated historical sessions and turn an
	// admitted worker into a banner-only stub.
	if backend == "codex" {
		args = append(args, "--codex-loop-gate", "off")
	}
	// Codex guard auto-detection must see the selected CODEX_HOME before choosing its
	// upstream. Forcing --provider openai turns an unrelated ambient OPENAI_API_KEY
	// into an API-billing opt-in and bypasses an otherwise-ready ChatGPT subscription.
	// Other backends still need their explicit wire selection.
	if backend != "codex" {
		args = append(args, "--provider", GuardProvider(backend))
	}
	if backend != "claude" {
		if strings.TrimSpace(baseURL) == "" && backend != "codex" {
			return append([]string(nil), command...), false
		}
		if strings.TrimSpace(baseURL) != "" {
			args = append(args, "--base-url", baseURL)
		}
	}
	// Headless dispatch workers launch with the curated fak_* tool surface (#3607): prune the
	// ~9.9k-token full-registry schema floor to the allowlist a single-issue worker uses; the
	// rest page in via the still-exposed fak_tools_search. The guard honors FAK_GUARD_EXPOSE_PROFILE
	// as the fleet opt-out (=full/off restores the whole registry).
	args = append(args, "--audit", GuardAuditPath(workspace, lane, backend), "--expose-profile", "headless", "--")
	args = append(args, command...)
	return args, true
}

// LaunchCommandShape returns a status-safe argv shape for reports and dry-runs.
// It preserves enough structure to debug backend/guard selection while scrubbing
// workspace paths, account identifiers, and token-like values.
func LaunchCommandShape(command []string, workspace string, account Account) []string {
	out := make([]string, 0, len(command))
	redactNext := false
	for _, arg := range command {
		if redactNext {
			out = append(out, "<redacted>")
			redactNext = false
			continue
		}
		shaped := redactLaunchArg(arg, workspace, account)
		out = append(out, shaped)
		if isSensitiveFlag(arg) && !strings.Contains(arg, "=") {
			redactNext = true
		}
	}
	return out
}

func redactLaunchArg(arg, workspace string, account Account) string {
	out := arg
	out = replaceLaunchSecret(out, workspace, "<workspace>")
	out = replaceLaunchSecret(out, account.Dir, "<account-dir>")
	out = replaceLaunchSecret(out, account.Tag, "<account>")
	if strings.Contains(out, "://") {
		out = RedactLaunchURL(out)
	}
	if idx := strings.Index(out, "="); idx > 0 && isSensitiveKey(out[:idx]) {
		return out[:idx+1] + "<redacted>"
	}
	return out
}

func replaceLaunchSecret(s, secret, marker string) string {
	secret = strings.TrimSpace(secret)
	if secret == "" {
		return s
	}
	for _, variant := range uniqueStrings(secret, filepath.Clean(secret), filepath.ToSlash(secret)) {
		if variant == "" || variant == "." {
			continue
		}
		s = strings.ReplaceAll(s, variant, marker)
	}
	return s
}

// RedactLaunchURL strips the userinfo and query string from a launch-argument URL, leaving
// scheme/host/path intact. A value that does not parse as an absolute URL is returned
// unchanged (it was never a URL, so there is nothing to redact).
//
// Exported because `fak dispatch tick`'s broker-side redactor needs the SAME rule: the two
// used to carry byte-identical copies, which is exactly how a redaction rule drifts and one
// surface starts leaking a credential the other strips.
func RedactLaunchURL(s string) string {
	u, err := url.Parse(s)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return s
	}
	u.User = nil
	u.RawQuery = ""
	return u.String()
}

func isSensitiveFlag(s string) bool {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "-") {
		return false
	}
	return isSensitiveKey(strings.TrimLeft(s, "-"))
}

func isSensitiveKey(s string) bool {
	low := strings.ToLower(s)
	for _, needle := range []string{"token", "oauth", "api-key", "apikey", "api_key", "authorization", "bearer", "secret"} {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}

func uniqueStrings(values ...string) []string {
	return stablePreferredByKey(values, func(value string) string { return value }, nil)
}

// stablePreferredByKey collapses duplicate keys without disturbing their first-seen
// order. When prefer is set, a later value may replace the representative in place;
// the key never moves, so routing output remains deterministic across replacements.
func stablePreferredByKey[T any](values []T, key func(T) string, prefer func(T, T) bool) []T {
	byKey := make(map[string]T, len(values))
	order := make([]string, 0, len(values))
	for _, value := range values {
		k := key(value)
		current, exists := byKey[k]
		if !exists {
			order = append(order, k)
			byKey[k] = value
			continue
		}
		if prefer != nil && prefer(value, current) {
			byKey[k] = value
		}
	}
	out := make([]T, 0, len(order))
	for _, k := range order {
		out = append(out, byKey[k])
	}
	return out
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
