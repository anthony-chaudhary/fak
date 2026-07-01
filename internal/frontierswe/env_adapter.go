package frontierswe

import (
	"fmt"
	"net"
	"net/url"
	"os/exec"
	"strconv"
	"strings"
)

// EnvAdapterSchema is the machine-readable contract emitted by
// `fak frontierswe env-adapter`.
const EnvAdapterSchema = "fak.frontierswe.env-adapter.v1"

const (
	DefaultGatewayBaseURL = "http://127.0.0.1:8080/v1"
	DefaultGatewayAddr    = "127.0.0.1:8080"
	DefaultUpstreamBase   = "http://127.0.0.1:11434/v1"
	DefaultModelEnv       = "${FRONTIERSWE_MODEL:-frontierswe-model}"
	DefaultWrappedAgent   = "harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch"
	DefaultFakBin         = "fak"
	DefaultRunCommand     = "python -m harbor run job.yaml"
)

// EnvAdapterConfig is the operator-facing shape of the FrontierSWE environment
// adapter. It deliberately plans a co-resident gateway rather than a remote one:
// for allow_internet=false tasks, the gateway and upstream base URL must be
// loopback or explicitly pinned.
type EnvAdapterConfig struct {
	Task            *Task
	FakBin          string
	GatewayAddr     string
	GatewayBaseURL  string
	UpstreamBaseURL string
	Model           string
	WrappedAgent    string
	RunCommand      string
	PinnedHosts     []string
}

// EnvAdapterPlan is the reproducible adapter recipe and local capability gate.
// Runnable=false is an honest local gate, not a failed recipe: Command remains
// populated so the operator can run it on a Docker/GHCR/Modal-capable box.
type EnvAdapterPlan struct {
	Schema           string               `json:"schema"`
	Task             string               `json:"task"`
	DockerImage      string               `json:"docker_image"`
	GatewayBaseURL   string               `json:"gateway_base_url"`
	GatewayAddr      string               `json:"gateway_addr"`
	HealthzURL       string               `json:"healthz_url"`
	UpstreamBaseURL  string               `json:"upstream_base_url"`
	Model            string               `json:"model"`
	FakBin           string               `json:"fak_bin"`
	WrappedAgent     string               `json:"wrapped_agent"`
	ShimImportPath   string               `json:"shim_import_path"`
	AllowInternet    bool                 `json:"allow_internet"`
	PinnedHosts      []string             `json:"pinned_hosts,omitempty"`
	NetworkMode      string               `json:"network_mode"`
	AgentTimeoutS    int64                `json:"agent_timeout_sec"`
	VerifierTimeoutS int64                `json:"verifier_timeout_sec"`
	Resources        EnvAdapterResources  `json:"resources"`
	Integrity        EnvAdapterIntegrity  `json:"integrity"`
	Capability       EnvAdapterCapability `json:"local_capability"`
	Steps            []EnvAdapterStep     `json:"steps"`
	Command          string               `json:"command"`
	JobYAML          string               `json:"job_yaml"`
	Notes            []string             `json:"notes,omitempty"`
}

// EnvAdapterResources mirrors the FrontierSWE task environment envelope.
type EnvAdapterResources struct {
	CPUs      int `json:"cpus"`
	MemoryMB  int `json:"memory_mb"`
	StorageMB int `json:"storage_mb"`
	GPUs      int `json:"gpus"`
}

// EnvAdapterIntegrity records whether the generated recipe honors the task's
// no-internet / pinned-host boundary.
type EnvAdapterIntegrity struct {
	OK     bool   `json:"ok"`
	Reason string `json:"reason,omitempty"`
}

// EnvAdapterCapability is the local-gate result for this host.
type EnvAdapterCapability struct {
	DockerPresent bool   `json:"docker_present"`
	Runnable      bool   `json:"runnable"`
	Reason        string `json:"reason,omitempty"`
}

// EnvAdapterStep is one human-readable stage in the emitted recipe.
type EnvAdapterStep struct {
	Name    string `json:"name"`
	Command string `json:"command"`
	Why     string `json:"why"`
}

// BuildEnvAdapterPlan returns the exact co-resident fak serve recipe for a
// FrontierSWE task. It does not start Docker; callers can run Command when
// Capability.Runnable is true, or copy it to a GPU/Docker/Modal host otherwise.
func BuildEnvAdapterPlan(cfg EnvAdapterConfig) EnvAdapterPlan {
	task := cfg.Task
	if task == nil {
		task = &Task{Name: "unknown"}
	}
	fakBin := defaultString(cfg.FakBin, DefaultFakBin)
	gatewayAddr := defaultString(cfg.GatewayAddr, DefaultGatewayAddr)
	gatewayBase := defaultString(cfg.GatewayBaseURL, DefaultGatewayBaseURL)
	upstreamBase := defaultString(cfg.UpstreamBaseURL, DefaultUpstreamBase)
	model := defaultString(cfg.Model, DefaultModelEnv)
	wrapped := defaultString(cfg.WrappedAgent, DefaultWrappedAgent)
	runCommand := defaultString(cfg.RunCommand, DefaultRunCommand)
	pins := normalizePins(cfg.PinnedHosts)

	healthz := HealthzURL(gatewayBase)
	integrity := validateEnvAdapterBoundary(task, gatewayBase, upstreamBase, pins)
	capability := detectEnvAdapterCapability(integrity.OK)

	serveCommand := fmt.Sprintf("%s serve --addr %s --provider openai --base-url %s --model \"$model\"",
		shWord(fakBin), shWord(gatewayAddr), shWord(upstreamBase))
	healthCommand := fmt.Sprintf("curl -fsS %s >/tmp/fak-healthz.json", shWord(healthz))
	smokeCommand := fmt.Sprintf("curl -fsS %s -H 'content-type: application/json' -d \"$smoke_payload\" >/tmp/fak-frontierswe-smoke.json",
		shWord(strings.TrimRight(gatewayBase, "/")+"/chat/completions"))

	steps := []EnvAdapterStep{
		{
			Name:    "start_gateway",
			Command: serveCommand,
			Why:     "starts fak serve inside the task sandbox, bound to loopback before the agent starts",
		},
		{
			Name:    "healthz_gate",
			Command: healthCommand,
			Why:     "proves the OpenAI-compatible gateway is live before turn 1",
		},
		{
			Name:    "one_turn_smoke",
			Command: smokeCommand,
			Why:     "drives one chat-completions turn through the same base URL the C6 shim uses",
		},
		{
			Name:    "run_frontierswe_harness",
			Command: runCommand,
			Why:     "hands control to FrontierSWE with only the model base URL rerouted through fak",
		},
	}

	plan := EnvAdapterPlan{
		Schema:           EnvAdapterSchema,
		Task:             task.Name,
		DockerImage:      task.Environment.DockerImage,
		GatewayBaseURL:   gatewayBase,
		GatewayAddr:      gatewayAddr,
		HealthzURL:       healthz,
		UpstreamBaseURL:  upstreamBase,
		Model:            model,
		FakBin:           fakBin,
		WrappedAgent:     wrapped,
		ShimImportPath:   "harbor_ext.fak_routed:FakRoutedAgent",
		AllowInternet:    task.Environment.AllowInternet,
		PinnedHosts:      pins,
		NetworkMode:      envAdapterNetworkMode(pins),
		AgentTimeoutS:    int64(task.AgentTimeoutSec()),
		VerifierTimeoutS: int64(task.VerifierTimeoutSec()),
		Resources: EnvAdapterResources{
			CPUs: task.Environment.CPUs, MemoryMB: task.Environment.MemoryMB,
			StorageMB: task.Environment.StorageMB, GPUs: task.Environment.GPUs,
		},
		Integrity:  integrity,
		Capability: capability,
		Steps:      steps,
		JobYAML:    envAdapterJobYAML(wrapped, gatewayBase, task.Environment.AllowInternet),
	}
	plan.Command = dockerCommand(task, plan, runCommand)
	if !capability.Runnable {
		plan.Notes = append(plan.Notes, "GATED locally: "+capability.Reason)
	}
	if !integrity.OK {
		plan.Notes = append(plan.Notes, "REFUSED: "+integrity.Reason)
	}
	if len(pins) > 0 {
		plan.Notes = append(plan.Notes, "pinned-host mode uses Docker bridge networking plus --add-host entries; the FrontierSWE/Modal sandbox must enforce pin_resolved_hosts")
	}
	return plan
}

// HealthzURL maps an OpenAI /v1 base URL to the gateway /healthz endpoint.
func HealthzURL(baseURL string) string {
	u, err := url.Parse(baseURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return strings.TrimRight(baseURL, "/") + "/healthz"
	}
	u.Path = strings.TrimSuffix(u.Path, "/")
	if strings.HasSuffix(u.Path, "/v1") {
		u.Path = strings.TrimSuffix(u.Path, "/v1")
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/healthz"
	u.RawQuery = ""
	u.Fragment = ""
	return u.String()
}

func validateEnvAdapterBoundary(task *Task, gatewayBase, upstreamBase string, pinnedHosts []string) EnvAdapterIntegrity {
	if task == nil || task.Environment.AllowInternet {
		return EnvAdapterIntegrity{OK: true}
	}
	var problems []string
	if !isSandboxURL(gatewayBase, pinnedHosts) {
		problems = append(problems, "gateway_base_url is not loopback or pinned")
	}
	if !isSandboxURL(upstreamBase, pinnedHosts) {
		problems = append(problems, "upstream_base_url is not loopback or pinned")
	}
	if len(problems) > 0 {
		return EnvAdapterIntegrity{
			OK:     false,
			Reason: "allow_internet=false requires loopback or pinned hosts: " + strings.Join(problems, "; "),
		}
	}
	return EnvAdapterIntegrity{OK: true}
}

func detectEnvAdapterCapability(integrityOK bool) EnvAdapterCapability {
	cap := EnvAdapterCapability{}
	if _, err := exec.LookPath("docker"); err == nil {
		cap.DockerPresent = true
	}
	cap.Runnable = integrityOK && cap.DockerPresent
	switch {
	case !integrityOK:
		cap.Reason = "adapter integrity gate failed; fix the no-internet boundary before running"
	case !cap.DockerPresent:
		cap.Reason = "Docker not found on this host; run the emitted command on a Docker/GHCR/Modal-capable box"
	}
	return cap
}

func dockerCommand(task *Task, plan EnvAdapterPlan, runCommand string) string {
	image := plan.DockerImage
	if image == "" {
		image = "ghcr.io/proximal-labs/frontier-swe/" + task.Name + ":v6"
	}
	args := []string{"docker", "run", "--rm", "--network", plan.NetworkMode}
	for _, host := range plan.PinnedHosts {
		args = append(args, "--add-host", host+":host-gateway")
	}
	if plan.Resources.CPUs > 0 {
		args = append(args, "--cpus", strconv.Itoa(plan.Resources.CPUs))
	}
	if plan.Resources.MemoryMB > 0 {
		args = append(args, "--memory", fmt.Sprintf("%dm", plan.Resources.MemoryMB))
	}
	if plan.Resources.GPUs > 0 {
		args = append(args, "--gpus", strconv.Itoa(plan.Resources.GPUs))
	}
	envs := map[string]string{
		"FAK_GATEWAY_URL":               plan.GatewayBaseURL,
		"OPENAI_BASE_URL":               plan.GatewayBaseURL,
		"OPENROUTER_BASE_URL":           plan.GatewayBaseURL,
		"QWEN_BASE_URL":                 plan.GatewayBaseURL,
		"FAK_FRONTIERSWE_WRAPPED_AGENT": plan.WrappedAgent,
		"FRONTIERSWE_RUN_CMD":           runCommand,
	}
	envKeys := []string{
		"FAK_GATEWAY_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL",
		"QWEN_BASE_URL", "FAK_FRONTIERSWE_WRAPPED_AGENT", "FRONTIERSWE_RUN_CMD",
	}
	for _, k := range envKeys {
		args = append(args, "-e", k+"="+envs[k])
	}
	args = append(args, image, "/bin/sh", "-lc", envAdapterShell(plan))
	for i, a := range args {
		args[i] = shWord(a)
	}
	return strings.Join(args, " ")
}

func envAdapterShell(plan EnvAdapterPlan) string {
	return strings.Join([]string{
		"set -eu",
		envAdapterModelLine(plan.Model),
		fmt.Sprintf("%s serve --addr %s --provider openai --base-url %s --model \"$model\" >/tmp/fak-serve.log 2>&1 &",
			shWord(defaultString(plan.FakBin, DefaultFakBin)), shWord(plan.GatewayAddr), shWord(plan.UpstreamBaseURL)),
		"fak_pid=$!",
		`trap 'kill "$fak_pid" 2>/dev/null || true' EXIT`,
		fmt.Sprintf("for i in $(seq 1 120); do curl -fsS %s >/tmp/fak-healthz.json && break; sleep 1; done", shWord(plan.HealthzURL)),
		"test -s /tmp/fak-healthz.json",
		`smoke_payload="{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"frontierswe env-adapter smoke\"}],\"max_tokens\":1}"`,
		fmt.Sprintf("curl -fsS %s -H 'content-type: application/json' -d \"$smoke_payload\" >/tmp/fak-frontierswe-smoke.json", shWord(strings.TrimRight(plan.GatewayBaseURL, "/")+"/chat/completions")),
		`exec /bin/sh -lc "${FRONTIERSWE_RUN_CMD:-` + DefaultRunCommand + `}"`,
	}, "\n")
}

func envAdapterModelLine(model string) string {
	if model == "" || model == DefaultModelEnv {
		return `model="${FRONTIERSWE_MODEL:-frontierswe-model}"`
	}
	return "model=" + shWord(model)
}

func envAdapterNetworkMode(pinnedHosts []string) string {
	if len(pinnedHosts) > 0 {
		return "bridge"
	}
	return "none"
}

func envAdapterJobYAML(wrapped, gatewayBase string, allowInternet bool) string {
	return fmt.Sprintf(`agents:
  - name: fak-routed-claude-code
    import_path: harbor_ext.fak_routed:FakRoutedAgent
    model_name: ${FRONTIERSWE_MODEL}
    override_timeout_sec: 72000
    kwargs:
      wrapped: %s
      fak_base_url: %s
      allow_internet: %t
`, wrapped, gatewayBase, allowInternet)
}

func isSandboxURL(raw string, pinnedHosts []string) bool {
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := u.Hostname()
	if host == "" {
		return false
	}
	if isLoopbackHost(host) {
		return true
	}
	for _, p := range pinnedHosts {
		if strings.EqualFold(host, p) {
			return true
		}
	}
	return false
}

func isLoopbackHost(host string) bool {
	h := strings.Trim(host, "[]")
	if strings.EqualFold(h, "localhost") {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

func normalizePins(pins []string) []string {
	out := make([]string, 0, len(pins))
	seen := map[string]bool{}
	for _, p := range pins {
		for _, part := range strings.Split(p, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			key := strings.ToLower(part)
			if !seen[key] {
				seen[key] = true
				out = append(out, part)
			}
		}
	}
	return out
}

func defaultString(s, d string) string {
	if strings.TrimSpace(s) == "" {
		return d
	}
	return s
}

func shWord(s string) string {
	if s == "" {
		return "''"
	}
	if strings.Contains(s, "\n") {
		return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
	}
	if !strings.ContainsAny(s, " \t'\"`$\\|&;()<>*?![]{}") {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
}
