package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	defaultClaudeMacGateway = "http://node-macos-a.local:8080"
	defaultClaudeMacModel   = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
	defaultClaudeMacSSHHost = "user@node-macos-a.local"
	defaultClaudeMacGrafana = "http://localhost:3000"
)

func cmdClaudeMacFak(argv []string) {
	os.Exit(runClaudeMacFak(os.Stdout, os.Stderr, argv))
}

func runClaudeMacFak(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("claude-mac-fak", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gatewayURL := fs.String("gateway-url", envOrDefault("FAK_MAC_GATEWAY", defaultClaudeMacGateway), "fak serve gateway on the Mac")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", defaultClaudeMacModel), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	fetchKey := fs.Bool("fetch-key", true, "when gateway key env is empty, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	configDir := fs.String("claude-config-dir", defaultClaudeMacConfigDir(), "isolated CLAUDE_CONFIG_DIR for this Mac gateway session")
	prompt := fs.String("prompt", "Reply with exactly: OK", "probe prompt; used with --probe")
	probe := fs.Bool("probe", false, "run one headless JSON probe instead of interactive Claude Code")
	interactive := fs.Bool("interactive", false, "explicitly open interactive Claude Code; this is the default")
	dryRun := fs.Bool("dry-run", false, "show the launch plan without starting Claude Code")
	asJSON := fs.Bool("json", false, "emit the launch model as JSON and do not start Claude Code")
	apiTimeoutMS := fs.Int("api-timeout-ms", 1800000, "Claude Code API_TIMEOUT_MS")
	width := fs.Int("width", 120, "dry-run width")
	command := fs.String("command", "claude", "Claude Code command or path to execute")
	debug := fs.Bool("debug", true, "before handoff, probe the gateway (/healthz + /debug/vars) and print a fak debug panel; aborts an interactive launch if the gateway is unreachable")
	overlay := fs.Bool("overlay", false, "do not launch Claude; instead poll /debug/vars and print one live fak line per tick (run in a second pane next to the session). Ctrl-C to stop")
	overlayInterval := fs.Duration("overlay-interval", 2*time.Second, "refresh interval for --overlay")
	grafanaURL := fs.String("grafana-url", envOrDefault("FAK_MAC_GRAFANA", defaultClaudeMacGrafana), "Grafana base URL shown in the debug panel (the shipped tools/grafana stack)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *gatewayURL == "" {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --gateway-url must not be empty")
		return 2
	}
	if *model == "" {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --model must not be empty")
		return 2
	}
	if *apiTimeoutMS < 0 {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --api-timeout-ms must be non-negative")
		return 2
	}
	if *probe && *interactive {
		fmt.Fprintln(stderr, "fak claude-mac-fak: pass either --probe or --interactive, not both")
		return 2
	}
	if *overlayInterval <= 0 {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --overlay-interval must be positive")
		return 2
	}
	if err := os.MkdirAll(*configDir, 0o700); err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: create --claude-config-dir: %v\n", err)
		return 1
	}
	if err := ensureClaudeMacGatewayKey(*keyEnv, *fetchKey, *sshHost, *sshKey); err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: %v\n", err)
		return 2
	}

	debugBase, err := normalizeTUIAgentGatewayURL(*gatewayURL)
	if err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: %v\n", err)
		return 2
	}
	dbg := &claudeMacDebugClient{
		base:    debugBase,
		key:     strings.TrimSpace(os.Getenv(strings.TrimSpace(*keyEnv))),
		grafana: strings.TrimSpace(*grafanaURL),
		hc:      &http.Client{Timeout: 10 * time.Second},
	}

	// --overlay is a self-contained watch loop: it never launches Claude. Run it in
	// a second pane next to the interactive session to see fak's live cache/throughput.
	if *overlay {
		return runClaudeMacOverlay(stdout, stderr, dbg, *model, *overlayInterval)
	}

	// Preflight debug panel (#claude-mac): probe the gateway and print what fak is
	// about to do BEFORE handing the terminal to Claude Code, which makes fak
	// otherwise invisible. A probe run stays quiet unless --debug is set explicitly;
	// an interactive launch defaults debug on and ABORTS on an unreachable gateway
	// rather than starting Claude against a dead/mock backend.
	auth := "gateway-bearer"
	if dbg.key == "" {
		auth = "none"
	}
	if *debug && !*asJSON && !*probe && !*dryRun {
		health, vars, err := dbg.probe()
		if err != nil {
			fmt.Fprintf(stderr, "fak debug: gateway unreachable: %v\n", err)
			if *interactive || (!*probe && !*dryRun) {
				return 1
			}
		} else {
			fmt.Fprint(stdout, renderClaudeMacPreflight(health, vars, dbg.base, *model, auth, dbg.grafana))
		}
	}

	tuiArgs := []string{
		"agent",
		"--command", *command,
		"--claude-config-dir", *configDir,
		"--gateway-url", *gatewayURL,
		"--gateway-key-env", *keyEnv,
		"--model", *model,
		"--api-timeout-ms", strconv.Itoa(*apiTimeoutMS),
		"--width", strconv.Itoa(*width),
	}
	if *dryRun {
		tuiArgs = append(tuiArgs, "--dry-run")
	}
	if *asJSON {
		tuiArgs = append(tuiArgs, "--json")
	}
	if *probe {
		tuiArgs = append(tuiArgs, "--prompt", *prompt)
	}
	passthrough := fs.Args()
	if len(passthrough) == 0 && *probe && !*asJSON {
		passthrough = []string{"--output-format", "json"}
	}
	if len(passthrough) > 0 {
		tuiArgs = append(tuiArgs, "--")
		tuiArgs = append(tuiArgs, passthrough...)
	}
	return runTUI(stdout, stderr, tuiArgs)
}

func envOrDefault(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

func defaultClaudeMacConfigDir() string {
	if v := strings.TrimSpace(os.Getenv("FAK_CLAUDE_CONFIG_DIR")); v != "" {
		return v
	}
	return filepath.Join(os.TempDir(), "fak-claude-mac")
}

func defaultClaudeMacSSHKey() string {
	if v := strings.TrimSpace(os.Getenv("FAK_MAC_SSH_KEY")); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(home, ".ssh", "id_ed25519_prod_to_laptop")
}

func ensureClaudeMacGatewayKey(envName string, fetch bool, host, keyPath string) error {
	envName = strings.TrimSpace(envName)
	if envName == "" {
		envName = "FAK_GATEWAY_KEY"
	}
	if strings.TrimSpace(os.Getenv(envName)) != "" {
		return nil
	}
	if !fetch {
		return fmt.Errorf("%s is empty; set it or leave --fetch-key enabled", envName)
	}
	host = strings.TrimSpace(host)
	if host == "" {
		return fmt.Errorf("%s is empty and --ssh-host is empty", envName)
	}
	args := []string{}
	if strings.TrimSpace(keyPath) != "" {
		args = append(args, "-i", keyPath)
	}
	args = append(args, host, "cat ~/.fak-gateway-key")
	cmd := exec.Command("ssh", args...)
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("fetch gateway key over ssh: %w", err)
	}
	secret := strings.TrimSpace(string(out))
	if secret == "" {
		return fmt.Errorf("fetch gateway key over ssh: empty key")
	}
	return os.Setenv(envName, secret)
}

// claudeMacHealth is the subset of the gateway's /healthz JSON we render. planner
// names the live /v1/chat/completions backend ("inkernel" | "proxy" | "mock"); a
// "mock" planner means the responses are scripted, not model output (see
// internal/gateway/http.go handleHealth), which the panel surfaces loudly.
type claudeMacHealth struct {
	OK      bool   `json:"ok"`
	Engine  string `json:"engine"`
	Model   string `json:"model"`
	Planner string `json:"planner"`
}

// claudeMacDebugVars is the subset of the gateway's /debug/vars JSON we render. The
// field/JSON-tag names mirror internal/gateway/debug.go debugVarsResponse; JSON
// decode tolerates the many extra fields we do not surface.
type claudeMacDebugVars struct {
	Gateway struct {
		Up               bool    `json:"up"`
		Version          string  `json:"version"`
		Engine           string  `json:"engine"`
		Model            string  `json:"model"`
		VDSO             bool    `json:"vdso"`
		UptimeSeconds    float64 `json:"uptime_seconds"`
		InflightRequests int64   `json:"inflight_requests"`
	} `json:"gateway"`
	Kernel struct {
		Submits      int64   `json:"submits"`
		VDSOHits     int64   `json:"vdso_hits"`
		EngineCalls  int64   `json:"engine_calls"`
		Denies       int64   `json:"denies"`
		Admitted     int64   `json:"admitted"`
		VDSOHitRatio float64 `json:"vdso_hit_ratio"`
	} `json:"kernel"`
	Runtime struct {
		NumGoroutine int `json:"num_goroutine"`
		Memory       struct {
			HeapAllocBytes uint64 `json:"heap_alloc_bytes"`
		} `json:"memory"`
	} `json:"runtime"`
}

// claudeMacDebugClient reads the Mac gateway's liveness + debug-vars surfaces. It
// reuses the same one-shot bearer-GET shape as sessionClient (session_cmd.go) but
// is kept local so the debug panel/overlay carry no extra coupling.
type claudeMacDebugClient struct {
	base    string
	key     string
	grafana string
	hc      *http.Client
}

func (c *claudeMacDebugClient) get(path string, out any) error {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, c.base+path, nil)
	if err != nil {
		return err
	}
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	hc := c.hc
	if hc == nil {
		hc = &http.Client{Timeout: 10 * time.Second}
	}
	resp, err := hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("GET %s: status %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (c *claudeMacDebugClient) health() (claudeMacHealth, error) {
	var h claudeMacHealth
	err := c.get("/healthz", &h)
	return h, err
}

func (c *claudeMacDebugClient) vars() (claudeMacDebugVars, error) {
	var v claudeMacDebugVars
	err := c.get("/debug/vars", &v)
	return v, err
}

// probe reads /healthz then /debug/vars. /healthz is the liveness gate; if it
// fails the gateway is unreachable and probe reports that error. A /debug/vars
// failure is non-fatal — health alone still proves the gateway is live — so the
// returned vars is simply zero-valued and the panel renders without counters.
func (c *claudeMacDebugClient) probe() (claudeMacHealth, claudeMacDebugVars, error) {
	h, err := c.health()
	if err != nil {
		return claudeMacHealth{}, claudeMacDebugVars{}, err
	}
	v, _ := c.vars()
	return h, v, nil
}

// renderClaudeMacPreflight renders the debug panel printed before fak hands the
// terminal to Claude Code. It proves the gateway is the live in-kernel fak serve
// (or warns loudly when planner=mock, i.e. scripted responses) and surfaces the
// vDSO cache-hit ratio, inflight requests, and uptime the launch path otherwise
// discards. The bearer is never printed.
func renderClaudeMacPreflight(h claudeMacHealth, v claudeMacDebugVars, gatewayURL, model, auth, grafanaURL string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak debug · gateway %s\n", gatewayURL)
	healthWord := "ok"
	if !h.OK {
		healthWord = "DOWN"
	}
	engine := firstNonEmpty(h.Engine, v.Gateway.Engine)
	fmt.Fprintf(&b, "health: %s  engine=%s  planner=%s\n", healthWord, blankDash(engine), blankDash(h.Planner))
	vdso := "off"
	if v.Gateway.VDSO {
		vdso = "on"
	}
	fmt.Fprintf(&b, "vdso=%s  cache-hit %.2f  inflight %d  up %s\n",
		vdso, v.Kernel.VDSOHitRatio, v.Gateway.InflightRequests, humanUptime(v.Gateway.UptimeSeconds))
	fmt.Fprintf(&b, "model %s  auth %s\n", blankDash(firstNonEmpty(model, h.Model, v.Gateway.Model)), blankDash(auth))
	fmt.Fprintf(&b, "metrics: %s/metrics · %s/debug/vars", gatewayURL, gatewayURL)
	if grafanaURL != "" {
		fmt.Fprintf(&b, " · grafana %s", grafanaURL)
	}
	fmt.Fprintln(&b)
	if strings.EqualFold(strings.TrimSpace(h.Planner), "mock") {
		fmt.Fprintln(&b, "WARN: planner=mock — responses are scripted, not model output")
	}
	fmt.Fprintln(&b, "-> launching claude ...")
	return b.String()
}

// renderClaudeMacOverlayLine renders one compact live line for the --overlay watch
// loop: kernel throughput (submits / vDSO hits / engine calls), inflight requests,
// heap, and goroutine count — the fak-specific signal an operator wants visible
// next to a running session.
func renderClaudeMacOverlayLine(v claudeMacDebugVars) string {
	pct := 0.0
	if v.Kernel.Submits > 0 {
		pct = float64(v.Kernel.VDSOHits) / float64(v.Kernel.Submits) * 100
	}
	return fmt.Sprintf("submits %d  hits %d (%.1f%%)  engine %d  inflight %d  heap %s  gor %d",
		v.Kernel.Submits, v.Kernel.VDSOHits, pct, v.Kernel.EngineCalls,
		v.Gateway.InflightRequests, humanBytes(int64(v.Runtime.Memory.HeapAllocBytes)), v.Runtime.NumGoroutine)
}

// runClaudeMacOverlay polls /debug/vars and prints one live line per tick until
// Ctrl-C, the second-pane companion to the interactive session. It mirrors the
// poll-loop shape of followGuardJournal (signal.NotifyContext + ticker) and never
// launches Claude. A transient fetch error prints a one-line note and keeps polling.
func runClaudeMacOverlay(stdout, stderr io.Writer, c *claudeMacDebugClient, model string, interval time.Duration) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	fmt.Fprintf(stdout, "fak debug overlay · %s  model=%s  (every %s, Ctrl-C to stop)\n", c.base, blankDash(model), interval)
	emit := func() {
		v, err := c.vars()
		if err != nil {
			fmt.Fprintf(stderr, "fak overlay: %v\n", err)
			return
		}
		fmt.Fprintf(stdout, "  %s\n", renderClaudeMacOverlayLine(v))
	}
	emit()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return 0
		case <-ticker.C:
			emit()
		}
	}
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func blankDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

// humanUptime renders gateway uptime seconds compactly (e.g. "3h12m", "45s").
func humanUptime(sec float64) string {
	if sec <= 0 {
		return "0s"
	}
	d := time.Duration(sec * float64(time.Second)).Round(time.Second)
	switch {
	case d >= time.Hour:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	case d >= time.Minute:
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	default:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
}
