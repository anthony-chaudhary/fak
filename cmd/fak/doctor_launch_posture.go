package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
)

const launchPostureSchema = "fak.launch-posture.v1"

type launchPostureMechanism struct {
	Name       string `json:"name"`
	Configured bool   `json:"configured"`
	State      string `json:"state"`
	Reason     string `json:"reason"`
	Action     string `json:"action,omitempty"`
	Disable    string `json:"disable,omitempty"`
}

type launchPostureReport struct {
	Schema     string                   `json:"schema"`
	OK         bool                     `json:"ok"`
	Entrypoint string                   `json:"entrypoint"`
	Harness    string                   `json:"harness,omitempty"`
	Wire       string                   `json:"wire"`
	Workspace  string                   `json:"workspace"`
	Mechanisms []launchPostureMechanism `json:"mechanisms"`
	Summary    map[string]int           `json:"summary"`
}

type launchPostureOptions struct {
	entrypoint      string
	harness         string
	provider        string
	baseURL         string
	workspace       string
	native          bool
	nativeCodeTools bool
	outputProfile   string
	workProfile     string
	compactHistory  int
	ctxViewBudget   int
	elideStaleReads bool
	deferColdTools  bool
	vcacheAnchor    bool
}

func runDoctorLaunchPosture(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak doctor launch-posture", flag.ContinueOnError)
	fs.SetOutput(stderr)
	entrypoint := fs.String("entrypoint", "guard", "launch entrypoint: agent|guard|serve")
	harness := fs.String("harness", "", "wrapped harness for guard (default: claude)")
	provider := fs.String("provider", "", "provider family used to derive the runtime wire")
	baseURL := fs.String("base-url", "", "upstream base URL used to derive the runtime wire")
	workspace := fs.String("workspace", "", "repository workspace (default: current directory)")
	native := fs.Bool("native", false, "model a native owned-loop serve")
	nativeCodeTools := fs.Bool("native-code-tools", true, "model bounded native code tools enabled")
	outputProfile := fs.String("output-profile", agentDefaultOutputStyle, "modeled response profile")
	workProfile := fs.String("work-profile", agentDefaultWorkProfile, "modeled work profile")
	compactHistory := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "modeled Anthropic byte-preserving history budget; 0 disables")
	ctxViewBudget := fs.Int("ctx-view-budget", agent.DefaultCtxViewBudget, "modeled provider-neutral decoded context-view budget; 0 disables")
	elideStaleReads := fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "model stale-read elision")
	deferColdTools := fs.Bool("defer-cold-tools", gateway.DefaultDeferColdTools, "model cold-tool deferral")
	vcacheAnchor := fs.Bool("vcache-anchor", gateway.DefaultVCacheAnchor, "model provider-cache anchoring")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak doctor launch-posture: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	report, err := deriveLaunchPosture(launchPostureOptions{
		entrypoint: *entrypoint, harness: *harness, provider: *provider, baseURL: *baseURL,
		workspace: *workspace, native: *native, nativeCodeTools: *nativeCodeTools,
		outputProfile: *outputProfile, workProfile: *workProfile, compactHistory: *compactHistory, ctxViewBudget: *ctxViewBudget,
		elideStaleReads: *elideStaleReads, deferColdTools: *deferColdTools, vcacheAnchor: *vcacheAnchor,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak doctor launch-posture: %v\n", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak doctor launch-posture: encode: %v\n", err)
			return 2
		}
	} else {
		renderLaunchPosture(stdout, report)
	}
	if report.OK {
		return 0
	}
	return 1
}

func deriveLaunchPosture(opts launchPostureOptions) (launchPostureReport, error) {
	opts.entrypoint = strings.ToLower(strings.TrimSpace(opts.entrypoint))
	if opts.entrypoint != "agent" && opts.entrypoint != "guard" && opts.entrypoint != "serve" {
		return launchPostureReport{}, fmt.Errorf("invalid --entrypoint %q; supported: agent, guard, serve", opts.entrypoint)
	}
	if opts.harness == "" && opts.entrypoint == "guard" {
		opts.harness = "claude"
	}
	if opts.harness != "" {
		opts.harness = strings.ToLower(strings.TrimSuffix(filepath.Base(strings.TrimSpace(opts.harness)), ".exe"))
	}
	if opts.provider == "" {
		switch {
		case opts.entrypoint == "guard" && opts.harness == "claude":
			opts.provider = "anthropic"
		case opts.entrypoint == "serve" && opts.native:
			opts.provider = "native"
		}
	}
	if opts.baseURL == "" && opts.provider != "" {
		opts.baseURL = guardDefaultBaseURL(opts.provider)
	}
	root := strings.TrimSpace(opts.workspace)
	if root == "" {
		var err error
		root, err = os.Getwd()
		if err != nil {
			return launchPostureReport{}, fmt.Errorf("workspace: %w", err)
		}
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return launchPostureReport{}, fmt.Errorf("workspace: %w", err)
	}
	wire := classifyShrinkWire(opts.provider, opts.baseURL, nil, func() string {
		if opts.entrypoint == "serve" && opts.native {
			return "owned-model"
		}
		return ""
	}())

	mechanisms := []launchPostureMechanism{
		postureCodeTools(opts, root),
		postureOutputProfile(opts),
		postureWorkProfile(opts),
		postureShrink("compact-history", opts.compactHistory > 0, wire, "--compact-history-budget 0", "use an Anthropic passthrough wire or set --provider anthropic"),
		postureShrinkDecoded(opts, wire),
		postureStaleReadElision(opts, wire),
		postureShrink("cold-tool-deferral", opts.deferColdTools, wire, "--defer-cold-tools=false", "use an Anthropic passthrough wire or set --provider anthropic"),
		postureShrink("vcache-anchor", opts.vcacheAnchor, wire, "--vcache-anchor=false", "use an Anthropic passthrough wire or set --provider anthropic"),
		postureVCacheSignals(opts, wire),
		postureVCacheCalibration(opts, wire),
	}
	summary := map[string]int{"active": 0, "inert": 0, "disabled": 0, "unsupported": 0}
	ok := true
	for _, mechanism := range mechanisms {
		summary[mechanism.State]++
		if mechanism.State == "inert" || mechanism.State == "unsupported" {
			ok = false
		}
	}
	return launchPostureReport{Schema: launchPostureSchema, OK: ok, Entrypoint: opts.entrypoint, Harness: opts.harness, Wire: wire, Workspace: root, Mechanisms: mechanisms, Summary: summary}, nil
}

func postureStaleReadElision(opts launchPostureOptions, wire string) launchPostureMechanism {
	m := launchPostureMechanism{Name: "stale-read-elision", Configured: opts.elideStaleReads, Disable: "--elide-stale-reads=false"}
	if !opts.elideStaleReads {
		m.State, m.Reason, m.Action = "disabled", "stale-read elision was explicitly disabled", "remove --elide-stale-reads=false"
		return m
	}
	if wire == shrinkWireAnthropicPassthrough {
		m.State, m.Reason = "active", "Anthropic request bytes elide superseded tool-read results before forwarding"
		return m
	}
	m.State, m.Reason = "active", "decoded provider-neutral history elides superseded tool-read results before context planning"
	return m
}

func postureVCacheSignals(opts launchPostureOptions, wire string) launchPostureMechanism {
	m := launchPostureMechanism{Name: "vcache-signals", Configured: true}
	if opts.entrypoint == "serve" && opts.native {
		m.State, m.Reason = "active", "owned-model usage feeds the provider-neutral cache-signal and per-family economics surface"
		return m
	}
	provider := strings.ToLower(strings.TrimSpace(opts.provider))
	if provider == "" && wire == shrinkWireMock {
		m.State, m.Reason, m.Action = "unsupported", "offline mock usage has no provider cache counters", "select a live provider wire when provider-cache feedback is expected"
		return m
	}
	m.State, m.Reason = "active", "normalized provider usage feeds cache-read/cache-write signals, warmth error, and per-family economics independently of request shaping"
	return m
}

func postureVCacheCalibration(opts launchPostureOptions, wire string) launchPostureMechanism {
	provider := strings.ToLower(strings.TrimSpace(opts.provider))
	if provider == "" {
		provider = strings.ToLower(strings.TrimSpace(wire))
	}
	if provider == "" || provider == "native" || provider == "owned-model" {
		return launchPostureMechanism{
			Name: "vcache-calibration", State: "unsupported",
			Reason: "the selected native/offline wire has no provider cache-feedback calibration",
			Action: "select anthropic, openai, gemini, or xai when provider-cache calibration is expected",
		}
	}
	path := nightrunLedgerPath(vcachecalibration.DefaultCalibrationRel)
	statuses, err := vcachecalibration.CalibrationStatuses(path, []string{provider}, time.Now(), vcachecalibration.DefaultCalibrationTTL)
	if err != nil {
		return launchPostureMechanism{
			Name: "vcache-calibration", Configured: true, State: "inert",
			Reason: "the provider calibration ledger cannot be read: " + err.Error(),
			Action: "repair the ledger, then run a live provider probe and fak vcache calibration-status",
		}
	}
	status := statuses[0]
	m := launchPostureMechanism{Name: "vcache-calibration", Configured: true}
	if status.State == "fresh" {
		runtime, steering, runtimeReason := vcachecalibration.FreshRuntimeConstants(path, provider, "", time.Now(), vcachecalibration.DefaultCalibrationTTL)
		if steering {
			m.State = "active"
			m.Reason = fmt.Sprintf("dated %s calibration is fresh and steering (min-prefix=%d measured=%t, read-mult=%.4g measured=%t)", provider, runtime.MinPrefixTokens, runtime.MinPrefixMeasured, runtime.ReadMult, runtime.ReadMultMeasured)
			return m
		}
		m.State = "inert"
		m.Reason = "dated provider feedback is fresh but observational only: " + runtimeReason
		m.Action = "run fak vcache calibrate --samples PROBES --ledger " + path + " with measured provider probes"
		return m
	}
	m.State = "inert"
	m.Reason = status.Reason
	m.Action = status.Action
	return m
}
func postureCodeTools(opts launchPostureOptions, workspace string) launchPostureMechanism {
	m := launchPostureMechanism{Name: "bounded-code-tools", Configured: opts.nativeCodeTools, Disable: "--native-code-tools=false"}
	if !opts.nativeCodeTools {
		m.State, m.Reason, m.Action = "disabled", "bounded repository tools were explicitly disabled", "remove --native-code-tools=false"
		return m
	}
	if opts.entrypoint == "guard" {
		var manifest policy.Manifest
		if err := json.Unmarshal(guardDefaultPolicyJSON, &manifest); err != nil || !containsAllTools(manifest.Allow, "Read", "Write", "Edit", "Bash", "Grep", "Glob") {
			m.State, m.Reason, m.Action = "inert", "the guard default policy does not expose the full bounded repository-tool floor", "repair the guard policy or pass a policy that allows Read/Write/Edit/Bash/Grep/Glob"
			return m
		}
		m.State, m.Reason, m.Disable = "active", "the wrapped harness repository tools are admitted by the guard default policy", "use --policy to narrow the tool floor"
		return m
	}
	if opts.entrypoint == "agent" {
		if markUnreadableWorkspace(&m, workspace) {
			return m
		}
		m.State, m.Reason, m.Disable = "active", "the owned agent loop arms bounded repository tools at the selected workspace", "--code-tools=false"
		return m
	}
	if opts.entrypoint == "serve" && opts.native {
		if markUnreadableWorkspace(&m, workspace) {
			return m
		}
		m.State, m.Reason = "active", "native serve arms the bounded catalog at the selected workspace"
		return m
	}
	if opts.entrypoint == "serve" {
		m.State, m.Reason, m.Action = "inert", "the bounded catalog requires the owned native loop", "add --native or use the wrapped harness's witnessed repository tools"
		return m
	}
	m.State, m.Reason, m.Action = "unsupported", "this entrypoint does not own the native coding-tool catalog", "use fak serve --native in this workspace or verify the wrapped harness tool catalog"
	return m
}

func markUnreadableWorkspace(m *launchPostureMechanism, workspace string) bool {
	if isReadableWorkspace(workspace) {
		return false
	}
	m.State, m.Reason, m.Action = "inert", "the selected workspace is not a readable directory", "pass --workspace with an existing repository directory"
	return true
}

func isReadableWorkspace(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func containsAllTools(have []string, want ...string) bool {
	set := make(map[string]bool, len(have))
	for _, name := range have {
		set[name] = true
	}
	for _, name := range want {
		if !set[name] {
			return false
		}
	}
	return true
}

func postureOutputProfile(opts launchPostureOptions) launchPostureMechanism {
	profile := syspromptmmu.DescribeStyle(opts.outputProfile)
	m := launchPostureMechanism{Name: "caveman-response-profile", Configured: profile.Applied, Disable: profile.DisableCommand}
	if !profile.Known {
		m.State, m.Reason, m.Action = "unsupported", "the selected response profile is unknown", "choose caveman:medium or full"
		return m
	}
	if !profile.Applied {
		m.State, m.Reason, m.Action = "disabled", "response shaping is off", "select caveman:medium"
		return m
	}
	switch {
	case opts.entrypoint == "agent":
		m.State, m.Reason = "active", "the owned agent loop composes the selected governed response segment"
	case opts.entrypoint == "guard" && opts.harness == "claude":
		m.State, m.Reason = "active", "guard injects the selected governed segment through claude --append-system-prompt"
	case opts.entrypoint == "guard" && opts.harness == "codex":
		m.State, m.Reason = "active", "guard injects the selected governed segment through codex -c developer_instructions"
	case opts.entrypoint == "guard":
		m.State, m.Reason, m.Action = "unsupported", "the wrapped harness has no witnessed response-profile injection seam", "use Claude or Codex, set --output-profile full, or add a witnessed adapter"
	case opts.entrypoint == "serve" && opts.native:
		m.State, m.Reason, m.Action = "inert", "native serve does not map this doctor selection into FAK_STYLE", "set FAK_STYLE="+profile.Style+" before fak serve --native"
	default:
		m.State, m.Reason, m.Action = "unsupported", "passthrough serve does not own the upstream harness response policy", "configure the client harness or use fak guard -- claude"
	}
	return m
}

func postureWorkProfile(opts launchPostureOptions) launchPostureMechanism {
	profile := syspromptmmu.DescribeWorkProfile(opts.workProfile)
	m := launchPostureMechanism{Name: "ponytail-work-profile", Configured: profile.Applied, Disable: "--work-profile standard"}
	if !profile.Known {
		m.State, m.Reason, m.Action = "unsupported", "the selected work profile is unknown", "choose ponytail:medium or standard"
		return m
	}
	if !profile.Applied {
		m.State, m.Reason, m.Action = "disabled", "work-policy shaping is off", "select ponytail:medium"
		return m
	}
	switch {
	case opts.entrypoint == "agent":
		m.State, m.Reason = "active", "the owned agent loop composes the selected governed work segment"
	case opts.entrypoint == "guard" && opts.harness == "claude":
		m.State, m.Reason = "active", "guard injects the selected governed work segment through claude --append-system-prompt"
	case opts.entrypoint == "guard" && opts.harness == "codex":
		m.State, m.Reason = "active", "guard injects the selected governed work segment through codex -c developer_instructions"
	case opts.entrypoint == "guard":
		m.State, m.Reason, m.Action = "unsupported", "the wrapped harness has no witnessed work-profile injection seam", "use Claude or Codex, set --work-profile standard, or add a witnessed adapter"
	case opts.entrypoint == "serve" && opts.native:
		m.State, m.Reason = "active", "the owned native loop resolves the same default work profile"
	default:
		m.State, m.Reason, m.Action = "unsupported", "passthrough serve does not own the upstream harness work policy", "configure the client harness or use fak guard -- claude"
	}
	return m
}

func postureShrinkDecoded(opts launchPostureOptions, wire string) launchPostureMechanism {
	m := launchPostureMechanism{Name: "decoded-context-view", Configured: opts.ctxViewBudget > 0, Disable: "--ctx-view-budget 0"}
	if opts.ctxViewBudget <= 0 {
		m.State, m.Reason, m.Action = "disabled", "provider-neutral decoded context planning was explicitly disabled", "remove --ctx-view-budget 0"
		return m
	}
	if wire == shrinkWireAnthropicPassthrough {
		m.State, m.Reason = "active", "configured on; buffered Anthropic turns re-plan through the decoded context-view seam after byte-preserving transforms"
		return m
	}
	m.State, m.Reason = "active", "configured on; OpenAI-compatible and owned-model turns re-plan decoded history before the provider/model call"
	return m
}

func postureShrink(name string, configured bool, wire, disable, action string) launchPostureMechanism {
	m := launchPostureMechanism{Name: name, Configured: configured, Disable: disable}
	if !configured {
		m.State, m.Reason, m.Action = "disabled", "the launch setting disables this mechanism", "remove the disabling override"
		return m
	}
	if wire == shrinkWireAnthropicPassthrough {
		m.State, m.Reason = "active", "configured on and reached by the Anthropic passthrough request seam"
		return m
	}
	m.State, m.Reason, m.Action = "inert", "configured on but runtime wire "+wire+" does not carry the Anthropic request-body seam", action
	return m
}

func renderLaunchPosture(w io.Writer, report launchPostureReport) {
	verdict := "READY"
	if !report.OK {
		verdict = "ATTENTION"
	}
	fmt.Fprintf(w, "launch posture: %s entrypoint=%s", verdict, report.Entrypoint)
	if report.Harness != "" {
		fmt.Fprintf(w, " harness=%s", report.Harness)
	}
	fmt.Fprintf(w, " wire=%s workspace=%s\n", report.Wire, report.Workspace)
	for _, mechanism := range report.Mechanisms {
		fmt.Fprintf(w, "  %-26s %-11s %s\n", mechanism.Name, strings.ToUpper(mechanism.State), mechanism.Reason)
		if mechanism.Action != "" && mechanism.State != "active" {
			fmt.Fprintf(w, "    action: %s\n", mechanism.Action)
		}
	}
	fmt.Fprintf(w, "summary: active=%d inert=%d disabled=%d unsupported=%d\n", report.Summary["active"], report.Summary["inert"], report.Summary["disabled"], report.Summary["unsupported"])
}
