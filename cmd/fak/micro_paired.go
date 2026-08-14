package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

type pairedArm struct {
	Mechanism       string   `json:"mechanism"`
	Provider        string   `json:"provider"`
	Model           string   `json:"model"`
	Completed       bool     `json:"completed"`
	Correct         bool     `json:"correct"`
	Answer          string   `json:"answer"`
	InputTokens     int      `json:"input_tokens"`
	OutputTokens    int      `json:"output_tokens"`
	CacheReadTokens int      `json:"cache_read_tokens,omitempty"`
	WallMS          int64    `json:"wall_ms"`
	CostUSD         *float64 `json:"cost_usd"`
	CostStatus      string   `json:"cost_status"`
	Managed         bool     `json:"managed"`
	Error           string   `json:"error,omitempty"`
}

const (
	pairedBaselineGuardTimeout  = 2 * time.Minute
	pairedBaselineParentTimeout = pairedBaselineGuardTimeout + 20*time.Second
)

type pairedReport struct {
	Schema           string    `json:"schema"`
	Task             string    `json:"task"`
	Expected         string    `json:"expected"`
	Complexity       string    `json:"complexity"`
	ExecutionVerdict string    `json:"execution_verdict"`
	ValueVerdict     string    `json:"value_verdict"`
	Reason           string    `json:"reason"`
	Micro            pairedArm `json:"microagent"`
	CLI              pairedArm `json:"managed_baseline"`
}

type claudeResult struct {
	IsError        bool     `json:"is_error"`
	APIErrorStatus int      `json:"api_error_status"`
	Result         string   `json:"result"`
	TotalCostUSD   *float64 `json:"total_cost_usd"`
	Usage          struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
	ModelUsage map[string]struct {
		InputTokens          int     `json:"inputTokens"`
		OutputTokens         int     `json:"outputTokens"`
		CacheReadInputTokens int     `json:"cacheReadInputTokens"`
		CostUSD              float64 `json:"costUSD"`
		Provider             string  `json:"provider"`
	} `json:"modelUsage"`
}

func cmdMicroPaired(args []string) {
	fs := flag.NewFlagSet("micro paired", flag.ExitOnError)
	gateway := fs.String("gateway", "", "running fak serve address for the microagent arm")
	model := fs.String("model", "", "model requested through the fak gateway")
	task := fs.String("task", "Reply with exactly READY", "identical task sent to both arms")
	expected := fs.String("expect", "READY", "exact expected answer")
	cliModel := fs.String("cli-model", "sonnet", "Claude model used by the tuned guarded-CLI arm")
	complexity := fs.String("complexity", "one-step", "task complexity bucket")
	jsonOut := fs.Bool("json", false, "emit the paired receipt as JSON")
	fs.Parse(args)
	if strings.TrimSpace(*gateway) == "" || strings.TrimSpace(*model) == "" {
		fmt.Fprintln(os.Stderr, "fak micro paired: --gateway and --model are required")
		os.Exit(2)
	}
	r := runMicroPaired(context.Background(), *gateway, *model, *task, *expected, *cliModel, *complexity)
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
	} else {
		fmt.Printf("%s execution=%s value=%s micro=%v cli=%v — %s\n", r.Schema, r.ExecutionVerdict, r.ValueVerdict, r.Micro.Correct, r.CLI.Correct, r.Reason)
	}
	if r.ExecutionVerdict != "PASS" {
		os.Exit(1)
	}
}

func runMicroPaired(ctx context.Context, gateway, model, task, expected, cliModel, complexity string) pairedReport {
	r := pairedReport{Schema: "fak-micro-paired/1", Task: task, Expected: expected, Complexity: complexity}
	cfg := defaultMicroConfig(false)
	cfg.Engine, cfg.Gateway, cfg.Model, cfg.Task = "gateway", gateway, model, task
	cfg.Agents, cfg.Turns = 1, 1
	start := time.Now()
	_, _, results, observations, err := driveMicroObserved(cfg)
	r.Micro = pairedArm{Mechanism: "shared-fak-gateway", Provider: "openai-compatible", Model: model, WallMS: time.Since(start).Milliseconds(), CostStatus: "provider-unsupported", Managed: true}
	if err != nil {
		r.Micro.Error = err.Error()
	} else if len(results) == 1 {
		obs := observations[results[0].ID]
		r.Micro.Completed = results[0].Done && results[0].Err == nil
		r.Micro.Answer, r.Micro.Model = strings.TrimSpace(obs.Answer), obs.Model
		r.Micro.InputTokens, r.Micro.OutputTokens = obs.Usage.PromptTokens, obs.Usage.CompletionTokens
		if obs.Usage.CostUSD != nil && obs.Usage.CostStatus == "provider-reported" {
			cost := *obs.Usage.CostUSD
			r.Micro.CostUSD = &cost
			r.Micro.CostStatus = "provider-reported"
		}
		r.Micro.Correct = r.Micro.Completed && r.Micro.Answer == expected
	}

	r.CLI = runPairedBaseline(ctx, task, expected, cliModel)
	return foldPaired(r)
}

func foldPaired(r pairedReport) pairedReport {
	r.ExecutionVerdict = "FAIL"
	if r.Micro.Correct && r.CLI.Correct && r.Micro.InputTokens+r.Micro.OutputTokens > 0 && r.CLI.InputTokens+r.CLI.OutputTokens > 0 {
		r.ExecutionVerdict = "PASS"
	}
	r.ValueVerdict = "NOT_YET"
	r.Reason = "paired execution is measured, but one or both providers did not report cost; no quality/$ winner is claimed"
	if r.ExecutionVerdict != "PASS" {
		r.Reason = "one or both identical-task arms failed correctness or provider-usage evidence"
	} else if r.Micro.CostUSD != nil && r.CLI.CostUSD != nil {
		switch {
		case *r.Micro.CostUSD < *r.CLI.CostUSD:
			r.ValueVerdict = "MICRO_WINS"
			r.Reason = "both identical-task arms were correct and provider-reported dollars favor the shared fak gateway"
		case *r.Micro.CostUSD > *r.CLI.CostUSD:
			r.ValueVerdict = "BASELINE_WINS"
			r.Reason = "both identical-task arms were correct and provider-reported dollars favor the managed CLI baseline"
		default:
			r.ValueVerdict = "TIE"
			r.Reason = "both identical-task arms were correct and provider-reported dollars are equal"
		}
	}
	return r
}

func filteredPairedEnv(env []string) []string {
	drop := map[string]bool{
		"FAK_REGISTRATION_ID": true, "FAK_ROOT_REGISTRATION_ID": true, "FAK_PARENT_REGISTRATION_ID": true,
		"FAK_SPAWN_GRANT_ID": true, "FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT": true, "FAK_GUARD_CAP_PARK": true, "ENABLE_TOOL_SEARCH": true, "FAK_GUARD_AFFORDANCE_MODE": true,
	}
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		name := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			name = kv[:i]
		}
		if !drop[strings.ToUpper(name)] {
			out = append(out, kv)
		}
	}
	return append(out, "FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT=0", "FAK_GUARD_CAP_PARK=off", "FAK_GUARD_AFFORDANCE_MODE=off")
}
func runPairedBaseline(ctx context.Context, task, expected, model string) pairedArm {
	arm := pairedArm{Mechanism: "fak-manage-claude", Provider: "anthropic", Model: model, CostStatus: "provider-unreported", Managed: true}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, pairedBaselineParentTimeout)
		defer cancel()
	}
	exe, err := os.Executable()
	if err != nil {
		arm.Error = err.Error()
		return arm
	}
	cmd := exec.CommandContext(ctx, exe, "manage", "--quiet", "--probe", "--rotate", "off", "--max-duration", pairedBaselineGuardTimeout.String(), "--lease", "mode=off", "--", "claude", "-p", task, "--output-format", "json", "--max-turns", "1", "--tools", "", "--model", model, "--setting-sources", "")
	env := filteredPairedEnv(os.Environ())
	if !pairedEnvHas(env, "CLAUDE_CONFIG_DIR") {
		if dir := pairedReadyClaudeConfigDir(ctx, exe); dir != "" {
			env = append(env, "CLAUDE_CONFIG_DIR="+dir)
		}
	}
	cmd.Env = env
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	start := time.Now()
	err = cmd.Run()
	arm.WallMS = time.Since(start).Milliseconds()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		arm.Error = fmt.Sprintf("managed baseline parent timed out after %s", pairedBaselineParentTimeout)
		return arm
	}
	var got claudeResult
	if decodeErr := json.Unmarshal(stdout.Bytes(), &got); decodeErr != nil {
		if err != nil {
			arm.Error = strings.TrimSpace(err.Error() + ": " + stdout.String() + " " + stderr.String())
		} else {
			arm.Error = "decode Claude JSON: " + decodeErr.Error()
		}
		return arm
	}
	arm.Completed, arm.Answer = !got.IsError, strings.TrimSpace(got.Result)
	if got.IsError {
		arm.Error = arm.Answer
		reconcilePairedBaselineCooldown(cmd.Env, got, time.Now())
	}
	arm.Correct = arm.Completed && arm.Answer == expected
	arm.InputTokens, arm.OutputTokens = got.Usage.InputTokens+got.Usage.CacheCreationInputTokens, got.Usage.OutputTokens
	arm.CacheReadTokens = got.Usage.CacheReadInputTokens
	if !got.IsError && got.TotalCostUSD != nil {
		cost := *got.TotalCostUSD
		arm.CostUSD = &cost
		arm.CostStatus = "provider-reported"
	}
	for name, usage := range got.ModelUsage {
		arm.Model = name
		if arm.InputTokens == 0 {
			arm.InputTokens = usage.InputTokens
		}
		if arm.OutputTokens == 0 {
			arm.OutputTokens = usage.OutputTokens
		}
		if arm.CacheReadTokens == 0 {
			arm.CacheReadTokens = usage.CacheReadInputTokens
		}
		if !got.IsError && arm.CostUSD == nil && usage.CostUSD > 0 {
			cost := usage.CostUSD
			arm.CostUSD = &cost
			arm.CostStatus = "provider-reported"
		}
		break
	}
	return arm
}

func reconcilePairedBaselineCooldown(env []string, got claudeResult, now time.Time) bool {
	if !got.IsError || got.APIErrorStatus != 429 {
		return false
	}
	kind := classifyLaunchModelUnavailable(got.Result, "")
	if kind != launchModelUsageLimit && kind != launchModelRateLimit {
		return false
	}
	configDir := pairedEnvValue(env, "CLAUDE_CONFIG_DIR")
	account := accountKeyForDir(configDir)
	if account == "" {
		return false
	}
	entry, ok := recordLaunchCooldown(io.Discard, account, got.Result, kind, now)
	return ok && entry.Account == account
}

func pairedEnvValue(env []string, name string) string {
	prefix := strings.ToUpper(strings.TrimSpace(name)) + "="
	for _, entry := range env {
		if strings.HasPrefix(strings.ToUpper(entry), prefix) {
			return strings.TrimSpace(entry[len(prefix):])
		}
	}
	return ""
}

func pairedReadyClaudeConfigDir(ctx context.Context, exe string) string {
	probeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(probeCtx, exe, "accounts", "next", "--json").Output()
	if err != nil {
		return ""
	}
	var seat accounts.RotationSeat
	if json.Unmarshal(out, &seat) != nil || !seat.CanServe {
		return ""
	}
	return strings.TrimSpace(seat.Dir)
}

func pairedEnvHas(env []string, name string) bool { return pairedEnvValue(env, name) != "" }
