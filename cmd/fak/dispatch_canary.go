package main

// dispatch_canary.go — `fak dispatch canary`, the dispatcher CANARY mode (#3036): launch
// EXACTLY ONE worker through the Claude Code dogfood harness against a chosen provider
// route, so an operator can prove a route end-to-end (Claude Code -> fak guard/serve -> the
// remote OpenAI-compatible route) before opening the floodgates to a full wave.
//
//	# proof-only: one worker, tiny proof prompt, no real GitHub issue
//	fak dispatch canary --base-url URL --model M [--provider P] [--account A] [--api-key-env ENV]
//	# single-issue: one worker pointed at ONE issue
//	fak dispatch canary --base-url URL --model M --issue 3036
//	# see the exact plan (argv + env + sidecars) without spawning
//	fak dispatch canary --base-url URL --model M --dry-run --json
//
// The canary is safety-gated BEFORE it spends a live worker slot: it runs the same cheap
// route-health smoke `fak dispatch route-health probe` runs (probeProviderRoute), persists
// the fak-route-health/1 row to the shared ledger, and REFUSES the live launch on any
// non-healthy class (timeout, auth, model_unavailable, rate_limited, provider_5xx,
// unsupported_route) — reusing the generic classifier so a canary never disagrees with the
// pre-spawn gate. When it does launch it writes the full set of normal worker sidecars —
// prompt file, transcript/proof path, guard-audit path, account/model/base-url metadata,
// lane-lease metadata, and the route-health probe result — under
// .dispatch-runs/canary/<run-id>/, so a canary run is as auditable as any dispatched worker.
//
// The probe and the launch are both behind seams (canaryRouteProbe, canaryLaunch), so the
// command/environment shape — proof-only and single-issue — is unit-tested without a live
// provider key and without spawning a real agent.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const canarySchema = "fak-dispatch-canary/1"

// Closed verdict vocabulary. Each is both an exit-code class and the machine-readable
// disposition echoed in the fak-dispatch-canary/1 payload.
const (
	canaryVerdictLaunched = "CANARY_LAUNCHED"           // gate passed, worker spawned
	canaryVerdictPlanned  = "CANARY_PLANNED"            // gate passed, --dry-run: sidecars written, not spawned
	canaryVerdictRefused  = "CANARY_REFUSED_UNHEALTHY"  // route smoke non-healthy: no live launch
)

// canaryModeProofOnly runs a tiny proof prompt with no real GitHub issue; canaryModeSingleIssue
// points the one worker at exactly one issue.
const (
	canaryModeProofOnly   = "proof-only"
	canaryModeSingleIssue = "single-issue"
)

// canaryRouteProbe is the smoke seam: production runs the live route-health probe; a test
// overrides it to inject a synthetic healthy/unhealthy record without a provider key.
var canaryRouteProbe = probeProviderRoute

// canaryLaunch is the spawn seam: production execs the guarded Claude Code worker; a test
// overrides it to capture the plan and return an exit code without spawning an agent.
var canaryLaunch = execCanaryWorker

// canarySidecars names the on-disk artifacts a canary worker run writes — the same family a
// normally dispatched worker leaves behind, so a canary is auditable by the same tooling.
type canarySidecars struct {
	Dir         string `json:"dir"`
	Prompt      string `json:"prompt,omitempty"`
	Transcript  string `json:"transcript,omitempty"`
	GuardAudit  string `json:"guard_audit,omitempty"`
	Metadata    string `json:"metadata"`
	Lease       string `json:"lease,omitempty"`
	RouteHealth string `json:"route_health"`
}

// canaryPlan is the fak-dispatch-canary/1 payload: the resolved command/environment shape,
// the gating verdict, the route-health record, and the sidecar paths.
type canaryPlan struct {
	Schema      string            `json:"schema"`
	Mode        string            `json:"mode"`
	Verdict     string            `json:"verdict"`
	Reason      string            `json:"reason"`
	Route       string            `json:"route"`
	Provider    string            `json:"provider"`
	Account     string            `json:"account,omitempty"`
	Model       string            `json:"model"`
	BaseURL     string            `json:"base_url"`
	APIKeyEnv   string            `json:"api_key_env,omitempty"`
	Lane        string            `json:"lane"`
	Issue       int               `json:"issue,omitempty"`
	Command     string            `json:"command"`
	RunID       string            `json:"run_id"`
	Argv        []string          `json:"argv"`
	Env         []string          `json:"env"`
	Prompt      string            `json:"-"`
	Sidecars    canarySidecars    `json:"sidecars"`
	RouteHealth routeHealthRecord `json:"route_health"`
}

// canaryParams are the resolved inputs to the pure plan builder.
type canaryParams struct {
	Workspace string
	Provider  string
	Account   string
	Model     string
	BaseURL   string
	APIKeyEnv string
	Lane      string
	Issue     int
	ProofOnly bool
	Command   string
	Now       int64
	FakBin    string
}

// canaryRunID is the deterministic run identity — mode + sanitized route + clock — so a test
// re-derives the sidecar dir hermetically and two canaries of the same route at the same
// second never collide across modes.
func canaryRunID(mode, route string, issue int, now int64) string {
	base := "canary"
	if mode == canaryModeSingleIssue {
		base = fmt.Sprintf("canary-issue%d", issue)
	}
	return fmt.Sprintf("%s-%s-%d", base, canarySanitize(route), now)
}

// canarySanitize maps a route key (provider/account/model, which carries slashes, colons, and
// dots) onto a filesystem-safe token for the run-id.
func canarySanitize(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "route"
	}
	return out
}

// canaryPromptText is the worker's seed prompt: a no-write proof for proof-only mode, or a
// pointer at exactly one issue for single-issue mode. The full issue body is fetched by the
// harness; this is the canary's own instruction seed.
func canaryPromptText(mode string, issue int, model, route string) string {
	if mode == canaryModeSingleIssue {
		return fmt.Sprintf(
			"You are a fak dispatcher canary worker on route %s (model %s).\n"+
				"Resolve exactly GitHub issue #%d in this repository, following the repo's\n"+
				"dispatch worker contract: the smallest correct change, tests, and a stamped\n"+
				"commit. Do not touch any other issue.\n",
			route, model, issue)
	}
	return fmt.Sprintf(
		"You are a fak dispatcher canary worker on route %s (model %s).\n"+
			"This is a PROOF-ONLY canary: do NOT modify any files or open any issue.\n"+
			"Confirm the route is live end-to-end by replying with one line containing the\n"+
			"model id, the current working directory, and the exact token CANARY-OK, then exit.\n",
		route, model)
}

// buildCanaryPlan is the pure shape builder: it resolves mode, run-id, the guard-wrapped
// Claude Code argv, the canary-namespaced route/metadata env, the prompt, and the sidecar
// paths. No I/O, no probe — the runner attaches the route-health record and verdict. Pure so
// the command/environment shape is unit-tested without a key or a spawn.
func buildCanaryPlan(p canaryParams) canaryPlan {
	route := routeHealthKey(p.Provider, p.Account, p.Model)
	mode := canaryModeSingleIssue
	if p.ProofOnly || p.Issue <= 0 {
		mode = canaryModeProofOnly
	}
	lane := strings.TrimSpace(p.Lane)
	if lane == "" {
		lane = "canary/" + p.Provider
	}
	runID := canaryRunID(mode, route, p.Issue, p.Now)
	dir := filepath.Join(p.Workspace, timeoutLedgerRunsDir, "canary", runID)

	command := strings.TrimSpace(p.Command)
	if command == "" {
		command = "claude"
	}

	// The route reaches the worker's guard/gateway via --api-key-env (the same knob route-health
	// and the managed-cache posture use); guard is on so the kernel adjudicates every tool call.
	var guardCacheArgs []string
	if strings.TrimSpace(p.APIKeyEnv) != "" {
		guardCacheArgs = guardCachePostureArgs(guardManagedCacheOn, p.APIKeyEnv)
	}
	argv := buildLaunchArgv(p.FakBin, launchOpts{
		command:         command,
		useGuard:        true,
		skipPermissions: true,
		model:           p.Model,
		guardCacheArgs:  guardCacheArgs,
		// Headless print mode: exactly one worker turn, prompt fed on stdin from the sidecar so a
		// large issue prompt never rides the argv.
		passthrough: []string{"-p"},
	})

	prompt := canaryPromptText(mode, p.Issue, p.Model, route)
	side := canarySidecars{
		Dir:         dir,
		Prompt:      filepath.Join(dir, "prompt.txt"),
		Transcript:  filepath.Join(dir, "transcript.jsonl"),
		GuardAudit:  filepath.Join(dir, "guard-audit.jsonl"),
		Metadata:    filepath.Join(dir, "metadata.json"),
		Lease:       filepath.Join(dir, "lease.json"),
		RouteHealth: filepath.Join(dir, "route-health.json"),
	}

	// Canary-namespaced worker/sidecar contract env. Route selection env is recorded here so the
	// launch seam (and any worker tooling) sees the exact route, run dir, and sidecar paths.
	envPairs := map[string]string{
		"FAK_CANARY_RUN_ID":      runID,
		"FAK_CANARY_ROUTE":       route,
		"FAK_CANARY_MODE":        mode,
		"FAK_CANARY_LANE":        lane,
		"FAK_CANARY_PROVIDER":    p.Provider,
		"FAK_CANARY_MODEL":       p.Model,
		"FAK_CANARY_BASE_URL":    p.BaseURL,
		"FAK_CANARY_RUN_DIR":     dir,
		"FAK_CANARY_TRANSCRIPT":  side.Transcript,
		"FAK_CANARY_GUARD_AUDIT": side.GuardAudit,
	}
	if strings.TrimSpace(p.Account) != "" {
		envPairs["FAK_CANARY_ACCOUNT"] = p.Account
	}
	if strings.TrimSpace(p.APIKeyEnv) != "" {
		envPairs["FAK_CANARY_API_KEY_ENV"] = p.APIKeyEnv
	}
	if mode == canaryModeSingleIssue {
		envPairs["FAK_CANARY_ISSUE"] = fmt.Sprintf("%d", p.Issue)
	}

	return canaryPlan{
		Schema:    canarySchema,
		Mode:      mode,
		Route:     route,
		Provider:  p.Provider,
		Account:   p.Account,
		Model:     p.Model,
		BaseURL:   p.BaseURL,
		APIKeyEnv: p.APIKeyEnv,
		Lane:      lane,
		Issue:     p.Issue,
		Command:   command,
		RunID:     runID,
		Argv:      argv,
		Env:       canarySortedEnv(envPairs),
		Prompt:    prompt,
		Sidecars:  side,
	}
}

// canarySortedEnv flattens the canary env map into a deterministic KEY=VALUE slice.
func canarySortedEnv(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+m[k])
	}
	return out
}

// canaryLeaseMetadata is the lane-lease sidecar body: enough to reconstruct which lane this
// canary held, in which mode, under which run-id.
type canaryLeaseMetadata struct {
	Schema    string `json:"schema"`
	Lane      string `json:"lane"`
	Mode      string `json:"mode"`
	RunID     string `json:"run_id"`
	Route     string `json:"route"`
	Issue     int    `json:"issue,omitempty"`
	Workspace string `json:"workspace"`
	Now       int64  `json:"acquired_at_unix"`
}

// writeCanarySidecars materializes the run's sidecars. metadata + route-health are ALWAYS
// written (a refusal is auditable too); the full worker set — prompt, lane lease, and the
// reserved transcript/guard-audit proof paths — is written only when the canary actually
// launches (or is planned in --dry-run), since a refused route never produced a worker turn.
func writeCanarySidecars(plan canaryPlan, workspace string, now int64, launching bool) error {
	if err := os.MkdirAll(plan.Sidecars.Dir, 0o755); err != nil {
		return err
	}
	metaBody, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(plan.Sidecars.Metadata, append(metaBody, '\n'), 0o644); err != nil {
		return err
	}
	rhBody, err := json.MarshalIndent(plan.RouteHealth, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(plan.Sidecars.RouteHealth, append(rhBody, '\n'), 0o644); err != nil {
		return err
	}
	if !launching {
		return nil
	}
	if err := os.WriteFile(plan.Sidecars.Prompt, []byte(plan.Prompt), 0o644); err != nil {
		return err
	}
	lease := canaryLeaseMetadata{
		Schema:    "fak-dispatch-canary-lease/1",
		Lane:      plan.Lane,
		Mode:      plan.Mode,
		RunID:     plan.RunID,
		Route:     plan.Route,
		Issue:     plan.Issue,
		Workspace: workspace,
		Now:       now,
	}
	leaseBody, err := json.MarshalIndent(lease, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(plan.Sidecars.Lease, append(leaseBody, '\n'), 0o644); err != nil {
		return err
	}
	// Reserve the transcript/guard-audit proof paths so the worker + guard append into the run
	// dir the plan already advertises (empty is a valid "no turn recorded yet").
	for _, p := range []string{plan.Sidecars.Transcript, plan.Sidecars.GuardAudit} {
		if err := reserveCanaryProofFile(p); err != nil {
			return err
		}
	}
	return nil
}

// reserveCanaryProofFile creates an empty proof file only if it does not already exist, so a
// re-run never truncates a transcript a worker is mid-writing.
func reserveCanaryProofFile(path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	return f.Close()
}

// execCanaryWorker is the production spawn seam: it execs the guarded Claude Code worker with
// the canary env layered over the process environment, feeds the prompt on stdin from the
// sidecar, and tees the worker's stdout into the transcript proof file.
func execCanaryWorker(stdout, stderr io.Writer, plan canaryPlan) int {
	if len(plan.Argv) == 0 {
		fmt.Fprintln(stderr, "fak dispatch canary: empty launch argv")
		return 2
	}
	promptFile, err := os.Open(plan.Sidecars.Prompt)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch canary: open prompt: %v\n", err)
		return 1
	}
	defer promptFile.Close()
	transcript, err := os.OpenFile(plan.Sidecars.Transcript, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch canary: open transcript: %v\n", err)
		return 1
	}
	defer transcript.Close()

	cmd := exec.Command(plan.Argv[0], plan.Argv[1:]...)
	cmd.Env = append(os.Environ(), plan.Env...)
	cmd.Stdin = promptFile
	cmd.Stdout = io.MultiWriter(stdout, transcript)
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak dispatch canary: launch: %v\n", err)
		return 1
	}
	return 0
}

// runDispatchCanary is the CLI entry. Exit 0 = launched (or planned in --dry-run); 3 = refused
// on a non-healthy route; 1/2 = usage/IO error.
func runDispatchCanary(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch canary", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", "", "OpenAI-compatible base URL of the route to canary (required)")
	model := fs.String("model", "", "model id the canary worker uses (required)")
	provider := fs.String("provider", "", "provider name for the route key (default: base-url host)")
	account := fs.String("account", "", "account/seat identity for the route key")
	apiKeyEnv := fs.String("api-key-env", "", "environment variable holding the route API key")
	issue := fs.Int("issue", 0, "single-issue mode: point the one worker at this issue number (0 = proof-only)")
	proofOnly := fs.Bool("proof-only", false, "force proof-only mode (no real issue) even if --issue is set")
	lane := fs.String("lane", "", "lane-lease identity for the canary worker (default: canary/<provider>)")
	command := fs.String("command", "claude", "agent command the harness launches")
	workspace := fs.String("workspace", ".", "workspace root sidecars and the route-health ledger live under")
	timeout := fs.Duration("timeout", routeHealthDefaultTimeout, "route smoke timeout")
	nowUnix := fs.Int64("now", 0, "clock as unix seconds for the run-id/probe (0 = current time)")
	dryRun := fs.Bool("dry-run", false, "gate + write sidecars but do not spawn the worker")
	asJSON := fs.Bool("json", false, "emit the fak-dispatch-canary/1 plan as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*baseURL) == "" || strings.TrimSpace(*model) == "" {
		fmt.Fprintln(stderr, "fak dispatch canary: --base-url and --model are required")
		return 2
	}
	prov := canaryProviderDefault(*provider, *baseURL)
	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak"
	}

	// The pre-spawn smoke: run the same route-health probe the gate consumes, and persist it to
	// the shared ledger so `route-health status`/`gate` see this canary's probe too.
	spec := routeProbeSpec{
		Provider:  prov,
		Account:   strings.TrimSpace(*account),
		Model:     strings.TrimSpace(*model),
		BaseURL:   strings.TrimSpace(*baseURL),
		APIKeyEnv: strings.TrimSpace(*apiKeyEnv),
		Timeout:   *timeout,
	}
	rec, err := canaryRouteProbe(spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch canary: route smoke: %v\n", err)
		return 1
	}
	if err := appendRouteHealthRecord(routeHealthLedgerPath(*workspace), rec); err != nil {
		fmt.Fprintf(stderr, "fak dispatch canary: persist route-health ledger: %v\n", err)
		return 1
	}

	plan := buildCanaryPlan(canaryParams{
		Workspace: *workspace,
		Provider:  prov,
		Account:   strings.TrimSpace(*account),
		Model:     strings.TrimSpace(*model),
		BaseURL:   strings.TrimSpace(*baseURL),
		APIKeyEnv: strings.TrimSpace(*apiKeyEnv),
		Lane:      strings.TrimSpace(*lane),
		Issue:     *issue,
		ProofOnly: *proofOnly,
		Command:   strings.TrimSpace(*command),
		Now:       now,
		FakBin:    fakBin,
	})
	plan.RouteHealth = rec

	// Refuse the live launch on any non-healthy class — the same classifier the pre-spawn gate
	// uses, so a canary never disagrees with the gate. The refusal is still auditable (metadata +
	// route-health sidecars written), but no worker is spawned.
	if rec.Class != string(routeClassHealthy) {
		plan.Verdict = canaryVerdictRefused
		plan.Reason = fmt.Sprintf("route smoke class %q is not healthy; refusing live launch (recheck: %s)", rec.Class, rec.Recheck)
		if werr := writeCanarySidecars(plan, *workspace, now, false); werr != nil {
			fmt.Fprintf(stderr, "fak dispatch canary: write sidecars: %v\n", werr)
			return 1
		}
		canaryReport(stdout, stderr, plan, *asJSON)
		return 3
	}

	launching := !*dryRun
	if launching {
		plan.Verdict = canaryVerdictLaunched
		plan.Reason = "route smoke healthy; launching one worker through the Claude Code harness"
	} else {
		plan.Verdict = canaryVerdictPlanned
		plan.Reason = "route smoke healthy; --dry-run: sidecars written, worker not spawned"
	}
	if werr := writeCanarySidecars(plan, *workspace, now, true); werr != nil {
		fmt.Fprintf(stderr, "fak dispatch canary: write sidecars: %v\n", werr)
		return 1
	}
	canaryReport(stdout, stderr, plan, *asJSON)
	if !launching {
		return 0
	}
	return canaryLaunch(stdout, stderr, plan)
}

// canaryProviderDefault mirrors route-health's provider defaulting: the base-url host names the
// route family when --provider is omitted.
func canaryProviderDefault(provider, baseURL string) string {
	prov := strings.TrimSpace(provider)
	if prov != "" {
		return prov
	}
	if u, err := url.Parse(strings.TrimSpace(baseURL)); err == nil && u.Host != "" {
		return u.Host
	}
	return "provider"
}

// canaryReport emits the plan as JSON (--json) or a compact human card.
func canaryReport(stdout, stderr io.Writer, plan canaryPlan, asJSON bool) {
	if asJSON {
		encodeJSONOrFail(stdout, stderr, plan, "fak dispatch canary")
		return
	}
	fmt.Fprintf(stdout, "canary %s [%s] route=%s model=%s\n", plan.Verdict, plan.Mode, plan.Route, plan.Model)
	fmt.Fprintf(stdout, "  reason: %s\n", plan.Reason)
	fmt.Fprintf(stdout, "  run-id: %s\n", plan.RunID)
	fmt.Fprintf(stdout, "  sidecars: %s\n", plan.Sidecars.Dir)
	if plan.Verdict != canaryVerdictRefused {
		fmt.Fprintf(stdout, "  command: %s\n", strings.Join(plan.Argv, " "))
	}
}
