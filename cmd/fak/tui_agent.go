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
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	acct "github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func runTUIAgent(stdout, stderr io.Writer, argv []string) int {
	// #938: a leading non-flag token may NAME a compute target (`fak c mac`). Resolve
	// it against the registry BEFORE flag parsing, because Go's flag package stops at
	// the first positional — a leading token would otherwise swallow every trailing
	// flag. A KNOWN target is stripped here and applied below; an UNKNOWN leading token
	// is left in place so it still forwards to `claude` verbatim (back-compat: the
	// `fak c mac`→`claude mac` footgun only changes once `mac` is a real target).
	reg, regErr := loadComputeTargets(defaultComputeTargetsFile())
	var leadingTarget string
	if regErr == nil {
		leadingTarget, argv = resolveLeadingTarget(argv, reg, stderr)
	}

	fs := flag.NewFlagSet("tui agent", flag.ContinueOnError)
	fs.SetOutput(stderr)
	defHome, _ := os.UserHomeDir()
	regDefault := os.Getenv("FAK_ACCOUNTS_REGISTRY")
	if regDefault == "" && defHome != "" {
		regDefault = filepath.Join(defHome, ".claude-accounts", "registry.json")
	}
	backend := fs.String("backend", "claude", "backend agent to launch (currently: claude)")
	command := fs.String("command", "claude", "Claude Code command or path to execute")
	account := fs.String("account", "", "Claude config-home account name from `fak accounts`")
	claudeConfigDir := fs.String("claude-config-dir", "", "explicit Claude config directory to pass as CLAUDE_CONFIG_DIR")
	registry := fs.String("registry", regDefault, "path to the fak accounts registry.json")
	home := fs.String("home", defHome, "home dir used when discovering Claude account homes")
	prompt := fs.String("prompt", "", "append `claude -p PROMPT` for a non-interactive backend run")
	permissionMode := fs.String("permission-mode", "bypassPermissions", "Claude --permission-mode for every spawned account session (default bypassPermissions so the guarded backend, not Claude's own prompt, mediates tools); pass \"\" to omit, or override it in the trailing `claude args`")
	policyPath := fs.String("policy", "", "capability-floor manifest for the guard child (default: built-in guard floor)")
	model := fs.String("model", "", "upstream Claude model override for the guard child")
	effort := fs.String("effort", "", "Claude reasoning effort for the next launch: low|medium|high|xhigh|max (a trailing Claude --effort remains an escape hatch)")
	sessionID := fs.String("session-id", "tui-agent", "trace/session id for the guard session")
	contextBudget := fs.Int("context-budget-tokens", 0, "seed a context-token budget in the guard session")
	restartOnBudget := fs.Bool("restart-on-budget", false, "ask guard to relaunch Claude on context-budget exhaustion")
	restartLimit := fs.Int("restart-limit", 0, "maximum guard relaunches for --restart-on-budget; 0 means unlimited")
	passthrough := fs.Bool("passthrough", false, "do not force subscription OAuth; let Claude Code forward its own credential")
	gatewayURL := fs.String("gateway-url", "", "existing fak serve gateway to use instead of starting a local guard, e.g. http://node:8080")
	gatewayKeyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the existing gateway bearer for --gateway-url")
	apiTimeoutMS := fs.Int("api-timeout-ms", 1800000, "API_TIMEOUT_MS for --gateway-url launches; 0 leaves it inherited")
	debugStats := fs.Bool("debug-stats", true, "print one compact per-turn line to stderr that leads with a verdict (ok/warming/degraded/cold) + the NET write-premium-aware token saving, then cache health and compaction; wired to fak guard --debug-stats")
	atText := fs.String("at", "", "snapshot time (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for dry-run human rendering")
	dryRun := fs.Bool("dry-run", false, "render the launch plan without starting the backend agent")
	asJSON := fs.Bool("json", false, "emit the launch model as JSON and do not start the backend agent")
	listTargets := fs.Bool("list-targets", false, "list the named compute targets (mac/gcp/local/anthropic + ~/.fak/targets.json) with a live /healthz column, then exit")
	targetFlag := fs.String("target", "", "named compute target to chat against (mac/gcp/local/anthropic + ~/.fak/targets.json); the explicit form of the leading `fak c <target>` token")
	auto := fs.Bool("auto", false, "health/cost/quota-aware automatic compute-target selection with failover (#939): probe every registered target, rank by the documented policy (healthy first, then cheapest/most-local), and launch the best one; --json emits the ranked decision instead of launching")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *listTargets {
		return runListComputeTargets(stdout, stderr, *asJSON)
	}
	// Which flags did the user set explicitly? A resolved target fills in only the
	// fields the user did NOT pass, so `fak c mac --model foo` keeps foo.
	setFlags := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { setFlags[f.Name] = true })
	// The explicit --target flag and the leading positional token must agree; pick the
	// one that is set (flag value wins if they are equal, errors if they conflict).
	selectedTarget := strings.TrimSpace(*targetFlag)
	if selectedTarget != "" && leadingTarget != "" && !strings.EqualFold(selectedTarget, leadingTarget) {
		fmt.Fprintf(stderr, "fak console agent: conflicting target: positional %q vs --target %q (pass one)\n", leadingTarget, selectedTarget)
		return 2
	}
	if selectedTarget == "" {
		selectedTarget = leadingTarget
	}
	// #939: --auto ranks the registered targets (healthy first, then cheapest/most-local)
	// and selects the winner, which then flows through the same resolution path as an
	// explicit target below. It is mutually exclusive with a named target or --gateway-url.
	selectedTarget, autoSelected, code, done := resolveAutoTarget(
		*auto, selectedTarget, setFlags, regErr, reg, *asJSON, stdout, stderr)
	if done {
		return code
	}
	if *width < 72 {
		*width = 72
	}
	if *contextBudget < 0 {
		fmt.Fprintln(stderr, "fak console agent: --context-budget-tokens must be non-negative")
		return 2
	}
	if *restartLimit < 0 {
		fmt.Fprintln(stderr, "fak console agent: --restart-limit must be non-negative")
		return 2
	}
	if *restartOnBudget && *contextBudget <= 0 {
		writeConfigBail(stderr, configBail{
			Verb:    "fak console agent",
			Reason:  bailBudgetFlagIncoherent,
			Summary: "--restart-on-budget has no budget to restart against",
			Knobs: []bailKnob{
				bailFlag("restart-on-budget", "true"),
				bailFlag("context-budget-tokens", strconv.Itoa(*contextBudget)).want("a positive token count, or drop --restart-on-budget"),
			},
			Check: "fak console agent --help   # both budget flags and their defaults",
		})
		return 2
	}
	if *apiTimeoutMS < 0 {
		fmt.Fprintln(stderr, "fak console agent: --api-timeout-ms must be non-negative")
		return 2
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console agent: %v\n", err)
		return 2
	}
	opts := tuiAgentOptions{
		Backend:             *backend,
		Command:             *command,
		CommandArgs:         fs.Args(),
		Prompt:              *prompt,
		PermissionMode:      *permissionMode,
		Account:             *account,
		ClaudeConfigDir:     *claudeConfigDir,
		Registry:            *registry,
		Home:                *home,
		Policy:              *policyPath,
		Model:               *model,
		Effort:              *effort,
		SessionID:           *sessionID,
		ContextBudgetTokens: *contextBudget,
		RestartOnBudget:     *restartOnBudget,
		RestartLimit:        *restartLimit,
		Passthrough:         *passthrough,
		GatewayURL:          *gatewayURL,
		GatewayKeyEnv:       *gatewayKeyEnv,
		APITimeoutMS:        *apiTimeoutMS,
		DebugStats:          *debugStats,
	}
	// #938: fold a resolved compute target into the launch options. A positional that
	// reached here is always a known target; an unknown --target value is an error
	// (unlike an unknown positional, which already passed through to claude above).
	var resolvedTarget *computeTarget
	if selectedTarget != "" {
		if regErr != nil {
			fmt.Fprintf(stderr, "fak console agent: load compute targets: %v\n", regErr)
			return 1
		}
		tgt, ok := reg.resolve(selectedTarget)
		if !ok {
			fmt.Fprintf(stderr, "fak console agent: unknown --target %q (see `fak c --list-targets`)\n", selectedTarget)
			if hint := reg.nearest(selectedTarget); hint != "" {
				fmt.Fprintf(stderr, "  did you mean %q?\n", hint)
			}
			return 2
		}
		applyComputeTarget(&opts, tgt, setFlags)
		resolvedTarget = &tgt
		// A target can be /healthz-up yet un-launchable because its bearer env var is
		// unset (the mac gateway's healthz is unauthenticated). Fail here with a
		// target-named, actionable message instead of the generic downstream
		// "--gateway-url requires FAK_GATEWAY_KEY to be set" — this covers both explicit
		// `fak c mac` and an `--auto` winner that resolved to a credential-gated target.
		if msg, missing := computeTargetCredMissing(tgt, opts.GatewayKeyEnv, os.Getenv); missing {
			fmt.Fprintln(stderr, msg)
			return 2
		}
	}
	report, err := buildTUIAgentReport(opts, at, tuiExecutable(), os.Getenv)
	if err != nil {
		fmt.Fprintf(stderr, "fak console agent: %v\n", err)
		return 2
	}
	report.Target = selectedTarget
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console agent")
	}
	if *dryRun {
		fmt.Fprint(stdout, renderTUIAgent(report, *width))
		return 0
	}
	// #938: gate an interactive launch against a resolved remote target on a reachable
	// gateway — mirror the claude-mac-fak preflight so `fak c mac/gcp/local` never hands
	// the terminal to Claude against a dead/mock backend. A target with no /healthz (the
	// real Anthropic API) is n/a and never blocks. --auto already probed every target and
	// picked a healthy winner, so it skips this redundant second probe.
	if resolvedTarget != nil && !autoSelected {
		if code, gated := preflightComputeTarget(stdout, stderr, *resolvedTarget); gated {
			return code
		}
	}
	return launchTUIAgent(stdout, stderr, report)
}

// resolveLeadingTarget peeks at the first arg. When it is a non-flag token that names a
// registered compute target, it returns that name and the remaining args (the token
// stripped) so the rest can be flag-parsed normally. An unknown token is left in place —
// back-compat: it forwards to `claude` exactly as today — with a one-line "did you mean"
// hint to stderr only when the token is lexically close to a real target name (#938).
func resolveLeadingTarget(argv []string, reg *targetRegistry, stderr io.Writer) (string, []string) {
	if len(argv) == 0 || reg == nil {
		return "", argv
	}
	tok := argv[0]
	if tok == "" || strings.HasPrefix(tok, "-") {
		return "", argv // a flag (or empty) — never a leading target token
	}
	if _, ok := reg.resolve(tok); ok {
		return tok, argv[1:]
	}
	if hint := reg.nearest(tok); hint != "" {
		fmt.Fprintf(stderr, "fak console agent: %q is not a known compute target (did you mean %q? — `fak c --list-targets`); forwarding it to claude\n", tok, hint)
	}
	return "", argv
}

// computeTargetCredMissing reports whether a resolved gateway-routed target declares a
// credential env var that is empty in the environment, and returns an actionable,
// target-named message. It exists because a target selected purely on a live /healthz
// (which is unauthenticated) can still be un-launchable when its bearer env var is unset:
// today that surfaces only as the generic downstream "--gateway-url requires FAK_GATEWAY_KEY
// to be set", which names neither the target nor how to supply the key. It fires ONLY for a
// REMOTE gateway-url/local-spawn target with a non-empty effective key env whose value is
// unset — a loopback local serve and the anthropic provider-proxy (OAuth via guard, empty
// GatewayURL) are exempt, matching buildTUIAgentGatewayReport's own bearer tolerance.
func computeTargetCredMissing(tgt computeTarget, keyEnv string, getenv func(string) string) (string, bool) {
	if tgt.Kind != targetGatewayURL && tgt.Kind != targetLocalSpawn {
		return "", false
	}
	if gatewayIsLocal(tgt.GatewayURL) {
		return "", false
	}
	env := strings.TrimSpace(keyEnv)
	if env == "" {
		env = strings.TrimSpace(tgt.CredEnv)
	}
	if env == "" || strings.TrimSpace(getenv(env)) != "" {
		return "", false // target needs no bearer, or the bearer is present
	}
	// Reachability and authorization fail differently, and the operator cannot tell
	// them apart from the outside: the mac gateway answers /healthz unauthenticated,
	// so the target looks up right until the first real call. Name the credential and
	// where to get it rather than letting the generic downstream message do it.
	check := fmt.Sprintf("export %s=$(ssh <gateway-host> 'cat ~/.fak-gateway-key')\n          fak c --list-targets   # every target and the credential it needs", env)
	if strings.EqualFold(tgt.Name, "mac") {
		check += "\n          fak claude-mac-fak     # fetches the Mac bearer over SSH for you"
	}
	var b strings.Builder
	writeConfigBail(&b, configBail{
		Verb:    "fak console agent",
		Reason:  bailKeyEnvUnset,
		Summary: fmt.Sprintf("target %q is reachable, but its gateway credential is empty", tgt.Name),
		Knobs: []bailKnob{
			bailFlag("gateway-key-env", env),
			bailEnv(env, "").want("the bearer token " + tgt.GatewayURL + " accepts"),
		},
		Check: check,
		Bind:  []string{"env=" + env},
	})
	// The caller Fprintln's this, and the block already ends in a newline.
	return strings.TrimRight(b.String(), "\n"), true
}

// applyComputeTarget folds a resolved target into the launch options WITHOUT clobbering
// any flag the user set explicitly (setFlags wins), so `fak c mac --model foo` keeps foo.
// A gateway-url / local-spawn target routes the launch through the existing --gateway-url
// path (gateway + model + the cred env-var NAME); the anthropic provider-proxy target IS
// the default guard path, so it leaves GatewayURL empty and carries only its model.
func applyComputeTarget(opt *tuiAgentOptions, tgt computeTarget, setFlags map[string]bool) {
	switch tgt.Kind {
	case targetGatewayURL, targetLocalSpawn:
		if !setFlags["gateway-url"] {
			opt.GatewayURL = tgt.GatewayURL
		}
		if !setFlags["model"] && tgt.Model != "" {
			opt.Model = tgt.Model
		}
		if !setFlags["gateway-key-env"] && tgt.CredEnv != "" {
			opt.GatewayKeyEnv = tgt.CredEnv
		}
	case targetProviderProxy:
		// The default guard path (provider anthropic, subscription OAuth): leave
		// GatewayURL empty so buildTUIAgentReport takes the guard branch, and carry
		// only the named model.
		if !setFlags["model"] && tgt.Model != "" {
			opt.Model = tgt.Model
		}
	}
}

// preflightComputeTarget probes a resolved target's /healthz before an interactive launch
// and gates a launch against a dead gateway (#938), reusing the registry probe. A target
// with no /healthz endpoint (the real Anthropic API) is n/a and never blocks. It returns
// gated=true with an exit code ONLY when the launch must be aborted.
func preflightComputeTarget(stdout, stderr io.Writer, tgt computeTarget) (int, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	health := tgt.probe(ctx, &http.Client{Timeout: 3 * time.Second})
	switch health.State {
	case "down":
		fmt.Fprintf(stderr, "fak console agent: target %q gateway is unreachable: %s\n", tgt.Name, health.Detail)
		fmt.Fprintf(stderr, "  not launching claude against a dead backend — check the gateway, or pick another target (`fak c --list-targets`)\n")
		return 1, true
	case "up":
		fmt.Fprintf(stdout, "fak console agent: target %q gateway is up — launching claude ...\n", tgt.Name)
	}
	// "up" and "n/a" both proceed; "n/a" (no /healthz, e.g. anthropic) prints nothing.
	return 0, false
}

func runTUIOverview(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui overview", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issuesJSON := fs.String("issues-json", "", "read gh issue JSON and include the issue pane card")
	epic := fs.Int("epic", 0, "highlight one epic issue number for the issue pane card")
	ledger := fs.String("ledger", "", "read loop JSONL ledger and include the loop pane card")
	sessionsJSON := fs.String("sessions-json", "", "read SessionListResponse JSON and include the session pane card")
	gardenJSON := fs.String("garden-json", "", "read fak garden JSON and include the garden pane card")
	savingsLedger := fs.String("savings-ledger", "", "Track-2 OBSERVED-$ savings ledger for the above-the-fold savings hero (default: live then published ledger)")
	var guardJSON stringList
	fs.Var(&guardJSON, "guard-json", "read a guard artifact JSON file and include the guard pane card (repeatable)")
	var paneList stringList
	fs.Var(&paneList, "pane", "overview pane id to include, in display order (repeatable; overrides overview_panes in console config)")
	paneSourceFlag := fs.String("pane-source", "", "source of default --pane values (internal)")
	check := fs.Bool("check", false, "include the garden gate decision when --garden-json is set")
	asOfText := fs.String("as-of", "", "date used for issue age/idle math (YYYY-MM-DD, default: today UTC)")
	atText := fs.String("at", "", "snapshot time for non-issue panes (RFC3339 or YYYY-MM-DD, default: now)")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	asJSON := fs.Bool("json", false, "emit the overview TUI model as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console overview: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *width < 72 {
		*width = 72
	}
	asOf, err := parseTUIDay(*asOfText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console overview: %v\n", err)
		return 2
	}
	at, err := parseTUITime(*atText)
	if err != nil {
		fmt.Fprintf(stderr, "fak console overview: %v\n", err)
		return 2
	}
	panes := normalizeTUIOverviewPaneList([]string(paneList))
	paneSource := ""
	if len(panes) > 0 {
		paneSource = strings.TrimSpace(*paneSourceFlag)
		if paneSource == "" {
			paneSource = "flag"
		}
	}
	report, err := loadTUIOverview(tuiOverviewOptions{
		IssuesJSON:    *issuesJSON,
		Epic:          *epic,
		Ledger:        *ledger,
		SessionsJSON:  *sessionsJSON,
		GardenJSON:    *gardenJSON,
		SavingsLedger: *savingsLedger,
		GuardJSON:     []string(guardJSON),
		CheckGarden:   *check,
		PaneOrder:     panes,
		PaneSource:    paneSource,
		AsOf:          asOf,
		At:            at,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak console overview: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak console overview")
	}
	fmt.Fprint(stdout, renderTUIOverview(report, *width))
	return 0
}

type tuiOverviewOptions struct {
	IssuesJSON    string
	Epic          int
	Ledger        string
	SessionsJSON  string
	GardenJSON    string
	SavingsLedger string
	GuardJSON     []string
	CheckGarden   bool
	PaneOrder     []string
	PaneSource    string
	AsOf          time.Time
	At            time.Time
}

func parseTUIDay(s string) (time.Time, error) {
	if strings.TrimSpace(s) == "" {
		now := time.Now().UTC()
		return time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC), nil
	}
	t, err := time.Parse("2006-01-02", strings.TrimSpace(s))
	if err != nil {
		return time.Time{}, fmt.Errorf("--as-of must be YYYY-MM-DD")
	}
	return t, nil
}

func parseTUITime(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Now().UTC(), nil
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t.UTC(), nil
	}
	return time.Time{}, fmt.Errorf("--at must be RFC3339 or YYYY-MM-DD")
}

// claudeArgsHavePermissionMode reports whether the operator already set a
// --permission-mode in the trailing `claude args`, in which case the default
// bypassPermissions must not be injected (Claude rejects a duplicated flag).
// It matches both the split form (`--permission-mode X`) and the joined form
// (`--permission-mode=X`).
func claudeArgsHavePermissionMode(args []string) bool {
	for _, a := range args {
		if a == "--permission-mode" || strings.HasPrefix(a, "--permission-mode=") {
			return true
		}
	}
	return false
}

var tuiAgentEffortOptions = []string{"low", "medium", "high", "xhigh", "max"}

func validTUIAgentEffort(value string) bool {
	for _, option := range tuiAgentEffortOptions {
		if value == option {
			return true
		}
	}
	return false
}

// claudeArgValue finds an operator-owned option in the trailing raw Claude argv.
// Raw arguments are intentionally not constrained by the console's finite picker:
// they are the compatibility escape hatch for a newer Claude CLI vocabulary.
// When aliases or repeated options are present, the last occurrence wins, matching
// ordinary CLI precedence.
func claudeArgValue(args []string, names ...string) (string, bool) {
	var value string
	found := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		matched := false
		for _, name := range names {
			if arg == name {
				found = true
				matched = true
				if i+1 < len(args) {
					value = args[i+1]
					i++
				}
				break
			}
			if strings.HasPrefix(arg, name+"=") {
				found = true
				matched = true
				value = strings.TrimPrefix(arg, name+"=")
				break
			}
		}
		if matched {
			continue
		}
	}
	return value, found
}

func buildTUIAgentReport(opt tuiAgentOptions, at time.Time, fakPath string, getenv func(string) string) (tuiAgentReport, error) {
	backend := strings.ToLower(strings.TrimSpace(opt.Backend))
	if backend == "" {
		backend = "claude"
	}
	if backend != "claude" {
		return tuiAgentReport{}, fmt.Errorf("unknown --backend %q (want claude)", opt.Backend)
	}
	commandName := strings.TrimSpace(opt.Command)
	if commandName == "" {
		return tuiAgentReport{}, fmt.Errorf("--command must not be empty")
	}
	sessionID := strings.TrimSpace(opt.SessionID)
	if sessionID == "" {
		sessionID = "tui-agent"
	}
	if strings.TrimSpace(opt.Account) != "" && strings.TrimSpace(opt.ClaudeConfigDir) != "" {
		return tuiAgentReport{}, fmt.Errorf("--account and --claude-config-dir are mutually exclusive")
	}
	if fakPath == "" {
		fakPath = "fak"
	}
	if rawModel, ok := claudeArgValue(opt.CommandArgs, "--model", "-m"); ok {
		// A raw Claude model is the final operator choice. Propagate it back through
		// the guard/provider layer too, otherwise a saved guard --model could silently
		// override the Claude argv on local launches or stale gateway model env vars.
		opt.Model = strings.TrimSpace(rawModel)
	}

	// Every spawned account session defaults to Claude's --permission-mode
	// bypassPermissions: the launch is already wrapped by `fak guard` (or pinned
	// at a fak serve gateway), so the reference monitor — not Claude's own
	// interactive permission prompt — mediates tool calls. Forcing it here means
	// ALL accounts spawn non-interactively-gated by default. An operator override
	// in the trailing `claude args` wins (we don't duplicate the flag), and
	// --permission-mode "" opts out entirely.
	permissionMode := strings.TrimSpace(opt.PermissionMode)
	if permissionMode != "" && claudeArgsHavePermissionMode(opt.CommandArgs) {
		permissionMode = "" // operator already set it in the passthrough args
	}
	command := []string{commandName}
	if permissionMode != "" {
		command = append(command, "--permission-mode", permissionMode)
	}
	effort := strings.ToLower(strings.TrimSpace(opt.Effort))
	if effort != "" && !validTUIAgentEffort(effort) {
		return tuiAgentReport{}, fmt.Errorf("invalid --effort %q (want %s)", opt.Effort, strings.Join(tuiAgentEffortOptions, "|"))
	}
	if rawEffort, ok := claudeArgValue(opt.CommandArgs, "--effort"); ok {
		// The trailing raw Claude option wins without duplication and remains free to
		// use values introduced after this console build's finite picker.
		effort = strings.TrimSpace(rawEffort)
	} else if effort != "" {
		command = append(command, "--effort", effort)
	}
	command = append(command, opt.CommandArgs...)
	if strings.TrimSpace(opt.Prompt) != "" {
		command = append(command, "-p", opt.Prompt)
	}

	env, cfgDir, cfgSource, resolvedAccount, identity, notes, err := resolveTUIAgentClaudeConfig(opt, getenv)
	if err != nil {
		return tuiAgentReport{}, err
	}
	if permissionMode != "" {
		notes = append(notes, fmt.Sprintf("permission-mode=%s: every spawned account session is launched with this Claude --permission-mode by default, so the guarded backend mediates tools instead of Claude's interactive prompt (override in the trailing claude args, or pass --permission-mode \"\" to omit)", permissionMode))
	}
	if strings.TrimSpace(opt.GatewayURL) != "" {
		return buildTUIAgentGatewayReport(opt, at, backend, command, permissionMode, effort, env, cfgDir, cfgSource, resolvedAccount, identity, notes, getenv)
	}
	guardArgs := []string{"guard", "--provider", "anthropic", "--session-id", sessionID}
	auth := "claude-subscription-oauth"
	if opt.Passthrough {
		auth = "passthrough"
		notes = append(notes, "Claude Code forwards its own credential through the gateway")
	} else {
		guardArgs = append(guardArgs, "--anthropic-oauth")
		notes = append(notes, "guard forces the Claude Pro/Max subscription OAuth path and fails loud if no token is available")
	}
	if strings.TrimSpace(opt.Policy) != "" {
		guardArgs = append(guardArgs, "--policy", strings.TrimSpace(opt.Policy))
	}
	if strings.TrimSpace(opt.Model) != "" {
		guardArgs = append(guardArgs, "--model", strings.TrimSpace(opt.Model))
	}
	if opt.ContextBudgetTokens > 0 {
		guardArgs = append(guardArgs, "--context-budget-tokens", strconv.Itoa(opt.ContextBudgetTokens))
	}
	if opt.RestartOnBudget {
		guardArgs = append(guardArgs, "--restart-on-budget")
	}
	if opt.RestartLimit > 0 {
		guardArgs = append(guardArgs, "--restart-limit", strconv.Itoa(opt.RestartLimit))
	}
	// Token-saving defaults: compact-history-budget and elide-result-bytes are already
	// default-on in guard, but we pass them explicitly so they appear in dry-run output
	// and the operator can see the active savings without reading guard's defaults.
	guardArgs = append(guardArgs,
		"--compact-history-budget", strconv.Itoa(gateway.DefaultCompactHistoryBudget),
		"--elide-result-bytes", strconv.Itoa(gateway.DefaultElideResultBytes),
	)
	notes = append(notes,
		fmt.Sprintf("compact-history-budget=%d: guard sheds un-cached middle turns once resident tokens exceed this threshold, preserving the upstream cache prefix", gateway.DefaultCompactHistoryBudget),
		fmt.Sprintf("elide-result-bytes=%d: guard shrinks oversized tool results outside the active working set to a bounded head+tail form", gateway.DefaultElideResultBytes),
	)
	if opt.DebugStats {
		guardArgs = append(guardArgs, "--debug-stats")
		notes = append(notes, "debug-stats=on: one compact per-turn line to stderr leading with a verdict (ok/warming/degraded/cold) + the NET write-premium-aware token saving, then cache health + compaction")
	}
	guardArgs = append(guardArgs, "--")
	launch := append([]string{fakPath}, guardArgs...)
	launch = append(launch, command...)

	return tuiAgentReport{
		Schema:              tuiAgentSchema,
		At:                  at.UTC().Format(time.RFC3339),
		Backend:             backend,
		Mode:                "launch",
		Provider:            "anthropic",
		Auth:                auth,
		Account:             strings.TrimSpace(opt.Account),
		ResolvedAccount:     resolvedAccount,
		AccountIdentity:     identity,
		ClaudeConfigDir:     cfgDir,
		ConfigSource:        cfgSource,
		SessionID:           sessionID,
		PermissionMode:      permissionMode,
		Policy:              strings.TrimSpace(opt.Policy),
		Model:               strings.TrimSpace(opt.Model),
		Effort:              effort,
		ContextBudget:       opt.ContextBudgetTokens,
		RestartOnBudget:     opt.RestartOnBudget,
		RestartLimit:        opt.RestartLimit,
		DebugStats:          opt.DebugStats,
		CompactHistoryLimit: gateway.DefaultCompactHistoryBudget,
		ElideResultBytes:    gateway.DefaultElideResultBytes,
		Command:             command,
		Launch:              launch,
		Env:                 env,
		Notes:               notes,
	}, nil
}

func buildTUIAgentGatewayReport(opt tuiAgentOptions, at time.Time, backend string, command []string, permissionMode, effort string, env []tuiAgentEnv, cfgDir, cfgSource, resolvedAccount, identity string, notes []string, getenv func(string) string) (tuiAgentReport, error) {
	if strings.TrimSpace(opt.Policy) != "" || opt.ContextBudgetTokens > 0 || opt.RestartOnBudget || opt.RestartLimit > 0 || opt.Passthrough {
		return tuiAgentReport{}, fmt.Errorf("--gateway-url launches an existing gateway; guard-only options (--policy, --context-budget-tokens, --restart-on-budget, --restart-limit, --passthrough) do not apply")
	}
	gatewayURL, err := normalizeTUIAgentGatewayURL(opt.GatewayURL)
	if err != nil {
		return tuiAgentReport{}, err
	}
	keyEnv := strings.TrimSpace(opt.GatewayKeyEnv)
	if keyEnv == "" {
		keyEnv = "FAK_GATEWAY_KEY"
	}
	bearer := strings.TrimSpace(getenv(keyEnv))
	// A loopback gateway is a local `fak serve` that, unless started with
	// --require-key-env, accepts unauthenticated requests — so tolerate an empty bearer
	// for it (mirrors claude_mac_fak's gatewayIsLocal tolerance), which is what makes
	// `fak c local` launch without demanding a bogus key. A REMOTE gateway still requires
	// the bearer to be set.
	localGateway := gatewayIsLocal(gatewayURL)
	if bearer == "" && !localGateway {
		return tuiAgentReport{}, fmt.Errorf("--gateway-url requires %s to be set (or pass --gateway-key-env VAR)", keyEnv)
	}
	notes = filterTUIAgentGatewayNotes(notes)
	env = append(env, tuiAgentEnv{Name: "ANTHROPIC_BASE_URL", Value: gatewayURL, Source: "gateway-url"})
	auth := "gateway-bearer"
	if bearer != "" {
		env = append(env, tuiAgentEnv{Name: "ANTHROPIC_API_KEY", Source: "env:" + keyEnv, FromEnv: keyEnv, Sensitive: true})
	} else {
		// loopback, no bearer: do not inject an empty ANTHROPIC_API_KEY (Claude Code
		// reads an empty value as "no key"); record the unauthenticated posture instead.
		auth = "none"
		notes = append(notes, fmt.Sprintf("loopback gateway %s with no %s set — launching unauthenticated (a local fak serve without --require-key-env)", gatewayURL, keyEnv))
	}
	if strings.TrimSpace(getenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")) == "" {
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", Value: "1", Source: "gateway-default"})
	}
	model := strings.TrimSpace(opt.Model)
	if model != "" {
		for _, name := range []string{
			"ANTHROPIC_MODEL",
			"ANTHROPIC_DEFAULT_OPUS_MODEL",
			"ANTHROPIC_DEFAULT_SONNET_MODEL",
			"ANTHROPIC_DEFAULT_HAIKU_MODEL",
			"ANTHROPIC_SMALL_FAST_MODEL",
		} {
			env = append(env, tuiAgentEnv{Name: name, Value: model, Source: "model"})
		}
	}
	if opt.APITimeoutMS > 0 && strings.TrimSpace(getenv("API_TIMEOUT_MS")) == "" {
		env = append(env, tuiAgentEnv{Name: "API_TIMEOUT_MS", Value: strconv.Itoa(opt.APITimeoutMS), Source: "gateway-default"})
	}
	notes = append(notes,
		"launches the agent directly against an existing fak serve gateway; no local fak guard is started",
		fmt.Sprintf("gateway bearer is read from %s at launch and is not printed in dry-run output", keyEnv),
	)
	sessionID := strings.TrimSpace(opt.SessionID)
	if sessionID == "" {
		sessionID = "tui-agent"
	}
	return tuiAgentReport{
		Schema:          tuiAgentSchema,
		At:              at.UTC().Format(time.RFC3339),
		Backend:         backend,
		Mode:            "launch",
		Provider:        "existing-fak-gateway",
		Auth:            auth,
		GatewayURL:      gatewayURL,
		Account:         strings.TrimSpace(opt.Account),
		ResolvedAccount: resolvedAccount,
		AccountIdentity: identity,
		ClaudeConfigDir: cfgDir,
		ConfigSource:    cfgSource,
		SessionID:       sessionID,
		PermissionMode:  permissionMode,
		Model:           model,
		Effort:          effort,
		Command:         command,
		Launch:          command,
		Env:             env,
		Notes:           notes,
	}, nil
}

func filterTUIAgentGatewayNotes(notes []string) []string {
	if len(notes) == 0 {
		return notes
	}
	out := notes[:0]
	for _, note := range notes {
		if strings.Contains(note, "Claude may prompt for login") {
			continue
		}
		out = append(out, note)
	}
	return out
}

func normalizeTUIAgentGatewayURL(raw string) (string, error) {
	u := strings.TrimSpace(raw)
	if u == "" {
		return "", fmt.Errorf("--gateway-url must not be empty")
	}
	u = strings.TrimRight(u, "/")
	if strings.HasSuffix(u, "/v1") {
		u = strings.TrimSuffix(u, "/v1")
	}
	if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
		return "", fmt.Errorf("--gateway-url must start with http:// or https://")
	}
	return u, nil
}

func resolveTUIAgentClaudeConfig(opt tuiAgentOptions, getenv func(string) string) ([]tuiAgentEnv, string, string, string, string, []string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	var env []tuiAgentEnv
	var notes []string
	account := strings.TrimSpace(opt.Account)
	if account != "" {
		reg, err := loadOrDiscover(opt.Registry, opt.Home)
		if err != nil {
			return nil, "", "", "", "", nil, err
		}
		reg = reg.Refresh()
		home, chain, err := reg.Serve(account)
		if err != nil {
			return nil, "", "", "", "", nil, err
		}
		for i, hop := range chain {
			to := home.Name
			if i+1 < len(chain) {
				to = chain[i+1]
			}
			notes = append(notes, fmt.Sprintf("%q can't serve; rehomed to %q", hop, to))
		}
		if note := tuiLoginNote(home); note != "" {
			notes = append(notes, note)
		}
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CONFIG_DIR", Value: home.Dir, Source: "account:" + home.Name})
		return env, home.Dir, "account:" + home.Name, home.Name, home.Identity.Email, notes, nil
	}
	if dir := strings.TrimSpace(opt.ClaudeConfigDir); dir != "" {
		env = append(env, tuiAgentEnv{Name: "CLAUDE_CONFIG_DIR", Value: dir, Source: "flag"})
		return resolveTUIConfigDir(env, dir, "flag", notes)
	}
	if dir := strings.TrimSpace(getenv("CLAUDE_CONFIG_DIR")); dir != "" {
		return resolveTUIConfigDir(nil, dir, "inherited-env", notes)
	}
	return resolveTUIConfigDir(nil, guardClaudeConfigDir(), "default", notes)
}

// resolveTUIConfigDir resolves a bare Claude config directory (no registry
// account) into the resolveTUIAgentClaudeConfig return fields: derive its
// identity, append its login note, and label the source. env is nil for the
// inherited/default paths, which rely on the ambient CLAUDE_CONFIG_DIR.
func resolveTUIConfigDir(env []tuiAgentEnv, dir, source string, notes []string) ([]tuiAgentEnv, string, string, string, string, []string, error) {
	id := acct.DeriveIdentity(dir)
	if note := tuiLoginNote(acct.Home{Dir: dir, Identity: id}); note != "" {
		notes = append(notes, note)
	}
	return env, dir, source, "", id.Email, notes, nil
}

func tuiLoginNote(home acct.Home) string {
	status := home.LoginStatus()
	if status == acct.LoginReady {
		return ""
	}
	reason, action := acct.LoginReasonAction(status, home)
	subject := home.Dir
	if home.Name != "" {
		subject = fmt.Sprintf("%q (%s)", home.Name, home.Dir)
	}
	if action != "" {
		return fmt.Sprintf("%s login=%s - %s; %s; Claude may prompt for login", subject, status, reason, action)
	}
	return fmt.Sprintf("%s login=%s - %s; Claude may prompt for login", subject, status, reason)
}

func tuiExecutable() string {
	if exe, err := os.Executable(); err == nil && strings.TrimSpace(exe) != "" {
		return exe
	}
	if len(os.Args) > 0 && strings.TrimSpace(os.Args[0]) != "" {
		return os.Args[0]
	}
	return "fak"
}

func tuiAgentPromptTransport(launch []string, goos string) ([]string, string, bool) {
	return guardPromptStdinTransportForOS(launch, goos)
}

func launchTUIAgent(stdout, stderr io.Writer, report tuiAgentReport) int {
	if len(report.Launch) == 0 {
		fmt.Fprintln(stderr, "fak console agent: empty launch command")
		return 1
	}
	launch, promptStdin, moved := tuiAgentPromptTransport(report.Launch, runtime.GOOS)
	child := exec.Command(launch[0], launch[1:]...)
	child.Stdin = os.Stdin
	if moved {
		child.Stdin = strings.NewReader(promptStdin)
	}
	child.Stdout = stdout
	child.Stderr = stderr
	child.Env = mergeTUIAgentEnv(os.Environ(), report.Env)
	if err := child.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak console agent: launch %q: %v\n", report.Launch[0], err)
		return 1
	}
	return 0
}

func mergeTUIAgentEnv(base []string, pairs []tuiAgentEnv) []string {
	out := append([]string{}, base...)
	for _, pair := range pairs {
		name := strings.TrimSpace(pair.Name)
		if name == "" {
			continue
		}
		value := pair.Value
		if strings.TrimSpace(pair.FromEnv) != "" {
			value = os.Getenv(strings.TrimSpace(pair.FromEnv))
		}
		line := name + "=" + value
		replaced := false
		for i, cur := range out {
			k, _, ok := strings.Cut(cur, "=")
			if ok && strings.EqualFold(k, name) {
				out[i] = line
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, line)
		}
	}
	return out
}

func loadTUISessions(path, addr, key string) (gateway.SessionListResponse, string, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return gateway.SessionListResponse{}, "", err
		}
		list, err := decodeTUISessions(b)
		return list, path, err
	}
	c := &sessionClient{
		base: strings.TrimRight(addr, "/"),
		key:  key,
		hc:   &http.Client{Timeout: 15 * time.Second},
	}
	list, err := c.list()
	if err != nil {
		return gateway.SessionListResponse{}, "", err
	}
	return list, c.base + "/v1/fak/sessions", nil
}

func decodeTUISessions(b []byte) (gateway.SessionListResponse, error) {
	var list gateway.SessionListResponse
	if err := json.Unmarshal(b, &list); err == nil && (list.Sessions != nil || list.Count != 0) {
		if list.Count == 0 {
			list.Count = len(list.Sessions)
		}
		return list, nil
	}
	var sessions []gateway.SessionState
	if err := json.Unmarshal(b, &sessions); err != nil {
		return gateway.SessionListResponse{}, fmt.Errorf("session JSON must be a SessionListResponse or SessionState array: %w", err)
	}
	return gateway.SessionListResponse{Sessions: sessions, Count: len(sessions)}, nil
}
