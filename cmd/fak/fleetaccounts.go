package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// `fak fleet-accounts` — the native Go port of the READ-ONLY roster/resolve/probe +
// status fold from tools/fleet_accounts.py. It answers "what is an account, and is it
// offered right now?" across Claude Code, Codex, and opencode by:
//   - DISCOVERING config dirs (<home>/.claude*, <home>/.codex*, <config_home>/opencode*),
//   - classifying each by the operator POLICY (accounts_policy.json) into
//     worker | excluded | non-account,
//   - reconciling Claude/Codex dirs that share one provider account (duplicate collapse),
//   - folding live runtime status (usage throttle / auth block / live sessions) from the
//     watchdog's sessions.json registry.
//
// It reuses the SAME single sources of truth the Python tool does — the same env-override
// path resolution, the same policy file, the same sessions.json — so it is a drop-in
// read surface, never a second account contract. The --json shapes are byte-compatible
// with the Python `json`/`seats` output.
//
// Subcommands:
//
//	fak fleet-accounts roster|list [--json]   the full classified roster + live status
//	fak fleet-accounts json                   alias for `roster --json` (Python `json` envelope)
//	fak fleet-accounts available              the account dirs safe to offer now (one per line)
//	fak fleet-accounts resolve [--account P] [--work-kind K] [--product P] [--t1|--t2|--t3]
//	                                          ONE flat record: config_dir + oauth_token + tier
//	fak fleet-accounts wave [--count N] [--work-kind K] [--product P] [--t1|--t2|--t3]
//	                                          allocate account session slots for fan-out
//	fak fleet-accounts seats [--product P] [--json]   the explicit seat pool (M distinct seats)
//	fak fleet-accounts status [--provider P] [--tier N|--t1|--t2|--t3] [--group-by provider,tier]
//	                                          compact rollups over provider/tier/model/state + mixed-limit warnings;
//	                                          repeat --snapshot FILE for all-node rollups
//
// NOT yet ported (documented follow-on, see issue #1415): the ACTIVE network probe
// (`probe`, which delegates to tools/account_probe.py), the probe-LEDGER freshness
// override inside runtime status, and the mutating ops (relogin / top-up / launch). The
// passive registry fold + roster/resolve/seats are the operator hot path and are fully
// ported here with tests; `probe` and the mutators remain on the Python shim. The Go
// roster keeps the legacy Python field order for the shared keys and adds the
// credential-safe login_status/can_serve fields for Claude switcher rows.
func cmdFleetAccounts(argv []string) { os.Exit(runFleetAccounts(os.Stdout, os.Stderr, argv)) }

func runFleetAccounts(stdout, stderr io.Writer, argv []string) int {
	mode := "list"
	rest := argv
	if len(argv) > 0 && len(argv[0]) > 0 && argv[0][0] != '-' {
		mode, rest = argv[0], argv[1:]
	}

	fs := flag.NewFlagSet("fleet-accounts "+mode, flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON instead of a human table")
	account := fs.String("account", "", "(resolve) pin to this account tag/basename; (status) filter tag/basename substring")
	workKind := fs.String("work-kind", "", "(resolve) operator work-kind (gardening|engineering|...) — wins over tier")
	task := fs.String("task", "", "(resolve) task text for the light/hard heuristic")
	product := fs.String("product", "", "(resolve/seats/status) scope to a product family (claude|codex|opencode)")
	provider := fs.String("provider", "", "(status) scope to a derived provider family (anthropic|groq|nvidia-nim|google|...)")
	tier := fs.Int("tier", 0, "(status) filter by model tier 1|2|3")
	state := fs.String("state", "", "(status) filter by state (ready|usage|auth|blocked|duplicate|excluded|non-account)")
	modelFilter := fs.String("model", "", "(status) filter by model substring")
	groupBy := fs.String("group-by", "provider,tier", "(status) comma-separated rollup dimensions: provider,product,tier,model,state,agent")
	showAccounts := fs.Bool("accounts", false, "(status) include matching per-account rows after rollups")
	nodeLabel := fs.String("node", "local", "(status) node label stamped into local JSON snapshots; never inferred from private hostnames")
	freshWithin := fs.String("fresh-within", "30m", "(status --snapshot) freshness window for node snapshots")
	includeStale := fs.Bool("include-stale", false, "(status --snapshot) include stale snapshots in free-now totals")
	var statusSnapshots fleetStatusSnapshotFlags
	fs.Var(&statusSnapshots, "snapshot", "(status) read a fleet-account-status/1 JSON snapshot; repeat for all-node rollups")
	t1 := fs.Bool("t1", false, "(resolve/status) pin/filter tier 1")
	t2 := fs.Bool("t2", false, "(resolve/status) pin/filter tier 2")
	t3 := fs.Bool("t3", false, "(resolve/status) pin/filter tier 3")
	allowFallback := fs.Bool("allow-tier-fallback", false, "(resolve) allow a tier-1 target to fall back to tier 2")
	faklocalOK := fs.Bool("faklocal-ok", false, "(resolve) synthesize the dogfood .claude-faklocal account when pinned")
	count := -1
	fs.IntVar(&count, "count", -1, "(wave) number of account session slots to allocate")
	fs.IntVar(&count, "n", -1, "(wave) shorthand for --count")
	explain := fs.Bool("explain", false, "(wave) emit the headroom witness projection")
	waveID := fs.String("wave-id", "", "(wave) override the deterministic wave id")
	taskTier := fs.Int("task-tier", 0, "(launch/exec) required task tier 1|2|3")
	invokedModel := fs.String("invoked-model", "", "(launch/exec) model passed to the worker; defaults to account model")
	speed := fs.String("speed", firstNonEmpty(strings.TrimSpace(os.Getenv("FAK_CLAUDE_SPEED")), "auto"), "(launch/exec) Claude speed posture auto|fast|standard; non-Claude ignored")
	prompt := fs.String("prompt", "", "(launch/exec) worker prompt")
	tier3Override := fs.Bool("allow-tier3-narrow", false, "(launch/exec) explicitly authorize a restricted tier-3 seat for narrow tier-3 work")
	launchLedger := fs.String("launch-ledger", ".fak/fleet-launches.jsonl", "(launch/exec) non-secret launch ledger path")
	if err := fs.Parse(rest); err != nil {
		return 2
	}

	cwd, _ := os.Getwd()
	repoRoot := findRepoRoot(cwd)
	toolsDir := filepath.Join(repoRoot, "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	rows := fleetaccounts.AnnotateWithProbes(fleetaccounts.Discover(paths.Home, paths.ConfigHome, pol), reg, paths.RegistryDir)

	switch mode {
	case "list", "roster":
		if *asJSON {
			return emitRosterJSON(stdout, stderr, paths, rows)
		}
		exampleNote := ""
		if !faFileExists(paths.PolicyPath) && faFileExists(paths.ExamplePath) {
			exampleNote = paths.ExamplePath + " (example; copy to _registry/ to customize)"
		}
		fmt.Fprint(stdout, fleetaccounts.RenderList(rows, paths.Home, paths.PolicyPath,
			faFileExists(paths.PolicyPath), exampleNote))
		return 0

	case "status":
		return runFleetAccountsStatus(stdout, stderr, fleetAccountsStatusRequest{
			paths: paths, rows: rows, repoRoot: repoRoot, product: product, provider: provider,
			tier: tier, state: state, account: account, modelFilter: modelFilter,
			t1: t1, t2: t2, t3: t3, snapshots: statusSnapshots, freshWithin: freshWithin,
			groupBy: groupBy, includeStale: includeStale, asJSON: asJSON,
			showAccounts: showAccounts, nodeLabel: nodeLabel,
		})

	case "json":
		return emitRosterJSON(stdout, stderr, paths, rows)

	case "available":
		for _, r := range fleetaccounts.Available(rows) {
			fmt.Fprintln(stdout, r.Account)
		}
		return 0

	case "launch", "exec":
		if *taskTier == 0 {
			fmt.Fprintln(stderr, "fleet-accounts launch: --task-tier is required")
			return 2
		}
		req := fleetaccounts.ResolveRequest{Pin: *account, WorkKind: *workKind, TaskText: *task, Product: *product, TaskClass: fmt.Sprintf("tier%d", *taskTier), StrictTier: true, AllowTierFallback: *allowFallback, FaklocalOK: *faklocalOK}
		resolved := fleetaccounts.Resolve(rows, paths.Home, req, pol)
		decision := fleetaccounts.DecideLaunch(fleetaccounts.LaunchRequest{Account: resolved, TaskTier: *taskTier, InvokedModel: *invokedModel, Prompt: *prompt, Tier3Override: *tier3Override, Speed: *speed})
		if err := appendFleetLaunchLedger(*launchLedger, decision); err != nil {
			fmt.Fprintf(stderr, "fleet-accounts launch ledger: %v\n", err)
			return 1
		}
		out, _ := json.MarshalIndent(decision, "", "  ")
		fmt.Fprintln(stdout, string(out))
		if !decision.OK {
			return 3
		}
		if mode == "launch" {
			return 0
		}
		outcome := executeFleetLaunchOutcome(decision, os.Stdin, stdout, stderr, os.Environ())
		if err := appendFleetTurnOutcome(*launchLedger, decision, outcome); err != nil {
			fmt.Fprintf(stderr, "fleet-accounts turn ledger: %v\n", err)
			return 1
		}
		return outcome.Code
	case "resolve":
		taskClass := fleetAccountsTaskClass(*t1, *t2, *t3)
		strict := *t1 || *t2 || *t3
		req := fleetaccounts.ResolveRequest{
			Pin: *account, TaskText: *task, TaskClass: taskClass, WorkKind: *workKind,
			Product: *product, AllowTierFallback: *allowFallback, StrictTier: strict,
			FaklocalOK: *faklocalOK,
		}
		resolved := fleetaccounts.Resolve(rows, paths.Home, req, pol)
		out, err := json.MarshalIndent(resolved, "", " ")
		if err != nil {
			fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
			return 1
		}
		fmt.Fprintln(stdout, string(out))
		if resolved.OK {
			return 0
		}
		return 1

	case "wave":
		explicitCount := count >= 0
		waveCount := count
		if !explicitCount {
			waveCount = 1_000_000
		}
		taskClass := fleetAccountsTaskClass(*t1, *t2, *t3)
		strict := *t1 || *t2 || *t3
		wave := fleetaccounts.AllocateWave(rows, fleetaccounts.WaveRequest{
			Count: waveCount, TaskText: *task, TaskClass: taskClass, WorkKind: *workKind,
			Product: *product, AllowTierFallback: *allowFallback, StrictTier: strict,
			Leases: fleetSeatLeases(repoRoot), WaveID: *waveID,
		}, pol)
		if !explicitCount {
			wave.Requested = wave.Granted
			wave.Shortfall = 0
			wave.Reason = fmt.Sprintf("all %d available session slot(s)", wave.Granted)
		}
		if *explain {
			return emitWaveExplainJSON(stdout, stderr, wave)
		}
		out, err := json.MarshalIndent(scrubFleetAccountWaveSecrets(wave), "", " ")
		if err != nil {
			fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
			return 1
		}
		fmt.Fprintln(stdout, string(out))
		if wave.OK {
			return 0
		}
		return 1

	case "seats":
		pool := fleetaccounts.BuildSeatPool(rows, fleetSeatLeases(repoRoot), *product)
		if *asJSON {
			out, err := json.MarshalIndent(pool, "", " ")
			if err != nil {
				fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
				return 1
			}
			fmt.Fprintln(stdout, string(out))
		} else {
			fmt.Fprint(stdout, fleetaccounts.RenderSeats(pool))
		}
		return 0

	default:
		fmt.Fprintln(stderr, "usage: fak fleet-accounts <roster|list|json|available|resolve|launch|exec|wave|seats|status> [flags]")
		fmt.Fprintln(stderr, "note: the active network probe + mutating ops (relogin/top-up/launch) remain on tools/fleet_accounts.py (issue #1415).")
		return 2
	}
}

type fleetStatusSnapshotFlags []string

func (f *fleetStatusSnapshotFlags) String() string { return strings.Join(*f, ",") }
func (f *fleetStatusSnapshotFlags) Set(v string) error {
	v = strings.TrimSpace(v)
	if v == "" {
		return fmt.Errorf("empty --snapshot")
	}
	*f = append(*f, v)
	return nil
}

func loadFleetAccountStatusSnapshots(paths []string) ([]fleetaccounts.StatusSnapshot, error) {
	out := make([]fleetaccounts.StatusSnapshot, 0, len(paths))
	for i, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read snapshot %s: %w", path, err)
		}
		var rep fleetaccounts.StatusReport
		if err := json.Unmarshal(data, &rep); err != nil {
			return nil, fmt.Errorf("decode snapshot %s: %w", path, err)
		}
		if rep.Schema != fleetaccounts.StatusReportSchema {
			return nil, fmt.Errorf("snapshot %s has schema %q, want %s", path, rep.Schema, fleetaccounts.StatusReportSchema)
		}
		node := strings.TrimSpace(rep.Node)
		if node == "" {
			node = fmt.Sprintf("node-%d", i+1)
		}
		out = append(out, fleetaccounts.StatusSnapshot{Node: node, Path: path, Report: rep})
	}
	return out, nil
}

func fleetAccountsSplitCSV(raw string) []string {
	var out []string
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func statusTierFilter(tier int, t1, t2, t3 bool) int {
	if tier > 0 {
		return tier
	}
	switch {
	case t1:
		return 1
	case t2:
		return 2
	case t3:
		return 3
	default:
		return 0
	}
}

func statusFilterRequested(f fleetaccounts.StatusFilter) bool {
	return strings.TrimSpace(f.Product) != "" ||
		strings.TrimSpace(f.Provider) != "" ||
		f.Tier > 0 ||
		strings.TrimSpace(f.State) != "" ||
		strings.TrimSpace(f.Account) != "" ||
		strings.TrimSpace(f.Model) != ""
}

func fleetSeatLeases(repoRoot string) []fleetaccounts.Lease {
	raw := dispatchLiveSeatLeases(filepath.Join(repoRoot, dispatchtick.RunsDirName))
	out := make([]fleetaccounts.Lease, 0, len(raw))
	for _, lease := range raw {
		row := fleetaccounts.Lease{Worker: lease.Worker, Tag: lease.Tag, Dir: lease.Dir}
		if lease.PID > 0 {
			pid := lease.PID
			row.PID = &pid
		}
		out = append(out, row)
	}
	return out
}

func scrubFleetAccountWaveSecrets(wave fleetaccounts.WaveResult) any {
	raw, err := json.Marshal(wave)
	if err != nil {
		return wave
	}
	var doc any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return wave
	}
	return scrubDispatchSecrets(doc)
}

func fleetAccountsTaskClass(t1, t2, t3 bool) string {
	switch {
	case t1:
		return "t1"
	case t2:
		return "t2"
	case t3:
		return "t3"
	default:
		return "auto"
	}
}

func emitWaveExplainJSON(stdout, stderr io.Writer, wave fleetaccounts.WaveResult) int {
	tags := make([]string, 0, len(wave.Lanes))
	pools := make([]string, 0, len(wave.Lanes))
	for _, lane := range wave.Lanes {
		tags = append(tags, lane.Tag)
		pools = append(pools, lane.Pool)
	}
	doc := map[string]any{
		"ok":                  wave.OK,
		"requested":           wave.Requested,
		"granted":             wave.Granted,
		"shortfall":           wave.Shortfall,
		"distinct_pools":      wave.DistinctPools,
		"target_tier":         wave.TargetTier,
		"naive_pools":         0,
		"headroom_multiplier": wave.DistinctPools,
		"reason":              wave.Reason,
		"lane_tags":           tags,
		"lane_pools":          pools,
	}
	if wave.Granted > 0 {
		doc["naive_pools"] = 1
	}
	out, err := json.MarshalIndent(doc, "", " ")
	if err != nil {
		fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	if wave.OK {
		return 0
	}
	return 1
}

// faFileExists reports whether a path exists (used for the policy/registry provenance
// flags in the JSON envelope + list footer).
func faFileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

// emitRosterJSON renders the `json` envelope (the Python json.dumps(indent=1) shape).
func emitRosterJSON(stdout, stderr io.Writer, paths fleetaccounts.Paths,
	rows []fleetaccounts.Account) int {
	env := fleetaccounts.BuildJSONEnvelope(paths.Home, paths.PolicyPath,
		faFileExists(paths.PolicyPath), paths.RegistryPath, faFileExists(paths.RegistryPath), rows)
	out, err := env.MarshalIndent()
	if err != nil {
		fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	return 0
}

type fleetTurnOutcome struct {
	Code   int
	OK     bool
	Reason string
}

type fleetTurnLedgerEntry struct {
	TS            string `json:"ts"`
	Phase         string `json:"phase"`
	Account       string `json:"account,omitempty"`
	Product       string `json:"product,omitempty"`
	LaunchOK      bool   `json:"launch_ok"`
	CompletedTurn bool   `json:"completed_turn_ok"`
	ExitCode      int    `json:"exit_code"`
	OutcomeReason string `json:"outcome_reason,omitempty"`
}

func appendFleetLaunchLedger(path string, decision fleetaccounts.LaunchDecision) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	record := struct {
		At               string `json:"at"`
		Account          string `json:"resolved_account,omitempty"`
		Product          string `json:"product,omitempty"`
		ConfiguredModel  string `json:"configured_model,omitempty"`
		InvokedModel     string `json:"invoked_model,omitempty"`
		EndpointClass    string `json:"endpoint_class,omitempty"`
		TaskTier         int    `json:"task_tier,omitempty"`
		OK               bool   `json:"ok"`
		Reason           string `json:"reason,omitempty"`
		OperatorOverride bool   `json:"operator_override,omitempty"`
	}{time.Now().UTC().Format(time.RFC3339Nano), decision.Account, decision.Product, decision.ConfiguredModel, decision.InvokedModel, decision.EndpointClass, decision.TaskTier, decision.OK, decision.Reason, decision.OperatorOverride}
	return json.NewEncoder(f).Encode(record)
}

var fleetLaunchExecCommand = exec.Command

func executeFleetLaunch(d fleetaccounts.LaunchDecision, stdin io.Reader, stdout, stderr io.Writer, environ []string) int {
	return executeFleetLaunchOutcome(d, stdin, stdout, stderr, environ).Code
}

func executeFleetLaunchOutcome(d fleetaccounts.LaunchDecision, stdin io.Reader, stdout, stderr io.Writer, environ []string) fleetTurnOutcome {
	if len(d.Argv) == 0 {
		fmt.Fprintln(stderr, "fleet-accounts exec: empty command")
		return fleetTurnOutcome{Code: 2, Reason: "EMPTY_COMMAND"}
	}
	cmd := fleetLaunchExecCommand(d.Argv[0], d.Argv[1:]...)
	cmd.Stdin, cmd.Stderr = stdin, stderr
	cmd.Env = overlayFleetLaunchEnv(environ, d.Env)

	var codexOutput bytes.Buffer
	if d.Product == "codex" {
		cmd.Stdout = io.MultiWriter(stdout, &codexOutput)
	} else {
		cmd.Stdout = stdout
	}
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return fleetTurnOutcome{Code: ee.ExitCode(), Reason: "PROCESS_EXIT"}
		}
		fmt.Fprintf(stderr, "fleet-accounts exec: %v\n", err)
		return fleetTurnOutcome{Code: 1, Reason: "PROCESS_START"}
	}
	if d.Product == "codex" && !codexAssistantCompleted(codexOutput.Bytes()) {
		reason := "ASSISTANT_RESPONSE_MISSING"
		kind := "access"
		reset := ""
		lower := bytes.ToLower(codexOutput.Bytes())
		if message, ok := codexUsageLimitFailure(codexOutput.Bytes()); ok {
			reason, kind = "PROVIDER_USAGE_LIMIT", "usage"
			reset = codexUsageLimitReset(message, time.Now())
		} else if bytes.Contains(lower, []byte("userpromptsubmit blocked")) {
			reason = "PROMPT_HOOK_BLOCK"
		} else if bytes.Contains(lower, []byte("401 unauthorized")) || bytes.Contains(lower, []byte("upstream rejected the credential")) {
			reason, kind = "UPSTREAM_AUTH", "auth"
		}
		recordFleetCodexProbe(d.Account, strings.ToUpper(kind), kind, reason, reset)
		fmt.Fprintf(stderr, "fleet-accounts exec: codex turn incomplete (%s): no completed assistant response\n", reason)
		return fleetTurnOutcome{Code: 70, Reason: reason}
	}
	if d.Product == "codex" {
		recordFleetCodexProbe(d.Account, "OK", "", "guarded assistant response completed", "")
	}
	return fleetTurnOutcome{OK: true}
}

func recordFleetCodexProbe(account, status, kind, reason, reset string) {
	if strings.TrimSpace(account) == "" {
		return
	}
	_ = accountprobe.AppendLedger("", accountprobe.LedgerEntry{
		TS:          time.Now().UTC().Format(time.RFC3339Nano),
		Account:     account,
		Status:      status,
		BlockReason: reason,
		Reset:       reset,
	})
}

func appendFleetTurnOutcome(path string, decision fleetaccounts.LaunchDecision, outcome fleetTurnOutcome) error {
	entry := fleetTurnLedgerEntry{
		TS: time.Now().UTC().Format(time.RFC3339Nano), Phase: "turn-outcome",
		Account: decision.Account, Product: decision.Product, LaunchOK: decision.OK,
		CompletedTurn: outcome.OK, ExitCode: outcome.Code, OutcomeReason: outcome.Reason,
	}
	return appendFleetLedgerJSON(path, entry)
}

func appendFleetLedgerJSON(path string, entry any) error {
	body, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = fmt.Fprintln(f, string(body))
	return err
}

type codexEvent struct {
	Type  string `json:"type"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

func codexUsageLimitFailure(output []byte) (string, bool) {
	for _, line := range bytes.Split(output, []byte{'\n'}) {
		var event codexEvent
		if json.Unmarshal(line, &event) == nil && event.Type == "turn.failed" && strings.Contains(strings.ToLower(event.Error.Message), "usage limit") {
			return event.Error.Message, true
		}
	}
	return "", false
}

var codexUsageResetPattern = regexp.MustCompile(`(?i)try again at\s+([A-Z][a-z]{2}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)`)

func codexUsageLimitReset(message string, now time.Time) string {
	m := codexUsageResetPattern.FindStringSubmatch(message)
	if len(m) != 2 {
		return ""
	}
	raw := regexp.MustCompile(`(?i)(\d{1,2})(st|nd|rd|th)`).ReplaceAllString(m[1], "$1")
	loc := now.Location()
	for _, layout := range []string{"Jan 2, 2006 3:04 PM", "Jan 2 2006 3:04 PM"} {
		if reset, err := time.ParseInLocation(layout, raw, loc); err == nil {
			return reset.Format(time.RFC3339)
		}
	}
	return ""
}

func overlayFleetLaunchEnv(base []string, overlay map[string]string) []string {
	out := append([]string(nil), base...)
	for key, value := range overlay {
		prefix := strings.ToUpper(key) + "="
		kept := out[:0]
		for _, entry := range out {
			if !strings.HasPrefix(strings.ToUpper(entry), prefix) {
				kept = append(kept, entry)
			}
		}
		out = append(kept, key+"="+value)
	}
	return out
}

func codexAssistantCompleted(output []byte) bool {
	for _, line := range bytes.Split(output, []byte{'\n'}) {
		var event struct {
			Type string `json:"type"`
			Item struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"item"`
		}
		if json.Unmarshal(bytes.TrimSpace(line), &event) == nil &&
			event.Type == "item.completed" && event.Item.Type == "agent_message" && strings.TrimSpace(event.Item.Text) != "" {
			return true
		}
	}
	return false
}
