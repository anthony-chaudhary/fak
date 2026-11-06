package main

// bench_h2h_bringup.go — the bring-up verbs of `fak bench h2h` for baseline
// runs on exclusive nodes: serve-wait polls every arm's OpenAI /models endpoint
// until it is ready (bounded by a deadline), plan renders the exact per-arm
// `run` argvs from the shared run-flag spec so parallel arm workers need no
// hand-substitution, and gpu-idle is an NVIDIA preflight that refuses to start
// until enough GPUs are idle.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/h2hbench"
)

// ---------------------------------------------------------------- serve-wait

const h2hServeWaitSchema = "fak.bench.h2h-serve-wait.v1"

type h2hArmRef struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

type h2hArmLastError struct {
	Name      string `json:"name"`
	URL       string `json:"url"`
	LastError string `json:"last_error"`
}

type h2hServeWaitEnvelope struct {
	Schema   string            `json:"schema"`
	Ready    []h2hArmRef       `json:"ready"`
	NotReady []h2hArmLastError `json:"not_ready"`
	Waited   string            `json:"waited"`
}

// runBenchH2HServeWait polls GET <armURL>/models on every --arm until each one
// answers 200 (and lists --model when set) or the deadline passes. Every error
// is retryable — connection refused included — until the deadline; the
// per-request timeout equals --interval so a hung server cannot eat the whole
// budget.
func runBenchH2HServeWait(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench h2h serve-wait", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var arms multiFlag
	fs.Var(&arms, "arm", "name=OpenAI-compatible /v1 base URL (repeatable)")
	model := fs.String("model", "", "when set, each arm must list this model id in GET <url>/models")
	timeout := fs.Duration("timeout", 30*time.Minute, "deadline before giving up")
	interval := fs.Duration("interval", 5*time.Second, "poll interval")
	jsonOut := fs.Bool("json", false, "emit the readiness envelope as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if len(arms) == 0 {
		fmt.Fprintln(stderr, "fak bench h2h serve-wait: at least one --arm is required")
		return 2
	}
	parsed := make([]h2hbench.Arm, 0, len(arms))
	for _, raw := range arms {
		arm, err := h2hbench.ParseArm(raw)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h serve-wait:", err)
			return 2
		}
		parsed = append(parsed, arm)
	}
	client := &http.Client{Timeout: *interval}
	type armState struct {
		arm   h2hbench.Arm
		err   string
		ready bool
	}
	states := make([]armState, len(parsed))
	for i, arm := range parsed {
		states[i] = armState{arm: arm}
	}
	start := time.Now()
	deadline := start.Add(*timeout)
	timedOut := false
	for {
		allReady := true
		for i := range states {
			if states[i].ready {
				continue
			}
			ok, err := probeH2HArmModels(client, states[i].arm, *model)
			if ok {
				states[i].ready = true
				states[i].err = ""
				continue
			}
			states[i].err = err.Error()
			allReady = false
		}
		if allReady {
			break
		}
		if time.Now().After(deadline) {
			timedOut = true
			break
		}
		sleep := *interval
		if remaining := time.Until(deadline); remaining < sleep {
			sleep = remaining
		}
		time.Sleep(sleep)
	}
	waited := time.Since(start)
	ready := make([]h2hArmRef, 0, len(states))
	notReady := make([]h2hArmLastError, 0, len(states))
	for _, s := range states {
		if s.ready {
			ready = append(ready, h2hArmRef{Name: s.arm.Name, URL: s.arm.BaseURL})
			continue
		}
		notReady = append(notReady, h2hArmLastError{Name: s.arm.Name, URL: s.arm.BaseURL, LastError: s.err})
	}
	if *jsonOut {
		env := h2hServeWaitEnvelope{Schema: h2hServeWaitSchema, Ready: ready, NotReady: notReady, Waited: waited.String()}
		_ = json.NewEncoder(stdout).Encode(env)
	} else if !timedOut {
		for _, r := range ready {
			fmt.Fprintf(stdout, "ready %s %s\n", r.Name, r.URL)
		}
	}
	if timedOut {
		parts := make([]string, 0, len(notReady))
		for _, n := range notReady {
			parts = append(parts, fmt.Sprintf("%s=%s: %s", n.Name, n.URL, n.LastError))
		}
		fmt.Fprintf(stderr, "serve-wait: timeout after %s waiting for %s\n", *timeout, strings.Join(parts, ", "))
		return 1
	}
	return 0
}

// probeH2HArmModels GETs <base>/models once and reports whether the arm answers
// 200 and, when wantModel is set, lists that model id. It owns no deadline: the
// caller retries until its own deadline passes.
func probeH2HArmModels(client *http.Client, arm h2hbench.Arm, wantModel string) (bool, error) {
	req, err := http.NewRequest(http.MethodGet, arm.BaseURL+"/models", nil)
	if err != nil {
		return false, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("HTTP %d from %s/models", resp.StatusCode, arm.BaseURL)
	}
	if wantModel == "" {
		return true, nil
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return false, fmt.Errorf("decode %s/models: %w", arm.BaseURL, err)
	}
	for _, m := range body.Data {
		if m.ID == wantModel {
			return true, nil
		}
	}
	return false, fmt.Errorf("model %q not listed by %s/models", wantModel, arm.BaseURL)
}

// ---------------------------------------------------------------- plan

const h2hPlanSchema = "fak.bench.h2h-plan.v1"

type h2hPlanArm struct {
	Name   string   `json:"name"`
	URL    string   `json:"url"`
	Ledger string   `json:"ledger"`
	Argv   []string `json:"argv"`
}

type h2hPlanEnvelope struct {
	Schema    string       `json:"schema"`
	Model     string       `json:"model"`
	LedgerDir string       `json:"ledger_dir"`
	Arms      []h2hPlanArm `json:"arms"`
}

// runBenchH2HPlan emits, for each --arm in the order given, the exact single-arm
// `fak bench h2h run` invocation with its own ledger under --ledger-dir. Flags
// left at their default values are omitted from the emitted argvs. Pure argv
// rendering: no network, no ledger writes.
func runBenchH2HPlan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench h2h plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f := registerH2HRunFlags(fs)
	ledger := fs.String("ledger", "", "combined-ledger path whose directory seeds --ledger-dir when --ledger-dir is unset (or \"off\")")
	ledgerDir := fs.String("ledger-dir", "", "directory for the per-arm ledgers (default: directory of --ledger when given, else .fak/nightrun)")
	jsonOut := fs.Bool("json", false, "emit the plan envelope as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *f.model == "" || len(f.arms) == 0 {
		fmt.Fprintln(stderr, "fak bench h2h plan: --model and at least one --arm are required")
		return 2
	}
	if _, err := h2hbench.ParseSpeculativeModes(*f.speculative); err != nil {
		fmt.Fprintln(stderr, "fak bench h2h plan: --speculative:", err)
		return 2
	}
	if strings.TrimSpace(*f.cold) != "" {
		if _, err := parsePositiveIntList(*f.cold); err != nil {
			fmt.Fprintln(stderr, "fak bench h2h plan: --cold:", err)
			return 2
		}
	}
	// Emitted argvs stay platform-neutral: ledger paths are rendered with
	// forward slashes (valid on every host fak runs on), so use the slash-only
	// path package rather than filepath, which follows the local separator.
	dir := strings.TrimSpace(*ledgerDir)
	if dir == "" {
		l := strings.TrimSpace(*ledger)
		if l != "" && !strings.EqualFold(l, "off") {
			dir = path.Dir(filepath.ToSlash(l))
		}
	}
	if dir == "" {
		dir = path.Dir(h2hbench.DefaultLedgerRel)
	}
	planArms := make([]h2hPlanArm, 0, len(f.arms))
	for _, raw := range f.arms {
		arm, err := h2hbench.ParseArm(raw)
		if err != nil {
			fmt.Fprintln(stderr, "fak bench h2h plan:", err)
			return 2
		}
		ledgerPath := path.Join(dir, "h2h-"+arm.Name+".jsonl")
		argv := []string{"fak", "bench", "h2h", "run", "--model", *f.model, "--arm", arm.Name + "=" + arm.BaseURL, "--ledger", ledgerPath}
		if *f.host != "" {
			argv = append(argv, "--host", *f.host)
		}
		if *f.runID != "" {
			argv = append(argv, "--run-id", *f.runID+"-"+arm.Name)
		}
		if *f.speculative != "0,2" {
			argv = append(argv, "--speculative", *f.speculative)
		}
		if *f.cold != "512,2048,8192" {
			argv = append(argv, "--cold", *f.cold)
		}
		if *f.reps != 3 {
			argv = append(argv, "--reps", strconv.Itoa(*f.reps))
		}
		if *f.decode != 256 {
			argv = append(argv, "--decode", strconv.Itoa(*f.decode))
		}
		if *f.turns != 6 {
			argv = append(argv, "--turns", strconv.Itoa(*f.turns))
		}
		if *f.system != 4096 {
			argv = append(argv, "--system", strconv.Itoa(*f.system))
		}
		if *f.turnTokens != 256 {
			argv = append(argv, "--turn-tokens", strconv.Itoa(*f.turnTokens))
		}
		if *f.answer != 16 {
			argv = append(argv, "--answer", strconv.Itoa(*f.answer))
		}
		if *f.timeout != 10*time.Minute {
			argv = append(argv, "--timeout", f.timeout.String())
		}
		if *f.keepThinking {
			argv = append(argv, "--keep-thinking")
		}
		if *f.slotsURL != "" {
			argv = append(argv, "--slots-url", *f.slotsURL)
		}
		if *f.gpuBusy != "auto" {
			argv = append(argv, "--gpu-busy", *f.gpuBusy)
		}
		if *f.apiKeyEnv != "" {
			argv = append(argv, "--api-key-env", *f.apiKeyEnv)
		}
		planArms = append(planArms, h2hPlanArm{Name: arm.Name, URL: arm.BaseURL, Ledger: ledgerPath, Argv: argv})
	}
	if *jsonOut {
		env := h2hPlanEnvelope{Schema: h2hPlanSchema, Model: *f.model, LedgerDir: dir, Arms: planArms}
		_ = json.NewEncoder(stdout).Encode(env)
		return 0
	}
	for _, a := range planArms {
		fmt.Fprintln(stdout, renderH2HArgvLine(a.Argv))
	}
	return 0
}

// renderH2HArgvLine joins one argv into a shell-safe line: args inside the safe
// charset stay bare, everything else is double-quoted with backslash and quote
// escaped.
func renderH2HArgvLine(argv []string) string {
	parts := make([]string, len(argv))
	for i, a := range argv {
		parts[i] = quoteH2HArgvArg(a)
	}
	return strings.Join(parts, " ")
}

func quoteH2HArgvArg(a string) string {
	if a != "" && h2hArgvSafe(a) {
		return a
	}
	return "\"" + strings.ReplaceAll(strings.ReplaceAll(a, `\`, `\\`), `"`, `\"`) + "\""
}

func h2hArgvSafe(a string) bool {
	for _, r := range a {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		case strings.ContainsRune("._:/=@+-", r):
		default:
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------- gpu-idle

const h2hGPUIdeSchema = "fak.bench.h2h-gpu-idle.v1"

// gpuIdleSample is the injectable nvidia-smi seam: the default execs the real
// binary through compute.NvidiaGPUStats; tests stub it. present=false is a typed
// failure (no nvidia-smi, unreadable output), never a zero-idle pass.
var gpuIdleSample = func() ([]compute.GPUStat, bool) { return compute.NvidiaGPUStats() }

type h2hGPUIdeRef struct {
	Index       int     `json:"index"`
	UtilPct     float64 `json:"util_pct"`
	MemUsedMiB  uint64  `json:"mem_used_mib"`
	MemTotalMiB uint64  `json:"mem_total_mib"`
}

type h2hGPUIdeEnvelope struct {
	Schema     string         `json:"schema"`
	Idle       []h2hGPUIdeRef `json:"idle"`
	LastSample []h2hGPUIdeRef `json:"last_sample,omitempty"`
	Error      string         `json:"error,omitempty"`
	Waited     string         `json:"waited"`
}

// runBenchH2HGPUIde polls nvidia-smi until at least --require GPUs sit at or
// below --max-util-pct utilization AND --max-mem-used-mib used VRAM, or the
// deadline passes. A host without a readable nvidia-smi fails typed, not green.
func runBenchH2HGPUIde(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench h2h gpu-idle", flag.ContinueOnError)
	fs.SetOutput(stderr)
	require := fs.Int("require", 1, "minimum number of idle GPUs required")
	maxUtil := fs.Float64("max-util-pct", 5, "maximum utilization.gpu percent for an idle GPU")
	maxMemMiB := fs.Int("max-mem-used-mib", 1024, "maximum memory.used MiB for an idle GPU")
	interval := fs.Duration("interval", 2*time.Second, "poll interval")
	timeout := fs.Duration("timeout", 60*time.Second, "deadline before giving up")
	jsonOut := fs.Bool("json", false, "emit the idle verdict as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *require < 1 {
		fmt.Fprintln(stderr, "fak bench h2h gpu-idle: --require must be >= 1")
		return 2
	}
	if *maxUtil < 0 || *maxMemMiB < 0 {
		fmt.Fprintln(stderr, "fak bench h2h gpu-idle: --max-util-pct and --max-mem-used-mib must be >= 0")
		return 2
	}
	maxMemBytes := uint64(*maxMemMiB) << 20
	start := time.Now()
	deadline := start.Add(*timeout)
	var last []compute.GPUStat
	for {
		stats, present := gpuIdleSample()
		if !present {
			msg := "gpu-idle: nvidia-smi absent or unreadable: cannot verify idle GPUs (typed failure, not a zero-idle pass)"
			if *jsonOut {
				env := h2hGPUIdeEnvelope{Schema: h2hGPUIdeSchema, Idle: []h2hGPUIdeRef{}, Error: msg, Waited: time.Since(start).String()}
				_ = json.NewEncoder(stdout).Encode(env)
			}
			fmt.Fprintln(stderr, msg)
			return 1
		}
		last = stats
		idle := h2hIdleGPUs(stats, *maxUtil, maxMemBytes)
		if len(idle) >= *require {
			if *jsonOut {
				env := h2hGPUIdeEnvelope{Schema: h2hGPUIdeSchema, Idle: h2hGPUIdeRefs(idle), Waited: time.Since(start).String()}
				_ = json.NewEncoder(stdout).Encode(env)
			} else {
				for _, g := range idle {
					fmt.Fprintf(stdout, "idle %d util=%s%% mem=%d/%dMiB\n", g.Index, formatH2HPct(g.UtilizationPct), g.VRAMUsedBytes>>20, g.VRAMTotalBytes>>20)
				}
			}
			return 0
		}
		if time.Now().After(deadline) {
			break
		}
		sleep := *interval
		if remaining := time.Until(deadline); remaining < sleep {
			sleep = remaining
		}
		time.Sleep(sleep)
	}
	idleCount := len(h2hIdleGPUs(last, *maxUtil, maxMemBytes))
	fmt.Fprintf(stderr, "gpu-idle: timeout after %s: %d/%d GPUs idle (need %d)\n", *timeout, idleCount, len(last), *require)
	for _, g := range last {
		fmt.Fprintf(stderr, "last sample: gpu %d util=%s%% mem=%d/%dMiB\n", g.Index, formatH2HPct(g.UtilizationPct), g.VRAMUsedBytes>>20, g.VRAMTotalBytes>>20)
	}
	if *jsonOut {
		env := h2hGPUIdeEnvelope{Schema: h2hGPUIdeSchema, Idle: []h2hGPUIdeRef{}, LastSample: h2hGPUIdeRefs(last), Waited: time.Since(start).String()}
		_ = json.NewEncoder(stdout).Encode(env)
	}
	return 1
}

// h2hIdleGPUs selects the GPUs idle enough for an exclusive-node run:
// utilization at or under maxUtilPct and VRAM used at or under maxMemBytes.
func h2hIdleGPUs(stats []compute.GPUStat, maxUtilPct float64, maxMemBytes uint64) []compute.GPUStat {
	var out []compute.GPUStat
	for _, s := range stats {
		if s.UtilizationPct <= maxUtilPct && s.VRAMUsedBytes <= maxMemBytes {
			out = append(out, s)
		}
	}
	return out
}

func h2hGPUIdeRefs(stats []compute.GPUStat) []h2hGPUIdeRef {
	out := make([]h2hGPUIdeRef, 0, len(stats))
	for _, s := range stats {
		out = append(out, h2hGPUIdeRef{Index: s.Index, UtilPct: s.UtilizationPct, MemUsedMiB: s.VRAMUsedBytes >> 20, MemTotalMiB: s.VRAMTotalBytes >> 20})
	}
	return out
}

// formatH2HPct renders a utilization percent without a trailing .0, so integer
// readings read as integers.
func formatH2HPct(p float64) string { return strconv.FormatFloat(p, 'f', -1, 64) }
