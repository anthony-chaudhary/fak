package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Receipts deliberately omit prompts, provider responses and credentials.
type opsRunReceipt struct {
	Schema   string    `json:"schema"`
	Harness  string    `json:"harness"`
	Status   string    `json:"status"`
	ExitCode int       `json:"exit_code"`
	Started  time.Time `json:"started_at"`
	Finished time.Time `json:"finished_at"`
}

var opsRunExecute = executeOpsRun

func runOpsRun(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("ops run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	harness := fs.String("harness", "opencode", "headless harness (opencode)")
	provider := fs.String("provider", "openai", "upstream wire: openai or gemini (native Google adapter)")
	promptFile := fs.String("prompt-file", "", "UTF-8 prompt file, delivered over stdin")
	timeout := fs.Duration("timeout", 5*time.Minute, "positive wall-clock deadline")
	receiptPath := fs.String("receipt", "", "execution metadata JSON file (required except dry-run)")
	model := fs.String("model", "", "upstream model identifier without an OpenCode provider prefix (required)")
	baseURL := fs.String("base-url", "", "upstream endpoint for guard (Gemini native uses /v1beta)")
	apiKeyEnv := fs.String("api-key-env", "", "environment variable holding the upstream key")
	policy := fs.String("policy", "", "guard capability-floor policy file")
	audit := fs.String("audit", "", "guard audit journal file")
	opencodeBin := fs.String("opencode-bin", "opencode", "OpenCode executable; use the native .exe on Windows")
	auto := fs.Bool("auto", false, "ask OpenCode to approve permissions not explicitly denied")
	pure := fs.Bool("pure", false, "disable OpenCode external plugins")
	dryRun := fs.Bool("dry-run", false, "validate and print metadata without launching")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || *harness != "opencode" || (*provider != "openai" && *provider != "gemini") || *timeout <= 0 || strings.TrimSpace(*model) == "" || *promptFile == "" || (!*dryRun && *receiptPath == "") {
		fmt.Fprintln(stderr, "ops run: require --harness opencode, --provider openai|gemini, --model, --prompt-file, --receipt and a positive --timeout; no positional arguments")
		return 2
	}
	if *provider == "gemini" && (*apiKeyEnv == "" || strings.TrimSpace(os.Getenv(*apiKeyEnv)) == "") {
		fmt.Fprintln(stderr, "ops run: --provider gemini requires --api-key-env naming a nonempty upstream key")
		return 2
	}
	prompt, err := os.ReadFile(*promptFile)
	if err != nil || len(bytes.TrimSpace(prompt)) == 0 {
		fmt.Fprintln(stderr, "ops run: prompt file must be readable and nonempty")
		return 2
	}
	if !*dryRun {
		for _, input := range []string{*promptFile, *policy, *audit} {
			if input != "" && opsRunSamePath(*receiptPath, input) {
				fmt.Fprintln(stderr, "ops run: receipt must be distinct from prompt, policy and audit files")
				return 2
			}
		}
	}
	if runtime.GOOS == "windows" && *opencodeBin == "opencode" {
		// npm exposes a shell shim on PATH; choose its installed native binary.
		if native := filepath.Join(os.Getenv("APPDATA"), "npm", "node_modules", "opencode-ai", "bin", "opencode.exe"); os.Getenv("APPDATA") != "" {
			if info, err := os.Stat(native); err == nil && !info.IsDir() {
				*opencodeBin = native
			}
		}
	}
	argv := []string{tuiExecutable(), "guard", "--provider", *provider, "--split", "off", "--model", *model}
	for _, pair := range [][2]string{{"--base-url", *baseURL}, {"--api-key-env", *apiKeyEnv}, {"--policy", *policy}, {"--audit", *audit}} {
		if pair[1] != "" {
			argv = append(argv, pair[0], pair[1])
		}
	}
	// A fresh provider name prevents deep-merging model/provider overrides from
	// global or project config into the route owned by this invocation.
	var nonce [12]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		fmt.Fprintln(stderr, "ops run: create provider identity failed")
		return 1
	}
	providerID := "fak_ops_" + hex.EncodeToString(nonce[:])
	modelID := providerID + "/" + *model
	argv = append(argv, "--", *opencodeBin, "run", "--format", "json", "--model", modelID)
	if *auto {
		argv = append(argv, "--auto")
	}
	if *pure {
		argv = append(argv, "--pure")
	}
	// Interpolation happens in the child after guard injects its local endpoint.
	config := map[string]any{}
	if existing := os.Getenv("OPENCODE_CONFIG_CONTENT"); strings.TrimSpace(existing) != "" {
		if json.Unmarshal([]byte(existing), &config) != nil || config == nil {
			fmt.Fprintln(stderr, "ops run: OPENCODE_CONFIG_CONTENT must be a JSON object")
			return 2
		}
	}
	providers, _ := config["provider"].(map[string]any)
	if providers == nil {
		providers = map[string]any{}
	}
	npm, childBase, childKey := "@ai-sdk/openai-compatible", "{env:OPENAI_BASE_URL}", "{env:OPENAI_API_KEY}"
	if *provider == "gemini" {
		// Native Google carries thoughtSignature across tool turns. The compatible
		// adapter used above lost that metadata in the witnessed Gemini tool loop.
		npm, childBase, childKey = "@ai-sdk/google", "{env:GOOGLE_GEMINI_BASE_URL}/v1beta", "fak-ops-guard"
	}
	providers[providerID] = map[string]any{"npm": npm, "options": map[string]string{"baseURL": childBase, "apiKey": childKey}, "models": map[string]any{*model: map[string]any{}}}
	config["provider"], config["model"] = providers, modelID
	config["small_model"] = modelID
	config["enabled_providers"] = []string{providerID}
	encoded, _ := json.Marshal(config)
	env := replaceOpsRunEnv(os.Environ(), "OPENCODE_CONFIG_CONTENT", string(encoded))
	if *dryRun {
		_ = json.NewEncoder(stdout).Encode(map[string]any{"schema": "fak-ops-run-plan/1", "harness": *harness, "provider": *provider, "guarded": true, "prompt_delivery": "stdin", "timeout": timeout.String(), "auto": *auto, "pure": *pure})
		return 0
	}
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	receipt := opsRunReceipt{Schema: "fak-ops-run/1", Harness: *harness, Status: "running", Started: time.Now().UTC()}
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run: write receipt: %v\n", err)
		return 1
	}
	code, complete, eventError := opsRunExecute(ctx, stdout, stderr, argv, env, prompt)
	receipt.Status = "failed"
	if ctx.Err() != nil {
		if ctx.Err() == context.DeadlineExceeded {
			code = 124
			receipt.Status = "timed_out"
		} else {
			code = 130
			receipt.Status = "cancelled"
		}
	} else if code == 0 {
		if eventError || !complete {
			code = 1
			fmt.Fprintln(stderr, "ops run: OpenCode did not report a successful completed turn")
		} else {
			receipt.Status = "succeeded"
		}
	}
	receipt.ExitCode, receipt.Finished = code, time.Now().UTC()
	if err := writeOpsRunReceipt(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "ops run: write receipt: %v\n", err)
		return 1
	}
	return code
}

func replaceOpsRunEnv(env []string, key, value string) []string {
	out := make([]string, 0, len(env)+1)
	for _, item := range env {
		name, _, _ := strings.Cut(item, "=")
		if !strings.EqualFold(name, key) {
			out = append(out, item)
		}
	}
	return append(out, key+"="+value)
}

func opsRunSamePath(a, b string) bool {
	left, le := filepath.Abs(a)
	right, re := filepath.Abs(b)
	if le == nil && re == nil && (left == right || (runtime.GOOS == "windows" && strings.EqualFold(left, right))) {
		return true
	}
	li, le := os.Stat(a)
	ri, re := os.Stat(b)
	return le == nil && re == nil && os.SameFile(li, ri)
}

// Forward output while retaining at most one bounded JSON event.
type opsRunEvents struct {
	output   io.Writer
	pending  []byte
	overflow bool
	complete bool
	failed   bool
}

func (w *opsRunEvents) Write(p []byte) (int, error) {
	n, err := w.output.Write(p)
	for _, b := range p[:n] {
		if b == '\n' {
			w.finishLine()
			continue
		}
		if len(w.pending) < 4*1024*1024 && !w.overflow {
			w.pending = append(w.pending, b)
		} else {
			w.overflow = true
			w.pending = nil
		}
	}
	return n, err
}
func (w *opsRunEvents) finishLine() {
	if w.overflow {
		w.failed = true
	}
	if !w.overflow {
		var ev struct {
			Type string `json:"type"`
			Part struct {
				Reason string `json:"reason"`
				State  struct {
					Status string `json:"status"`
				} `json:"state"`
			} `json:"part"`
		}
		if json.Unmarshal(w.pending, &ev) == nil {
			if ev.Type == "error" || (ev.Type == "tool_use" && ev.Part.State.Status == "error") {
				w.failed = true
			}
			if ev.Type == "step_start" {
				w.complete = false
			}
			if ev.Type == "step_finish" {
				w.complete = ev.Part.Reason == "stop"
			}
		} else if bytes.HasPrefix(bytes.TrimSpace(w.pending), []byte("{")) {
			w.failed = true
		}
	}
	w.pending = nil
	w.overflow = false
}

func executeOpsRun(ctx context.Context, stdout, stderr io.Writer, argv, env []string, prompt []byte) (int, bool, bool) {
	events := &opsRunEvents{output: stdout}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin = bytes.NewReader(prompt)
	cmd.Stdout, cmd.Stderr = events, stderr
	cmd.WaitDelay = 5 * time.Second
	procguard.ConfigureProcessTreeCancel(cmd)
	windowgate.ConfigureBackgroundCommand(cmd)
	err := cmd.Run()
	events.finishLine()
	if err != nil {
		fmt.Fprintf(stderr, "ops run: child: %v\n", err)
		return childprocess.ExitCode(err, 1), events.complete, events.failed
	}
	return 0, events.complete, events.failed
}

func writeOpsRunReceipt(path string, receipt opsRunReceipt) error {
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".ops-run-*.json")
	if err != nil {
		return err
	}
	temp := f.Name()
	defer os.Remove(temp)
	if _, err = f.Write(append(data, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(temp, path)
}
