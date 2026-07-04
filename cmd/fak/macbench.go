package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macbench"
)

func cmdMacBench(argv []string) { os.Exit(runMacBench(os.Stdout, os.Stderr, argv)) }

func runMacBench(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "watch" {
		return runMacBenchWatch(stdout, stderr, argv[1:])
	}
	suite := macbench.SuiteAll
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		suite = macbench.Suite(argv[0])
		argv = argv[1:]
	}
	def := macbench.DefaultOptions()
	fs := flag.NewFlagSet("macbench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gateway := fs.String("gateway", envOrDefault("FAK_MAC_GATEWAY", def.Gateway), "fak serve gateway on the Mac; defaults to loopback for on-node runs")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", def.Model), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	keyFile := fs.String("gateway-key-file", "~/.fak-gateway-key", "file holding the gateway bearer when the env var is empty; empty disables file lookup")
	fetchKey := fs.Bool("fetch-key", true, "when env/file key lookup is empty for a remote gateway, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	timeout := fs.Duration("timeout", 2*time.Hour, "overall benchmark timeout")
	decodeTokens := fs.String("decode-tokens", "256,512", "comma-separated max_tokens for decode-longgen")
	prefillTokens := fs.String("prefill-tokens", "128,512,2048,4096", "comma-separated prompt-token targets for prefill-sweep")
	concurrency := fs.Int("concurrency", 2, "concurrent requests for the 2stream suite")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	key, err := resolveMacBenchKeyForRun(*keyEnv, *keyFile, *fetchKey, *sshHost, *sshKey, *gateway, suite)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: %v\n", err)
		return 2
	}
	dec, err := parseIntCSV(*decodeTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: --decode-tokens: %v\n", err)
		return 2
	}
	pre, err := parseIntCSV(*prefillTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: --prefill-tokens: %v\n", err)
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak macbench: --timeout must be positive")
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	rep, err := macbench.Run(ctx, macbench.Options{
		Gateway:       *gateway,
		Model:         *model,
		Key:           key,
		Suite:         suite,
		DecodeTokens:  dec,
		PrefillTokens: pre,
		Concurrency:   *concurrency,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: %v\n", err)
		return 1
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep)
	} else {
		renderMacBench(stdout, rep)
	}
	if rep.HasErrors() {
		return 1
	}
	return 0
}

func runMacBenchWatch(stdout, stderr io.Writer, argv []string) int {
	def := macbench.DefaultOptions()
	fs := flag.NewFlagSet("macbench watch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gateway := fs.String("gateway", envOrDefault("FAK_MAC_GATEWAY", def.Gateway), "fak serve gateway on the Mac")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", def.Model), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	keyFile := fs.String("gateway-key-file", "~/.fak-gateway-key", "file holding the gateway bearer when the env var is empty; empty disables file lookup")
	fetchKey := fs.Bool("fetch-key", true, "when env/file key lookup is empty for a remote gateway, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	duration := fs.Duration("duration", 12*time.Hour, "maximum time to poll before giving up")
	interval := fs.Duration("interval", 5*time.Minute, "delay between health polls")
	healthTimeout := fs.Duration("health-timeout", 20*time.Second, "timeout for each health poll")
	runTimeout := fs.Duration("run-timeout", 2*time.Hour, "timeout for the full macbench run after health turns green")
	resultPath := fs.String("result", "", "optional path for the full macbench result JSON")
	logPath := fs.String("log", "", "optional append-only log path for health and full macbench JSON reports")
	decodeTokens := fs.String("decode-tokens", "256,512", "comma-separated max_tokens for decode-longgen")
	prefillTokens := fs.String("prefill-tokens", "128,512,2048,4096", "comma-separated prompt-token targets for prefill-sweep")
	concurrency := fs.Int("concurrency", 2, "concurrent requests for the 2stream suite")
	maxPolls := fs.Int("max-polls", 0, "maximum health polls before giving up; 0 means bounded by --duration only")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *duration <= 0 || *interval <= 0 || *healthTimeout <= 0 || *runTimeout <= 0 {
		fmt.Fprintln(stderr, "fak macbench watch: durations must be positive")
		return 2
	}
	dec, err := parseIntCSV(*decodeTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench watch: --decode-tokens: %v\n", err)
		return 2
	}
	pre, err := parseIntCSV(*prefillTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench watch: --prefill-tokens: %v\n", err)
		return 2
	}

	deadline := time.Now().Add(*duration)
	polls := 0
	for {
		polls++
		healthCtx, cancel := context.WithTimeout(context.Background(), *healthTimeout)
		health, err := macbench.Run(healthCtx, macbench.Options{
			Gateway: *gateway,
			Model:   *model,
			Suite:   macbench.SuiteHealth,
		})
		cancel()
		if err != nil {
			fmt.Fprintf(stderr, "fak macbench watch: %v\n", err)
			return 1
		}
		if err := writeMacBenchWatchReport(stdout, *logPath, health); err != nil {
			fmt.Fprintf(stderr, "fak macbench watch: write --log: %v\n", err)
			return 1
		}
		if health.Health.OK {
			return runMacBenchWatchFull(stdout, stderr, macBenchWatchRunOptions{
				gateway:       *gateway,
				model:         *model,
				keyEnv:        *keyEnv,
				keyFile:       *keyFile,
				fetchKey:      *fetchKey,
				sshHost:       *sshHost,
				sshKey:        *sshKey,
				timeout:       *runTimeout,
				resultPath:    *resultPath,
				logPath:       *logPath,
				decodeTokens:  dec,
				prefillTokens: pre,
				concurrency:   *concurrency,
			})
		}
		if (*maxPolls > 0 && polls >= *maxPolls) || time.Now().Add(*interval).After(deadline) {
			fmt.Fprintf(stderr, "fak macbench watch: gateway did not become healthy after %d poll(s)\n", polls)
			return 124
		}
		time.Sleep(*interval)
	}
}

type macBenchWatchRunOptions struct {
	gateway       string
	model         string
	keyEnv        string
	keyFile       string
	fetchKey      bool
	sshHost       string
	sshKey        string
	timeout       time.Duration
	resultPath    string
	logPath       string
	decodeTokens  []int
	prefillTokens []int
	concurrency   int
}

func runMacBenchWatchFull(stdout, stderr io.Writer, opts macBenchWatchRunOptions) int {
	key, err := resolveMacBenchKeyForRun(opts.keyEnv, opts.keyFile, opts.fetchKey, opts.sshHost, opts.sshKey, opts.gateway, macbench.SuiteAll)
	if err != nil {
		_ = writeMacBenchWatchError(opts.logPath, "key", err)
		fmt.Fprintf(stderr, "fak macbench watch: %v\n", err)
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), opts.timeout)
	defer cancel()
	rep, err := macbench.Run(ctx, macbench.Options{
		Gateway:       opts.gateway,
		Model:         opts.model,
		Key:           key,
		Suite:         macbench.SuiteAll,
		DecodeTokens:  opts.decodeTokens,
		PrefillTokens: opts.prefillTokens,
		Concurrency:   opts.concurrency,
	})
	if err != nil {
		_ = writeMacBenchWatchError(opts.logPath, "run", err)
		fmt.Fprintf(stderr, "fak macbench watch: %v\n", err)
		return 1
	}
	if err := writeMacBenchWatchReport(stdout, opts.logPath, rep); err != nil {
		fmt.Fprintf(stderr, "fak macbench watch: write --log: %v\n", err)
		return 1
	}
	if strings.TrimSpace(opts.resultPath) != "" {
		if err := writeMacBenchResultFile(opts.resultPath, rep); err != nil {
			fmt.Fprintf(stderr, "fak macbench watch: write --result: %v\n", err)
			return 1
		}
	}
	if rep.HasErrors() {
		return 1
	}
	return 0
}

func writeMacBenchWatchReport(stdout io.Writer, logPath string, rep macbench.Report) error {
	_ = writeIndentedJSONNoEscape(stdout, rep)
	logPath = strings.TrimSpace(logPath)
	if logPath == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(rep)
}

func writeMacBenchWatchError(logPath, phase string, err error) error {
	logPath = strings.TrimSpace(logPath)
	if logPath == "" || err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}
	f, errOpen := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if errOpen != nil {
		return errOpen
	}
	defer f.Close()
	event := struct {
		Schema      string `json:"schema"`
		GeneratedAt string `json:"generated_at"`
		Phase       string `json:"phase"`
		Error       string `json:"error"`
	}{
		Schema:      "fak.macbench.watch.event.v1",
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Phase:       phase,
		Error:       err.Error(),
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(event)
}

func writeMacBenchResultFile(path string, rep macbench.Report) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(rep)
}

func resolveMacBenchKeyForRun(envName, keyFile string, fetch bool, sshHost, sshKey, gateway string, suite macbench.Suite) (string, error) {
	key, err := resolveMacBenchKey(envName, keyFile)
	if err != nil {
		return "", err
	}
	if key != "" || !fetch || suite == macbench.SuiteHealth {
		return key, nil
	}
	if err := ensureClaudeMacGatewayKey(envName, true, sshHost, sshKey, gateway); err != nil {
		return "", err
	}
	return strings.TrimSpace(os.Getenv(nonEmptyString(strings.TrimSpace(envName), "FAK_GATEWAY_KEY"))), nil
}

func nonEmptyString(s, fallback string) string {
	if s != "" {
		return s
	}
	return fallback
}

func renderMacBench(w io.Writer, rep macbench.Report) {
	fmt.Fprintf(w, "macbench %s gateway=%s model=%s health=%v\n", rep.Suite, rep.Gateway, rep.Model, rep.Health.OK)
	for _, row := range rep.Rows {
		if row.Error != "" {
			fmt.Fprintf(w, "- %s: ERROR %s\n", row.Name, row.Error)
			continue
		}
		switch {
		case row.PrefillTokensPerSecond > 0:
			fmt.Fprintf(w, "- %s: %.2f tok/s prefill (prompt=%d ttft=%.3fs completion=%d)\n",
				row.Name, row.PrefillTokensPerSecond, row.PromptTokens, row.TTFTSeconds, row.CompletionTokens)
		case row.TokensPerSecond > 0:
			fmt.Fprintf(w, "- %s: %.2f tok/s decode (completion=%d wall=%.3fs streams=%d)\n",
				row.Name, row.TokensPerSecond, row.CompletionTokens, row.WallSeconds, row.Streams)
		default:
			fmt.Fprintf(w, "- %s: no metric\n", row.Name)
		}
	}
	if rep.Headline != "" {
		fmt.Fprintf(w, "headline: %s\n", rep.Headline)
	}
	for _, err := range rep.Errors {
		fmt.Fprintf(w, "error: %s\n", err)
	}
}

func resolveMacBenchKey(envName, keyFile string) (string, error) {
	envName = strings.TrimSpace(envName)
	if envName == "" {
		envName = "FAK_GATEWAY_KEY"
	}
	if key := strings.TrimSpace(os.Getenv(envName)); key != "" {
		return key, nil
	}
	keyFile = strings.TrimSpace(keyFile)
	if keyFile == "" {
		return "", nil
	}
	if strings.HasPrefix(keyFile, "~/") || strings.HasPrefix(keyFile, `~\`) {
		if home, err := os.UserHomeDir(); err == nil {
			keyFile = filepath.Join(home, keyFile[2:])
		}
	}
	b, err := os.ReadFile(keyFile)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read --gateway-key-file: %w", err)
	}
	return strings.TrimSpace(string(b)), nil
}

func parseIntCSV(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("%q is not a positive integer", part)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one value is required")
	}
	return out, nil
}
