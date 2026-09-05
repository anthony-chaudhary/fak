package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macbench"
)

func cmdMacBench(argv []string) { os.Exit(runMacBench(os.Stdout, os.Stderr, argv)) }

func runMacBench(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "validate-comparison" {
		return runMacBenchValidateComparison(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "validate-agentic-comparison" {
		return runMacBenchValidateAgenticComparison(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "watch-status" {
		return runMacBenchWatchStatus(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "recover" {
		return runMacBenchRecover(stdout, stderr, argv[1:])
	}
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
	decodeTokens := fs.String("decode-tokens", "16,32,64,128,256,512", "comma-separated max_tokens for decode-longgen")
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

func runMacBenchValidateComparison(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench validate-comparison", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "three-way comparison packet JSON")
	asJSON := fs.Bool("json", false, "emit machine-readable validation result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "fak macbench validate-comparison: --input is required")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: read --input: %v\n", err)
		return 1
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var packet macbench.ComparisonPacket
	if err := dec.Decode(&packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: decode packet: %v\n", err)
		return 1
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak macbench validate-comparison: decode packet: %v\n", err)
		return 1
	}
	if err := macbench.ValidateComparisonPacket(packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: %v\n", err)
		return 1
	}
	if err := verifyMacBenchComparisonEvidenceFiles(packet, *input); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-comparison: %v\n", err)
		return 1
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	result := struct {
		Schema       string `json:"schema"`
		Valid        bool   `json:"valid"`
		PacketSHA256 string `json:"packet_sha256"`
	}{
		Schema:       "fak.macbench.comparison.validation.v1",
		Valid:        true,
		PacketSHA256: digest,
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result)
	} else {
		fmt.Fprintf(stdout, "VALID packet_sha256=%s\n", result.PacketSHA256)
	}
	return 0
}

func runMacBenchValidateAgenticComparison(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench validate-agentic-comparison", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "agentic comparison packet JSON")
	asJSON := fs.Bool("json", false, "emit machine-readable validation result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "fak macbench validate-agentic-comparison: --input is required")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: read --input: %v\n", err)
		return 1
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var packet macbench.AgenticComparisonPacket
	if err := dec.Decode(&packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: decode packet: %v\n", err)
		return 1
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: decode packet: %v\n", err)
		return 1
	}
	if err := macbench.ValidateAgenticComparisonPacket(packet); err != nil {
		fmt.Fprintf(stderr, "fak macbench validate-agentic-comparison: %v\n", err)
		return 1
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	result := struct {
		Schema       string  `json:"schema"`
		Valid        bool    `json:"valid"`
		PacketSHA256 string  `json:"packet_sha256"`
		SpeedupRatio float64 `json:"speedup_ratio"`
	}{
		Schema:       "fak.macbench.agentic-comparison.validation.v1",
		Valid:        true,
		PacketSHA256: digest,
		SpeedupRatio: packet.Summary.SpeedupRatio,
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, result)
	} else {
		fmt.Fprintf(stdout, "VALID packet_sha256=%s speedup=%.2fx\n", result.PacketSHA256, result.SpeedupRatio)
	}
	return 0
}

func verifyMacBenchComparisonEvidenceFiles(packet macbench.ComparisonPacket, packetPath string) error {
	base, err := filepath.Abs(filepath.Dir(packetPath))
	if err != nil {
		return fmt.Errorf("resolve packet directory: %w", err)
	}
	base, err = filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("resolve packet directory symlinks: %w", err)
	}
	for _, arm := range packet.Arms {
		raw, err := verifyMacBenchComparisonEvidenceFile(base, arm.RawResult.Path, arm.RawResult.SHA256)
		if err != nil {
			return fmt.Errorf("arm %s raw_result: %w", arm.Name, err)
		}
		var rawFile macbench.ComparisonRawSamplesFile
		if err := decodeStrictMacBenchComparisonJSON(raw, &rawFile); err != nil {
			return fmt.Errorf("arm %s raw_result: decode: %w", arm.Name, err)
		}
		wantRaw := macbench.ComparisonRawSamplesFile{
			Schema: macbench.ComparisonRawSamplesSchema, Arm: arm.Name,
			CampaignID: packet.CampaignID, RunID: arm.RunID, HostID: arm.HostID,
			StartedAt: arm.StartedAt, FinishedAt: arm.FinishedAt, Samples: arm.Samples,
		}
		if !reflect.DeepEqual(rawFile, wantRaw) {
			return fmt.Errorf("arm %s raw_result: content does not match packet samples", arm.Name)
		}

		quality, err := verifyMacBenchComparisonEvidenceFile(base, arm.Quality.ResultPath, arm.Quality.ResultSHA256)
		if err != nil {
			return fmt.Errorf("arm %s quality: %w", arm.Name, err)
		}
		var qualityFile macbench.ComparisonQualityEvidenceFile
		if err := decodeStrictMacBenchComparisonJSON(quality, &qualityFile); err != nil {
			return fmt.Errorf("arm %s quality: decode: %w", arm.Name, err)
		}
		wantQuality := macbench.ComparisonQualityEvidenceFile{
			Schema: macbench.ComparisonQualityEvidenceSchema, Arm: arm.Name, RunID: arm.RunID,
			PolicyRef: arm.Quality.PolicyRef, PolicyVersion: arm.Quality.PolicyVersion,
			PolicySHA256: arm.Quality.PolicySHA256,
			Passed:       arm.Quality.Passed, Score: arm.Quality.Score,
			ArtifactSHA256: arm.Artifact.SHA256, PromptSetSHA256: arm.PromptSetSHA256,
		}
		if qualityFile != wantQuality {
			return fmt.Errorf("arm %s quality: content does not match packet quality result", arm.Name)
		}
	}
	return nil
}

func verifyMacBenchComparisonEvidenceFile(base, relative, wantDigest string) ([]byte, error) {
	relative = strings.TrimSpace(relative)
	if relative == "" || filepath.IsAbs(relative) {
		return nil, fmt.Errorf("path must be relative to the packet")
	}
	clean := filepath.Clean(relative)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("path escapes the packet directory")
	}
	resolved, err := filepath.EvalSymlinks(filepath.Join(base, clean))
	if err != nil {
		return nil, fmt.Errorf("resolve %q: %w", relative, err)
	}
	inside, err := filepath.Rel(base, resolved)
	if err != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("path escapes the packet directory")
	}
	f, err := os.Open(resolved)
	if err != nil {
		return nil, fmt.Errorf("open %q: %w", relative, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %q: %w", relative, err)
	}
	if info.Size() > 64<<20 {
		return nil, fmt.Errorf("%q exceeds 64 MiB evidence limit", relative)
	}
	raw, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", relative, err)
	}
	got := fmt.Sprintf("%x", sha256.Sum256(raw))
	if got != wantDigest {
		return nil, fmt.Errorf("sha256 mismatch for %q", relative)
	}
	return raw, nil
}

func decodeStrictMacBenchComparisonJSON(raw []byte, out any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
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
	decodeTokens := fs.String("decode-tokens", "16,32,64,128,256,512", "comma-separated max_tokens for decode-longgen")
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

const macBenchWatchStatusSchema = "fak.macbench.watch.status.v1"

type macBenchWatchEvent struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Phase       string `json:"phase"`
	Error       string `json:"error"`
}

type macBenchWatchStatus struct {
	Schema          string              `json:"schema"`
	GeneratedAt     string              `json:"generated_at"`
	LogPath         string              `json:"log_path,omitempty"`
	ResultPath      string              `json:"result_path,omitempty"`
	LogPresent      bool                `json:"log_present"`
	ResultPresent   bool                `json:"result_present"`
	Reports         int                 `json:"reports"`
	Events          int                 `json:"events"`
	State           string              `json:"state"`
	LastGeneratedAt string              `json:"last_generated_at,omitempty"`
	LastError       string              `json:"last_error,omitempty"`
	NextAction      string              `json:"next_action"`
	LatestReport    *macbench.Report    `json:"latest_report,omitempty"`
	LatestEvent     *macBenchWatchEvent `json:"latest_event,omitempty"`
	Result          *macbench.Report    `json:"result,omitempty"`
}

func runMacBenchWatchStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench watch-status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	logPath := fs.String("log", "", "macbench watch append-only log path")
	resultPath := fs.String("result", "", "macbench watch full result JSON path")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*logPath) == "" && strings.TrimSpace(*resultPath) == "" {
		fmt.Fprintln(stderr, "fak macbench watch-status: pass --log and/or --result")
		return 2
	}
	status, err := loadMacBenchWatchStatus(*logPath, *resultPath, time.Now().UTC())
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench watch-status: %v\n", err)
		return 1
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, status)
		return 0
	}
	renderMacBenchWatchStatus(stdout, status)
	return 0
}

func loadMacBenchWatchStatus(logPath, resultPath string, now time.Time) (macBenchWatchStatus, error) {
	s := macBenchWatchStatus{
		Schema:      macBenchWatchStatusSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		LogPath:     filepath.ToSlash(strings.TrimSpace(logPath)),
		ResultPath:  filepath.ToSlash(strings.TrimSpace(resultPath)),
		State:       "no_reports",
		NextAction:  "wait for the first macbench watch poll",
	}
	if strings.TrimSpace(logPath) != "" {
		if err := readMacBenchWatchLog(logPath, &s); err != nil {
			return s, err
		}
	}
	if strings.TrimSpace(resultPath) != "" {
		result, ok, err := readMacBenchResult(resultPath)
		if err != nil {
			return s, err
		}
		s.ResultPresent = ok
		if ok {
			s.Result = &result
		}
	}
	classifyMacBenchWatchStatus(&s)
	return s, nil
}

func readMacBenchWatchLog(path string, s *macBenchWatchStatus) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read --log: %w", err)
	}
	defer f.Close()
	s.LogPresent = true
	dec := json.NewDecoder(f)
	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("parse --log: %w", err)
		}
		var hdr struct {
			Schema      string `json:"schema"`
			GeneratedAt string `json:"generated_at"`
		}
		if err := json.Unmarshal(raw, &hdr); err != nil {
			return fmt.Errorf("parse --log header: %w", err)
		}
		switch hdr.Schema {
		case macbench.Schema:
			var rep macbench.Report
			if err := json.Unmarshal(raw, &rep); err != nil {
				return fmt.Errorf("parse macbench report: %w", err)
			}
			s.Reports++
			repCopy := rep
			s.LatestReport = &repCopy
			s.LastGeneratedAt = nonEmptyString(rep.GeneratedAt, s.LastGeneratedAt)
		case "fak.macbench.watch.event.v1":
			var ev macBenchWatchEvent
			if err := json.Unmarshal(raw, &ev); err != nil {
				return fmt.Errorf("parse watch event: %w", err)
			}
			s.Events++
			evCopy := ev
			s.LatestEvent = &evCopy
			s.LastGeneratedAt = nonEmptyString(ev.GeneratedAt, s.LastGeneratedAt)
		default:
			return fmt.Errorf("parse --log: unknown schema %q", hdr.Schema)
		}
	}
	return nil
}

func readMacBenchResult(path string) (macbench.Report, bool, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return macbench.Report{}, false, nil
		}
		return macbench.Report{}, false, fmt.Errorf("read --result: %w", err)
	}
	var rep macbench.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		return macbench.Report{}, true, fmt.Errorf("parse --result: %w", err)
	}
	return rep, true, nil
}

func classifyMacBenchWatchStatus(s *macBenchWatchStatus) {
	if s.Result != nil {
		s.LastGeneratedAt = nonEmptyString(s.Result.GeneratedAt, s.LastGeneratedAt)
		if s.Result.HasErrors() {
			s.State = "completed_with_errors"
			s.LastError = firstMacBenchError(*s.Result)
			s.NextAction = "inspect the full macbench result and fix the failing suite"
			return
		}
		s.State = "completed"
		s.NextAction = "record or publish the full macbench result"
		return
	}
	if s.LatestReport != nil {
		rep := *s.LatestReport
		s.LastGeneratedAt = nonEmptyString(rep.GeneratedAt, s.LastGeneratedAt)
		if rep.Suite == macbench.SuiteAll {
			if rep.HasErrors() {
				s.State = "completed_with_errors"
				s.LastError = firstMacBenchError(rep)
				s.NextAction = "inspect the full macbench report in the watch log"
				return
			}
			s.State = "completed"
			s.NextAction = "persist the full macbench report with --result or fold it into nightrun"
			return
		}
		if rep.Health.OK {
			s.State = "healthy_waiting_for_full_run"
			s.NextAction = "wait for the full macbench suite to finish"
			return
		}
		s.State = "waiting_for_gateway"
		s.LastError = firstMacBenchError(rep)
		s.NextAction = "keep the watcher running; gateway health is still false"
		return
	}
	if s.LatestEvent != nil {
		s.State = "watch_error"
		s.LastError = s.LatestEvent.Error
		s.NextAction = "inspect the watch event and restart after fixing the phase"
		return
	}
	if !s.LogPresent && s.LogPath != "" {
		s.State = "missing_log"
		s.NextAction = "confirm the watch process started and wrote its first poll"
	}
}

func firstMacBenchError(rep macbench.Report) string {
	if rep.Health.Error != "" {
		return rep.Health.Error
	}
	if len(rep.Errors) > 0 {
		return rep.Errors[0]
	}
	for _, row := range rep.Rows {
		if row.Error != "" {
			return row.Error
		}
	}
	return ""
}

func renderMacBenchWatchStatus(w io.Writer, s macBenchWatchStatus) {
	fmt.Fprintf(w, "macbench watch-status: %s\n", s.State)
	if s.LastGeneratedAt != "" {
		fmt.Fprintf(w, "last: %s\n", s.LastGeneratedAt)
	}
	fmt.Fprintf(w, "log: present=%v reports=%d events=%d\n", s.LogPresent, s.Reports, s.Events)
	fmt.Fprintf(w, "result: present=%v\n", s.ResultPresent)
	if s.LatestReport != nil {
		fmt.Fprintf(w, "latest: suite=%s gateway=%s model=%s health=%v\n",
			s.LatestReport.Suite, s.LatestReport.Gateway, s.LatestReport.Model, s.LatestReport.Health.OK)
	}
	if s.LastError != "" {
		fmt.Fprintf(w, "error: %s\n", s.LastError)
	}
	fmt.Fprintf(w, "next: %s\n", s.NextAction)
}

func runMacBenchRecover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench recover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	logPath := fs.String("log", "", "macbench watch append-only log path")
	resultPath := fs.String("result", "", "macbench watch full result JSON path")
	watcherRunning := fs.Bool("watcher-running", true, "whether the macbench watcher process is still running")
	tailnetOnline := fs.String("tailnet-online", "unknown", "Mac peer status: true|false|unknown (also online|offline)")
	sshReachable := fs.String("ssh-reachable", "unknown", "Mac control path status: true|false|unknown (also reachable|unreachable)")
	wakeHelper := fs.String("wake-helper", "unknown", "wake/restart helper availability: true|false|unknown (also present|absent)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*logPath) == "" && strings.TrimSpace(*resultPath) == "" {
		fmt.Fprintln(stderr, "fak macbench recover: pass --log and/or --result")
		return 2
	}
	status, err := loadMacBenchWatchStatus(*logPath, *resultPath, time.Now().UTC())
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 1
	}
	tailnet, err := parseOptionalMacBenchBool("tailnet-online", *tailnetOnline)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 2
	}
	ssh, err := parseOptionalMacBenchBool("ssh-reachable", *sshReachable)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 2
	}
	wake, err := parseOptionalMacBenchBool("wake-helper", *wakeHelper)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench recover: %v\n", err)
		return 2
	}
	// Only claim log presence when a --log path was actually named; otherwise
	// leave it unknown so a --result-only call keeps its existing verdict.
	var logPresent *bool
	if strings.TrimSpace(*logPath) != "" {
		present := status.LogPresent
		logPresent = &present
	}
	plan := macbench.PlanRecovery(macbench.RecoverySignals{
		WatcherRunning: *watcherRunning,
		ResultPresent:  status.ResultPresent,
		LatestReport:   status.LatestReport,
		LogPresent:     logPresent,
		TailnetOnline:  tailnet,
		SSHReachable:   ssh,
		WakeHelper:     wake,
	})
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, plan)
		return 0
	}
	renderMacBenchRecovery(stdout, plan)
	return 0
}

func parseOptionalMacBenchBool(name, raw string) (*bool, error) {
	v := strings.ToLower(strings.TrimSpace(raw))
	switch v {
	case "", "unknown":
		return nil, nil
	case "1", "t", "true", "y", "yes", "online", "reachable", "present", "available":
		b := true
		return &b, nil
	case "0", "f", "false", "n", "no", "offline", "unreachable", "absent", "missing", "unavailable":
		b := false
		return &b, nil
	default:
		return nil, fmt.Errorf("--%s must be true, false, or unknown", name)
	}
}

func renderMacBenchRecovery(w io.Writer, plan macbench.RecoveryPlan) {
	fmt.Fprintf(w, "macbench recovery: %s (%s)\n", plan.State, plan.Severity)
	fmt.Fprintf(w, "%s\n", plan.Summary)
	for _, ev := range plan.Evidence {
		fmt.Fprintf(w, "evidence: %s\n", ev)
	}
	for _, action := range plan.Actions {
		fmt.Fprintf(w, "- %s: %s\n", action.ID, action.Title)
		if action.Detail != "" {
			fmt.Fprintf(w, "  %s\n", action.Detail)
		}
	}
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
