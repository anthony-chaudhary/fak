package qwen38quantrun

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

const AdapterSchema = "fak.qwen38-quant-adapter/1"

var maintainedAdapterArms = map[string][]string{
	qwen38quant.EngineLlamaCpp: {"q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs"},
	qwen38quant.EngineVLLM:     {"bf16", "fp8", "awq_int4", "gptq_int4"},
}

var maintainedAdapterSources = map[string]string{
	qwen38quant.EngineLlamaCpp: "https://github.com/ggml-org/llama.cpp/commit/",
	qwen38quant.EngineVLLM:     "https://github.com/vllm-project/vllm/commit/",
}

var maintainedAdapterLicenses = map[string]string{
	qwen38quant.EngineLlamaCpp: "MIT",
	qwen38quant.EngineVLLM:     "Apache-2.0",
}

type EndpointConfig struct {
	Endpoint    string        `json:"endpoint"`
	APIKey      string        `json:"api_key,omitempty"`
	Model       string        `json:"model"`
	Repetitions int           `json:"repetitions,omitempty"`
	Timeout     time.Duration `json:"timeout,omitempty"`
}

func (c EndpointConfig) runnerConfig() Config {
	return Config{Endpoint: c.Endpoint, APIKey: c.APIKey, Model: c.Model, Repetitions: c.Repetitions, Timeout: c.Timeout}
}

// AdapterConfig is the file-backed operator contract for a live campaign.
// ObservationCommand must emit one Observation JSON object on stdout; lifecycle
// commands are argv arrays and are never interpreted by a shell.
type AdapterConfig struct {
	Schema           string         `json:"schema,omitempty"`
	RuntimeSource    string         `json:"runtime_source,omitempty"`
	SourceObservedAt string         `json:"source_observed_at,omitempty"`
	SourceLicense    string         `json:"source_license,omitempty"`
	SupportedArms    []string       `json:"supported_arms,omitempty"`
	APIKeyEnv        string         `json:"api_key_env,omitempty"`
	Endpoint         EndpointConfig `json:"endpoint"`
	// ExecutionEngine carries the model-math runtime identity used for evidence
	// promotion, distinct from endpoint, planner, transport, and hardware backend identity.
	ExecutionEngine    string               `json:"execution_engine"`
	Arm                string               `json:"arm"`
	Expected           qwen38quant.Identity `json:"expected"`
	Command            []string             `json:"command"`
	RequireDevice      string               `json:"require_device"`
	StaleAfter         string               `json:"stale_after"`
	RollbackThreshold  string               `json:"rollback_threshold"`
	ObservationCommand []string             `json:"observation_command"`
	RestartCommand     []string             `json:"restart_command"`
	ReadyCommand       []string             `json:"ready_command"`
	CleanupCommand     []string             `json:"cleanup_command"`
}

// SelfcheckAdapterConfig validates one maintained example without executing a
// command, contacting an endpoint, or downloading a model.
func SelfcheckAdapterConfig(path string) (AdapterConfig, error) {
	var cfg AdapterConfig
	if err := decodeFile(path, &cfg); err != nil {
		return AdapterConfig{}, err
	}
	if err := validateMaintainedAdapter(cfg); err != nil {
		return AdapterConfig{}, err
	}
	return cfg, nil
}

func validateMaintainedAdapter(cfg AdapterConfig) error {
	if cfg.Schema != AdapterSchema {
		return fmt.Errorf("schema: got %q", cfg.Schema)
	}
	wantArms, ok := maintainedAdapterArms[cfg.ExecutionEngine]
	if !ok {
		return fmt.Errorf("execution_engine %q has no maintained comparison adapter", cfg.ExecutionEngine)
	}
	if !slices.Equal(cfg.SupportedArms, wantArms) {
		return fmt.Errorf("supported_arms drift for %s: got %v want %v", cfg.ExecutionEngine, cfg.SupportedArms, wantArms)
	}
	if !slices.Contains(cfg.SupportedArms, cfg.Arm) {
		return fmt.Errorf("arm %q is not covered by the maintained %s adapter", cfg.Arm, cfg.ExecutionEngine)
	}
	if cfg.Endpoint.Endpoint == "" || cfg.Endpoint.Model == "" {
		return errors.New("endpoint and endpoint.model are required")
	}
	if cfg.Endpoint.APIKey != "" {
		return errors.New("inline api_key is refused; use api_key_env")
	}
	if !validEnvName(cfg.APIKeyEnv) {
		return fmt.Errorf("api_key_env %q is not a valid environment variable name", cfg.APIKeyEnv)
	}
	if !validHex(cfg.Expected.RuntimeRevision, 20) {
		return errors.New("expected.RuntimeRevision must be a full 40-character source revision")
	}
	if cfg.RuntimeSource != maintainedAdapterSources[cfg.ExecutionEngine]+cfg.Expected.RuntimeRevision {
		return errors.New("runtime_source must be the immutable upstream commit URL for expected.RuntimeRevision")
	}
	if cfg.SourceLicense != maintainedAdapterLicenses[cfg.ExecutionEngine] || cfg.SourceObservedAt == "" {
		return errors.New("source_observed_at and the upstream source_license are required")
	}
	if _, err := time.Parse("2006-01-02", cfg.SourceObservedAt); err != nil {
		return fmt.Errorf("source_observed_at: %w", err)
	}
	if missing := missingAdapterIdentity(cfg.Expected); len(missing) != 0 {
		return fmt.Errorf("expected identity is incomplete: %s", strings.Join(missing, ", "))
	}
	for name, value := range map[string]string{
		"CheckpointSHA256": cfg.Expected.CheckpointSHA256,
		"ArtifactSHA256":   cfg.Expected.ArtifactSHA256,
		"TokenizerSHA256":  cfg.Expected.TokenizerSHA256,
		"TemplateSHA256":   cfg.Expected.TemplateSHA256,
	} {
		if !validHex(value, sha256.Size) {
			return fmt.Errorf("expected.%s must be SHA-256", name)
		}
	}
	commands := map[string][]string{
		"command": cfg.Command, "observation_command": cfg.ObservationCommand,
		"restart_command": cfg.RestartCommand, "ready_command": cfg.ReadyCommand,
		"cleanup_command": cfg.CleanupCommand,
	}
	for name, argv := range commands {
		if err := validatePinnedArgv(name, argv, cfg.Expected.RuntimeRevision); err != nil {
			return err
		}
	}
	if !slices.Contains(cfg.Command, cfg.APIKeyEnv) {
		return errors.New("command must forward api_key_env by name without embedding its value")
	}
	if cfg.ExecutionEngine == qwen38quant.EngineLlamaCpp {
		if !argvContains(cfg.Command, "llama.cpp:"+cfg.Expected.RuntimeRevision) || !argvPair(cfg.Command, "--pull", "never") || !argvPair(cfg.Command, "--gpus", "all") || !argvPair(cfg.Command, "--model", "/models/") || !argvPair(cfg.Command, "--n-gpu-layers", "all") {
			return errors.New("llama.cpp command must use a local revision-tagged image with --pull never and a /models/ artifact")
		}
	} else {
		if !argvContains(cfg.Command, "vllm:"+cfg.Expected.RuntimeRevision) || !argvPair(cfg.Command, "--pull", "never") || !argvPair(cfg.Command, "--gpus", "all") || !argvPair(cfg.Command, "serve", "/models/") || !argvPair(cfg.Command, "--model-impl", "vllm") {
			return errors.New("vLLM command must use a local revision-tagged image, a /models/ artifact, and --model-impl vllm")
		}
	}
	if !argvContains(cfg.ObservationCommand, ".State.Running") || !argvContains(cfg.ObservationCommand, ".HostConfig.DeviceRequests") {
		return errors.New("observation_command must derive residency and fallback state from the running container")
	}
	if !argvContains(cfg.ReadyCommand, "/health") {
		return errors.New("ready_command must call the maintained runtime health endpoint")
	}
	if cfg.RequireDevice == "" || cfg.StaleAfter == "" || cfg.RollbackThreshold == "" {
		return errors.New("require_device, stale_after, and rollback_threshold are required")
	}
	if _, err := time.Parse("2006-01-02", cfg.StaleAfter); err != nil {
		return fmt.Errorf("stale_after: %w", err)
	}
	return nil
}

func validatePinnedArgv(name string, argv []string, revision string) error {
	if len(argv) == 0 {
		return fmt.Errorf("%s is required", name)
	}
	for _, arg := range argv {
		if arg == "" {
			return fmt.Errorf("%s contains an empty argument", name)
		}
	}
	switch filepath.Base(strings.ToLower(argv[0])) {
	case "sh", "bash", "zsh", "cmd", "cmd.exe", "powershell", "pwsh":
		return fmt.Errorf("%s must be an argv command, not a shell", name)
	}
	if !argvContains(argv, revision) {
		return fmt.Errorf("%s does not carry runtime revision %s", name, revision)
	}
	return nil
}

func argvContains(argv []string, fragment string) bool {
	for _, arg := range argv {
		if strings.Contains(arg, fragment) {
			return true
		}
	}
	return false
}

func argvPair(argv []string, key, valueFragment string) bool {
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == key && strings.Contains(argv[i+1], valueFragment) {
			return true
		}
	}
	return false
}

func validEnvName(value string) bool {
	if value == "" || !(value[0] == '_' || value[0] >= 'A' && value[0] <= 'Z' || value[0] >= 'a' && value[0] <= 'z') {
		return false
	}
	for i := 1; i < len(value); i++ {
		if c := value[i]; !(c == '_' || c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}

func validHex(value string, bytes int) bool {
	if len(value) != bytes*2 {
		return false
	}
	for _, c := range value {
		if !strings.ContainsRune("0123456789abcdef", c) {
			return false
		}
	}
	return true
}

func missingAdapterIdentity(id qwen38quant.Identity) []string {
	values := []struct{ name, value string }{
		{"Model", id.Model}, {"CheckpointSHA256", id.CheckpointSHA256}, {"ArtifactSHA256", id.ArtifactSHA256},
		{"TokenizerSHA256", id.TokenizerSHA256}, {"TemplateSHA256", id.TemplateSHA256},
		{"QuantizerRevision", id.QuantizerRevision}, {"RuntimeRevision", id.RuntimeRevision}, {"FakModuleRev", id.FakModuleRev},
	}
	var missing []string
	for _, value := range values {
		if value.value == "" {
			missing = append(missing, value.name)
		}
	}
	return missing
}

// RunAdapter loads the frozen corpus and operator config, executes the real
// endpoint campaign, validates it, and atomically writes the archive and report.
func RunAdapter(ctx context.Context, configPath, corpusPath, reportPath, archivePath string) error {
	var cfg AdapterConfig
	if err := decodeFile(configPath, &cfg); err != nil {
		return fmt.Errorf("config: %w", err)
	}
	if _, maintained := maintainedAdapterArms[cfg.ExecutionEngine]; maintained {
		if err := validateMaintainedAdapter(cfg); err != nil {
			return fmt.Errorf("maintained adapter contract: %w; run SelfcheckAdapterConfig(%q) and repair the named field before retrying", err, configPath)
		}
	}
	corpusBytes, err := os.ReadFile(corpusPath)
	if err != nil {
		return fmt.Errorf("corpus: %w", err)
	}
	corpus, err := qwen38quant.DecodeCorpus(corpusBytes)
	if err != nil {
		return fmt.Errorf("corpus: %w", err)
	}
	if len(cfg.ObservationCommand) == 0 {
		return errors.New("observation_command is required")
	}
	if len(cfg.RestartCommand) == 0 || len(cfg.ReadyCommand) == 0 || len(cfg.CleanupCommand) == 0 {
		return errors.New("restart_command, ready_command, and cleanup_command are required")
	}
	if cfg.Endpoint.APIKey != "" {
		return errors.New("inline api_key is refused; use api_key_env")
	}
	if cfg.APIKeyEnv != "" {
		if !validEnvName(cfg.APIKeyEnv) {
			return fmt.Errorf("api_key_env %q is not a valid environment variable name", cfg.APIKeyEnv)
		}
		cfg.Endpoint.APIKey = os.Getenv(cfg.APIKeyEnv)
		if cfg.Endpoint.APIKey == "" {
			return fmt.Errorf("API key environment %s is empty", cfg.APIKeyEnv)
		}
	}
	campaign, err := (Runner{}).RunCampaign(ctx, CampaignConfig{
		Endpoint: cfg.Endpoint.runnerConfig(), ExecutionEngine: cfg.ExecutionEngine, Arm: cfg.Arm, Expected: cfg.Expected, Command: cfg.Command,
		RequireDevice: cfg.RequireDevice, StaleAfter: cfg.StaleAfter,
		RollbackThreshold: cfg.RollbackThreshold,
		Probe:             commandProbe{argv: cfg.ObservationCommand},
		Lifecycle:         commandLifecycle{restart: cfg.RestartCommand, ready: cfg.ReadyCommand, cleanup: cfg.CleanupCommand},
	}, corpus)
	if err != nil {
		return err
	}
	if err := qwen38quant.Validate(campaign.Report, corpus); err != nil {
		return fmt.Errorf("validate report: %w", err)
	}
	report, err := json.MarshalIndent(campaign.Report, "", "  ")
	if err != nil {
		return err
	}
	report = append(report, '\n')
	if err := writeAtomic(archivePath, campaign.Archive, 0o600); err != nil {
		return fmt.Errorf("archive: %w", err)
	}
	if err := writeAtomic(reportPath, report, 0o644); err != nil {
		_ = os.Remove(archivePath)
		return fmt.Errorf("report: %w", err)
	}
	return nil
}

func decodeFile(path string, dst any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

type commandProbe struct{ argv []string }

func (p commandProbe) Observe(ctx context.Context) (Observation, error) {
	out, err := runArgv(ctx, p.argv)
	if err != nil {
		return Observation{}, err
	}
	var got Observation
	d := json.NewDecoder(strings.NewReader(string(out)))
	d.DisallowUnknownFields()
	if err := d.Decode(&got); err != nil {
		return Observation{}, fmt.Errorf("decode observation: %w", err)
	}
	return got, nil
}

type commandLifecycle struct{ restart, ready, cleanup []string }

func (l commandLifecycle) Restart(ctx context.Context) error {
	_, err := runArgv(ctx, l.restart)
	return err
}
func (l commandLifecycle) Ready(ctx context.Context) error {
	_, err := runArgv(ctx, l.ready)
	return err
}
func (l commandLifecycle) Cleanup(ctx context.Context) error {
	_, err := runArgv(ctx, l.cleanup)
	return err
}

func runArgv(ctx context.Context, argv []string) ([]byte, error) {
	if len(argv) == 0 {
		return nil, errors.New("empty command")
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s: %w: %s", argv[0], err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

func writeAtomic(path string, data []byte, mode os.FileMode) error {
	if path == "" {
		return errors.New("output path is required")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".qwen38-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err = f.Write(data); err == nil {
		err = f.Chmod(mode)
	}
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	for i := 0; i < 5; i++ {
		err = os.Rename(tmp, path)
		if err == nil {
			return nil
		}
		time.Sleep(time.Duration(i+1) * 10 * time.Millisecond)
	}
	return err
}
