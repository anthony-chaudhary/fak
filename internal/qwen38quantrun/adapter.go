package qwen38quantrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processstart"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
	"github.com/anthony-chaudhary/fak/internal/serverproduct"
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

// ManagedServerConfig makes a READY lifecycle receipt, rather than a hand-copied
// URL, authoritative for the measured endpoint.
type ManagedServerConfig struct {
	Directory         string   `json:"directory"`
	MinimumGeneration uint64   `json:"minimum_generation,omitempty"`
	ReceiptDigest     string   `json:"receipt_digest,omitempty"`
	ProtocolFamily    string   `json:"protocol_family"`
	ProtocolRevision  string   `json:"protocol_revision"`
	Capabilities      []string `json:"capabilities"`
	BaseURL           string   `json:"base_url,omitempty"`
	ModelAlias        string   `json:"model_alias"`
}

type ManagedServerIdentity struct {
	ReceiptDigest        string   `json:"receipt_digest"`
	Generation           uint64   `json:"generation"`
	ProcessID            int      `json:"process_id"`
	ProcessStartIdentity string   `json:"process_start_identity"`
	BaseURL              string   `json:"base_url"`
	ModelAlias           string   `json:"model_alias"`
	ProtocolFamily       string   `json:"protocol_family"`
	ProtocolRevision     string   `json:"protocol_revision"`
	Capabilities         []string `json:"capabilities"`
}

type managedArchive struct {
	Schema         string                  `json:"schema"`
	Campaign       json.RawMessage         `json:"campaign"`
	ServerIdentity []ManagedServerIdentity `json:"server_identity_chain"`
}

// AdapterConfig is the file-backed operator contract for a live campaign.
// ObservationCommand must emit one Observation JSON object on stdout; lifecycle
// commands are argv arrays and are never interpreted by a shell.
type AdapterConfig struct {
	Schema           string               `json:"schema,omitempty"`
	RuntimeSource    string               `json:"runtime_source,omitempty"`
	SourceObservedAt string               `json:"source_observed_at,omitempty"`
	SourceLicense    string               `json:"source_license,omitempty"`
	SupportedArms    []string             `json:"supported_arms,omitempty"`
	APIKeyEnv        string               `json:"api_key_env,omitempty"`
	Endpoint         EndpointConfig       `json:"endpoint"`
	ManagedServer    *ManagedServerConfig `json:"managed_server,omitempty"`
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
	if cfg.ManagedServer == nil && (cfg.Endpoint.Endpoint == "" || cfg.Endpoint.Model == "") {
		return errors.New("endpoint and endpoint.model are required")
	}
	if cfg.ManagedServer != nil {
		if err := validateManagedServerConfig(*cfg.ManagedServer); err != nil {
			return err
		}
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

func validateManagedServerConfig(cfg ManagedServerConfig) error {
	if strings.TrimSpace(cfg.Directory) == "" {
		return errors.New("managed_server.directory is required")
	}
	if cfg.ProtocolFamily == "" || cfg.ProtocolRevision == "" || len(cfg.Capabilities) == 0 || cfg.ModelAlias == "" {
		return errors.New("managed_server protocol, capabilities, and model_alias are required")
	}
	if cfg.ProtocolFamily != serverproduct.ProtocolOpenAIHTTP {
		return fmt.Errorf("managed_server.protocol_family must be %q", serverproduct.ProtocolOpenAIHTTP)
	}
	return nil
}

const (
	managedStateFilename   = "server-state.json"
	managedReceiptFilename = "server-receipt.json"
	managedStateSchema     = "fak.server-lifecycle-state/v1"
	managedStateReady      = "ready"
)

type managedReadyExpectation struct {
	Generation           uint64
	MinimumGeneration    uint64
	ProcessID            int
	ProcessStartIdentity string
	ReceiptDigest        string
	ProtocolFamily       string
	ProtocolRevision     string
	Capabilities         []string
	BaseURL              string
	ModelAlias           string
}

type managedReadyBinding struct {
	Receipt       serverproduct.ServerReceipt
	ReceiptDigest string
	ReceiptBytes  []byte
}

type managedStateRecord struct {
	Schema               string `json:"schema"`
	State                string `json:"state"`
	InstanceID           string `json:"instance_id"`
	Generation           uint64 `json:"generation"`
	ProcessID            int    `json:"process_id,omitempty"`
	ProcessStartIdentity string `json:"process_start_identity,omitempty"`
	BaseURL              string `json:"base_url,omitempty"`
	Error                string `json:"error,omitempty"`
	UpdatedAt            string `json:"updated_at"`
	ReadinessDeadline    string `json:"readiness_deadline,omitempty"`
}

// consumeManagedReady is the runner-side registration seam for the lifecycle
// wire contract. Keeping the read-only consumer here prevents the tier-2 runner
// from depending on the tier-3 process manager while preserving identity checks.
func consumeManagedReady(dir string, want managedReadyExpectation) (managedReadyBinding, error) {
	if strings.TrimSpace(dir) == "" {
		return managedReadyBinding{}, errors.New("lifecycle directory is required")
	}
	statePath := filepath.Join(dir, managedStateFilename)
	receiptPath := filepath.Join(dir, managedReceiptFilename)
	stateRaw, err := os.ReadFile(statePath)
	if err != nil {
		return managedReadyBinding{}, fmt.Errorf("read lifecycle state: %w", err)
	}
	state, err := decodeManagedReadyState(stateRaw)
	if err != nil {
		return managedReadyBinding{}, err
	}
	receiptRaw, err := os.ReadFile(receiptPath)
	if err != nil {
		return managedReadyBinding{}, fmt.Errorf("read ready receipt: %w", err)
	}
	receipt, err := serverproduct.DecodeReceipt(receiptRaw)
	if err != nil {
		return managedReadyBinding{}, fmt.Errorf("decode ready receipt: %w", err)
	}
	digest := managedReceiptDigest(receiptRaw)
	if err := matchManagedReadyState(state, receipt, digest, want); err != nil {
		return managedReadyBinding{}, err
	}
	started, ok := processstart.Start(state.ProcessID)
	if !ok {
		return managedReadyBinding{}, errors.New("revalidate process start identity: process is not live")
	}
	if started.UTC().Format(time.RFC3339Nano) != state.ProcessStartIdentity {
		return managedReadyBinding{}, errors.New("live process start identity mismatch")
	}
	stateAfter, err := os.ReadFile(statePath)
	if err != nil {
		return managedReadyBinding{}, fmt.Errorf("reread lifecycle state: %w", err)
	}
	receiptAfter, err := os.ReadFile(receiptPath)
	if err != nil {
		return managedReadyBinding{}, fmt.Errorf("reread ready receipt: %w", err)
	}
	if !bytes.Equal(stateRaw, stateAfter) || !bytes.Equal(receiptRaw, receiptAfter) {
		return managedReadyBinding{}, errors.New("lifecycle READY identity changed during validation")
	}
	return managedReadyBinding{Receipt: receipt, ReceiptDigest: digest, ReceiptBytes: append([]byte(nil), receiptRaw...)}, nil
}

func decodeManagedReadyState(raw []byte) (managedStateRecord, error) {
	var state managedStateRecord
	if err := decodeManagedStrict(raw, &state); err != nil {
		return managedStateRecord{}, fmt.Errorf("decode lifecycle state: %w", err)
	}
	if state.Schema != managedStateSchema {
		return managedStateRecord{}, fmt.Errorf("lifecycle state schema must be %q", managedStateSchema)
	}
	if state.State != managedStateReady {
		return managedStateRecord{}, fmt.Errorf("lifecycle state = %q, want %q", state.State, managedStateReady)
	}
	if state.Generation == 0 || state.ProcessID <= 0 || state.ProcessStartIdentity == "" || state.InstanceID == "" || state.BaseURL == "" {
		return managedStateRecord{}, errors.New("ready lifecycle state identity is incomplete")
	}
	return state, nil
}

func matchManagedReadyState(state managedStateRecord, receipt serverproduct.ServerReceipt, digest string, want managedReadyExpectation) error {
	if receipt.Identity.InstanceID != state.InstanceID || receipt.Generation != state.Generation ||
		receipt.Ownership.InstanceID != state.InstanceID || receipt.Ownership.ProcessID != state.ProcessID ||
		receipt.Ownership.ProcessStartIdentity != state.ProcessStartIdentity || receipt.Endpoint.BaseURL != state.BaseURL {
		return errors.New("ready receipt identity does not match lifecycle state")
	}
	if want.Generation != 0 && state.Generation != want.Generation {
		return fmt.Errorf("generation = %d, want %d", state.Generation, want.Generation)
	}
	if want.MinimumGeneration != 0 && state.Generation < want.MinimumGeneration {
		return fmt.Errorf("generation = %d, want at least %d", state.Generation, want.MinimumGeneration)
	}
	if want.ProcessID != 0 && state.ProcessID != want.ProcessID {
		return fmt.Errorf("process id = %d, want %d", state.ProcessID, want.ProcessID)
	}
	if want.ProcessStartIdentity != "" && state.ProcessStartIdentity != want.ProcessStartIdentity {
		return errors.New("process start identity mismatch")
	}
	if want.ReceiptDigest != "" && digest != want.ReceiptDigest {
		return fmt.Errorf("receipt digest = %q, want %q", digest, want.ReceiptDigest)
	}
	if want.BaseURL != "" && receipt.Endpoint.BaseURL != want.BaseURL {
		return fmt.Errorf("base URL = %q, want %q", receipt.Endpoint.BaseURL, want.BaseURL)
	}
	if want.ModelAlias != "" && receipt.ModelAlias != want.ModelAlias {
		return fmt.Errorf("model alias = %q, want %q", receipt.ModelAlias, want.ModelAlias)
	}
	if want.ProtocolFamily != "" && receipt.Protocol.Family != want.ProtocolFamily {
		return fmt.Errorf("protocol family = %q, want %q", receipt.Protocol.Family, want.ProtocolFamily)
	}
	if want.ProtocolRevision != "" && receipt.Protocol.Revision != want.ProtocolRevision {
		return fmt.Errorf("protocol revision = %q, want %q", receipt.Protocol.Revision, want.ProtocolRevision)
	}
	if len(want.Capabilities) > 0 {
		got, expected := slices.Clone(receipt.Protocol.Capabilities), slices.Clone(want.Capabilities)
		slices.Sort(got)
		slices.Sort(expected)
		if !slices.Equal(got, expected) {
			return errors.New("protocol capabilities mismatch")
		}
	}
	return nil
}

func managedReceiptDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func decodeManagedStrict(raw []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}

type managedServerLifecycle struct {
	delegate   Lifecycle
	config     ManagedServerConfig
	mu         sync.Mutex
	current    managedReadyBinding
	identities []ManagedServerIdentity
}

func newManagedServerLifecycle(delegate Lifecycle, cfg ManagedServerConfig) (*managedServerLifecycle, error) {
	if err := validateManagedServerConfig(cfg); err != nil {
		return nil, err
	}
	binding, err := consumeManagedReady(cfg.Directory, managedExpectation(cfg, true))
	if err != nil {
		return nil, fmt.Errorf("managed server READY identity: %w", err)
	}
	managed := &managedServerLifecycle{delegate: delegate, config: cfg, current: binding}
	managed.identities = append(managed.identities, managedIdentity(binding))
	return managed, nil
}

func managedExpectation(cfg ManagedServerConfig, pinDigest bool) managedReadyExpectation {
	want := managedReadyExpectation{
		MinimumGeneration: cfg.MinimumGeneration,
		ProtocolFamily:    cfg.ProtocolFamily, ProtocolRevision: cfg.ProtocolRevision,
		Capabilities: slices.Clone(cfg.Capabilities), BaseURL: cfg.BaseURL, ModelAlias: cfg.ModelAlias,
	}
	if pinDigest {
		want.ReceiptDigest = cfg.ReceiptDigest
	}
	return want
}

func exactManagedExpectation(binding managedReadyBinding, cfg ManagedServerConfig) managedReadyExpectation {
	return managedReadyExpectation{
		Generation: binding.Receipt.Generation, ProcessID: binding.Receipt.Ownership.ProcessID,
		ProcessStartIdentity: binding.Receipt.Ownership.ProcessStartIdentity, ReceiptDigest: binding.ReceiptDigest,
		ProtocolFamily: cfg.ProtocolFamily, ProtocolRevision: cfg.ProtocolRevision,
		Capabilities: slices.Clone(cfg.Capabilities), BaseURL: binding.Receipt.Endpoint.BaseURL, ModelAlias: binding.Receipt.ModelAlias,
	}
}

func (m *managedServerLifecycle) Restart(ctx context.Context) error { return m.delegate.Restart(ctx) }

func (m *managedServerLifecycle) Ready(ctx context.Context) error {
	if err := m.delegate.Ready(ctx); err != nil {
		return err
	}
	m.mu.Lock()
	previous := m.current
	m.mu.Unlock()
	want := managedExpectation(m.config, false)
	want.MinimumGeneration = previous.Receipt.Generation + 1
	want.BaseURL = previous.Receipt.Endpoint.BaseURL
	want.ModelAlias = previous.Receipt.ModelAlias
	binding, err := consumeManagedReady(m.config.Directory, want)
	if err != nil {
		return fmt.Errorf("managed server READY identity after restart: %w", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.current = binding
	identity := managedIdentity(binding)
	if len(m.identities) == 0 || m.identities[len(m.identities)-1].ReceiptDigest != identity.ReceiptDigest {
		m.identities = append(m.identities, identity)
	}
	return nil
}

func (m *managedServerLifecycle) Revalidate(context.Context) error {
	m.mu.Lock()
	current := m.current
	m.mu.Unlock()
	_, err := consumeManagedReady(m.config.Directory, exactManagedExpectation(current, m.config))
	if err != nil {
		return fmt.Errorf("managed server READY identity changed: %w", err)
	}
	return nil
}

func (m *managedServerLifecycle) Cleanup(ctx context.Context) error {
	if err := m.Revalidate(ctx); err != nil {
		return fmt.Errorf("refuse cleanup of unverified managed process: %w", err)
	}
	return m.delegate.Cleanup(ctx)
}

func (m *managedServerLifecycle) endpoint() EndpointConfig {
	m.mu.Lock()
	defer m.mu.Unlock()
	return EndpointConfig{Endpoint: m.current.Receipt.Endpoint.BaseURL, Model: m.current.Receipt.ModelAlias}
}

func (m *managedServerLifecycle) identityChain() []ManagedServerIdentity {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ManagedServerIdentity(nil), m.identities...)
}

func managedIdentity(binding managedReadyBinding) ManagedServerIdentity {
	r := binding.Receipt
	return ManagedServerIdentity{
		ReceiptDigest: binding.ReceiptDigest, Generation: r.Generation, ProcessID: r.Ownership.ProcessID,
		ProcessStartIdentity: r.Ownership.ProcessStartIdentity, BaseURL: r.Endpoint.BaseURL, ModelAlias: r.ModelAlias,
		ProtocolFamily: r.Protocol.Family, ProtocolRevision: r.Protocol.Revision, Capabilities: slices.Clone(r.Protocol.Capabilities),
	}
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
	var lifecycle Lifecycle = commandLifecycle{restart: cfg.RestartCommand, ready: cfg.ReadyCommand, cleanup: cfg.CleanupCommand}
	endpoint := cfg.Endpoint.runnerConfig()
	var managed *managedServerLifecycle
	if cfg.ManagedServer != nil {
		managed, err = newManagedServerLifecycle(lifecycle, *cfg.ManagedServer)
		if err != nil {
			return err
		}
		resolved := managed.endpoint()
		endpoint.Endpoint, endpoint.Model = resolved.Endpoint, resolved.Model
		endpoint.BeforeTrial = func(trialCtx context.Context, _ qwen38quant.Fixture, _ int) error {
			return managed.Revalidate(trialCtx)
		}
		lifecycle = managed
	}
	campaign, err := (Runner{}).RunCampaign(ctx, CampaignConfig{
		Endpoint: endpoint, ExecutionEngine: cfg.ExecutionEngine, Arm: cfg.Arm, Expected: cfg.Expected, Command: cfg.Command,
		RequireDevice: cfg.RequireDevice, StaleAfter: cfg.StaleAfter,
		RollbackThreshold: cfg.RollbackThreshold,
		Probe:             commandProbe{argv: cfg.ObservationCommand},
		Lifecycle:         lifecycle,
	}, corpus)
	if err != nil {
		return err
	}
	if managed != nil {
		archive, wrapErr := canonicalJSON(managedArchive{Schema: "fak.qwen38-quant-managed-raw/1", Campaign: json.RawMessage(campaign.Archive), ServerIdentity: managed.identityChain()})
		if wrapErr != nil {
			return fmt.Errorf("managed archive: %w", wrapErr)
		}
		campaign.Archive = archive
		sum := sha256.Sum256(archive)
		campaign.Report.RawArchiveSHA256 = hex.EncodeToString(sum[:])
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
