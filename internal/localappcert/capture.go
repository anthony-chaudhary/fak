package localappcert

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const CaptureSchema = "fak.local-app-certification-capture/1"

const (
	ReasonCommandExitNonzero = "COMMAND_EXIT_NONZERO"
	ReasonCommandStartFailed = "COMMAND_START_FAILED"
	ReasonCommandTimeout     = "COMMAND_TIMEOUT"
	ReasonCaptureCancelled   = "CAPTURE_CANCELLED"
)

// CaptureSpec describes direct argv commands for one complete certification run.
// RuntimeRevision and Artifact may instead be supplied by CaptureOptions, but a
// value supplied in both places must agree.
type CaptureSpec struct {
	Schema          string
	RuntimeRevision string
	Artifact        string
	Scenarios       []CaptureScenarioSpec
}

type CaptureScenarioSpec struct {
	Name    string
	Command []string
	Timeout string
}

type CaptureOptions struct {
	EvidenceDir     string
	RuntimeRevision string
	Artifact        string
}

// LoadCaptureSpec reads a strict JSON capture specification. Scenario commands
// are a JSON object so the scenario name is declared exactly once at its point
// of use; duplicate object keys are detected rather than silently overwritten.
func LoadCaptureSpec(path string) (CaptureSpec, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return CaptureSpec{}, err
	}
	return ParseCaptureSpec(b)
}

// ParseCaptureSpec parses the capture format without executing any commands.
func ParseCaptureSpec(b []byte) (CaptureSpec, error) {
	var raw struct {
		Schema          string          `json:"schema"`
		RuntimeRevision string          `json:"runtime_revision"`
		Artifact        string          `json:"artifact"`
		Scenarios       json.RawMessage `json:"scenarios"`
	}
	if err := decodeStrict(b, &raw); err != nil {
		return CaptureSpec{}, fmt.Errorf("localappcert: decode capture specification: %w", err)
	}

	scenarios, err := decodeCaptureScenarios(raw.Scenarios)
	if err != nil {
		return CaptureSpec{}, err
	}
	return CaptureSpec{
		Schema:          raw.Schema,
		RuntimeRevision: raw.RuntimeRevision,
		Artifact:        raw.Artifact,
		Scenarios:       scenarios,
	}, nil
}

func decodeCaptureScenarios(b []byte) ([]CaptureScenarioSpec, error) {
	if len(bytes.TrimSpace(b)) == 0 {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("localappcert: decode capture scenarios: %w", err)
	}
	if tok != json.Delim('{') {
		return nil, errors.New("localappcert: capture scenarios must be an object keyed by required scenario name")
	}
	seen := make(map[string]bool)
	var scenarios []CaptureScenarioSpec
	for dec.More() {
		nameToken, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("localappcert: decode capture scenario name: %w", err)
		}
		name, ok := nameToken.(string)
		if !ok {
			return nil, errors.New("localappcert: capture scenario name is not a string")
		}
		if seen[name] {
			return nil, fmt.Errorf("localappcert: duplicate capture scenario %q", name)
		}
		seen[name] = true

		var commandRaw json.RawMessage
		if err := dec.Decode(&commandRaw); err != nil {
			return nil, fmt.Errorf("localappcert: decode capture scenario %q: %w", name, err)
		}
		var command struct {
			Command []string `json:"command"`
			Timeout string   `json:"timeout"`
		}
		if err := decodeStrict(commandRaw, &command); err != nil {
			return nil, fmt.Errorf("localappcert: decode capture scenario %q: %w", name, err)
		}
		scenarios = append(scenarios, CaptureScenarioSpec{Name: name, Command: command.Command, Timeout: command.Timeout})
	}
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("localappcert: decode capture scenarios: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return nil, fmt.Errorf("localappcert: decode capture scenarios: %w", err)
	}
	return scenarios, nil
}

func decodeStrict(b []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	return requireJSONEOF(dec)
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

// ValidateCaptureSpec rejects an incomplete or ambiguous capture before any
// evidence directory is created or command is started.
func ValidateCaptureSpec(spec CaptureSpec, opts CaptureOptions) error {
	if spec.Schema != CaptureSchema {
		return fmt.Errorf("localappcert: capture schema %q, want %q", spec.Schema, CaptureSchema)
	}
	if strings.TrimSpace(opts.EvidenceDir) == "" {
		return errors.New("localappcert: capture evidence directory is required")
	}
	if _, err := resolveMetadata("runtime revision", spec.RuntimeRevision, opts.RuntimeRevision); err != nil {
		return err
	}
	if _, err := resolveMetadata("artifact", spec.Artifact, opts.Artifact); err != nil {
		return err
	}

	required := make(map[string]bool, len(RequiredScenarios))
	for _, name := range RequiredScenarios {
		required[name] = true
	}
	seen := make(map[string]bool, len(spec.Scenarios))
	var duplicates, unknown []string
	for _, scenario := range spec.Scenarios {
		if seen[scenario.Name] {
			duplicates = append(duplicates, scenario.Name)
			continue
		}
		seen[scenario.Name] = true
		if !required[scenario.Name] {
			unknown = append(unknown, scenario.Name)
			continue
		}
		if len(scenario.Command) == 0 || strings.TrimSpace(scenario.Command[0]) == "" {
			return fmt.Errorf("localappcert: capture scenario %q has no command argv", scenario.Name)
		}
		timeout, err := time.ParseDuration(scenario.Timeout)
		if err != nil || timeout <= 0 {
			return fmt.Errorf("localappcert: capture scenario %q has invalid timeout %q", scenario.Name, scenario.Timeout)
		}
	}
	if len(duplicates) != 0 {
		sort.Strings(duplicates)
		return fmt.Errorf("localappcert: duplicate capture scenarios %v", uniqueStrings(duplicates))
	}
	if len(unknown) != 0 {
		sort.Strings(unknown)
		return fmt.Errorf("localappcert: unknown capture scenarios %v", unknown)
	}
	var missing []string
	for _, name := range RequiredScenarios {
		if !seen[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) != 0 {
		return fmt.Errorf("localappcert: missing capture scenarios %v", missing)
	}
	return nil
}

func uniqueStrings(values []string) []string {
	if len(values) < 2 {
		return values
	}
	result := values[:1]
	for _, value := range values[1:] {
		if value != result[len(result)-1] {
			result = append(result, value)
		}
	}
	return result
}

func resolveMetadata(field, specValue, optionValue string) (string, error) {
	specValue = strings.TrimSpace(specValue)
	optionValue = strings.TrimSpace(optionValue)
	if specValue != "" && optionValue != "" && specValue != optionValue {
		return "", fmt.Errorf("localappcert: capture %s conflicts between specification and CLI option", field)
	}
	if optionValue != "" {
		return optionValue, nil
	}
	if specValue != "" {
		return specValue, nil
	}
	return "", fmt.Errorf("localappcert: capture %s is required", field)
}

// Capture runs every required scenario in canonical order. Each command is
// launched directly from argv, without shell interpretation, and all output is
// written even when the command fails or reaches its deadline.
func Capture(ctx context.Context, spec CaptureSpec, opts CaptureOptions) ([]Scenario, error) {
	if err := ValidateCaptureSpec(spec, opts); err != nil {
		return nil, err
	}
	runtimeRevision, err := resolveMetadata("runtime revision", spec.RuntimeRevision, opts.RuntimeRevision)
	if err != nil {
		return nil, err
	}
	artifact, err := resolveMetadata("artifact", spec.Artifact, opts.Artifact)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(opts.EvidenceDir, 0o755); err != nil {
		return nil, fmt.Errorf("localappcert: create evidence directory: %w", err)
	}

	commands := make(map[string]CaptureScenarioSpec, len(spec.Scenarios))
	for _, scenario := range spec.Scenarios {
		commands[scenario.Name] = scenario
	}
	rows := make([]Scenario, 0, len(RequiredScenarios))
	for index, name := range RequiredScenarios {
		scenario := commands[name]
		timeout, _ := time.ParseDuration(scenario.Timeout)
		commandContext, cancel := context.WithTimeout(ctx, timeout)
		command := exec.CommandContext(commandContext, scenario.Command[0], scenario.Command[1:]...)
		output, commandErr := command.CombinedOutput()
		contextErr := commandContext.Err()
		cancel()

		evidence := filepath.Join(opts.EvidenceDir, fmt.Sprintf("%02d-%s.log", index+1, name))
		if err := os.WriteFile(evidence, output, 0o644); err != nil {
			return rows, fmt.Errorf("localappcert: write evidence for scenario %q: %w", name, err)
		}
		digest := sha256.Sum256(output)
		row := Scenario{
			Name:           name,
			Status:         StatusPass,
			Evidence:       evidence,
			EvidenceSHA256: hex.EncodeToString(digest[:]),
			EvidenceBytes:  int64(len(output)),
			Receipt: &Receipt{
				Engine:          "fak-native",
				Fallback:        "none",
				Artifact:        artifact,
				RuntimeRevision: runtimeRevision,
			},
		}
		if commandErr != nil || contextErr != nil {
			row.Status = StatusFail
			row.Reason = commandFailureReason(commandErr, contextErr)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func commandFailureReason(commandErr, contextErr error) string {
	if errors.Is(contextErr, context.DeadlineExceeded) {
		return ReasonCommandTimeout
	}
	if errors.Is(contextErr, context.Canceled) {
		return ReasonCaptureCancelled
	}
	var exitErr *exec.ExitError
	if errors.As(commandErr, &exitErr) {
		return ReasonCommandExitNonzero
	}
	return ReasonCommandStartFailed
}

// SixGateOptions configures an ordered six-gate serving acceptance run.
type SixGateOptions struct {
	BaseURL               string
	APIKey                string
	EvidenceDir           string
	Client                *http.Client
	RequestTimeout        time.Duration
	LoadTimeout           time.Duration
	LoadPollInterval      time.Duration
	ContextMaxRetries     int
	DisableContextRetries bool
	SpeedWindow           time.Duration
	RuntimeRevision       string
	Artifact              string
	Served                string
	Receipt               *Receipt
	Now                   func() time.Time
	Nonce                 func() (string, error)
}

// ValidateSixGateSpec rejects invalid or capped six-gate specifications before
// making network requests or creating evidence directories.
func ValidateSixGateSpec(spec SixGateSpec, opts SixGateOptions) error {
	if strings.TrimSpace(spec.Model) == "" {
		return errors.New("localappcert: six-gate model is required")
	}
	if spec.ContextWindow <= 0 {
		return errors.New("localappcert: six-gate context_window must be positive")
	}
	if spec.AnswerCapped {
		return errors.New("localappcert: six-gate answers must never be capped")
	}
	if spec.SpeedFloorTokensPerSecond < 0 || math.IsNaN(spec.SpeedFloorTokensPerSecond) || math.IsInf(spec.SpeedFloorTokensPerSecond, 0) {
		return errors.New("localappcert: six-gate speed floor must be finite and non-negative")
	}
	if spec.SpeedWindowSeconds < 0 || math.IsNaN(spec.SpeedWindowSeconds) || math.IsInf(spec.SpeedWindowSeconds, 0) {
		return errors.New("localappcert: six-gate speed window must be finite and non-negative")
	}
	endpoint, err := resolveSixGateRequired("endpoint", spec.Endpoint, opts.BaseURL)
	if err != nil {
		return err
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return fmt.Errorf("localappcert: invalid six-gate endpoint %q", endpoint)
	}
	rev, err := resolveSixGateOptional("runtime revision", spec.RuntimeRevision, opts.RuntimeRevision)
	if err != nil {
		return err
	}
	if _, err := resolveSixGateOptional("artifact", spec.Artifact, opts.Artifact); err != nil {
		return err
	}
	if opts.Receipt != nil && !hasValidNativeReceipt(opts.Receipt, rev) {
		return errors.New("localappcert: six-gate receipt lacks exact fak-native fallback-free identity")
	}
	return nil
}

func resolveSixGateRequired(field, specVal, optVal string) (string, error) {
	val, err := resolveSixGateOptional(field, specVal, optVal)
	if err != nil {
		return "", err
	}
	if val == "" {
		return "", fmt.Errorf("localappcert: six-gate %s is required", field)
	}
	return val, nil
}

func resolveSixGateOptional(field, specVal, optVal string) (string, error) {
	specVal = strings.TrimSpace(specVal)
	optVal = strings.TrimSpace(optVal)
	if specVal != "" && optVal != "" && specVal != optVal {
		return "", fmt.Errorf("localappcert: six-gate %s conflicts between specification and option", field)
	}
	if optVal != "" {
		return optVal, nil
	}
	return specVal, nil
}

// RunSixGates runs the six serving acceptance gates in canonical order
// (load -> chat -> reasoning -> tools -> context -> speed), stopping at the
// first non-answering failure ("blew up") while recording spec metadata and
// per-gate evidence even on failure.
func RunSixGates(ctx context.Context, spec SixGateSpec, opts SixGateOptions) (SixGateReport, error) {
	if err := ValidateSixGateSpec(spec, opts); err != nil {
		return SixGateReport{}, err
	}

	endpoint, _ := resolveSixGateRequired("endpoint", spec.Endpoint, opts.BaseURL)
	endpoint = strings.TrimRight(endpoint, "/")
	runtimeRevision, _ := resolveSixGateOptional("runtime revision", spec.RuntimeRevision, opts.RuntimeRevision)
	artifact, _ := resolveSixGateOptional("artifact", spec.Artifact, opts.Artifact)

	speedWindow := opts.SpeedWindow
	if speedWindow <= 0 && spec.SpeedWindowSeconds > 0 {
		speedWindow = time.Duration(spec.SpeedWindowSeconds * float64(time.Second))
	}
	if speedWindow <= 0 {
		speedWindow = DefaultSpeedWindow
	}
	floor := effectiveSpeedFloor(spec.SpeedFloorTokensPerSecond)

	normalizedSpec := spec
	normalizedSpec.Endpoint = endpoint
	normalizedSpec.RuntimeRevision = runtimeRevision
	normalizedSpec.Artifact = artifact
	normalizedSpec.AnswerCapped = false
	normalizedSpec.SpeedFloorTokensPerSecond = floor
	normalizedSpec.SpeedWindowSeconds = speedWindow.Seconds()

	if opts.EvidenceDir != "" {
		if err := os.MkdirAll(opts.EvidenceDir, 0o755); err != nil {
			return SixGateReport{}, fmt.Errorf("localappcert: create evidence directory: %w", err)
		}
	}

	nowFn := opts.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	nonceFn := opts.Nonce
	if nonceFn == nil {
		nonceFn = defaultHexNonce
	}
	reqTimeout := opts.RequestTimeout
	if reqTimeout <= 0 {
		reqTimeout = 60 * time.Second
	}
	client := opts.Client
	if client == nil {
		client = &http.Client{Timeout: reqTimeout}
	}

	runner := sixGateRunner{
		spec:            normalizedSpec,
		opts:            opts,
		endpoint:        endpoint,
		client:          client,
		requestTimeout:  reqTimeout,
		speedWindow:     speedWindow,
		floorTPS:        floor,
		now:             nowFn,
		nonce:           nonceFn,
		contextRetries:  resolveContextRetries(opts),
		expectedInteger: resolveReasoningExpectedInt(spec),
	}

	report := SixGateReport{
		Schema: SixGateSchema,
		Spec:   normalizedSpec,
		Receipt: func() *Receipt {
			if opts.Receipt == nil {
				return nil
			}
			c := *opts.Receipt
			return &c
		}(),
	}

	var passedNames []string
	var contextGateResult *GateResult
	var speedGateResult *GateResult

	for index, gate := range SixGates {
		res, evPayload := runner.executeGate(ctx, gate)
		if err := persistGateEvidence(&res, evPayload, index, opts.EvidenceDir); err != nil {
			report.Gates = append(report.Gates, res)
			return report, err
		}
		report.Gates = append(report.Gates, res)

		if res.Status == StatusPass {
			passedNames = append(passedNames, string(gate))
		}
		if gate == GateContext {
			copied := res
			contextGateResult = &copied
		}
		if gate == GateSpeed {
			copied := res
			speedGateResult = &copied
		}
		if res.BlewUp {
			report.AbortedGate = gate
			break
		}
	}

	served := strings.TrimSpace(opts.Served)
	if served == "" {
		served = normalizedSpec.Model
	}
	report.Passed = len(passedNames) == len(SixGates) && report.AbortedGate == ""
	report.Proof = SixGateProof{
		Gates:  strings.Join(passedNames, ","),
		Served: served,
	}
	if contextGateResult != nil && contextGateResult.PrefillTokensPerSecond != nil && !contextGateResult.PrefillSuppressed && contextGateResult.CachedTokens == 0 && !contextGateResult.PrefixCached {
		v := *contextGateResult.PrefillTokensPerSecond
		report.Proof.PrefillTokensPerSecond = &v
	}
	if speedGateResult != nil && speedGateResult.Status == StatusPass && speedGateResult.Source == SourceTokenizer {
		v := speedGateResult.DecodeTokensPerSecond
		report.Proof.DecodeTokensPerSecond = &v
	}

	if err := finalizeReportEvidence(&report, opts.EvidenceDir); err != nil {
		return report, err
	}
	return report, nil
}

func resolveContextRetries(opts SixGateOptions) int {
	if opts.DisableContextRetries || opts.ContextMaxRetries < 0 {
		return 0
	}
	if opts.ContextMaxRetries == 0 {
		return 2
	}
	return opts.ContextMaxRetries
}

func resolveReasoningExpectedInt(spec SixGateSpec) int64 {
	if spec.ReasoningExpectedInt != nil {
		return *spec.ReasoningExpectedInt
	}
	return 391
}

func defaultHexNonce() (string, error) {
	var raw [16]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(raw[:]), nil
}

type sixGateRunner struct {
	spec            SixGateSpec
	opts            SixGateOptions
	endpoint        string
	client          *http.Client
	requestTimeout  time.Duration
	speedWindow     time.Duration
	floorTPS        float64
	now             func() time.Time
	nonce           func() (string, error)
	contextRetries  int
	expectedInteger int64
}

func (r *sixGateRunner) executeGate(ctx context.Context, gate GateName) (GateResult, map[string]any) {
	switch gate {
	case GateLoad:
		return r.runLoadGate(ctx)
	case GateChat:
		return r.runChatGate(ctx)
	case GateReasoning:
		return r.runReasoningGate(ctx)
	case GateTools:
		return r.runToolsGate(ctx)
	case GateContext:
		return r.runContextGate(ctx)
	case GateSpeed:
		return r.runSpeedGate(ctx)
	default:
		return GateResult{Gate: gate, Status: StatusFail, BlewUp: true, Reason: ReasonRequestFailed}, map[string]any{"error": "unknown gate"}
	}
}

type chatCompletionRequest struct {
	Model         string         `json:"model"`
	Messages      []chatMessage  `json:"messages"`
	Tools         []chatTool     `json:"tools,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatTool struct {
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

type chatToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type chatToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatMessage struct {
	Role             string         `json:"role"`
	Content          string         `json:"content,omitempty"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	Reasoning        string         `json:"reasoning,omitempty"`
	ToolCalls        []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	Name             string         `json:"name,omitempty"`
}

func (m chatMessage) reasoningText() string {
	if strings.TrimSpace(m.ReasoningContent) != "" {
		return m.ReasoningContent
	}
	return m.Reasoning
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

type completionTokensDetails struct {
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
	DraftTokens              int `json:"draft_tokens,omitempty"`
	AcceptedTokens           int `json:"accepted_tokens,omitempty"`
	RejectedTokens           int `json:"rejected_tokens,omitempty"`
}

type chatUsage struct {
	PromptTokens            int                      `json:"prompt_tokens,omitempty"`
	CompletionTokens        int                      `json:"completion_tokens,omitempty"`
	CachedTokens            int                      `json:"cached_tokens,omitempty"`
	DraftTokens             int                      `json:"draft_tokens,omitempty"`
	SpeculativeDraftTokens  int                      `json:"speculative_draft_tokens,omitempty"`
	AcceptedTokens          int                      `json:"accepted_tokens,omitempty"`
	RejectedTokens          int                      `json:"rejected_tokens,omitempty"`
	Source                  string                   `json:"source,omitempty"`
	PromptTokensDetails     *promptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type serverTimings struct {
	PromptN            int     `json:"prompt_n,omitempty"`
	PromptMS           float64 `json:"prompt_ms,omitempty"`
	PromptPerSecond    float64 `json:"prompt_per_second,omitempty"`
	PredictedN         int     `json:"predicted_n,omitempty"`
	PredictedMS        float64 `json:"predicted_ms,omitempty"`
	PredictedPerSecond float64 `json:"predicted_per_second,omitempty"`
	CacheN             int     `json:"cache_n,omitempty"`
	DraftN             int     `json:"draft_n,omitempty"`
	AcceptedN          int     `json:"accepted_n,omitempty"`
	RejectedN          int     `json:"rejected_n,omitempty"`
	Source             string  `json:"source,omitempty"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	Delta        chatMessage `json:"delta"`
	FinishReason string      `json:"finish_reason"`
}

type chatCompletionResponse struct {
	ID             string         `json:"id,omitempty"`
	Model          string         `json:"model,omitempty"`
	Choices        []chatChoice   `json:"choices"`
	Usage          *chatUsage     `json:"usage,omitempty"`
	Timings        *serverTimings `json:"timings,omitempty"`
	Source         string         `json:"source,omitempty"`
	PrefixCacheHit bool           `json:"prefix_cache_hit,omitempty"`
}

func (u *chatUsage) cachedTokensCount() int {
	if u == nil {
		return 0
	}
	c := u.CachedTokens
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > c {
		c = u.PromptTokensDetails.CachedTokens
	}
	return c
}

func (u *chatUsage) draftTokensCount() int {
	if u == nil {
		return 0
	}
	d := maxInt(u.DraftTokens, u.SpeculativeDraftTokens)
	if u.CompletionTokensDetails != nil {
		d = maxInt(d, u.CompletionTokensDetails.DraftTokens)
	}
	return d
}

func (u *chatUsage) acceptedTokensCount() int {
	if u == nil {
		return 0
	}
	a := u.AcceptedTokens
	if u.CompletionTokensDetails != nil {
		a = maxInt(a, u.CompletionTokensDetails.AcceptedTokens)
		a = maxInt(a, u.CompletionTokensDetails.AcceptedPredictionTokens)
	}
	return a
}

func (u *chatUsage) rejectedTokensCount() int {
	if u == nil {
		return 0
	}
	rej := u.RejectedTokens
	if u.CompletionTokensDetails != nil {
		rej = maxInt(rej, u.CompletionTokensDetails.RejectedTokens)
		rej = maxInt(rej, u.CompletionTokensDetails.RejectedPredictionTokens)
	}
	return rej
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (r *sixGateRunner) newRequest(ctx context.Context, method, path string, body io.Reader) (*http.Request, context.CancelFunc, error) {
	reqCtx, cancel := context.WithTimeout(ctx, r.requestTimeout)
	req, err := http.NewRequestWithContext(reqCtx, method, r.endpoint+path, body)
	if err != nil {
		cancel()
		return nil, nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if strings.TrimSpace(r.opts.APIKey) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(r.opts.APIKey))
	}
	return req, cancel, nil
}

func (r *sixGateRunner) runLoadGate(ctx context.Context) (GateResult, map[string]any) {
	started := r.now()
	pollInterval := r.opts.LoadPollInterval
	if pollInterval <= 0 {
		pollInterval = 100 * time.Millisecond
	}

	var deadline time.Time
	polling := r.opts.LoadTimeout > 0
	if polling {
		deadline = started.Add(r.opts.LoadTimeout)
	}

	var attempts int
	var lastListed []string
	var lastStatus int
	var lastErr string

	for {
		attempts++
		listed, status, err := r.fetchModels(ctx)
		if len(listed) > 0 {
			lastListed = listed
		}
		lastStatus = status
		if err == nil {
			for _, id := range listed {
				if id == r.spec.Model {
					dur := r.now().Sub(started).Seconds()
					res := GateResult{
						Gate:            GateLoad,
						Status:          StatusPass,
						DurationSeconds: dur,
						ListedModels:    listed,
					}
					return res, map[string]any{
						"gate":          GateLoad,
						"status":        StatusPass,
						"attempts":      attempts,
						"http_status":   status,
						"target_model":  r.spec.Model,
						"listed_models": listed,
					}
				}
			}
			lastErr = fmt.Sprintf("model %q not in /v1/models list", r.spec.Model)
		} else {
			lastErr = err.Error()
		}

		if errors.Is(ctx.Err(), context.Canceled) {
			dur := r.now().Sub(started).Seconds()
			return GateResult{
					Gate:            GateLoad,
					Status:          StatusFail,
					BlewUp:          true,
					Reason:          ReasonCaptureCancelled,
					DurationSeconds: dur,
					ListedModels:    lastListed,
				}, map[string]any{
					"gate":          GateLoad,
					"status":        StatusFail,
					"blew_up":       true,
					"attempts":      attempts,
					"http_status":   lastStatus,
					"target_model":  r.spec.Model,
					"listed_models": lastListed,
					"error":         lastErr,
				}
		}
		if !polling || !r.now().Before(deadline) {
			break
		}
		timer := time.NewTimer(pollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
		case <-timer.C:
		}
	}

	reason := ReasonLoadFailed
	if lastStatus == http.StatusOK && len(lastListed) >= 0 && strings.Contains(lastErr, "not in /v1/models list") {
		reason = ReasonModelNotListed
	}
	dur := r.now().Sub(started).Seconds()
	return GateResult{
			Gate:            GateLoad,
			Status:          StatusFail,
			BlewUp:          true,
			Reason:          reason,
			DurationSeconds: dur,
			ListedModels:    lastListed,
		}, map[string]any{
			"gate":          GateLoad,
			"status":        StatusFail,
			"blew_up":       true,
			"attempts":      attempts,
			"http_status":   lastStatus,
			"target_model":  r.spec.Model,
			"listed_models": lastListed,
			"error":         lastErr,
		}
}

func (r *sixGateRunner) fetchModels(ctx context.Context) ([]string, int, error) {
	req, cancel, err := r.newRequest(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return nil, 0, err
	}
	defer cancel()

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, resp.StatusCode, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, resp.StatusCode, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var parsed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, resp.StatusCode, fmt.Errorf("decode /v1/models: %w", err)
	}
	ids := make([]string, 0, len(parsed.Data))
	for _, entry := range parsed.Data {
		ids = append(ids, entry.ID)
	}
	return ids, resp.StatusCode, nil
}

func (r *sixGateRunner) doChatCompletion(ctx context.Context, payload chatCompletionRequest) (chatCompletionResponse, http.Header, int, error) {
	rawReq, err := json.Marshal(payload)
	if err != nil {
		return chatCompletionResponse{}, nil, 0, err
	}
	req, cancel, err := r.newRequest(ctx, http.MethodPost, "/v1/chat/completions", bytes.NewReader(rawReq))
	if err != nil {
		return chatCompletionResponse{}, nil, 0, err
	}
	defer cancel()

	resp, err := r.client.Do(req)
	if err != nil {
		return chatCompletionResponse{}, nil, 0, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return chatCompletionResponse{}, resp.Header, resp.StatusCode, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return chatCompletionResponse{}, resp.Header, resp.StatusCode, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var parsed chatCompletionResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return chatCompletionResponse{}, resp.Header, resp.StatusCode, fmt.Errorf("decode chat completion: %w", err)
	}
	return parsed, resp.Header, resp.StatusCode, nil
}

func (r *sixGateRunner) runChatGate(ctx context.Context) (GateResult, map[string]any) {
	started := r.now()
	prompt := strings.TrimSpace(r.spec.ChatPrompt)
	if prompt == "" {
		prompt = "What is the capital of France? Answer concisely."
	}
	reqPayload := chatCompletionRequest{
		Model:    r.spec.Model,
		Messages: []chatMessage{{Role: "user", Content: prompt}},
	}
	resp, _, statusCode, err := r.doChatCompletion(ctx, reqPayload)
	dur := r.now().Sub(started).Seconds()
	if err != nil || len(resp.Choices) == 0 {
		reason := classifyTransportFailure(ctx, err)
		errMsg := ""
		if err != nil {
			errMsg = err.Error()
		} else {
			errMsg = "response contained zero choices"
		}
		return GateResult{
				Gate:            GateChat,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          reason,
				DurationSeconds: dur,
			}, map[string]any{
				"gate":        GateChat,
				"status":      StatusFail,
				"blew_up":     true,
				"http_status": statusCode,
				"uncapped":    true,
				"error":       errMsg,
			}
	}

	choice := resp.Choices[0]
	content := strings.TrimSpace(choice.Message.Content)
	if content == "" {
		return GateResult{
				Gate:            GateChat,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          ReasonEmptyAnswer,
				DurationSeconds: dur,
				FinishReason:    choice.FinishReason,
			}, map[string]any{
				"gate":          GateChat,
				"status":        StatusFail,
				"blew_up":       true,
				"finish_reason": choice.FinishReason,
				"uncapped":      true,
				"error":         "empty answer content",
			}
	}

	if choice.FinishReason != "stop" {
		return GateResult{
				Gate:            GateChat,
				Status:          StatusFail,
				BlewUp:          false,
				Reason:          ReasonFinishReasonNotStop,
				DurationSeconds: dur,
				FinishReason:    choice.FinishReason,
			}, map[string]any{
				"gate":          GateChat,
				"status":        StatusFail,
				"blew_up":       false,
				"finish_reason": choice.FinishReason,
				"uncapped":      true,
				"content_bytes": len(choice.Message.Content),
			}
	}

	return GateResult{
			Gate:            GateChat,
			Status:          StatusPass,
			DurationSeconds: dur,
			FinishReason:    choice.FinishReason,
		}, map[string]any{
			"gate":          GateChat,
			"status":        StatusPass,
			"finish_reason": choice.FinishReason,
			"uncapped":      true,
			"content_bytes": len(choice.Message.Content),
		}
}

func (r *sixGateRunner) runReasoningGate(ctx context.Context) (GateResult, map[string]any) {
	started := r.now()
	prompt := strings.TrimSpace(r.spec.ReasoningPrompt)
	if prompt == "" {
		prompt = "What is 17 * 23? Think carefully and give only the exact result in your final answer."
	}
	reqPayload := chatCompletionRequest{
		Model:    r.spec.Model,
		Messages: []chatMessage{{Role: "user", Content: prompt}},
	}
	resp, _, statusCode, err := r.doChatCompletion(ctx, reqPayload)
	dur := r.now().Sub(started).Seconds()
	if err != nil || len(resp.Choices) == 0 {
		reason := classifyTransportFailure(ctx, err)
		errMsg := "response contained zero choices"
		if err != nil {
			errMsg = err.Error()
		}
		return GateResult{
				Gate:            GateReasoning,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          reason,
				DurationSeconds: dur,
			}, map[string]any{
				"gate":        GateReasoning,
				"status":      StatusFail,
				"blew_up":     true,
				"http_status": statusCode,
				"uncapped":    true,
				"error":       errMsg,
			}
	}

	choice := resp.Choices[0]
	reasoning := strings.TrimSpace(choice.Message.reasoningText())
	content := strings.TrimSpace(choice.Message.Content)
	if content == "" && reasoning == "" && len(choice.Message.ToolCalls) == 0 {
		return GateResult{
				Gate:            GateReasoning,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          ReasonEmptyAnswer,
				DurationSeconds: dur,
				FinishReason:    choice.FinishReason,
			}, map[string]any{
				"gate":          GateReasoning,
				"status":        StatusFail,
				"blew_up":       true,
				"finish_reason": choice.FinishReason,
				"error":         "empty completion",
			}
	}

	intSet := ExtractExactIntegers(choice.Message.Content)
	ints := sortedIntKeys(intSet)
	expected := r.expectedInteger
	separateReasoning := reasoning != ""

	res := GateResult{
		Gate:              GateReasoning,
		DurationSeconds:   dur,
		FinishReason:      choice.FinishReason,
		SeparateReasoning: separateReasoning,
		ReasoningText:     reasoning,
		ExpectedInteger:   &expected,
		ExtractedIntegers: ints,
	}

	switch {
	case !separateReasoning:
		res.Status = StatusFail
		res.Reason = ReasonReasoningFieldMissing
	case !intSet[expected]:
		res.Status = StatusFail
		res.Reason = ReasonReasoningMismatch
	case choice.FinishReason != "" && choice.FinishReason != "stop":
		res.Status = StatusFail
		res.Reason = ReasonFinishReasonNotStop
	default:
		res.Status = StatusPass
	}

	return res, map[string]any{
		"gate":               GateReasoning,
		"status":             res.Status,
		"reason":             res.Reason,
		"finish_reason":      choice.FinishReason,
		"separate_reasoning": separateReasoning,
		"expected_integer":   expected,
		"extracted_integers": ints,
		"content":            choice.Message.Content,
		"uncapped":           true,
	}
}

func sortedIntKeys(m map[int64]bool) []int64 {
	if len(m) == 0 {
		return nil
	}
	keys := make([]int64, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func (r *sixGateRunner) runToolsGate(ctx context.Context) (GateResult, map[string]any) {
	started := r.now()
	toolName := strings.TrimSpace(r.spec.ToolName)
	if toolName == "" {
		toolName = "get_weather"
	}
	argKey := strings.TrimSpace(r.spec.ToolArgKey)
	if argKey == "" {
		argKey = "city"
	}
	expectedArg := strings.TrimSpace(r.spec.ToolExpectedArg)
	if expectedArg == "" {
		expectedArg = "Tokyo"
	}

	prompt := fmt.Sprintf("Call the %s tool for %s, and then include the exact verification_code returned by the tool in your reply.", toolName, expectedArg)
	tools := []chatTool{{
		Type: "function",
		Function: chatToolFunction{
			Name:        toolName,
			Description: "Look up weather conditions and verification token for a city.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					argKey: map[string]any{"type": "string"},
				},
				"required": []string{argKey},
			},
		},
	}}

	turn1Req := chatCompletionRequest{
		Model:    r.spec.Model,
		Messages: []chatMessage{{Role: "user", Content: prompt}},
		Tools:    tools,
	}
	turn1Resp, _, statusCode, err := r.doChatCompletion(ctx, turn1Req)
	if err != nil || len(turn1Resp.Choices) == 0 {
		dur := r.now().Sub(started).Seconds()
		reason := classifyTransportFailure(ctx, err)
		errMsg := "response contained zero choices"
		if err != nil {
			errMsg = err.Error()
		}
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          reason,
				DurationSeconds: dur,
				ToolName:        toolName,
			}, map[string]any{
				"gate":        GateTools,
				"turn":        1,
				"status":      StatusFail,
				"blew_up":     true,
				"http_status": statusCode,
				"error":       errMsg,
			}
	}

	turn1Msg := turn1Resp.Choices[0].Message
	if strings.TrimSpace(turn1Msg.Content) == "" && strings.TrimSpace(turn1Msg.reasoningText()) == "" && len(turn1Msg.ToolCalls) == 0 {
		dur := r.now().Sub(started).Seconds()
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          ReasonEmptyAnswer,
				DurationSeconds: dur,
				ToolName:        toolName,
			}, map[string]any{
				"gate":    GateTools,
				"turn":    1,
				"status":  StatusFail,
				"blew_up": true,
				"error":   "empty turn 1 completion",
			}
	}

	var matchedCall *chatToolCall
	for i := range turn1Msg.ToolCalls {
		if turn1Msg.ToolCalls[i].Function.Name == toolName {
			matchedCall = &turn1Msg.ToolCalls[i]
			break
		}
	}
	if matchedCall == nil {
		dur := r.now().Sub(started).Seconds()
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				Reason:          ReasonToolNotCalled,
				DurationSeconds: dur,
				ToolName:        toolName,
			}, map[string]any{
				"gate":       GateTools,
				"turn":       1,
				"status":     StatusFail,
				"reason":     ReasonToolNotCalled,
				"tool_calls": turn1Msg.ToolCalls,
				"content":    turn1Msg.Content,
			}
	}

	var argsMap map[string]any
	if err := json.Unmarshal([]byte(matchedCall.Function.Arguments), &argsMap); err != nil {
		dur := r.now().Sub(started).Seconds()
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				Reason:          ReasonToolArgMismatch,
				DurationSeconds: dur,
				ToolName:        toolName,
				ToolArgs:        matchedCall.Function.Arguments,
			}, map[string]any{
				"gate":      GateTools,
				"turn":      1,
				"status":    StatusFail,
				"reason":    ReasonToolArgMismatch,
				"tool_args": matchedCall.Function.Arguments,
				"error":     err.Error(),
			}
	}
	gotArg, ok := argsMap[argKey].(string)
	if !ok || !strings.EqualFold(strings.TrimSpace(gotArg), expectedArg) {
		dur := r.now().Sub(started).Seconds()
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				Reason:          ReasonToolArgMismatch,
				DurationSeconds: dur,
				ToolName:        toolName,
				ToolArgs:        matchedCall.Function.Arguments,
			}, map[string]any{
				"gate":         GateTools,
				"turn":         1,
				"status":       StatusFail,
				"reason":       ReasonToolArgMismatch,
				"tool_args":    matchedCall.Function.Arguments,
				"expected_arg": expectedArg,
			}
	}

	nonce, err := r.nonce()
	if err != nil {
		dur := r.now().Sub(started).Seconds()
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          ReasonRequestFailed,
				DurationSeconds: dur,
				ToolName:        toolName,
			}, map[string]any{
				"gate":   GateTools,
				"status": StatusFail,
				"error":  fmt.Sprintf("generate nonce: %v", err),
			}
	}

	toolResultBytes, _ := json.Marshal(map[string]any{
		argKey:              gotArg,
		"condition":         "clear",
		"temperature_c":     21,
		"verification_code": nonce,
	})
	turn2Messages := []chatMessage{
		{Role: "user", Content: prompt},
		{Role: "assistant", Content: turn1Msg.Content, ToolCalls: turn1Msg.ToolCalls},
		{Role: "tool", ToolCallID: matchedCall.ID, Name: toolName, Content: string(toolResultBytes)},
	}
	turn2Req := chatCompletionRequest{
		Model:    r.spec.Model,
		Messages: turn2Messages,
		Tools:    tools,
	}
	turn2Resp, _, turn2Status, err := r.doChatCompletion(ctx, turn2Req)
	dur := r.now().Sub(started).Seconds()
	if err != nil || len(turn2Resp.Choices) == 0 {
		reason := classifyTransportFailure(ctx, err)
		errMsg := "turn 2 response contained zero choices"
		if err != nil {
			errMsg = err.Error()
		}
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          reason,
				DurationSeconds: dur,
				ToolName:        toolName,
				ToolArgs:        matchedCall.Function.Arguments,
				ToolNonce:       nonce,
			}, map[string]any{
				"gate":        GateTools,
				"turn":        2,
				"status":      StatusFail,
				"blew_up":     true,
				"http_status": turn2Status,
				"tool_nonce":  nonce,
				"error":       errMsg,
			}
	}

	turn2Choice := turn2Resp.Choices[0]
	turn2Content := strings.TrimSpace(turn2Choice.Message.Content)
	if turn2Content == "" && strings.TrimSpace(turn2Choice.Message.reasoningText()) == "" {
		return GateResult{
				Gate:            GateTools,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          ReasonEmptyAnswer,
				DurationSeconds: dur,
				ToolName:        toolName,
				ToolArgs:        matchedCall.Function.Arguments,
				ToolNonce:       nonce,
				FinishReason:    turn2Choice.FinishReason,
			}, map[string]any{
				"gate":       GateTools,
				"turn":       2,
				"status":     StatusFail,
				"blew_up":    true,
				"tool_nonce": nonce,
				"error":      "empty turn 2 completion",
			}
	}

	nonceMatched := strings.Contains(strings.ToLower(turn2Choice.Message.Content), strings.ToLower(nonce))
	res := GateResult{
		Gate:            GateTools,
		DurationSeconds: dur,
		FinishReason:    turn2Choice.FinishReason,
		ToolName:        toolName,
		ToolArgs:        matchedCall.Function.Arguments,
		ToolNonce:       nonce,
		NonceMatched:    nonceMatched,
	}
	switch {
	case !nonceMatched:
		res.Status = StatusFail
		res.Reason = ReasonToolNonceMissing
	case turn2Choice.FinishReason != "" && turn2Choice.FinishReason != "stop":
		res.Status = StatusFail
		res.Reason = ReasonFinishReasonNotStop
	default:
		res.Status = StatusPass
	}

	return res, map[string]any{
		"gate":          GateTools,
		"status":        res.Status,
		"reason":        res.Reason,
		"tool_name":     toolName,
		"tool_args":     matchedCall.Function.Arguments,
		"tool_nonce":    nonce,
		"nonce_matched": nonceMatched,
		"finish_reason": turn2Choice.FinishReason,
		"reply":         turn2Choice.Message.Content,
		"uncapped":      true,
	}
}

func (r *sixGateRunner) runContextGate(ctx context.Context) (GateResult, map[string]any) {
	started := r.now()
	nonce, err := r.nonce()
	if err != nil {
		return GateResult{
				Gate:            GateContext,
				Status:          StatusFail,
				BlewUp:          true,
				Reason:          ReasonRequestFailed,
				DurationSeconds: r.now().Sub(started).Seconds(),
			}, map[string]any{
				"gate":   GateContext,
				"status": StatusFail,
				"error":  fmt.Sprintf("generate access code: %v", err),
			}
	}
	accessCode := "CODE-" + nonce
	targetPromptTokens := maxInt(1, int(math.Round(DefaultContextDepthRatio*float64(r.spec.ContextWindow))))
	minTokens := minContextPromptTokens(r.spec.ContextWindow)
	prompt := buildContextPrompt(targetPromptTokens, accessCode)

	reqPayload := chatCompletionRequest{
		Model:    r.spec.Model,
		Messages: []chatMessage{{Role: "user", Content: prompt}},
	}

	var dropSeconds []float64
	var lastErr error
	var lastStatus int

	for attempt := 0; attempt <= r.contextRetries; attempt++ {
		attemptStart := r.now()
		resp, hdr, statusCode, callErr := r.doChatCompletion(ctx, reqPayload)
		attemptElapsed := r.now().Sub(attemptStart)
		lastStatus = statusCode

		if callErr != nil {
			lastErr = callErr
			if ctx.Err() == nil && isRetryableConnectionDrop(callErr, statusCode) {
				dropSeconds = append(dropSeconds, attemptElapsed.Seconds())
				if attempt < r.contextRetries {
					continue
				}
			}
			dur := r.now().Sub(started).Seconds()
			return GateResult{
					Gate:               GateContext,
					Status:             StatusFail,
					BlewUp:             true,
					Reason:             classifyTransportFailure(ctx, callErr),
					DurationSeconds:    dur,
					TargetPromptTokens: targetPromptTokens,
					MinPromptTokens:    minTokens,
					DropSeconds:        dropSeconds,
				}, map[string]any{
					"gate":                 GateContext,
					"status":               StatusFail,
					"blew_up":              true,
					"http_status":          lastStatus,
					"attempts":             attempt + 1,
					"drop_seconds":         dropSeconds,
					"target_prompt_tokens": targetPromptTokens,
					"min_prompt_tokens":    minTokens,
					"error":                callErr.Error(),
				}
		}

		if len(resp.Choices) == 0 {
			dur := r.now().Sub(started).Seconds()
			return GateResult{
					Gate:               GateContext,
					Status:             StatusFail,
					BlewUp:             true,
					Reason:             ReasonRequestFailed,
					DurationSeconds:    dur,
					TargetPromptTokens: targetPromptTokens,
					MinPromptTokens:    minTokens,
					DropSeconds:        dropSeconds,
				}, map[string]any{
					"gate":         GateContext,
					"status":       StatusFail,
					"blew_up":      true,
					"http_status":  statusCode,
					"attempts":     attempt + 1,
					"drop_seconds": dropSeconds,
					"error":        "response contained zero choices",
				}
		}

		choice := resp.Choices[0]
		if strings.TrimSpace(choice.Message.Content) == "" && strings.TrimSpace(choice.Message.reasoningText()) == "" {
			dur := r.now().Sub(started).Seconds()
			return GateResult{
					Gate:               GateContext,
					Status:             StatusFail,
					BlewUp:             true,
					Reason:             ReasonEmptyAnswer,
					DurationSeconds:    dur,
					TargetPromptTokens: targetPromptTokens,
					MinPromptTokens:    minTokens,
					DropSeconds:        dropSeconds,
					FinishReason:       choice.FinishReason,
				}, map[string]any{
					"gate":         GateContext,
					"status":       StatusFail,
					"blew_up":      true,
					"attempts":     attempt + 1,
					"drop_seconds": dropSeconds,
					"error":        "empty answer content",
				}
		}

		promptTokens := 0
		cachedTokens := 0
		if resp.Usage != nil {
			promptTokens = resp.Usage.PromptTokens
			cachedTokens = resp.Usage.cachedTokensCount()
		}
		if resp.Timings != nil {
			if promptTokens == 0 && resp.Timings.PromptN > 0 {
				promptTokens = resp.Timings.PromptN
			}
			if resp.Timings.CacheN > cachedTokens {
				cachedTokens = resp.Timings.CacheN
			}
		}

		prefixCached := cachedTokens > 0 ||
			resp.PrefixCacheHit ||
			strings.EqualFold(hdr.Get("X-Prefix-Cache-Hit"), "true") ||
			strings.EqualFold(hdr.Get("X-Cache"), "HIT")

		var prefillPtr *float64
		var prefillSuppressed bool
		var suppressionReason string

		if prefixCached {
			prefillSuppressed = true
			if len(dropSeconds) > 0 {
				suppressionReason = "prefix_cache_retry"
			} else {
				suppressionReason = "prefix_cache"
			}
		} else {
			var rate float64
			switch {
			case resp.Timings != nil && resp.Timings.PromptPerSecond > 0:
				rate = resp.Timings.PromptPerSecond
			case resp.Timings != nil && resp.Timings.PromptMS > 0 && promptTokens > 0:
				rate = float64(promptTokens) / (resp.Timings.PromptMS / 1000.0)
			case attemptElapsed > 0 && promptTokens > 0:
				rate = float64(promptTokens) / attemptElapsed.Seconds()
			}
			if rate > 0 {
				v := rate
				prefillPtr = &v
			}
		}

		codeRecovered := strings.Contains(strings.ToLower(choice.Message.Content), strings.ToLower(accessCode))
		dur := r.now().Sub(started).Seconds()

		res := GateResult{
			Gate:                     GateContext,
			DurationSeconds:          dur,
			FinishReason:             choice.FinishReason,
			TargetPromptTokens:       targetPromptTokens,
			MinPromptTokens:          minTokens,
			PromptTokens:             promptTokens,
			CodeRecovered:            codeRecovered,
			DropSeconds:              dropSeconds,
			CachedTokens:             cachedTokens,
			PrefixCached:             prefixCached,
			PrefillTokensPerSecond:   prefillPtr,
			PrefillSuppressed:        prefillSuppressed,
			PrefillSuppressionReason: suppressionReason,
		}

		switch {
		case !codeRecovered:
			res.Status = StatusFail
			res.Reason = ReasonContextCodeMissing
		case promptTokens < minTokens:
			res.Status = StatusFail
			res.Reason = ReasonContextTokensLow
		case choice.FinishReason != "" && choice.FinishReason != "stop":
			res.Status = StatusFail
			res.Reason = ReasonFinishReasonNotStop
		default:
			res.Status = StatusPass
		}

		ev := map[string]any{
			"gate":                       GateContext,
			"status":                     res.Status,
			"reason":                     res.Reason,
			"attempts":                   attempt + 1,
			"target_prompt_tokens":       targetPromptTokens,
			"min_prompt_tokens":          minTokens,
			"prompt_tokens":              promptTokens,
			"code_recovered":             codeRecovered,
			"drop_seconds":               dropSeconds,
			"cached_tokens":              cachedTokens,
			"prefix_cached":              prefixCached,
			"prefill_suppressed":         prefillSuppressed,
			"prefill_suppression_reason": suppressionReason,
			"finish_reason":              choice.FinishReason,
			"uncapped":                   true,
		}
		if prefillPtr != nil {
			ev["prefill"] = *prefillPtr
		}
		return res, ev
	}

	dur := r.now().Sub(started).Seconds()
	errText := ""
	if lastErr != nil {
		errText = lastErr.Error()
	}
	return GateResult{
			Gate:               GateContext,
			Status:             StatusFail,
			BlewUp:             true,
			Reason:             ReasonRequestFailed,
			DurationSeconds:    dur,
			TargetPromptTokens: targetPromptTokens,
			MinPromptTokens:    minTokens,
			DropSeconds:        dropSeconds,
		}, map[string]any{
			"gate":         GateContext,
			"status":       StatusFail,
			"blew_up":      true,
			"drop_seconds": dropSeconds,
			"error":        errText,
		}
}

func buildContextPrompt(targetTokens int, accessCode string) string {
	// Target ~3.5 bytes/token across English subword tokenizers so the planted
	// prompt lands near 85% of context_window and safely clears the 0.6x floor.
	const chunkWords = "the quick brown fox jumps over the lazy dog near the quiet river "
	targetBytes := targetTokens * 35 / 10
	var b strings.Builder
	b.Grow(targetBytes + 128)
	for index := 1; b.Len() < targetBytes; index++ {
		fmt.Fprintf(&b, "Line %04d: %s", index, chunkWords)
	}
	fmt.Fprintf(&b, "\nAccess code: %s\nRepeat only the exact access code from above.", accessCode)
	return b.String()
}

func isRetryableConnectionDrop(err error, statusCode int) bool {
	if statusCode == http.StatusBadGateway || statusCode == http.StatusServiceUnavailable || statusCode == http.StatusGatewayTimeout {
		return true
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr)
}

func (r *sixGateRunner) runSpeedGate(ctx context.Context) (GateResult, map[string]any) {
	started := r.now()
	reqPayload := chatCompletionRequest{
		Model: r.spec.Model,
		Messages: []chatMessage{{
			Role:    "user",
			Content: "Write a detailed technical explanation of how memory bandwidth bounds autoregressive LLM decode.",
		}},
		Stream:        true,
		StreamOptions: &streamOptions{IncludeUsage: true},
	}

	rawReq, err := json.Marshal(reqPayload)
	if err != nil {
		return GateResult{
				Gate:                 GateSpeed,
				Status:               StatusFail,
				BlewUp:               true,
				Reason:               ReasonRequestFailed,
				FloorTokensPerSecond: r.floorTPS,
			}, map[string]any{
				"gate":   GateSpeed,
				"status": StatusFail,
				"error":  err.Error(),
			}
	}

	req, cancel, err := r.newRequest(ctx, http.MethodPost, "/v1/chat/completions", bytes.NewReader(rawReq))
	if err != nil {
		return GateResult{
				Gate:                 GateSpeed,
				Status:               StatusFail,
				BlewUp:               true,
				Reason:               ReasonRequestFailed,
				FloorTokensPerSecond: r.floorTPS,
			}, map[string]any{
				"gate":   GateSpeed,
				"status": StatusFail,
				"error":  err.Error(),
			}
	}
	defer cancel()

	resp, err := r.client.Do(req)
	if err != nil {
		dur := r.now().Sub(started).Seconds()
		return GateResult{
				Gate:                 GateSpeed,
				Status:               StatusFail,
				BlewUp:               true,
				Reason:               classifyTransportFailure(ctx, err),
				DurationSeconds:      dur,
				FloorTokensPerSecond: r.floorTPS,
			}, map[string]any{
				"gate":    GateSpeed,
				"status":  StatusFail,
				"blew_up": true,
				"error":   err.Error(),
			}
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		dur := r.now().Sub(started).Seconds()
		return GateResult{
				Gate:                 GateSpeed,
				Status:               StatusFail,
				BlewUp:               true,
				Reason:               ReasonRequestFailed,
				DurationSeconds:      dur,
				FloorTokensPerSecond: r.floorTPS,
			}, map[string]any{
				"gate":        GateSpeed,
				"status":      StatusFail,
				"blew_up":     true,
				"http_status": resp.StatusCode,
				"error":       strings.TrimSpace(string(body)),
			}
	}

	var (
		content        strings.Builder
		reasoning      strings.Builder
		finishReason   string
		usage          *chatUsage
		timings        *serverTimings
		explicitSource string
		firstTokenSeen bool
		firstTokenAt   time.Time
		lastTokenAt    time.Time
		windowCutoff   bool
	)

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if strings.Contains(contentType, "text/event-stream") {
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "[DONE]" {
				break
			}
			var chunk chatCompletionResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				dur := r.now().Sub(started).Seconds()
				return GateResult{
						Gate:                 GateSpeed,
						Status:               StatusFail,
						BlewUp:               true,
						Reason:               ReasonRequestFailed,
						DurationSeconds:      dur,
						FloorTokensPerSecond: r.floorTPS,
					}, map[string]any{
						"gate":    GateSpeed,
						"status":  StatusFail,
						"blew_up": true,
						"error":   fmt.Sprintf("decode SSE chunk: %v", err),
					}
			}
			if chunk.Source != "" {
				explicitSource = chunk.Source
			}
			if chunk.Usage != nil {
				usage = chunk.Usage
				if chunk.Usage.Source != "" {
					explicitSource = chunk.Usage.Source
				}
			}
			if chunk.Timings != nil {
				timings = chunk.Timings
				if chunk.Timings.Source != "" {
					explicitSource = chunk.Timings.Source
				}
			}
			for _, c := range chunk.Choices {
				if c.Delta.Content != "" || c.Delta.reasoningText() != "" || c.Message.Content != "" || c.Message.reasoningText() != "" {
					now := r.now()
					if !firstTokenSeen {
						firstTokenSeen = true
						firstTokenAt = now
					}
					lastTokenAt = now
					content.WriteString(c.Delta.Content)
					content.WriteString(c.Message.Content)
					reasoning.WriteString(c.Delta.reasoningText())
					reasoning.WriteString(c.Message.reasoningText())
				}
				if c.FinishReason != "" {
					finishReason = c.FinishReason
				}
			}
			if firstTokenSeen && r.speedWindow > 0 && !lastTokenAt.Before(firstTokenAt.Add(r.speedWindow)) {
				windowCutoff = true
				break
			}
		}
		if err := scanner.Err(); err != nil && !windowCutoff {
			dur := r.now().Sub(started).Seconds()
			return GateResult{
					Gate:                 GateSpeed,
					Status:               StatusFail,
					BlewUp:               true,
					Reason:               classifyTransportFailure(ctx, err),
					DurationSeconds:      dur,
					FloorTokensPerSecond: r.floorTPS,
				}, map[string]any{
					"gate":    GateSpeed,
					"status":  StatusFail,
					"blew_up": true,
					"error":   err.Error(),
				}
		}
	} else {
		body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		if err != nil {
			dur := r.now().Sub(started).Seconds()
			return GateResult{
					Gate:                 GateSpeed,
					Status:               StatusFail,
					BlewUp:               true,
					Reason:               classifyTransportFailure(ctx, err),
					DurationSeconds:      dur,
					FloorTokensPerSecond: r.floorTPS,
				}, map[string]any{
					"gate":    GateSpeed,
					"status":  StatusFail,
					"blew_up": true,
					"error":   err.Error(),
				}
		}
		var parsed chatCompletionResponse
		if err := json.Unmarshal(body, &parsed); err != nil {
			dur := r.now().Sub(started).Seconds()
			return GateResult{
					Gate:                 GateSpeed,
					Status:               StatusFail,
					BlewUp:               true,
					Reason:               ReasonRequestFailed,
					DurationSeconds:      dur,
					FloorTokensPerSecond: r.floorTPS,
				}, map[string]any{
					"gate":    GateSpeed,
					"status":  StatusFail,
					"blew_up": true,
					"error":   err.Error(),
				}
		}
		explicitSource = parsed.Source
		usage = parsed.Usage
		if usage != nil && usage.Source != "" {
			explicitSource = usage.Source
		}
		timings = parsed.Timings
		if timings != nil && timings.Source != "" {
			explicitSource = timings.Source
		}
		for _, c := range parsed.Choices {
			content.WriteString(c.Message.Content)
			reasoning.WriteString(c.Message.reasoningText())
			if c.FinishReason != "" {
				finishReason = c.FinishReason
			}
		}
	}

	streamEnd := r.now()
	dur := streamEnd.Sub(started).Seconds()

	if strings.TrimSpace(content.String()) == "" && strings.TrimSpace(reasoning.String()) == "" && usage == nil && timings == nil {
		return GateResult{
				Gate:                 GateSpeed,
				Status:               StatusFail,
				BlewUp:               true,
				Reason:               ReasonEmptyAnswer,
				DurationSeconds:      dur,
				FloorTokensPerSecond: r.floorTPS,
			}, map[string]any{
				"gate":    GateSpeed,
				"status":  StatusFail,
				"blew_up": true,
				"error":   "empty speed completion",
			}
	}

	serverCompletion := 0
	if usage != nil {
		serverCompletion = usage.CompletionTokens
	}
	if timings != nil && timings.PredictedN > serverCompletion {
		serverCompletion = timings.PredictedN
	}

	draftTokens := usage.draftTokensCount()
	acceptedTokens := usage.acceptedTokensCount()
	rejectedTokens := usage.rejectedTokensCount()
	if timings != nil {
		draftTokens = maxInt(draftTokens, timings.DraftN)
		acceptedTokens = maxInt(acceptedTokens, timings.AcceptedN)
		rejectedTokens = maxInt(rejectedTokens, timings.RejectedN)
	}

	source := strings.ToLower(strings.TrimSpace(explicitSource))
	if source == "" {
		if serverCompletion > 0 || acceptedTokens > 0 {
			source = SourceTokenizer
		} else {
			source = SourceChars
		}
	}

	countedTokens := serverCompletion
	draftsUnverified := false
	if draftTokens > 0 || rejectedTokens > 0 {
		if acceptedTokens <= 0 {
			draftsUnverified = true
			countedTokens = 0
		} else if serverCompletion == 0 ||
			serverCompletion == draftTokens ||
			(rejectedTokens > 0 && serverCompletion == acceptedTokens+rejectedTokens) ||
			(rejectedTokens == 0 && serverCompletion > acceptedTokens) ||
			serverCompletion < acceptedTokens {
			// Never count unaccepted speculative drafts toward decode tok/s.
			countedTokens = acceptedTokens
		}
	}

	var decodeDuration time.Duration
	if timings != nil && timings.PredictedMS > 0 {
		decodeDuration = time.Duration(timings.PredictedMS * float64(time.Millisecond))
	} else if firstTokenSeen {
		decodeDuration = lastTokenAt.Sub(firstTokenAt)
		if decodeDuration <= 0 {
			decodeDuration = streamEnd.Sub(firstTokenAt)
		}
		if decodeDuration <= 0 {
			decodeDuration = streamEnd.Sub(started)
		}
	} else {
		decodeDuration = streamEnd.Sub(started)
	}

	var decodeTPS float64
	if decodeDuration > 0 && countedTokens > 0 {
		decodeTPS = float64(countedTokens) / decodeDuration.Seconds()
	}

	res := GateResult{
		Gate:                  GateSpeed,
		DurationSeconds:       dur,
		FinishReason:          finishReason,
		DecodeTokensPerSecond: decodeTPS,
		FloorTokensPerSecond:  r.floorTPS,
		Source:                source,
		DraftTokens:           draftTokens,
		AcceptedTokens:        acceptedTokens,
		RejectedTokens:        rejectedTokens,
		CountedTokens:         countedTokens,
	}

	switch {
	case source != SourceTokenizer:
		res.Status = StatusFail
		res.Reason = ReasonSpeedSourceInvalid
	case draftsUnverified:
		res.Status = StatusFail
		res.Reason = ReasonSpeedDraftsCounted
	case decodeDuration <= 0:
		res.Status = StatusFail
		res.Reason = ReasonSpeedZeroDuration
	case !windowCutoff && finishReason != "" && finishReason != "stop":
		res.Status = StatusFail
		res.Reason = ReasonFinishReasonNotStop
	case decodeTPS < r.floorTPS:
		res.Status = StatusFail
		res.Reason = ReasonSpeedBelowFloor
	default:
		res.Status = StatusPass
	}

	return res, map[string]any{
		"gate":                     GateSpeed,
		"status":                   res.Status,
		"reason":                   res.Reason,
		"source":                   source,
		"decode_tokens_per_second": decodeTPS,
		"floor_tokens_per_second":  r.floorTPS,
		"decode_duration_seconds":  decodeDuration.Seconds(),
		"server_completion_tokens": serverCompletion,
		"draft_tokens":             draftTokens,
		"accepted_tokens":          acceptedTokens,
		"rejected_tokens":          rejectedTokens,
		"counted_tokens":           countedTokens,
		"window_cutoff":            windowCutoff,
		"finish_reason":            finishReason,
		"uncapped":                 true,
	}
}

func classifyTransportFailure(ctx context.Context, err error) string {
	if errors.Is(ctx.Err(), context.Canceled) || errors.Is(err, context.Canceled) {
		return ReasonCaptureCancelled
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
		return ReasonCommandTimeout
	}
	return ReasonRequestFailed
}

func persistGateEvidence(res *GateResult, payload map[string]any, index int, evidenceDir string) error {
	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("localappcert: marshal evidence for gate %q: %w", res.Gate, err)
	}
	sum := sha256.Sum256(b)
	res.EvidenceSHA256 = hex.EncodeToString(sum[:])
	res.EvidenceBytes = int64(len(b))
	if evidenceDir != "" {
		path := filepath.Join(evidenceDir, fmt.Sprintf("%02d-%s.json", index+1, res.Gate))
		if err := os.WriteFile(path, b, 0o644); err != nil {
			return fmt.Errorf("localappcert: write evidence for gate %q: %w", res.Gate, err)
		}
		res.Evidence = path
	}
	return nil
}

func finalizeReportEvidence(report *SixGateReport, evidenceDir string) error {
	logBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("localappcert: marshal six-gate report: %w", err)
	}
	sum := sha256.Sum256(logBytes)
	digest := hex.EncodeToString(sum[:])
	report.Proof.LogSHA256 = "sha256:" + digest
	report.EvidenceSHA256 = digest
	report.EvidenceBytes = int64(len(logBytes))
	if evidenceDir != "" {
		finalBytes, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			return fmt.Errorf("localappcert: marshal six-gate report: %w", err)
		}
		path := filepath.Join(evidenceDir, "six-gate-report.json")
		if err := os.WriteFile(path, finalBytes, 0o644); err != nil {
			return fmt.Errorf("localappcert: write six-gate report: %w", err)
		}
		report.Evidence = path
	}
	return nil
}
