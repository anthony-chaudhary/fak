package apihostprobe

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	ReadinessSchema  = "fak.api-host-readiness.v1"
	AcceptanceSchema = "fak.api-host-acceptance.v1"
	defaultVersion   = "dev"
)

var (
	defaultReadinessTargets = []ReadinessTarget{
		{
			Name:      "gemini_openai_compatible",
			BaseURL:   "https://generativelanguage.googleapis.com/v1beta/openai",
			APIKeyEnv: "GEMINI_API_KEY",
			ModelHint: "gemini-2.5-flash",
		},
		{
			Name:      "glama_gateway",
			BaseURL:   "https://gateway.glama.ai/v1",
			APIKeyEnv: "GLAMA_API_KEY",
			ModelHint: "openai/gpt-4.1-nano-2025-04-14",
		},
		{
			Name:      "pollinations_no_key",
			BaseURL:   "https://gen.pollinations.ai/v1",
			APIKeyEnv: "",
			ModelHint: "openai-fast",
		},
	}
	defaultAcceptanceTargets = []AcceptanceTarget{
		{
			Name:      "gemini_openai_compatible",
			Provider:  "openai-compatible",
			BaseURL:   "https://generativelanguage.googleapis.com/v1beta/openai",
			APIKeyEnv: "GEMINI_API_KEY",
			ModelHint: "gemini-2.5-flash",
		},
		{
			Name:      "glama_gateway",
			Provider:  "openai-compatible",
			BaseURL:   "https://gateway.glama.ai/v1",
			APIKeyEnv: "GLAMA_API_KEY",
			ModelHint: "openai/gpt-4.1-nano-2025-04-14",
		},
		{
			Name:      "pollinations_no_key",
			Provider:  "openai-compatible",
			BaseURL:   "https://gen.pollinations.ai/v1",
			APIKeyEnv: "",
			ModelHint: "openai-fast",
		},
	}
	openAICompatibleClasses = map[string]bool{
		"openai_compatible_upstream": true,
	}
	knownReadinessStatuses = map[string]bool{
		"MODELS_CONFIRMED":    true,
		"HTTP_OK_NO_MODELS":   true,
		"AUTH_ENV_MISSING":    true,
		"AUTH_REQUIRED":       true,
		"ACCESS_DENIED":       true,
		"BILLING_REQUIRED":    true,
		"RATE_LIMITED":        true,
		"TRANSIENT_TRANSPORT": true,
		"HTTP_STATUS":         true,
		"INVALID_TARGET":      true,
	}
	openAICompatibleProviders = map[string]bool{
		"openai-compatible": true,
		"openai":            true,
		"xai":               true,
	}
	nativeProviders = map[string]bool{
		"anthropic": true,
		"gemini":    true,
	}
	directProviders = map[string]bool{
		"direct-http": true,
		"direct-mcp":  true,
	}
	externalBlockers = map[string]bool{
		"NEEDS_AUTH_ENV":      true,
		"AUTH_REQUIRED":       true,
		"ACCESS_DENIED":       true,
		"BILLING_REQUIRED":    true,
		"RATE_LIMITED":        true,
		"TRANSIENT_TRANSPORT": true,
	}
)

type ReadinessTarget struct {
	Name      string `json:"name"`
	BaseURL   string `json:"base_url"`
	APIKeyEnv string `json:"api_key_env"`
	ModelHint string `json:"model_hint"`
}

type AcceptanceTarget struct {
	Name      string `json:"name"`
	Provider  string `json:"provider"`
	BaseURL   string `json:"base_url"`
	APIKeyEnv string `json:"api_key_env"`
	ModelHint string `json:"model_hint"`
}

type ReadinessProbe struct {
	Version     string   `json:"version"`
	Name        string   `json:"name"`
	BaseURL     string   `json:"base_url"`
	APIKeyEnv   string   `json:"api_key_env"`
	ModelHint   string   `json:"model_hint"`
	URL         string   `json:"url"`
	Status      string   `json:"status"`
	HTTPStatus  *int     `json:"http_status"`
	Models      []string `json:"models"`
	BodyExcerpt string   `json:"body_excerpt"`
	Error       string   `json:"error"`
}

type ReadinessSummary struct {
	Targets            int  `json:"targets"`
	ModelsConfirmed    int  `json:"models_confirmed"`
	AuthEnvMissing     int  `json:"auth_env_missing"`
	AuthRequired       int  `json:"auth_required"`
	AccessDenied       int  `json:"access_denied"`
	BillingRequired    int  `json:"billing_required"`
	TransientTransport int  `json:"transient_transport"`
	InvalidTargets     int  `json:"invalid_targets"`
	Unclassified       int  `json:"unclassified"`
	ReadinessGate      bool `json:"readiness_gate"`
}

type ReadinessReport struct {
	Schema      string           `json:"schema"`
	AppVersion  string           `json:"app_version"`
	GeneratedAt string           `json:"generated_at"`
	ScopeNote   string           `json:"scope_note"`
	Summary     ReadinessSummary `json:"summary"`
	Probes      []ReadinessProbe `json:"probes"`
}

type AcceptanceRow struct {
	Version         string          `json:"version"`
	Name            string          `json:"name"`
	Provider        string          `json:"provider"`
	BaseURL         string          `json:"base_url"`
	APIKeyEnv       string          `json:"api_key_env"`
	ModelHint       string          `json:"model_hint"`
	ContractClass   string          `json:"contract_class"`
	Status          string          `json:"status"`
	Reason          string          `json:"reason"`
	ReadinessStatus string          `json:"readiness_status"`
	Probe           *ReadinessProbe `json:"probe"`
	LatestSweep     map[string]any  `json:"latest_sweep"`
	NextLiveCommand string          `json:"next_live_command"`
}

type ArtifactError struct {
	Path     string `json:"path"`
	RowIndex *int   `json:"row_index,omitempty"`
	Error    string `json:"error"`
}

type AcceptanceSummary struct {
	Targets               int  `json:"targets"`
	KnownStatuses         int  `json:"known_statuses"`
	ReadyForLiveBridgeRun int  `json:"ready_for_live_bridge_run"`
	LiveBridgeConfirmed   int  `json:"live_bridge_confirmed"`
	WireSupportedUnprobed int  `json:"wire_supported_unprobed"`
	TypedExternalBlockers int  `json:"typed_external_blockers"`
	UnsupportedWire       int  `json:"unsupported_wire"`
	InvalidTargets        int  `json:"invalid_targets"`
	Unclassified          int  `json:"unclassified"`
	SweepArtifactErrors   int  `json:"sweep_artifact_errors"`
	AcceptanceGate        bool `json:"acceptance_gate"`
}

type AcceptanceReport struct {
	Schema         string            `json:"schema"`
	AppVersion     string            `json:"app_version"`
	GeneratedAt    string            `json:"generated_at"`
	ScopeNote      string            `json:"scope_note"`
	Summary        AcceptanceSummary `json:"summary"`
	Targets        []AcceptanceRow   `json:"targets"`
	ArtifactErrors []ArtifactError   `json:"artifact_errors"`
}

type ReadinessOptions struct {
	Timeout          time.Duration
	ProbeMissingAuth bool
	AppVersion       string
}

type AcceptanceOptions struct {
	Timeout          time.Duration
	ProbeMissingAuth bool
	Root             string
	AppVersion       string
}

func DefaultReadinessTargets() []ReadinessTarget {
	return append([]ReadinessTarget(nil), defaultReadinessTargets...)
}

func DefaultAcceptanceTargets() []AcceptanceTarget {
	return append([]AcceptanceTarget(nil), defaultAcceptanceTargets...)
}

func AppVersion(start string) string {
	if v := strings.TrimSpace(os.Getenv("FAK_APP_VERSION")); v != "" {
		return v
	}
	root := repoRoot(start)
	if root == "" {
		return defaultVersion
	}
	raw, err := os.ReadFile(filepath.Join(root, "VERSION"))
	if err != nil {
		return defaultVersion
	}
	version := strings.TrimSpace(string(raw))
	for _, line := range strings.Split(version, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.HasPrefix(line, "<<<<<<<") || strings.HasPrefix(line, "=======") || strings.HasPrefix(line, ">>>>>>>") {
			return defaultVersion
		}
	}
	if version == "" {
		return defaultVersion
	}
	return version
}

func ParseReadinessTarget(spec string) (ReadinessTarget, error) {
	parts := splitSpec(spec)
	if len(parts) < 2 || len(parts) > 4 {
		return ReadinessTarget{}, fmt.Errorf("target must be name|base_url[|api_key_env[|model_hint]]")
	}
	target := ReadinessTarget{
		Name:    parts[0],
		BaseURL: parts[1],
	}
	if len(parts) >= 3 {
		target.APIKeyEnv = parts[2]
	}
	if len(parts) >= 4 {
		target.ModelHint = parts[3]
	}
	if err := ReadinessTargetError(target); err != "" {
		return ReadinessTarget{}, fmt.Errorf("%s", err)
	}
	return target, nil
}

func ParseAcceptanceTarget(spec string) (AcceptanceTarget, error) {
	parts := splitSpec(spec)
	if len(parts) < 3 || len(parts) > 5 {
		return AcceptanceTarget{}, fmt.Errorf("target must be name|provider|base_url[|api_key_env[|model_hint]]")
	}
	target := AcceptanceTarget{
		Name:     parts[0],
		Provider: NormalizeProvider(parts[1]),
		BaseURL:  parts[2],
	}
	if len(parts) >= 4 {
		target.APIKeyEnv = parts[3]
	}
	if len(parts) >= 5 {
		target.ModelHint = parts[4]
	}
	if err := AcceptanceTargetError(target); err != "" {
		return AcceptanceTarget{}, fmt.Errorf("%s", err)
	}
	return target, nil
}

// loadRosterTargets reads a roster JSON file, extracts its "targets" list (an
// absent field is an empty roster, a non-list field is an error), and folds each
// object row into a T via build. build returns (target, keep, err): keep=false
// skips the row (a filtered/ineligible entry), a non-nil err aborts the load.
// It folds the read+list-extract+row-loop scaffold shared by the readiness and
// acceptance roster loaders.
func loadRosterTargets[T any](path string, build func(idx int, row map[string]any) (T, bool, error)) ([]T, error) {
	var data map[string]any
	if err := readJSONFile(path, &data); err != nil {
		return nil, err
	}
	rawRows, ok := data["targets"].([]any)
	if !ok {
		if _, exists := data["targets"]; !exists {
			rawRows = nil
		} else {
			return nil, fmt.Errorf("roster targets field is not a JSON list: %s", path)
		}
	}
	targets := make([]T, 0, len(rawRows))
	for idx, raw := range rawRows {
		row, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("roster target %d is not a JSON object", idx)
		}
		target, keep, err := build(idx, row)
		if err != nil {
			return nil, err
		}
		if keep {
			targets = append(targets, target)
		}
	}
	return targets, nil
}

func LoadReadinessRosterTargets(path string) ([]ReadinessTarget, error) {
	return loadRosterTargets(path, func(idx int, row map[string]any) (ReadinessTarget, bool, error) {
		contractClass := stringField(row, "contract_class")
		status := stringField(row, "status")
		if !openAICompatibleClasses[contractClass] || status == "INVALID_TARGET" {
			return ReadinessTarget{}, false, nil
		}
		target := ReadinessTarget{
			Name:      stringField(row, "name"),
			BaseURL:   stringField(row, "base_url"),
			APIKeyEnv: stringField(row, "api_key_env"),
			ModelHint: stringField(row, "model_hint"),
		}
		if err := ReadinessTargetError(target); err != "" {
			label := target.Name
			if label == "" {
				label = fmt.Sprint(idx)
			}
			return ReadinessTarget{}, false, fmt.Errorf("roster target %s: %s", label, err)
		}
		return target, true, nil
	})
}

func LoadAcceptanceRosterTargets(path string) ([]AcceptanceTarget, error) {
	return loadRosterTargets(path, func(idx int, row map[string]any) (AcceptanceTarget, bool, error) {
		return AcceptanceTarget{
			Name:      stringField(row, "name"),
			Provider:  NormalizeProvider(stringField(row, "provider")),
			BaseURL:   stringField(row, "base_url"),
			APIKeyEnv: stringField(row, "api_key_env"),
			ModelHint: stringField(row, "model_hint"),
		}, true, nil
	})
}

func ReadinessTargetError(target ReadinessTarget) string {
	name := strings.TrimSpace(target.Name)
	baseURL := strings.TrimSpace(target.BaseURL)
	if name == "" {
		return "target name is empty"
	}
	if baseURL == "" {
		return "target base_url is empty"
	}
	parsed, err := url.Parse(baseURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return fmt.Sprintf("target base_url must be an absolute http(s) URL: %q", baseURL)
	}
	return ""
}

func AcceptanceTargetError(target AcceptanceTarget) string {
	if strings.TrimSpace(target.Name) == "" {
		return "target name is empty"
	}
	if strings.TrimSpace(target.BaseURL) == "" {
		return "target base_url is empty"
	}
	return ""
}

func ModelsURL(baseURL string) string {
	return strings.TrimRight(baseURL, "/") + "/models"
}

func ClassifyHTTP(status int, body string) string {
	lower := strings.ToLower(body)
	if status == 200 {
		var data map[string]any
		if err := json.Unmarshal([]byte(body), &data); err != nil {
			return "HTTP_OK_NO_MODELS"
		}
		if _, ok := data["data"].([]any); ok {
			return "MODELS_CONFIRMED"
		}
		return "HTTP_OK_NO_MODELS"
	}
	if status == 403 && (strings.Contains(lower, "access denied") || strings.Contains(lower, "error 1010") || strings.Contains(lower, "browser_signature_banned")) {
		return "ACCESS_DENIED"
	}
	if status == 401 || status == 403 {
		return "AUTH_REQUIRED"
	}
	if status == 402 || strings.Contains(lower, "no_payment_method") || strings.Contains(lower, "no payment method") {
		return "BILLING_REQUIRED"
	}
	if status == 429 {
		return "RATE_LIMITED"
	}
	if status >= 500 && status <= 599 {
		return "TRANSIENT_TRANSPORT"
	}
	return "HTTP_STATUS"
}

func ModelIDs(body string) []string {
	var data map[string]any
	if err := json.Unmarshal([]byte(body), &data); err != nil {
		return nil
	}
	rawItems, ok := data["data"].([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, raw := range rawItems {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		id, ok := item["id"].(string)
		if ok {
			out = append(out, id)
		}
	}
	return out
}

func ProbeReadinessTarget(ctx context.Context, target ReadinessTarget, opts ReadinessOptions) ReadinessProbe {
	version := opts.AppVersion
	if version == "" {
		version = AppVersion("")
	}
	out := ReadinessProbe{
		Version:   version,
		Name:      target.Name,
		BaseURL:   target.BaseURL,
		APIKeyEnv: target.APIKeyEnv,
		ModelHint: target.ModelHint,
		Models:    []string{},
	}
	if err := ReadinessTargetError(target); err != "" {
		out.Status = "INVALID_TARGET"
		out.Error = err
		return out
	}
	out.URL = ModelsURL(target.BaseURL)
	key := ""
	if strings.TrimSpace(target.APIKeyEnv) != "" {
		key = os.Getenv(target.APIKeyEnv)
	}
	if strings.TrimSpace(target.APIKeyEnv) != "" && key == "" && !opts.ProbeMissingAuth {
		out.Status = "AUTH_ENV_MISSING"
		out.Error = fmt.Sprintf("environment variable %s is not set", target.APIKeyEnv)
		return out
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, out.URL, nil)
	if err != nil {
		out.Status = "TRANSIENT_TRANSPORT"
		out.Error = err.Error()
		return out
	}
	req.Header.Set("Accept", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		out.Status = "TRANSIENT_TRANSPORT"
		out.Error = err.Error()
		return out
	}
	defer resp.Body.Close()
	raw, readErr := io.ReadAll(resp.Body)
	body := string(raw)
	status := resp.StatusCode
	out.HTTPStatus = &status
	out.Status = ClassifyHTTP(status, body)
	if readErr != nil {
		out.Status = "TRANSIENT_TRANSPORT"
		out.HTTPStatus = nil
		out.Error = readErr.Error()
		return out
	}
	out.BodyExcerpt = excerpt(body, 500)
	if status == 200 {
		ids := ModelIDs(body)
		if len(ids) > 25 {
			ids = ids[:25]
		}
		out.Models = ids
	} else {
		out.Error = out.BodyExcerpt
	}
	return out
}

func BuildReadinessReport(ctx context.Context, targets []ReadinessTarget, opts ReadinessOptions) ReadinessReport {
	version := opts.AppVersion
	if version == "" {
		version = AppVersion("")
	}
	if len(targets) == 0 {
		targets = DefaultReadinessTargets()
	}
	opts.AppVersion = version
	probes := make([]ReadinessProbe, 0, len(targets))
	for _, target := range targets {
		probes = append(probes, ProbeReadinessTarget(ctx, target, opts))
	}
	summary := ReadinessSummary{Targets: len(probes)}
	for _, probe := range probes {
		switch probe.Status {
		case "MODELS_CONFIRMED":
			summary.ModelsConfirmed++
		case "AUTH_ENV_MISSING":
			summary.AuthEnvMissing++
		case "AUTH_REQUIRED":
			summary.AuthRequired++
		case "ACCESS_DENIED":
			summary.AccessDenied++
		case "BILLING_REQUIRED":
			summary.BillingRequired++
		case "TRANSIENT_TRANSPORT":
			summary.TransientTransport++
		case "INVALID_TARGET":
			summary.InvalidTargets++
		}
		if !knownReadinessStatuses[probe.Status] {
			summary.Unclassified++
		}
	}
	summary.ReadinessGate = summary.Unclassified == 0 && summary.InvalidTargets == 0 && summary.Targets > 0
	return ReadinessReport{
		Schema:      ReadinessSchema,
		AppVersion:  version,
		GeneratedAt: utcNow(),
		ScopeNote:   "Cheap current-state /models probe. It verifies endpoint shape or records typed auth/billing/transport states; it does not run chat completions or spend model tokens.",
		Summary:     summary,
		Probes:      probes,
	}
}

func ReadinessMarkdown(report ReadinessReport) string {
	s := report.Summary
	lines := []string{
		"# API-Host Readiness Probe",
		"",
		"> Current-state `/models` probe for compatible API hosts.",
		"",
		"## Summary",
		"",
		fmt.Sprintf("- Targets: %d", s.Targets),
		fmt.Sprintf("- Models confirmed: %d", s.ModelsConfirmed),
		fmt.Sprintf("- Auth env missing: %d", s.AuthEnvMissing),
		fmt.Sprintf("- Auth required: %d", s.AuthRequired),
		fmt.Sprintf("- Access denied: %d", s.AccessDenied),
		fmt.Sprintf("- Billing required: %d", s.BillingRequired),
		fmt.Sprintf("- Transient transport: %d", s.TransientTransport),
		fmt.Sprintf("- Invalid targets: %d", s.InvalidTargets),
		fmt.Sprintf("- Readiness gate: %s", yesNo(s.ReadinessGate)),
		"",
		"| target | status | HTTP | models |",
		"|---|---|---:|---:|",
	}
	for _, probe := range report.Probes {
		httpStatus := ""
		if probe.HTTPStatus != nil {
			httpStatus = fmt.Sprint(*probe.HTTPStatus)
		}
		lines = append(lines, fmt.Sprintf("| `%s` | %s | %s | %d |", probe.Name, probe.Status, httpStatus, len(probe.Models)))
	}
	lines = append(lines, "")
	return strings.Join(lines, "\n")
}

func NormalizeProvider(provider string) string {
	p := strings.ToLower(strings.TrimSpace(provider))
	switch p {
	case "":
		return "openai-compatible"
	case "gpt":
		return "openai"
	case "chat-completions":
		return "openai-compatible"
	case "claude":
		return "anthropic"
	case "google":
		return "gemini"
	case "grok":
		return "xai"
	case "http":
		return "direct-http"
	case "mcp":
		return "direct-mcp"
	default:
		return p
	}
}

func ContractClass(provider string) string {
	if openAICompatibleProviders[provider] {
		return "openai_compatible_upstream"
	}
	if nativeProviders[provider] {
		return "native_provider_transcript_adapters"
	}
	if provider == "direct-http" {
		return "direct_kernel_http_syscall"
	}
	if provider == "direct-mcp" {
		return "direct_kernel_mcp_syscall"
	}
	return "unsupported"
}

func AcceptanceStatus(provider, probeStatus string) (string, string) {
	if !supportedProvider(provider) {
		return "UNSUPPORTED_WIRE", "Host does not match a covered API-host wire."
	}
	if directProviders[provider] {
		return "WIRE_SUPPORTED_UNPROBED", "Direct HTTP/MCP integration is covered by executable source witnesses; no remote /models probe applies."
	}
	if nativeProviders[provider] {
		return "WIRE_SUPPORTED_UNPROBED", "Native provider wire is covered by adapter and gateway witnesses; no generic no-spend /models endpoint is assumed."
	}
	switch probeStatus {
	case "MODELS_CONFIRMED":
		return "READY_FOR_LIVE_BRIDGE_RUN", "Compatible /models surface is reachable; candidate is ready for a live bridge smoke run."
	case "AUTH_ENV_MISSING":
		return "NEEDS_AUTH_ENV", "Required API key environment variable is not set."
	case "AUTH_REQUIRED", "ACCESS_DENIED", "BILLING_REQUIRED", "RATE_LIMITED", "TRANSIENT_TRANSPORT":
		return probeStatus, "Candidate host is compatible-shaped but blocked by a typed external state."
	case "HTTP_OK_NO_MODELS":
		return "MODELS_SHAPE_MISMATCH", "HTTP 200 response did not expose an OpenAI-compatible model list."
	default:
		return "UNCLASSIFIED", "Candidate host returned an unclassified readiness state."
	}
}

func LoadSweepRows(root string) ([]map[string]any, []ArtifactError) {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	pattern := filepath.Join(root, "fak", "experiments", "agent-live", "transcript-adapter-sweep*", "sweep-summary.json")
	paths, err := filepath.Glob(pattern)
	if err != nil {
		return nil, []ArtifactError{{Path: filepath.ToSlash(pattern), Error: "invalid sweep glob: " + err.Error()}}
	}
	sort.Strings(paths)
	var rows []map[string]any
	var errors []ArtifactError
	for _, path := range paths {
		rel := relPath(root, path)
		raw, err := os.ReadFile(path)
		if err != nil {
			errors = append(errors, ArtifactError{Path: rel, Error: "cannot read artifact: " + err.Error()})
			continue
		}
		raw = bytes.TrimPrefix(raw, []byte{0xEF, 0xBB, 0xBF})
		var decoded any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			errors = append(errors, ArtifactError{Path: rel, Error: "invalid JSON: " + err.Error()})
			continue
		}
		list, ok := decoded.([]any)
		if !ok {
			errors = append(errors, ArtifactError{Path: rel, Error: "sweep summary is not a JSON list"})
			continue
		}
		for idx, rawRow := range list {
			row, ok := rawRow.(map[string]any)
			if !ok {
				rowIndex := idx
				errors = append(errors, ArtifactError{Path: rel, RowIndex: &rowIndex, Error: "sweep row is not a JSON object"})
				continue
			}
			cp := copyMap(row)
			cp["_summary_path"] = rel
			rows = append(rows, cp)
		}
	}
	return rows, errors
}

func ClassifySweepError(err string) string {
	lower := strings.ToLower(err)
	if strings.Contains(lower, "http 402") || strings.Contains(lower, "no_payment_method") || strings.Contains(lower, "no payment method") {
		return "BILLING_REQUIRED"
	}
	if strings.Contains(lower, "http 401") || strings.Contains(lower, "authentication required") || strings.Contains(lower, "unauthorized") {
		return "AUTH_REQUIRED"
	}
	if strings.Contains(lower, "access denied") || strings.Contains(lower, "browser_signature_banned") {
		return "ACCESS_DENIED"
	}
	if strings.Contains(lower, "http 429") || strings.Contains(lower, "rate limit") {
		return "RATE_LIMITED"
	}
	if strings.Contains(lower, "http 5") || strings.Contains(lower, "timeout") || strings.Contains(lower, "connection") {
		return "TRANSIENT_TRANSPORT"
	}
	return "UNCLASSIFIED"
}

func LatestSweepRow(target AcceptanceTarget, rows []map[string]any) map[string]any {
	model := target.ModelHint
	var matched []map[string]any
	for _, row := range rows {
		if stringValue(row["kind"]) != "api" {
			continue
		}
		if stringValue(row["base_url"]) != target.BaseURL {
			continue
		}
		if model != "" && stringValue(row["model"]) != model {
			continue
		}
		matched = append(matched, row)
	}
	if len(matched) == 0 {
		return nil
	}
	sort.SliceStable(matched, func(i, j int) bool {
		return stringValue(matched[i]["generated_at"]) < stringValue(matched[j]["generated_at"])
	})
	return matched[len(matched)-1]
}

func StatusFromLiveSweep(row map[string]any) (string, string) {
	if row == nil {
		return "", ""
	}
	if stringValue(row["status"]) == "ok" && row["live"] == true && strings.TrimSpace(stringValue(row["transcript_sha"])) != "" {
		return "LIVE_BRIDGE_CONFIRMED", "A committed live sweep row confirms this host drove the bridge."
	}
	if stringValue(row["status"]) == "failed" {
		status := ClassifySweepError(stringValue(row["error"]))
		if status != "UNCLASSIFIED" {
			return status, "Latest live bridge smoke hit a typed external blocker."
		}
		return "UNCLASSIFIED", "Latest live bridge smoke failed with an unclassified error."
	}
	return "", ""
}

func ClassifyAcceptanceTarget(ctx context.Context, target AcceptanceTarget, opts AcceptanceOptions, sweepRows []map[string]any) AcceptanceRow {
	version := opts.AppVersion
	if version == "" {
		version = AppVersion(opts.Root)
	}
	provider := NormalizeProvider(target.Provider)
	normalized := AcceptanceTarget{
		Name:      target.Name,
		Provider:  provider,
		BaseURL:   target.BaseURL,
		APIKeyEnv: target.APIKeyEnv,
		ModelHint: target.ModelHint,
	}
	cls := ContractClass(provider)
	base := AcceptanceRow{
		Version:         version,
		Name:            normalized.Name,
		Provider:        normalized.Provider,
		BaseURL:         normalized.BaseURL,
		APIKeyEnv:       normalized.APIKeyEnv,
		ModelHint:       normalized.ModelHint,
		ContractClass:   cls,
		ReadinessStatus: "NOT_PROBED",
		NextLiveCommand: "",
	}
	if err := AcceptanceTargetError(normalized); err != "" {
		base.Status = "INVALID_TARGET"
		base.Reason = err
		return base
	}
	sweepRow := LatestSweepRow(normalized, sweepRows)
	liveStatus, liveReason := StatusFromLiveSweep(sweepRow)
	var probe *ReadinessProbe
	probeStatus := "NOT_PROBED"
	if openAICompatibleProviders[provider] {
		p := ProbeReadinessTarget(ctx, ReadinessTarget{
			Name:      normalized.Name,
			BaseURL:   normalized.BaseURL,
			APIKeyEnv: normalized.APIKeyEnv,
			ModelHint: normalized.ModelHint,
		}, ReadinessOptions{
			Timeout:          opts.Timeout,
			ProbeMissingAuth: opts.ProbeMissingAuth,
			AppVersion:       version,
		})
		probe = &p
		probeStatus = p.Status
	}
	status, reason := AcceptanceStatus(provider, probeStatus)
	if liveStatus != "" {
		status, reason = liveStatus, liveReason
	}
	base.Status = status
	base.Reason = reason
	base.ReadinessStatus = probeStatus
	base.Probe = probe
	base.LatestSweep = sweepRow
	if status == "READY_FOR_LIVE_BRIDGE_RUN" {
		base.NextLiveCommand = NextLiveCommand(normalized)
	}
	return base
}

func NextLiveCommand(target AcceptanceTarget) string {
	model := target.ModelHint
	if model == "" {
		model = "<model>"
	}
	keyEnv := target.APIKeyEnv
	if keyEnv == "" {
		keyEnv = "<api_key_env>"
	}
	return "pwsh tools/run_transcript_adapter_sweep.ps1 " +
		fmt.Sprintf("-ApiBaseUrl %s -ApiKeyEnv %s -ApiModels %s ", target.BaseURL, keyEnv, model) +
		"-SkipOffline -SkipLocalShim -SkipMicrobench"
}

func BuildAcceptanceReport(ctx context.Context, targets []AcceptanceTarget, opts AcceptanceOptions) AcceptanceReport {
	if strings.TrimSpace(opts.Root) == "" {
		opts.Root = "."
	}
	version := opts.AppVersion
	if version == "" {
		version = AppVersion(opts.Root)
	}
	if len(targets) == 0 {
		targets = DefaultAcceptanceTargets()
	}
	opts.AppVersion = version
	sweepRows, artifactErrors := LoadSweepRows(opts.Root)
	rows := make([]AcceptanceRow, 0, len(targets))
	for _, target := range targets {
		rows = append(rows, ClassifyAcceptanceTarget(ctx, target, opts, sweepRows))
	}
	summary := AcceptanceSummary{Targets: len(rows), SweepArtifactErrors: len(artifactErrors)}
	for _, row := range rows {
		if row.Status != "UNCLASSIFIED" {
			summary.KnownStatuses++
		}
		switch row.Status {
		case "READY_FOR_LIVE_BRIDGE_RUN":
			summary.ReadyForLiveBridgeRun++
		case "LIVE_BRIDGE_CONFIRMED":
			summary.LiveBridgeConfirmed++
		case "WIRE_SUPPORTED_UNPROBED":
			summary.WireSupportedUnprobed++
		case "UNSUPPORTED_WIRE":
			summary.UnsupportedWire++
		case "INVALID_TARGET":
			summary.InvalidTargets++
		}
		if externalBlockers[row.Status] {
			summary.TypedExternalBlockers++
		}
	}
	summary.Unclassified = len(rows) - summary.KnownStatuses
	summary.AcceptanceGate = len(rows) > 0 &&
		len(rows) == summary.KnownStatuses &&
		summary.UnsupportedWire == 0 &&
		summary.InvalidTargets == 0 &&
		len(artifactErrors) == 0
	return AcceptanceReport{
		Schema:         AcceptanceSchema,
		AppVersion:     version,
		GeneratedAt:    utcNow(),
		ScopeNote:      "No-spend candidate-host acceptance gate. READY_FOR_LIVE_BRIDGE_RUN means a host is compatible-shaped and reachable enough to run a live bridge smoke; typed blockers are external state, not bridge failures.",
		Summary:        summary,
		Targets:        rows,
		ArtifactErrors: artifactErrors,
	}
}

func AcceptanceMarkdown(report AcceptanceReport) string {
	s := report.Summary
	lines := []string{
		"# API-Host Acceptance Probe",
		"",
		"> Candidate-host gate for the scoped FAK/DOS API-host bridge contract.",
		"",
		"## Summary",
		"",
		fmt.Sprintf("- Targets: %d", s.Targets),
		fmt.Sprintf("- Ready for live bridge run: %d", s.ReadyForLiveBridgeRun),
		fmt.Sprintf("- Live bridge confirmed: %d", s.LiveBridgeConfirmed),
		fmt.Sprintf("- Wire supported but unprobed: %d", s.WireSupportedUnprobed),
		fmt.Sprintf("- Typed external blockers: %d", s.TypedExternalBlockers),
		fmt.Sprintf("- Unsupported wire: %d", s.UnsupportedWire),
		fmt.Sprintf("- Invalid targets: %d", s.InvalidTargets),
		fmt.Sprintf("- Sweep artifact errors: %d", s.SweepArtifactErrors),
		fmt.Sprintf("- Acceptance gate: %s", yesNo(s.AcceptanceGate)),
		"",
		"| target | provider | class | status | readiness |",
		"|---|---|---|---|---|",
	}
	for _, row := range report.Targets {
		lines = append(lines, fmt.Sprintf("| `%s` | `%s` | `%s` | %s | %s |", row.Name, row.Provider, row.ContractClass, row.Status, row.ReadinessStatus))
	}
	if len(report.ArtifactErrors) > 0 {
		lines = append(lines, "", "## Artifact Errors", "", "| path | detail |", "|---|---|")
		for _, err := range report.ArtifactErrors {
			lines = append(lines, fmt.Sprintf("| `%s` | %s |", err.Path, err.Error))
		}
	}
	lines = append(lines, "")
	return strings.Join(lines, "\n")
}

func MarshalReport(report any) ([]byte, error) {
	raw, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

func splitSpec(spec string) []string {
	raw := strings.Split(spec, "|")
	for i := range raw {
		raw[i] = strings.TrimSpace(raw[i])
	}
	return raw
}

func repoRoot(start string) string {
	if strings.TrimSpace(start) == "" {
		wd, err := os.Getwd()
		if err != nil {
			return ""
		}
		start = wd
	}
	abs, err := filepath.Abs(start)
	if err == nil {
		start = abs
	}
	if info, err := os.Stat(start); err == nil && !info.IsDir() {
		start = filepath.Dir(start)
	}
	dir := start
	for {
		if _, err := os.Stat(filepath.Join(dir, "VERSION")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func readJSONFile(path string, dst any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	raw = bytes.TrimPrefix(raw, []byte{0xEF, 0xBB, 0xBF})
	return json.Unmarshal(raw, dst)
}

func stringField(row map[string]any, key string) string {
	return strings.TrimSpace(stringValue(row[key]))
}

func stringValue(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	default:
		return fmt.Sprint(x)
	}
}

func excerpt(s string, n int) string {
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n])
}

func utcNow() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func yesNo(v bool) string {
	if v {
		return "yes"
	}
	return "no"
}

func supportedProvider(provider string) bool {
	return openAICompatibleProviders[provider] || nativeProviders[provider] || directProviders[provider]
}

func relPath(root, path string) string {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return filepath.ToSlash(path)
	}
	return filepath.ToSlash(rel)
}

func copyMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in)+1)
	for k, v := range in {
		out[k] = v
	}
	return out
}
