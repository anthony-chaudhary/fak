package macbench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const Schema = "fak.macbench.result.v1"

type Suite string

const (
	SuiteAll          Suite = "all"
	SuiteDecodeLong   Suite = "decode-longgen"
	SuitePrefillSweep Suite = "prefill-sweep"
	SuiteTwoStream    Suite = "2stream"
	SuiteHealth       Suite = "health"
)

type Options struct {
	Gateway       string
	Model         string
	Key           string
	Suite         Suite
	DecodeTokens  []int
	PrefillTokens []int
	Concurrency   int
	HTTPClient    *http.Client
	Now           func() time.Time
}

type Report struct {
	Schema      string   `json:"schema"`
	GeneratedAt string   `json:"generated_at"`
	Suite       Suite    `json:"suite"`
	Gateway     string   `json:"gateway"`
	Model       string   `json:"model"`
	Health      Health   `json:"health"`
	Rows        []Row    `json:"rows,omitempty"`
	Headline    string   `json:"headline,omitempty"`
	Errors      []string `json:"errors,omitempty"`
}

type Health struct {
	OK      bool   `json:"ok"`
	Engine  string `json:"engine,omitempty"`
	Planner string `json:"planner,omitempty"`
	Model   string `json:"model,omitempty"`
	Error   string `json:"error,omitempty"`
}

const RecoverySchema = "fak.macbench.recovery.v1"

type RecoverySignals struct {
	WatcherRunning bool
	ResultPresent  bool
	LatestReport   *Report
	// LogPresent reports whether the watch log the caller named actually
	// exists. nil means no log path was inspected, so presence is unknown.
	// A known-false value is NOT the same as "the watcher has not polled
	// yet": there is no log to poll into, so waiting cannot make progress.
	LogPresent    *bool
	TailnetOnline *bool
	SSHReachable  *bool
	WakeHelper    *bool
}

type RecoveryPlan struct {
	Schema   string           `json:"schema"`
	State    string           `json:"state"`
	Severity string           `json:"severity"`
	Summary  string           `json:"summary"`
	Evidence []string         `json:"evidence,omitempty"`
	Actions  []RecoveryAction `json:"actions"`
}

type RecoveryAction struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Detail string `json:"detail"`
}

type Row struct {
	Name                   string  `json:"name"`
	Kind                   string  `json:"kind"`
	Streams                int     `json:"streams,omitempty"`
	PromptRequested        int     `json:"prompt_requested,omitempty"`
	PromptTokens           int     `json:"prompt_tokens,omitempty"`
	MaxTokens              int     `json:"max_tokens,omitempty"`
	CompletionTokens       int     `json:"completion_tokens,omitempty"`
	WallSeconds            float64 `json:"wall_seconds,omitempty"`
	TTFTSeconds            float64 `json:"ttft_seconds,omitempty"`
	TokensPerSecond        float64 `json:"tokens_per_second,omitempty"`
	PrefillTokensPerSecond float64 `json:"prefill_tokens_per_second,omitempty"`
	FinishReason           string  `json:"finish_reason,omitempty"`
	HTTPStatus             int     `json:"http_status,omitempty"`
	Headline               string  `json:"headline,omitempty"`
	Error                  string  `json:"error,omitempty"`
}

func DefaultOptions() Options {
	return Options{
		Gateway:       "http://127.0.0.1:8080",
		Model:         "qwen3.6-27b",
		Suite:         SuiteAll,
		DecodeTokens:  []int{256, 512},
		PrefillTokens: []int{128, 512, 2048, 4096},
		Concurrency:   2,
	}
}

func PlanRecovery(sig RecoverySignals) RecoveryPlan {
	plan := RecoveryPlan{
		Schema:   RecoverySchema,
		State:    "no_health_report",
		Severity: "info",
		Summary:  "macbench has not observed a gateway health report yet",
		Actions: []RecoveryAction{{
			ID:     "wait-first-poll",
			Title:  "wait for first health poll",
			Detail: "Keep the watcher running until it writes the first sanitized health report.",
		}},
	}
	if sig.ResultPresent {
		plan.State = "result_present"
		plan.Severity = "info"
		plan.Summary = "macbench result artifact is present"
		plan.Actions = []RecoveryAction{{
			ID:     "record-result",
			Title:  "record result",
			Detail: "Fold the result artifact into the nightrun ledger or benchmark summary.",
		}}
		return plan
	}
	if !sig.WatcherRunning {
		plan.State = "watcher_not_running"
		plan.Severity = "action"
		plan.Summary = "macbench watcher is not running"
		plan.Actions = []RecoveryAction{{
			ID:     "restart-watch",
			Title:  "restart macbench watch",
			Detail: "Start `fak macbench watch` with the same sanitized log and result paths.",
		}}
		return plan
	}
	if boolKnownFalse(sig.LogPresent) {
		plan.State = "log_missing"
		plan.Severity = "action"
		plan.Summary = "macbench watch log is absent at the named path; there is no evidence to recover from"
		plan.Actions = []RecoveryAction{
			{
				ID:     "confirm-log-path",
				Title:  "confirm the watch log path",
				Detail: "Check the run id and box directory: the watch log does not exist at the path passed to --log.",
			},
			{
				ID:     "start-fresh-watch",
				Title:  "start a fresh watch run",
				Detail: "Nightrun artifacts are host-local and rotate, so a run whose log has aged out cannot be revived. Start `fak macbench watch` and bind follow-up work to the NEW run id.",
			},
		}
		return plan
	}
	if sig.LatestReport == nil {
		return plan
	}

	rep := *sig.LatestReport
	gateway := sanitizedReportGateway(rep.Gateway)
	plan.Evidence = append(plan.Evidence, fmt.Sprintf("latest suite=%s gateway=%s health=%t", rep.Suite, gateway, rep.Health.OK))
	if rep.Health.Error != "" {
		plan.Evidence = append(plan.Evidence, "health_error="+sanitizeGatewayInText(rep.Health.Error, rep.Gateway))
	}
	if rep.Health.OK {
		plan.State = "gateway_ready"
		plan.Severity = "info"
		plan.Summary = "gateway health is OK; waiting for full benchmark result"
		plan.Actions = []RecoveryAction{{
			ID:     "wait-full-suite",
			Title:  "wait for full suite",
			Detail: "Let the watcher run the full `all` suite and write the result artifact.",
		}}
		return plan
	}

	errText := strings.ToLower(rep.Health.Error + "\n" + strings.Join(rep.Errors, "\n"))
	switch {
	case boolKnownFalse(sig.TailnetOnline):
		plan.State = "tailnet_offline"
		plan.Severity = "operator"
		plan.Summary = "Mac benchmark peer is offline; gateway cannot be recovered from this host"
		plan.Actions = []RecoveryAction{
			{
				ID:     "wake-or-power-mac",
				Title:  "wake or power the Mac",
				Detail: "Use the private lab control path or physical access to bring the Mac back onto the tailnet.",
			},
			{
				ID:     "confirm-tailnet-online",
				Title:  "confirm tailnet peer online",
				Detail: "Re-check the tailnet peer status before restarting the gateway.",
			},
			{
				ID:     "restart-gateway",
				Title:  "restart fak gateway",
				Detail: "Once reachable, start or restart the Mac `fak serve --metal` gateway and keep the watcher running.",
			},
		}
	case boolKnownFalse(sig.SSHReachable):
		plan.State = "control_path_down"
		plan.Severity = "operator"
		plan.Summary = "Mac control path is unreachable; gateway restart cannot be attempted"
		plan.Actions = []RecoveryAction{
			{
				ID:     "restore-control-path",
				Title:  "restore control path",
				Detail: "Bring SSH/tailnet connectivity back before trying to read the gateway key or restart the gateway.",
			},
			{
				ID:     "keep-watch-running",
				Title:  "keep watcher running",
				Detail: "The watcher will write the full result automatically once gateway health succeeds.",
			},
		}
	case strings.Contains(errText, "deadline exceeded") ||
		strings.Contains(errText, "timeout") ||
		strings.Contains(errText, "connection refused") ||
		strings.Contains(errText, "no route to host"):
		plan.State = "gateway_unreachable"
		plan.Severity = "operator"
		plan.Summary = "gateway health probe is timing out or refusing connections"
		plan.Actions = []RecoveryAction{
			{
				ID:     "check-peer-online",
				Title:  "check Mac peer",
				Detail: "Confirm the Mac is awake and visible on the private network.",
			},
			{
				ID:     "restart-gateway",
				Title:  "restart fak gateway",
				Detail: "Restart the Mac gateway, then re-run `fak macbench watch-status` against the existing log/result paths.",
			},
		}
	default:
		plan.State = "waiting_for_gateway"
		plan.Severity = "watch"
		plan.Summary = "gateway health is still false"
		plan.Actions = []RecoveryAction{{
			ID:     "keep-watch-running",
			Title:  "keep watcher running",
			Detail: "Continue polling while investigating the latest sanitized health error.",
		}}
	}
	if boolKnownFalse(sig.WakeHelper) {
		plan.Actions = append(plan.Actions, RecoveryAction{
			ID:     "document-wake-helper-gap",
			Title:  "document wake helper gap",
			Detail: "Track the missing wake/restart helper so the next run has an operator-usable recovery path.",
		})
	}
	return plan
}

func (r Report) HasErrors() bool {
	if len(r.Errors) > 0 {
		return true
	}
	for _, row := range r.Rows {
		if row.Error != "" {
			return true
		}
	}
	return false
}

func Run(ctx context.Context, opts Options) (Report, error) {
	opts = normalizeOptions(opts)
	base, err := NormalizeGateway(opts.Gateway)
	if err != nil {
		return Report{}, err
	}
	opts.Gateway = base
	now := opts.Now()
	rep := Report{
		Schema:      Schema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Suite:       opts.Suite,
		Gateway:     SanitizeGatewayForReport(base),
		Model:       opts.Model,
	}
	rep.Health = probeHealth(ctx, opts)
	rep.Health.Error = sanitizeGatewayInText(rep.Health.Error, base)
	if !rep.Health.OK {
		rep.Errors = append(rep.Errors, "healthz failed: "+rep.Health.Error)
		if opts.Suite == SuiteHealth {
			return rep, nil
		}
	}

	switch opts.Suite {
	case SuiteHealth:
	case SuiteAll:
		rep.Rows = append(rep.Rows, runDecodeLong(ctx, opts)...)
		rep.Rows = append(rep.Rows, runPrefillSweep(ctx, opts)...)
		rep.Rows = append(rep.Rows, runTwoStream(ctx, opts)...)
	case SuiteDecodeLong:
		rep.Rows = append(rep.Rows, runDecodeLong(ctx, opts)...)
	case SuitePrefillSweep:
		rep.Rows = append(rep.Rows, runPrefillSweep(ctx, opts)...)
	case SuiteTwoStream:
		rep.Rows = append(rep.Rows, runTwoStream(ctx, opts)...)
	default:
		return Report{}, fmt.Errorf("unknown suite %q", opts.Suite)
	}
	sanitizeReportErrors(&rep, base)
	rep.Headline = headline(rep.Rows)
	return rep, nil
}

func normalizeOptions(opts Options) Options {
	def := DefaultOptions()
	if strings.TrimSpace(opts.Gateway) == "" {
		opts.Gateway = def.Gateway
	}
	if strings.TrimSpace(opts.Model) == "" {
		opts.Model = def.Model
	}
	if opts.Suite == "" {
		opts.Suite = def.Suite
	}
	if len(opts.DecodeTokens) == 0 {
		opts.DecodeTokens = def.DecodeTokens
	}
	if len(opts.PrefillTokens) == 0 {
		opts.PrefillTokens = def.PrefillTokens
	}
	if opts.Concurrency <= 0 {
		opts.Concurrency = def.Concurrency
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = &http.Client{}
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	return opts
}

func NormalizeGateway(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		raw = "http://127.0.0.1:8080"
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" {
		u.Scheme = "http"
	}
	if u.Host == "" {
		return "", fmt.Errorf("gateway %q has no host", raw)
	}
	u.Path = strings.TrimSuffix(u.Path, "/")
	u.Path = strings.TrimSuffix(u.Path, "/v1")
	u.RawQuery = ""
	u.Fragment = ""
	return strings.TrimSuffix(u.String(), "/"), nil
}

func SanitizeGatewayForReport(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return "<gateway>"
	}
	host := u.Hostname()
	if strings.EqualFold(host, "localhost") {
		return u.Scheme + "://localhost" + portSuffix(u)
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return u.Scheme + "://" + host + portSuffix(u)
	}
	return "<remote-gateway>"
}

func sanitizeReportErrors(rep *Report, rawGateway string) {
	rep.Health.Error = sanitizeGatewayInText(rep.Health.Error, rawGateway)
	for i := range rep.Errors {
		rep.Errors[i] = sanitizeGatewayInText(rep.Errors[i], rawGateway)
	}
	for i := range rep.Rows {
		rep.Rows[i].Error = sanitizeGatewayInText(rep.Rows[i].Error, rawGateway)
	}
}

func sanitizeGatewayInText(s, rawGateway string) string {
	if s == "" {
		return s
	}
	return strings.ReplaceAll(s, rawGateway, SanitizeGatewayForReport(rawGateway))
}

func sanitizedReportGateway(raw string) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "<") && strings.HasSuffix(raw, ">") {
		return raw
	}
	return SanitizeGatewayForReport(raw)
}

func boolKnownFalse(v *bool) bool {
	return v != nil && !*v
}

func portSuffix(u *url.URL) string {
	if p := u.Port(); p != "" {
		return ":" + p
	}
	return ""
}

func probeHealth(ctx context.Context, opts Options) Health {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, opts.Gateway+"/healthz", nil)
	if err != nil {
		return Health{Error: err.Error()}
	}
	resp, err := opts.HTTPClient.Do(req)
	if err != nil {
		return Health{Error: err.Error()}
	}
	defer resp.Body.Close()
	var h Health
	if resp.StatusCode/100 != 2 {
		h.Error = fmt.Sprintf("status %d", resp.StatusCode)
		return h
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&h); err != nil {
		return Health{Error: err.Error()}
	}
	return h
}

func runDecodeLong(ctx context.Context, opts Options) []Row {
	rows := make([]Row, 0, len(opts.DecodeTokens))
	for _, maxTok := range opts.DecodeTokens {
		row := runBuffered(ctx, opts, "decode-longgen", fmt.Sprintf("decode-%d", maxTok), 25, maxTok)
		rows = append(rows, row)
	}
	return rows
}

func runPrefillSweep(ctx context.Context, opts Options) []Row {
	rows := make([]Row, 0, len(opts.PrefillTokens))
	for _, promptTok := range opts.PrefillTokens {
		row := runStreamed(ctx, opts, "prefill-sweep", fmt.Sprintf("prefill-%d", promptTok), promptTok, 16)
		rows = append(rows, row)
	}
	return rows
}

func runTwoStream(ctx context.Context, opts Options) []Row {
	streams := opts.Concurrency
	rows := make([]Row, streams)
	var wg sync.WaitGroup
	start := time.Now()
	for i := 0; i < streams; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			rows[i] = runBuffered(ctx, opts, "2stream", fmt.Sprintf("2stream-%d", i+1), 25, 128)
			rows[i].Streams = 1
		}()
	}
	wg.Wait()
	wall := elapsedSeconds(start)
	agg := Row{Name: "2stream-aggregate", Kind: "2stream", Streams: streams, PromptRequested: 25, MaxTokens: 128, WallSeconds: round(wall)}
	for _, row := range rows {
		agg.CompletionTokens += row.CompletionTokens
		if row.Error != "" {
			if agg.Error != "" {
				agg.Error += "; "
			}
			agg.Error += row.Name + ": " + row.Error
		}
	}
	if wall > 0 && agg.CompletionTokens > 0 {
		agg.TokensPerSecond = round(float64(agg.CompletionTokens) / wall)
		agg.Headline = fmt.Sprintf("%.2f tok/s", agg.TokensPerSecond)
	}
	out := []Row{agg}
	out = append(out, rows...)
	return out
}

func runBuffered(ctx context.Context, opts Options, kind, name string, promptTokens, maxTokens int) Row {
	row := Row{Name: name, Kind: kind, PromptRequested: promptTokens, MaxTokens: maxTokens}
	body := chatBody(opts.Model, prompt(promptTokens), maxTokens, false)
	start := time.Now()
	resp, err := doChat(ctx, opts, body)
	wall := elapsedSeconds(start)
	row.WallSeconds = round(wall)
	if err != nil {
		row.Error = err.Error()
		return row
	}
	defer resp.Body.Close()
	row.HTTPStatus = resp.StatusCode
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if resp.StatusCode/100 != 2 {
		row.Error = fmt.Sprintf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
		return row
	}
	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		row.Error = err.Error()
		return row
	}
	row.PromptTokens = cr.Usage.PromptTokens
	row.CompletionTokens = cr.Usage.CompletionTokens
	if len(cr.Choices) > 0 {
		row.FinishReason = cr.Choices[0].FinishReason
	}
	if row.CompletionTokens > 0 && wall > 0 {
		row.TokensPerSecond = round(float64(row.CompletionTokens) / wall)
		row.Headline = fmt.Sprintf("%.2f tok/s", row.TokensPerSecond)
	}
	return row
}

func runStreamed(ctx context.Context, opts Options, kind, name string, promptTokens, maxTokens int) Row {
	row := Row{Name: name, Kind: kind, PromptRequested: promptTokens, MaxTokens: maxTokens}
	body := chatBody(opts.Model, prompt(promptTokens), maxTokens, true)
	start := time.Now()
	resp, err := doChat(ctx, opts, body)
	if err != nil {
		row.Error = err.Error()
		return row
	}
	defer resp.Body.Close()
	row.HTTPStatus = resp.StatusCode
	if resp.StatusCode/100 != 2 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		row.Error = fmt.Sprintf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
		return row
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	chunkTokens := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var sr streamResponse
		if err := json.Unmarshal([]byte(payload), &sr); err != nil {
			continue
		}
		if sr.Usage != nil {
			row.PromptTokens = sr.Usage.PromptTokens
			row.CompletionTokens = sr.Usage.CompletionTokens
		}
		for _, choice := range sr.Choices {
			if choice.FinishReason != nil {
				row.FinishReason = *choice.FinishReason
			}
			if choice.Delta.Content != "" {
				chunkTokens++
				if row.TTFTSeconds == 0 {
					row.TTFTSeconds = elapsedSeconds(start)
				}
			}
		}
	}
	row.WallSeconds = elapsedSeconds(start)
	if row.CompletionTokens == 0 {
		row.CompletionTokens = chunkTokens
	}
	if row.PromptTokens == 0 {
		row.PromptTokens = promptTokens
	}
	if err := sc.Err(); err != nil {
		row.Error = err.Error()
		return row
	}
	if row.TTFTSeconds > 0 && row.PromptTokens > 0 {
		row.PrefillTokensPerSecond = round(float64(row.PromptTokens) / row.TTFTSeconds)
		row.Headline = fmt.Sprintf("%.2f tok/s", row.PrefillTokensPerSecond)
	}
	if row.CompletionTokens > 0 && row.WallSeconds > 0 {
		row.TokensPerSecond = round(float64(row.CompletionTokens) / row.WallSeconds)
	}
	return row
}

func doChat(ctx context.Context, opts Options, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, opts.Gateway+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(opts.Key) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(opts.Key))
	}
	return opts.HTTPClient.Do(req)
}

func chatBody(model, prompt string, maxTokens int, stream bool) []byte {
	body := map[string]any{
		"model":       model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":  maxTokens,
		"temperature": 0,
	}
	if stream {
		body["stream"] = true
	}
	b, _ := json.Marshal(body)
	return b
}

func prompt(tokens int) string {
	if tokens <= 0 {
		tokens = 25
	}
	var b strings.Builder
	b.WriteString("Continue with plain short filler text. Context:")
	for i := 0; i < tokens; i++ {
		fmt.Fprintf(&b, " token%d", i%97)
	}
	b.WriteString("\nAnswer with neutral text only.")
	return b.String()
}

func headline(rows []Row) string {
	for _, row := range rows {
		if row.Headline != "" {
			return row.Headline
		}
	}
	return ""
}

func round(v float64) float64 {
	if v > 0 && v < 0.001 {
		return 0.001
	}
	return float64(int(v*1000+0.5)) / 1000
}

func elapsedSeconds(start time.Time) float64 {
	v := time.Since(start).Seconds()
	if v <= 0 {
		return 0.001
	}
	return round(v)
}

type chatResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type streamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}
