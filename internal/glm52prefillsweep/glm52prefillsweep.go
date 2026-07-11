// Package glm52prefillsweep is the GLM-5.2 pure-fak PREFILL-latency sweep driver —
// the Go port of the retired tools/glm52_prefill_sweep.py (lever L9; #3085/#3086).
//
// It is the prefill analogue of the pure-fak DECODE sweep (cmd/glmdsatput): it runs a
// prefill-dominant request (a large prompt, max_tokens≈1) at each prompt length in
// {128, 512, 2048, 4096, 8192} against a GLM-5.2 `fak serve` endpoint, records TTFT /
// prefill tok/s per length, and lands each length as a DISCOVERABLE benchmark-ledger
// artifact under experiments/benchmark/runs/by-machine/<node>/<UTC>-glm52-prefill-sweep/
// p<len>/ (the same manifest/result/RESULTS.md shape the decode sweep lands via benchcli).
//
// HONESTY FENCE (load-bearing): running the driver produces NO measured prefill number
// by itself — it ENABLES the measurement an on-box sm_80 GPU peer runs. The metric is
// single-stream prefill tok/s = prompt_tokens / TTFT, served FULL-MLA on sm_80 (GLM-5.2's
// native DSA sparse-attention kernel is sm_90-floored), NOT the 753B MoE aggregate serving
// rate. Lengths ≥ 4096 may hit the sm_80 DSA illegal-memory-access; the live sweep
// TOLERATES and RECORDS a per-length FAIL and continues — one bad length never sinks the
// sweep or discards the smaller lengths that succeeded.
//
// The planner (BuildPlan / BuildDryRunReport) and the ledger-land helpers are pure: no
// network, no GPU, no subprocess. --dry-run (or omitting --endpoint) exercises only that
// pure path. The live HTTP sweep is reached only with an explicit --endpoint.
package glm52prefillsweep

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	// SchemaVersion is the top-level sweep-report schema.
	SchemaVersion = "fak.glm52-prefill-sweep.v1"
	// RecordSchema is the per-length record schema — deliberately distinct from the
	// decode sweep's "glm-throughput/1" so a reader never mistakes a prefill row for a
	// decode row.
	RecordSchema = "glm-prefill/1"
	// LineageSchema / BenchmarkArtifactSchema are the provenance envelopes
	// benchcli.DecodeArtifact keys on.
	LineageSchema           = "fak-bench-lineage/1"
	BenchmarkArtifactSchema = "fak-benchmark-artifact/1"

	// DefaultModel is the served GLM-5.2 model id.
	DefaultModel = "zai-org/GLM-5.2"
	// DefaultMaxTokens keeps a request prefill-dominant: a large prompt with (near-)zero
	// generation, so the measured latency is the prompt's forward pass, not decode. 1
	// keeps TTFT well-defined for non-streaming endpoints.
	DefaultMaxTokens = 1
	// FragileMinLen: lengths at/above this are flagged fragile on sm_80 (the note's P0
	// DSA illegal-memory-access).
	FragileMinLen = 4096

	fillerWord = "word"
)

// PrefillLengths is the #3085 sweep axis: establish the currently-unmeasured prefill
// baseline.
var PrefillLengths = []int{128, 512, 2048, 4096, 8192}

// Scope is the load-bearing caveat that travels WITH every landed number.
const Scope = "single-stream prefill tok/s = prompt_tokens / TTFT (time to first token); " +
	"served full-MLA on sm_80 (GLM-5.2 DSA sparse-attn kernel is sm_90-floored, so " +
	"this is the full-MLA context curve, not native DSA); NOT the 753B MoE " +
	"aggregate serving rate"

// execGit runs a git subprocess; it is a package var so a test can prove the dry-run
// path never shells out (override it to fail, then run --dry-run).
var execGit = func(root string, args ...string) (string, int) {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return out.String(), ee.ExitCode()
		}
		return "", 1
	}
	return out.String(), 0
}

// httpDo performs one HTTP request; a package var for the same dry-run-purity reason.
var httpDo = func(req *http.Request, timeout time.Duration) (*http.Response, error) {
	client := &http.Client{Timeout: timeout}
	return client.Do(req)
}

// --------------------------------------------------------------------------- //
// JSON shapes (field order is documentary; the ledger reader is order-independent)
// --------------------------------------------------------------------------- //

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// RequestBody is one OpenAI-compatible chat/completions body. StreamOptions is a pointer
// so it is emitted only in streaming mode (Python omits the key for a blocking request).
type RequestBody struct {
	Model         string         `json:"model"`
	Messages      []chatMessage  `json:"messages"`
	MaxTokens     int            `json:"max_tokens"`
	Temperature   int            `json:"temperature"`
	Stream        bool           `json:"stream"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

// PlanStep is one prompt length's request body and land path. Pure — no I/O to build.
type PlanStep struct {
	PromptLen          int         `json:"prompt_len"`
	TargetPromptTokens int         `json:"target_prompt_tokens"`
	MaxTokens          int         `json:"max_tokens"`
	Stream             bool        `json:"stream"`
	PromptChars        int         `json:"prompt_chars"`
	RequestBody        RequestBody `json:"request_body"`
	LandSubdir         string      `json:"land_subdir"`
	FragileOnSM80      bool        `json:"fragile_on_sm80"`
}

// Lineage is the four-axis provenance benchcli.DecodeArtifact recognizes.
type Lineage struct {
	LineageSchema string `json:"lineage_schema"`
	AppVersion    string `json:"app_version"`
	UTC           string `json:"utc"`
	GitCommit     string `json:"git_commit"`
	GoVersion     string `json:"go_version"`
	Node          string `json:"node"`
}

// Record is one discoverable per-length result. Nullable fields are pointers so a FAIL
// row carries JSON null (not a zero) where the Python emits None.
type Record struct {
	Schema             string   `json:"schema"`
	Model              string   `json:"model"`
	ServedModel        string   `json:"served_model"`
	Endpoint           string   `json:"endpoint"`
	Backend            string   `json:"backend"`
	PromptLen          int      `json:"prompt_len"`
	TargetPromptTokens int      `json:"target_prompt_tokens"`
	PromptTokens       int      `json:"prompt_tokens"`
	MaxTokens          int      `json:"max_tokens"`
	Stream             bool     `json:"stream"`
	TTFTSeconds        *float64 `json:"ttft_s"`
	PrefillSeconds     *float64 `json:"prefill_seconds"`
	PrefillTokS        *float64 `json:"prefill_tok_s"`
	CompletionTokens   *int     `json:"completion_tokens"`
	TTFTSource         *string  `json:"ttft_source"`
	Status             string   `json:"status"`
	Scope              string   `json:"scope"`
	Error              string   `json:"error,omitempty"`
	HTTPStatus         *int     `json:"http_status,omitempty"`
}

type artifactHarness struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type artifactModel struct {
	Name      string `json:"name"`
	Precision string `json:"precision"`
}

type artifactMetrics struct {
	PrefillTokS *float64 `json:"prefill_tok_s"`
}

type artifactResults struct {
	Metrics artifactMetrics `json:"metrics"`
}

type artifactWitness struct {
	TestPath            string `json:"test_path"`
	ReproductionCommand string `json:"reproduction_command"`
}

type benchmarkArtifact struct {
	Schema     string          `json:"schema"`
	RunID      string          `json:"run_id"`
	Timestamp  string          `json:"timestamp"`
	FakCommit  string          `json:"fak_commit"`
	FakVersion string          `json:"fak_version"`
	Harness    artifactHarness `json:"harness"`
	Model      artifactModel   `json:"model"`
	Results    artifactResults `json:"results"`
	Witness    artifactWitness `json:"witness"`
}

// Manifest embeds Record so its fields sit at the manifest top level (mirroring Python's
// `manifest = dict(record)` then adding the lineage + envelope), then adds the two
// provenance blocks DecodeArtifact keys on.
type Manifest struct {
	Record
	Lineage           Lineage           `json:"lineage"`
	BenchmarkArtifact benchmarkArtifact `json:"benchmark_artifact"`
}

// --------------------------------------------------------------------------- //
// Time / lineage helpers
// --------------------------------------------------------------------------- //

func utcNow() string {
	return time.Now().UTC().Format("2006-01-02T15:04:05.000000Z")
}

// compactStamp is the UTC stamp for the land-dir name, matching the decode sweep's
// `date -u +%Y%m%dT%H%M%SZ`.
func compactStamp() string {
	return time.Now().UTC().Format("20060102T150405Z")
}

func envTrim(key string) string {
	return strings.TrimSpace(os.Getenv(key))
}

func resolveNode(override string) string {
	if s := strings.TrimSpace(override); s != "" {
		return s
	}
	if node := envTrim("FAK_BENCH_NODE"); node != "" {
		return node
	}
	if host, err := os.Hostname(); err == nil {
		if h := strings.TrimSpace(host); h != "" {
			return h
		}
	}
	return "node"
}

// gitCommit is the HEAD sha, honoring the FAK_BENCH_COMMIT override the decode sweep
// exports. Fail-soft to "unknown" so a lineage-free artifact can't ship.
func gitCommit(root string) string {
	if override := envTrim("FAK_BENCH_COMMIT"); override != "" {
		return override
	}
	out, rc := execGit(root, "rev-parse", "HEAD")
	if rc != 0 {
		return "unknown"
	}
	if sha := strings.TrimSpace(out); sha != "" {
		return sha
	}
	return "unknown"
}

func benchLineage(root, node string) Lineage {
	appVersion := envTrim("FAK_APP_VERSION")
	if appVersion == "" {
		appVersion = "unknown"
	}
	utc := envTrim("FAK_BENCH_UTC")
	if utc == "" {
		utc = utcNow()
	}
	return Lineage{
		LineageSchema: LineageSchema,
		AppVersion:    appVersion,
		UTC:           utc,
		GitCommit:     gitCommit(root),
		GoVersion:     runtime.Version(), // this emitter is now the Go harness
		Node:          node,
	}
}

func runID(l Lineage, promptLen int) string {
	stamp := strings.NewReplacer(":", "", "-", "").Replace(l.UTC)
	stamp = strings.TrimRight(stamp, "Z")
	if stamp == "" {
		stamp = "unknown-time"
	}
	commit := l.GitCommit
	if commit == "" {
		commit = "unknown"
	}
	if len(commit) > 12 {
		commit = commit[:12]
	}
	raw := strings.ToLower(fmt.Sprintf("%s-glm52-prefill-p%d-%s", stamp, promptLen, commit))
	var b strings.Builder
	for _, c := range raw {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' {
			b.WriteRune(c)
		} else {
			b.WriteRune('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

// --------------------------------------------------------------------------- //
// Pure planning (what --dry-run and the tests exercise; no I/O)
// --------------------------------------------------------------------------- //

// syntheticPrompt is a prompt that tokenizes to ~target tokens: a repeated short word,
// which most BPE tokenizers map to ~1 token each. The ACHIEVED prompt_tokens is read
// back from the endpoint usage in live mode, so this only needs to be prefill-dominant
// at scale, not exact.
func syntheticPrompt(target int) string {
	n := target
	if n < 1 {
		n = 1
	}
	parts := make([]string, n)
	for i := range parts {
		parts[i] = fillerWord
	}
	return strings.Join(parts, " ")
}

func prefillPayload(model string, promptLen, maxTokens int, stream bool) RequestBody {
	body := RequestBody{
		Model:       model,
		Messages:    []chatMessage{{Role: "user", Content: syntheticPrompt(promptLen)}},
		MaxTokens:   maxTokens,
		Temperature: 0,
		Stream:      stream,
	}
	if stream {
		// Ask for a final usage chunk so prompt_tokens is authoritative for tok/s.
		body.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return body
}

// landSubdir is one discoverable subdir per length, mirroring the decode sweep's
// per-config subdir convention. Empty land root → empty (land disabled).
func landSubdir(landRoot string, promptLen int) string {
	root := strings.TrimRight(landRoot, "/")
	return fmt.Sprintf("%s/p%d", root, promptLen)
}

// BuildPlan is the full sweep plan: one step per prompt length with its request body and
// land path. Pure — no network, no GPU, no filesystem.
func BuildPlan(model, landRoot string, lengths []int, maxTokens int, stream bool, fragileMinLen int) []PlanStep {
	if lengths == nil {
		lengths = append([]int(nil), PrefillLengths...)
	}
	plan := make([]PlanStep, 0, len(lengths))
	for _, promptLen := range lengths {
		body := prefillPayload(model, promptLen, maxTokens, stream)
		content := body.Messages[0].Content
		sub := ""
		if landRoot != "" {
			sub = landSubdir(landRoot, promptLen)
		}
		plan = append(plan, PlanStep{
			PromptLen:          promptLen,
			TargetPromptTokens: promptLen,
			MaxTokens:          maxTokens,
			Stream:             stream,
			PromptChars:        len(content),
			RequestBody:        body,
			LandSubdir:         sub,
			FragileOnSM80:      promptLen >= fragileMinLen,
		})
	}
	return plan
}

// --------------------------------------------------------------------------- //
// HTTP (live mode only — never reached in --dry-run)
// --------------------------------------------------------------------------- //

func asFloat(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case json.Number:
		f, err := t.Float64()
		return f, err == nil
	case string:
		f, err := strconv.ParseFloat(t, 64)
		return f, err == nil
	default:
		return 0, false
	}
}

func asInt(v any) (int, bool) {
	if f, ok := asFloat(v); ok {
		return int(f), true
	}
	return 0, false
}

func parseJSONObject(raw string) map[string]any {
	var data any
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		return nil
	}
	if m, ok := data.(map[string]any); ok {
		return m
	}
	return nil
}

type endpointGate struct {
	URL         string   `json:"url"`
	Reachable   bool     `json:"reachable"`
	HTTPStatus  int      `json:"http_status"`
	ModelIDs    []string `json:"model_ids"`
	BodyExcerpt string   `json:"body_excerpt"`
}

func excerpt(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func jsonGet(url string, timeout time.Duration) (int, map[string]any, string) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return 0, nil, err.Error()
	}
	req.Header.Set("Accept", "application/json")
	resp, err := httpDo(req, timeout)
	if err != nil {
		return 0, nil, err.Error()
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, parseJSONObject(string(raw)), excerpt(string(raw), 2000)
}

func endpointReachable(baseURL string, timeout time.Duration) endpointGate {
	url := strings.TrimRight(baseURL, "/") + "/models"
	status, data, body := jsonGet(url, timeout)
	ids := []string{}
	if data != nil {
		if rows, ok := data["data"].([]any); ok {
			for _, r := range rows {
				if row, ok := r.(map[string]any); ok {
					if id, ok := row["id"].(string); ok {
						ids = append(ids, id)
					}
				}
			}
		}
	}
	return endpointGate{
		URL:         url,
		Reachable:   status == 200,
		HTTPStatus:  status,
		ModelIDs:    ids,
		BodyExcerpt: excerpt(body, 500),
	}
}

// measurement is the raw per-request outcome the record is folded from.
type measurement struct {
	ok               bool
	httpStatus       int
	ttftSeconds      *float64
	totalSeconds     *float64
	promptTokens     *int
	completionTokens *int
	source           string
	errMsg           string
}

func fptr(f float64) *float64 { return &f }
func iptr(i int) *int         { return &i }

func round3(f float64) float64 { return math.Round(f*1000) / 1000 }
func round6(f float64) float64 { return math.Round(f*1e6) / 1e6 }

// measurePrefillStream times to the first content delta (~= the prefill forward pass);
// prompt_tokens comes from the final usage chunk when present.
func measurePrefillStream(url string, payload RequestBody, timeout time.Duration) measurement {
	data, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return measurement{ok: false, httpStatus: 0, errMsg: err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	started := time.Now()
	resp, err := httpDo(req, timeout)
	if err != nil {
		return measurement{ok: false, httpStatus: 0, errMsg: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		return measurement{ok: false, httpStatus: resp.StatusCode, errMsg: excerpt(string(raw), 500)}
	}
	var ttft *float64
	var promptTokens, completionTokens *int
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		chunkRaw := strings.TrimSpace(line[len("data:"):])
		if chunkRaw == "[DONE]" {
			break
		}
		chunk := parseJSONObject(chunkRaw)
		if chunk == nil {
			continue
		}
		if usage, ok := chunk["usage"].(map[string]any); ok {
			if pt, ok := asInt(usage["prompt_tokens"]); ok {
				promptTokens = iptr(pt)
			}
			if ct, ok := asInt(usage["completion_tokens"]); ok {
				completionTokens = iptr(ct)
			}
		}
		if ttft == nil {
			if choices, ok := chunk["choices"].([]any); ok && len(choices) > 0 {
				if c0, ok := choices[0].(map[string]any); ok {
					if delta, ok := c0["delta"].(map[string]any); ok {
						if content, ok := delta["content"].(string); ok && content != "" {
							ttft = fptr(round6(time.Since(started).Seconds()))
						}
					}
				}
			}
		}
	}
	total := round6(time.Since(started).Seconds())
	return measurement{
		ok:               ttft != nil,
		httpStatus:       200,
		ttftSeconds:      ttft,
		totalSeconds:     fptr(total),
		promptTokens:     promptTokens,
		completionTokens: completionTokens,
		source:           "stream-ttft",
	}
}

// measurePrefillBlocking is the non-streaming fallback: the whole request duration. With
// max_tokens=1 this is prefill-dominant; prompt_tokens comes from the response usage.
func measurePrefillBlocking(url string, payload RequestBody, timeout time.Duration) measurement {
	data, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return measurement{ok: false, httpStatus: 0, errMsg: err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	started := time.Now()
	resp, err := httpDo(req, timeout)
	if err != nil {
		return measurement{ok: false, httpStatus: 0, errMsg: err.Error()}
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return measurement{ok: false, httpStatus: resp.StatusCode, errMsg: excerpt(string(raw), 500)}
	}
	dur := round6(time.Since(started).Seconds())
	obj := parseJSONObject(string(raw))
	var promptTokens, completionTokens *int
	hasChoices := false
	if obj != nil {
		if _, ok := obj["choices"]; ok {
			hasChoices = true
		}
		if usage, ok := obj["usage"].(map[string]any); ok {
			if pt, ok := asInt(usage["prompt_tokens"]); ok {
				promptTokens = iptr(pt)
			}
			if ct, ok := asInt(usage["completion_tokens"]); ok {
				completionTokens = iptr(ct)
			}
		}
	}
	return measurement{
		ok:               hasChoices,
		httpStatus:       200,
		ttftSeconds:      fptr(dur),
		totalSeconds:     fptr(dur),
		promptTokens:     promptTokens,
		completionTokens: completionTokens,
		source:           "blocking-duration",
	}
}

// RecordForLength folds one raw measurement into the discoverable per-length record. The
// scope field is load-bearing — it travels with the number.
func RecordForLength(m measurement, model, endpoint string, promptLen, maxTokens int, stream bool) Record {
	promptTokens := promptLen
	if m.promptTokens != nil && *m.promptTokens != 0 {
		promptTokens = *m.promptTokens
	}
	var prefillTokS *float64
	if m.ok && m.ttftSeconds != nil && *m.ttftSeconds > 0 {
		prefillTokS = fptr(round3(float64(promptTokens) / *m.ttftSeconds))
	}
	status := "FAIL"
	if m.ok {
		status = "OK"
	}
	var source *string
	if m.source != "" {
		s := m.source
		source = &s
	}
	rec := Record{
		Schema:             RecordSchema,
		Model:              "glm_moe_dsa",
		ServedModel:        model,
		Endpoint:           endpoint,
		Backend:            fmt.Sprintf("fak serve @ %s", endpoint),
		PromptLen:          promptLen,
		TargetPromptTokens: promptLen,
		PromptTokens:       promptTokens,
		MaxTokens:          maxTokens,
		Stream:             stream,
		TTFTSeconds:        m.ttftSeconds,
		PrefillSeconds:     m.ttftSeconds,
		PrefillTokS:        prefillTokS,
		CompletionTokens:   m.completionTokens,
		TTFTSource:         source,
		Status:             status,
		Scope:              Scope,
	}
	if !m.ok {
		errMsg := m.errMsg
		if errMsg == "" {
			errMsg = "no first token observed"
		}
		rec.Error = errMsg
		rec.HTTPStatus = iptr(m.httpStatus)
	}
	return rec
}

// --------------------------------------------------------------------------- //
// Ledger land (mirror cmd/glmdsatput writeLedgerArtifact)
// --------------------------------------------------------------------------- //

// BuildManifest is the discoverable manifest: the record body verbatim + a top-level
// lineage block (what benchcli.DecodeArtifact keys on) + a benchmark_artifact envelope
// carrying a run_id.
func BuildManifest(record Record, lineage Lineage, promptLen int) Manifest {
	return Manifest{
		Record:  record,
		Lineage: lineage,
		BenchmarkArtifact: benchmarkArtifact{
			Schema:     BenchmarkArtifactSchema,
			RunID:      runID(lineage, promptLen),
			Timestamp:  lineage.UTC,
			FakCommit:  lineage.GitCommit,
			FakVersion: lineage.AppVersion,
			Harness:    artifactHarness{Name: "glm52_prefill_sweep", Version: "1.0.0"},
			Model:      artifactModel{Name: "glm_moe_dsa", Precision: "served"},
			Results:    artifactResults{Metrics: artifactMetrics{PrefillTokS: record.PrefillTokS}},
			Witness: artifactWitness{
				TestPath:            "internal/glm52prefillsweep/glm52prefillsweep_test.go",
				ReproductionCommand: "fak glm52-prefill-sweep --endpoint <url>/v1 --model " + record.ServedModel,
			},
		},
	}
}

func marshalIndent(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return append(b, '\n')
}

func resultsMarkdown(record Record) string {
	val := func(f *float64) string {
		if f == nil {
			return ""
		}
		return strconv.FormatFloat(*f, 'g', -1, 64)
	}
	tokS := "n/a"
	if record.PrefillTokS != nil {
		tokS = strconv.FormatFloat(*record.PrefillTokS, 'g', -1, 64)
	}
	lines := []string{
		"# fak NATIVE glm_moe_dsa PREFILL latency (pure-fak)",
		"",
		fmt.Sprintf("> **Scope (load-bearing):** `%s`", record.Scope),
		"",
		"| field | value |",
		"|---|---|",
		fmt.Sprintf("| endpoint | %s |", record.Endpoint),
		fmt.Sprintf("| served_model | %s |", record.ServedModel),
		fmt.Sprintf("| prompt_len (target) | %d |", record.TargetPromptTokens),
		fmt.Sprintf("| prompt_tokens (measured) | %d |", record.PromptTokens),
		fmt.Sprintf("| TTFT (prefill s) | %s |", val(record.TTFTSeconds)),
		fmt.Sprintf("| **PREFILL** | **%s tok/s** |", tokS),
		fmt.Sprintf("| status | %s |", record.Status),
		"",
		"This artifact carries a benchcli lineage + benchmark_artifact envelope, so it " +
			"is discoverable by fak's lineage index and bindable by `dos verify`.",
		"",
	}
	return strings.Join(lines, "\n")
}

// WriteLedgerArtifact lands one length as a discoverable artifact: manifest.json (lineage
// + envelope around the record), result.json (raw record, NO lineage so it is not
// double-counted by BuildLineageIndex), RESULTS.md. Returns the manifest path.
func WriteLedgerArtifact(landPath string, record Record, lineage Lineage) (string, error) {
	if err := os.MkdirAll(landPath, 0o755); err != nil {
		return "", err
	}
	manifest := BuildManifest(record, lineage, record.PromptLen)
	manifestPath := landPath + "/manifest.json"
	if err := os.WriteFile(manifestPath, marshalIndent(manifest), 0o644); err != nil {
		return "", err
	}
	if err := os.WriteFile(landPath+"/result.json", marshalIndent(record), 0o644); err != nil {
		return "", err
	}
	if err := os.WriteFile(landPath+"/RESULTS.md", []byte(resultsMarkdown(record)), 0o644); err != nil {
		return "", err
	}
	return manifestPath, nil
}

// --------------------------------------------------------------------------- //
// Report assembly
// --------------------------------------------------------------------------- //

// DryRunReport is the pure plan report (no endpoint or GPU contacted).
type DryRunReport struct {
	Schema      string     `json:"schema"`
	GeneratedAt string     `json:"generated_at"`
	Mode        string     `json:"mode"`
	DryRun      bool       `json:"dry_run"`
	Model       string     `json:"model"`
	BaseURL     string     `json:"base_url"`
	LandRoot    string     `json:"land_root"`
	LandEnabled bool       `json:"land_enabled"`
	Lengths     []int      `json:"lengths"`
	MaxTokens   int        `json:"max_tokens"`
	Stream      bool       `json:"stream"`
	Scope       string     `json:"scope"`
	Plan        []PlanStep `json:"plan"`
	Notes       []string   `json:"notes"`
}

// BuildDryRunReport builds the pure plan report.
func BuildDryRunReport(model, baseURL, landRoot string, lengths []int, maxTokens int, stream bool) DryRunReport {
	plan := BuildPlan(model, landRoot, lengths, maxTokens, stream, FragileMinLen)
	return DryRunReport{
		Schema:      SchemaVersion,
		GeneratedAt: utcNow(),
		Mode:        "plan",
		DryRun:      true,
		Model:       model,
		BaseURL:     baseURL,
		LandRoot:    landRoot,
		LandEnabled: landRoot != "",
		Lengths:     lengths,
		MaxTokens:   maxTokens,
		Stream:      stream,
		Scope:       Scope,
		Plan:        plan,
		Notes: []string{
			"PLAN ONLY: no endpoint or GPU was contacted; no prefill number is produced.",
			"The on-box sm_80 peer runs the live sweep (--endpoint) to produce the numbers.",
			fmt.Sprintf("Lengths >= %d are flagged fragile_on_sm80 (the note's P0 DSA "+
				"illegal-memory-access); the live sweep records a per-length FAIL and continues.", FragileMinLen),
		},
	}
}

type liveResult struct {
	Record   Record `json:"record"`
	Manifest string `json:"manifest"`
}

type liveSummary struct {
	Lengths int `json:"lengths"`
	OK      int `json:"ok"`
	Failed  int `json:"failed"`
}

// LiveReport is the live-sweep report.
type LiveReport struct {
	Schema       string       `json:"schema"`
	GeneratedAt  string       `json:"generated_at"`
	Mode         string       `json:"mode"`
	DryRun       bool         `json:"dry_run"`
	Model        string       `json:"model"`
	BaseURL      string       `json:"base_url"`
	Node         string       `json:"node"`
	LandRoot     string       `json:"land_root"`
	LandEnabled  bool         `json:"land_enabled"`
	Lengths      []int        `json:"lengths"`
	MaxTokens    int          `json:"max_tokens"`
	Stream       bool         `json:"stream"`
	Scope        string       `json:"scope"`
	EndpointGate endpointGate `json:"endpoint_gate"`
	Results      []liveResult `json:"results"`
	Aborted      string       `json:"aborted,omitempty"`
	Summary      *liveSummary `json:"summary,omitempty"`
}

func runLiveSweep(root, model, baseURL, landRoot string, lengths []int, maxTokens int, stream bool, httpTimeout, requestTimeout time.Duration, node string) LiveReport {
	reach := endpointReachable(baseURL, httpTimeout)
	report := LiveReport{
		Schema:       SchemaVersion,
		GeneratedAt:  utcNow(),
		Mode:         "live",
		DryRun:       false,
		Model:        model,
		BaseURL:      baseURL,
		Node:         node,
		LandRoot:     landRoot,
		LandEnabled:  landRoot != "",
		Lengths:      lengths,
		MaxTokens:    maxTokens,
		Stream:       stream,
		Scope:        Scope,
		EndpointGate: reach,
		Results:      []liveResult{},
	}
	if !reach.Reachable {
		report.Aborted = "endpoint not reachable"
		return report
	}
	lineage := benchLineage(root, node)
	chatURL := strings.TrimRight(baseURL, "/") + "/chat/completions"
	okCount := 0
	for _, promptLen := range lengths {
		payload := prefillPayload(model, promptLen, maxTokens, stream)
		var m measurement
		if stream {
			m = measurePrefillStream(chatURL, payload, requestTimeout)
		} else {
			m = measurePrefillBlocking(chatURL, payload, requestTimeout)
		}
		record := RecordForLength(m, model, baseURL, promptLen, maxTokens, stream)
		landed := ""
		if landRoot != "" {
			path, err := WriteLedgerArtifact(landSubdir(landRoot, promptLen), record, lineage)
			if err != nil {
				landed = "LAND_FAILED: " + err.Error()
			} else {
				landed = path
			}
		}
		if record.Status == "OK" {
			okCount++
		}
		report.Results = append(report.Results, liveResult{Record: record, Manifest: landed})
	}
	report.Summary = &liveSummary{Lengths: len(lengths), OK: okCount, Failed: len(lengths) - okCount}
	return report
}

// --------------------------------------------------------------------------- //
// CLI
// --------------------------------------------------------------------------- //

func parseLengths(raw string) ([]int, error) {
	if strings.TrimSpace(raw) == "" {
		return append([]int(nil), PrefillLengths...), nil
	}
	var out []int
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid length %q: %w", part, err)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return append([]int(nil), PrefillLengths...), nil
	}
	return out, nil
}

func defaultLandRoot(node, stamp string) string {
	return fmt.Sprintf("experiments/benchmark/runs/by-machine/%s/%s-glm52-prefill-sweep", node, stamp)
}

func writeOut(path string, v any) error {
	if dir := parentDir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, marshalIndent(v), 0o644)
}

// parentDir returns the directory portion of a path, using both separators so it works
// on Windows and POSIX.
func parentDir(path string) string {
	i := strings.LastIndexAny(path, `/\`)
	if i < 0 {
		return ""
	}
	return path[:i]
}

// Run is the CLI entry (Go port of the Python main). Returns the process exit code.
func Run(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("glm52-prefill-sweep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		endpoint    string
		model       string
		lengthsRaw  string
		maxTokens   int
		noStream    bool
		nodeArg     string
		stampArg    string
		outPath     string
		httpTimeout float64
		reqTimeout  float64
		dryRun      bool
	)
	fs.StringVar(&endpoint, "endpoint", "", "OpenAI-compatible GLM-5.2 fak serve endpoint (omit with --dry-run to only print the plan)")
	fs.StringVar(&endpoint, "base-url", "", "alias for --endpoint")
	fs.StringVar(&model, "model", DefaultModel, "served model id")
	fs.StringVar(&lengthsRaw, "lengths", "", "comma list of prompt lengths (default 128,512,2048,4096,8192)")
	fs.IntVar(&maxTokens, "max-tokens", DefaultMaxTokens, "generation cap per request; small/zero keeps the request prefill-dominant")
	fs.BoolVar(&noStream, "no-stream", false, "use a blocking request (whole-duration) instead of streaming TTFT")
	fs.StringVar(&nodeArg, "node", "", "override the land-dir node segment (default FAK_BENCH_NODE or hostname)")
	fs.StringVar(&stampArg, "stamp", "", "override the land-dir UTC stamp (default now, %Y%m%dT%H%M%SZ)")
	fs.StringVar(&outPath, "out", "experiments/glm52/prefill-sweep.json", "write the sweep report/plan JSON here")
	fs.Float64Var(&httpTimeout, "http-timeout-s", 15.0, "endpoint-reachability timeout")
	fs.Float64Var(&reqTimeout, "request-timeout-s", 900.0, "per-request timeout")
	fs.BoolVar(&dryRun, "dry-run", false, "print the full sweep plan WITHOUT hitting any endpoint or GPU")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	lengths, err := parseLengths(lengthsRaw)
	if err != nil {
		fmt.Fprintf(stderr, "glm52-prefill-sweep: %v\n", err)
		return 2
	}
	stream := !noStream
	node := resolveNode(nodeArg)
	stamp := strings.TrimSpace(stampArg)
	if stamp == "" {
		stamp = compactStamp()
	}
	// Default-on land; opt out with GLM_LAND_DIR="" (mirrors the decode sweep's
	// ${GLM_LAND_DIR-<default>} semantics: absent -> default, present -> value).
	landRoot, ok := os.LookupEnv("GLM_LAND_DIR")
	if !ok {
		landRoot = defaultLandRoot(node, stamp)
	}

	if dryRun || endpoint == "" {
		// Pure path: no git, no network. (The dry-run purity test patches execGit +
		// httpDo to fail and asserts neither is reached from here.)
		base := endpoint
		if base == "" {
			base = "(none: --dry-run)"
		}
		report := BuildDryRunReport(model, base, landRoot, lengths, maxTokens, stream)
		if err := writeOut(outPath, report); err != nil {
			fmt.Fprintf(stderr, "glm52-prefill-sweep: write %s: %v\n", outPath, err)
			return 1
		}
		printPlanSummary(stdout, report)
		return 0
	}

	// Live mode only: resolve the repo root for the lineage's git_commit.
	root := "."
	if top, rc := execGit(".", "rev-parse", "--show-toplevel"); rc == 0 {
		if t := strings.TrimSpace(top); t != "" {
			root = t
		}
	}
	report := runLiveSweep(root, model, endpoint, landRoot, lengths, maxTokens, stream,
		time.Duration(httpTimeout*float64(time.Second)), time.Duration(reqTimeout*float64(time.Second)), node)
	if err := writeOut(outPath, report); err != nil {
		fmt.Fprintf(stderr, "glm52-prefill-sweep: write %s: %v\n", outPath, err)
		return 1
	}
	if report.Summary != nil {
		fmt.Fprintln(stdout, string(mustIndent(report.Summary)))
	} else {
		fmt.Fprintln(stdout, string(mustIndent(map[string]any{"endpoint_gate": report.EndpointGate})))
	}
	if report.Aborted != "" {
		return 1
	}
	if report.Summary != nil && report.Summary.OK > 0 {
		return 0
	}
	return 1
}

func mustIndent(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return b
}

func printPlanSummary(stdout io.Writer, report DryRunReport) {
	fmt.Fprintf(stdout, "GLM-5.2 prefill sweep PLAN (dry-run) -- model=%s land_root=%s\n", report.Model, report.LandRoot)
	fmt.Fprintf(stdout, "scope: %s\n", report.Scope)
	for _, step := range report.Plan {
		fragile := ""
		if step.FragileOnSM80 {
			fragile = "  [fragile_on_sm80]"
		}
		fmt.Fprintf(stdout, "  P=%-5d max_tokens=%d prompt_chars=%-7d -> %s%s\n",
			step.PromptLen, step.MaxTokens, step.PromptChars, step.LandSubdir, fragile)
	}
}
