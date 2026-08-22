package trajectory

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// AuditSchema versions every row emitted by the cross-harness efficiency audit.
const AuditSchema = "fak-trajectory-audit/1"

const (
	AuditSourceClaude = "claude"
	AuditSourceCodex  = "codex"
)

// AuditSource is one supported harness transcript root. RootLabel is safe to
// persist; Root may be machine-private and is never written to an artifact.
type AuditSource struct {
	Name      string
	Root      string
	RootLabel string
}

// AuditOptions selects the transcript roots and comparison window.
type AuditOptions struct {
	Sources  []AuditSource
	Since    time.Duration
	Now      time.Time
	Baseline *AuditSummaryRow
}

// AuditTokens are the four disjoint, exact billing buckets normalized across
// harnesses. Codex's input_tokens includes its cached/cache-write subsets, so
// the Codex parser subtracts those exact fields before populating InputTokens.
type AuditTokens struct {
	InputTokens       int64 `json:"input_tokens"`
	OutputTokens      int64 `json:"output_tokens"`
	CacheCreateTokens int64 `json:"cache_create_tokens"`
	CacheReadTokens   int64 `json:"cache_read_tokens"`
}

func (t AuditTokens) inputTotal() int64 {
	return t.InputTokens + t.CacheCreateTokens + t.CacheReadTokens
}

func (t AuditTokens) accountedTotal() int64 {
	return t.inputTotal() + t.OutputTokens
}

func (t *AuditTokens) add(other AuditTokens) {
	t.InputTokens += other.InputTokens
	t.OutputTokens += other.OutputTokens
	t.CacheCreateTokens += other.CacheCreateTokens
	t.CacheReadTokens += other.CacheReadTokens
}

// AuditDenominatorRow is the denominator for one source parser. RecordTypes makes
// schema drift observable without treating harmless metadata records as usage.
type AuditDenominatorRow struct {
	Schema                string         `json:"schema"`
	Kind                  string         `json:"kind"`
	Source                string         `json:"source"`
	Root                  string         `json:"root"`
	RootPresent           bool           `json:"root_present"`
	FilesDiscovered       int            `json:"files_discovered"`
	FilesScanned          int            `json:"files_scanned"`
	Records               int            `json:"records"`
	UsageRecordsSeen      int            `json:"usage_records_seen"`
	UsageRecordsExact     int            `json:"usage_records_exact"`
	UsageRecordsApplied   int            `json:"usage_records_applied"`
	DuplicateUsageRecords int            `json:"duplicate_usage_records"`
	RefusedRecords        int            `json:"refused_records"`
	RecordTypes           map[string]int `json:"record_types"`
	TokenSemantics        string         `json:"token_semantics"`
}

// AuditTranscriptRow is one queryable, content-free transcript rollup.
type AuditTranscriptRow struct {
	Schema           string      `json:"schema"`
	Kind             string      `json:"kind"`
	Source           string      `json:"source"`
	TranscriptID     string      `json:"session_id"`
	SourcePath       string      `json:"source_path"`
	Models           []string    `json:"models"`
	Tokens           AuditTokens `json:"tokens"`
	ToolCalls        int         `json:"tool_calls"`
	ToolErrors       int         `json:"tool_errors"`
	RepeatedFailures int         `json:"repeated_failures"`
	MutationChurn    int         `json:"mutation_churn"`
	HookP95MS        *int64      `json:"hook_p95_ms"`
	UsageRecords     int         `json:"usage_records"`
}

// AuditBottleneckRow ranks sessions by exact accounted tokens. DominantBucket
// explains the largest token component without fabricating a cross-vendor price.
type AuditBottleneckRow struct {
	Schema          string `json:"schema"`
	Kind            string `json:"kind"`
	Rank            int    `json:"rank"`
	Source          string `json:"source"`
	TranscriptID    string `json:"session_id"`
	SourcePath      string `json:"source_path"`
	AccountedTokens int64  `json:"accounted_tokens"`
	DominantBucket  string `json:"dominant_bucket"`
	DominantTokens  int64  `json:"dominant_tokens"`
}

// AuditSummaryRow is the machine-wide exact rollup and baseline input.
type AuditSummaryRow struct {
	Schema              string      `json:"schema"`
	Kind                string      `json:"kind"`
	Sources             int         `json:"sources"`
	Transcripts         int         `json:"sessions"`
	FilesDiscovered     int         `json:"files_discovered"`
	FilesScanned        int         `json:"files_scanned"`
	Records             int         `json:"records"`
	UsageRecordsExact   int         `json:"usage_records_exact"`
	RefusedRecords      int         `json:"refused_records"`
	Tokens              AuditTokens `json:"tokens"`
	InputOutputRatio    *float64    `json:"input_output_ratio"`
	PromptWriteFraction *float64    `json:"prompt_write_fraction"`
	RepeatedFailures    int         `json:"repeated_failures"`
	MutationChurn       int         `json:"mutation_churn"`
	HookP95MS           *int64      `json:"hook_p95_ms"`
}

// AuditDeltaRow compares one higher-is-worse metric with a prior summary.
type AuditDeltaRow struct {
	Schema     string   `json:"schema"`
	Kind       string   `json:"kind"`
	Metric     string   `json:"metric"`
	Current    *float64 `json:"current"`
	Baseline   *float64 `json:"baseline"`
	Delta      *float64 `json:"delta"`
	Comparable bool     `json:"comparable"`
	Regression bool     `json:"regression"`
}

// AuditRefusalRow names a transcript shape the parser refused to estimate.
type AuditRefusalRow struct {
	Schema     string `json:"schema"`
	Kind       string `json:"kind"`
	Source     string `json:"source"`
	SourcePath string `json:"source_path"`
	Line       int    `json:"line"`
	Code       string `json:"code"`
	Detail     string `json:"detail"`
}

// AuditResult is the complete versioned artifact before rendering.
type AuditResult struct {
	Summary      AuditSummaryRow
	Denominators []AuditDenominatorRow
	Transcripts  []AuditTranscriptRow
	Bottlenecks  []AuditBottleneckRow
	Baseline     []AuditDeltaRow
	Refusals     []AuditRefusalRow
}

// DefaultAuditSources discovers the two supported harness homes.
func DefaultAuditSources() []AuditSource {
	home, _ := os.UserHomeDir()
	claudeHome := strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR"))
	if claudeHome == "" {
		claudeHome = filepath.Join(home, ".claude")
	}
	codexHome := strings.TrimSpace(os.Getenv("CODEX_HOME"))
	if codexHome == "" {
		codexHome = filepath.Join(home, ".codex")
	}
	return []AuditSource{
		{Name: AuditSourceClaude, Root: filepath.Join(claudeHome, "projects"), RootLabel: "claude/projects"},
		{Name: AuditSourceCodex, Root: filepath.Join(codexHome, "sessions"), RootLabel: "codex/sessions"},
	}
}

// RunAudit reads each selected file once and returns a deterministic rollup.
func RunAudit(opts AuditOptions) (AuditResult, error) {
	if len(opts.Sources) == 0 {
		opts.Sources = DefaultAuditSources()
	}
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}

	result := AuditResult{}
	allHookDurations := []int64{}
	for _, source := range opts.Sources {
		denominator, transcripts, refusals, hookDurations, err := auditSource(source, opts)
		if err != nil {
			return AuditResult{}, err
		}
		result.Denominators = append(result.Denominators, denominator)
		result.Transcripts = append(result.Transcripts, transcripts...)
		result.Refusals = append(result.Refusals, refusals...)
		allHookDurations = append(allHookDurations, hookDurations...)
	}

	sort.Slice(result.Denominators, func(i, j int) bool {
		if result.Denominators[i].Source != result.Denominators[j].Source {
			return result.Denominators[i].Source < result.Denominators[j].Source
		}
		return result.Denominators[i].Root < result.Denominators[j].Root
	})
	sort.Slice(result.Transcripts, func(i, j int) bool {
		if result.Transcripts[i].Source != result.Transcripts[j].Source {
			return result.Transcripts[i].Source < result.Transcripts[j].Source
		}
		if result.Transcripts[i].TranscriptID != result.Transcripts[j].TranscriptID {
			return result.Transcripts[i].TranscriptID < result.Transcripts[j].TranscriptID
		}
		return result.Transcripts[i].SourcePath < result.Transcripts[j].SourcePath
	})
	sort.Slice(result.Refusals, func(i, j int) bool {
		a, b := result.Refusals[i], result.Refusals[j]
		if a.Source != b.Source {
			return a.Source < b.Source
		}
		if a.SourcePath != b.SourcePath {
			return a.SourcePath < b.SourcePath
		}
		if a.Line != b.Line {
			return a.Line < b.Line
		}
		return a.Code < b.Code
	})

	result.Summary = summarizeAudit(result.Denominators, result.Transcripts, allHookDurations)
	result.Bottlenecks = rankAuditBottlenecks(result.Transcripts)
	if opts.Baseline != nil {
		result.Baseline = auditBaselineDeltas(result.Summary, *opts.Baseline)
	}
	return result, nil
}

func auditSource(source AuditSource, opts AuditOptions) (AuditDenominatorRow, []AuditTranscriptRow, []AuditRefusalRow, []int64, error) {
	denominator := AuditDenominatorRow{
		Schema: AuditSchema, Kind: "source_denominator", Source: source.Name,
		Root: source.RootLabel, RecordTypes: map[string]int{},
	}
	switch source.Name {
	case AuditSourceClaude:
		denominator.TokenSemantics = "message usage buckets are disjoint; duplicate message ids are counted once"
	case AuditSourceCodex:
		denominator.TokenSemantics = "final cumulative input per segment; only a versioned task_started boundary may begin a segment after a decrease; cached/cache-write subsets remain exact subtraction"
	default:
		return denominator, nil, nil, nil, fmt.Errorf("trajectory audit: source %q has no parser", source.Name)
	}

	info, err := os.Stat(source.Root)
	if err != nil {
		if os.IsNotExist(err) {
			return denominator, nil, nil, nil, nil
		}
		return denominator, nil, nil, nil, fmt.Errorf("trajectory audit: stat %s root: %w", source.Name, err)
	}
	if !info.IsDir() {
		return denominator, nil, nil, nil, fmt.Errorf("trajectory audit: %s root is not a directory", source.Name)
	}
	denominator.RootPresent = true

	var files []string
	err = filepath.WalkDir(source.Root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !entry.IsDir() && strings.EqualFold(filepath.Ext(entry.Name()), ".jsonl") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return denominator, nil, nil, nil, fmt.Errorf("trajectory audit: discover %s transcripts: %w", source.Name, err)
	}
	sort.Strings(files)
	denominator.FilesDiscovered = len(files)

	var transcripts []AuditTranscriptRow
	var refusals []AuditRefusalRow
	var hookDurations []int64
	for _, path := range files {
		if opts.Since > 0 {
			stat, statErr := os.Stat(path)
			if statErr != nil {
				return denominator, nil, nil, nil, fmt.Errorf("trajectory audit: stat transcript: %w", statErr)
			}
			if stat.ModTime().Before(opts.Now.Add(-opts.Since)) {
				continue
			}
		}
		rel, relErr := filepath.Rel(source.Root, path)
		if relErr != nil {
			return denominator, nil, nil, nil, fmt.Errorf("trajectory audit: relativize transcript: %w", relErr)
		}
		rel = filepath.ToSlash(rel)
		row, fileRefusals, fileHooks, parseErr := parseAuditFile(source.Name, path, rel, &denominator)
		if parseErr != nil {
			return denominator, nil, nil, nil, parseErr
		}
		denominator.FilesScanned++
		transcripts = append(transcripts, row)
		refusals = append(refusals, fileRefusals...)
		hookDurations = append(hookDurations, fileHooks...)
	}
	return denominator, transcripts, refusals, hookDurations, nil
}

func summarizeAudit(denominators []AuditDenominatorRow, transcripts []AuditTranscriptRow, hookDurations []int64) AuditSummaryRow {
	summary := AuditSummaryRow{Schema: AuditSchema, Kind: "summary", Sources: len(denominators), Transcripts: len(transcripts)}
	for _, row := range denominators {
		summary.FilesDiscovered += row.FilesDiscovered
		summary.FilesScanned += row.FilesScanned
		summary.Records += row.Records
		summary.UsageRecordsExact += row.UsageRecordsExact
		summary.RefusedRecords += row.RefusedRecords
	}
	for _, transcript := range transcripts {
		summary.Tokens.add(transcript.Tokens)
		summary.RepeatedFailures += transcript.RepeatedFailures
		summary.MutationChurn += transcript.MutationChurn
	}
	if summary.Tokens.OutputTokens > 0 {
		v := float64(summary.Tokens.inputTotal()) / float64(summary.Tokens.OutputTokens)
		summary.InputOutputRatio = &v
	}
	if summary.Tokens.inputTotal() > 0 {
		v := float64(summary.Tokens.CacheCreateTokens) / float64(summary.Tokens.inputTotal())
		summary.PromptWriteFraction = &v
	}
	summary.HookP95MS = auditPercentile(hookDurations, 95)
	return summary
}

func rankAuditBottlenecks(transcripts []AuditTranscriptRow) []AuditBottleneckRow {
	rows := make([]AuditBottleneckRow, 0, len(transcripts))
	for _, transcript := range transcripts {
		bucket, tokens := dominantAuditBucket(transcript.Tokens)
		rows = append(rows, AuditBottleneckRow{
			Schema: AuditSchema, Kind: "bottleneck", Source: transcript.Source,
			TranscriptID: transcript.TranscriptID, SourcePath: transcript.SourcePath,
			AccountedTokens: transcript.Tokens.accountedTotal(), DominantBucket: bucket, DominantTokens: tokens,
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].AccountedTokens != rows[j].AccountedTokens {
			return rows[i].AccountedTokens > rows[j].AccountedTokens
		}
		if rows[i].Source != rows[j].Source {
			return rows[i].Source < rows[j].Source
		}
		if rows[i].TranscriptID != rows[j].TranscriptID {
			return rows[i].TranscriptID < rows[j].TranscriptID
		}
		return rows[i].SourcePath < rows[j].SourcePath
	})
	for i := range rows {
		rows[i].Rank = i + 1
	}
	return rows
}

func dominantAuditBucket(tokens AuditTokens) (string, int64) {
	values := []struct {
		name  string
		value int64
	}{
		{"input", tokens.InputTokens},
		{"cache_create", tokens.CacheCreateTokens},
		{"cache_read", tokens.CacheReadTokens},
		{"output", tokens.OutputTokens},
	}
	best := values[0]
	for _, value := range values[1:] {
		if value.value > best.value {
			best = value
		}
	}
	return best.name, best.value
}

func auditBaselineDeltas(current, baseline AuditSummaryRow) []AuditDeltaRow {
	floatOfInt := func(v int) *float64 { f := float64(v); return &f }
	floatOfInt64 := func(v *int64) *float64 {
		if v == nil {
			return nil
		}
		f := float64(*v)
		return &f
	}
	return []AuditDeltaRow{
		newAuditDelta("input_output_ratio", current.InputOutputRatio, baseline.InputOutputRatio),
		newAuditDelta("prompt_write_fraction", current.PromptWriteFraction, baseline.PromptWriteFraction),
		newAuditDelta("repeated_failures", floatOfInt(current.RepeatedFailures), floatOfInt(baseline.RepeatedFailures)),
		newAuditDelta("hook_p95_ms", floatOfInt64(current.HookP95MS), floatOfInt64(baseline.HookP95MS)),
	}
}

func newAuditDelta(metric string, current, baseline *float64) AuditDeltaRow {
	row := AuditDeltaRow{Schema: AuditSchema, Kind: "baseline_delta", Metric: metric, Current: current, Baseline: baseline}
	if current == nil || baseline == nil {
		return row
	}
	delta := *current - *baseline
	row.Delta = &delta
	row.Comparable = true
	row.Regression = delta > 0
	return row
}

// ReadAuditBaseline loads the versioned summary row from a previous JSONL artifact.
func ReadAuditBaseline(path string) (*AuditSummaryRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("trajectory audit: open baseline: %w", err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		var envelope struct {
			Schema string `json:"schema"`
			Kind   string `json:"kind"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &envelope); err != nil {
			return nil, fmt.Errorf("trajectory audit: decode baseline row: %w", err)
		}
		if envelope.Schema != AuditSchema {
			return nil, fmt.Errorf("trajectory audit: baseline schema %q has no reader", envelope.Schema)
		}
		if envelope.Kind != "summary" {
			continue
		}
		var summary AuditSummaryRow
		if err := json.Unmarshal(scanner.Bytes(), &summary); err != nil {
			return nil, fmt.Errorf("trajectory audit: decode baseline summary: %w", err)
		}
		return &summary, nil
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("trajectory audit: scan baseline: %w", err)
	}
	return nil, fmt.Errorf("trajectory audit: baseline has no %s summary row", AuditSchema)
}

// WriteAuditJSONL writes stable, versioned rows suitable for direct querying.
func WriteAuditJSONL(w io.Writer, result AuditResult) error {
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	rows := []any{result.Summary}
	for _, row := range result.Denominators {
		rows = append(rows, row)
	}
	for _, row := range result.Transcripts {
		rows = append(rows, row)
	}
	for _, row := range result.Bottlenecks {
		rows = append(rows, row)
	}
	for _, row := range result.Baseline {
		rows = append(rows, row)
	}
	for _, row := range result.Refusals {
		rows = append(rows, row)
	}
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			return fmt.Errorf("trajectory audit: encode JSONL: %w", err)
		}
	}
	return nil
}
