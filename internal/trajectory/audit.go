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

const qwenTopContributorConcentrationThreshold = 0.50

const (
	auditRepeatedFailureSemantics     = "actionable-identical-failed-action/2"
	auditRepeatedFailureNormalization = "per_session"
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
	Sources        []AuditSource
	Since          time.Duration
	Now            time.Time
	Baseline       *AuditSummaryRow
	SchemaBaseline *AuditSchemaBaseline
	// UserContains keeps only transcripts whose user-authored prompts contain this case-insensitive literal. Tool output, system text, and injected context never select a transcript; there is no raw-byte fallback.
	UserContains string
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

// AuditCodexCacheObservation keeps bounded per-request cache telemetry separate
// from cumulative accounting. The provider path is configuration provenance,
// not proof of where provider-side cache state physically resided.
type AuditCodexCacheObservation struct {
	TranscriptProducer               string `json:"transcript_producer"`
	ModelProvider                    string `json:"model_provider,omitempty"`
	ModelProviderSource              string `json:"model_provider_source"`
	LastTokenUsageCachedInputSamples int    `json:"last_token_usage_cached_input_samples"`
	LastTokenUsageCachedInputMin     *int64 `json:"last_token_usage_cached_input_min,omitempty"`
	LastTokenUsageCachedInputMax     *int64 `json:"last_token_usage_cached_input_max,omitempty"`
	PhysicalProviderCacheResidency   string `json:"physical_provider_cache_residency"`
	FakOwnedCacheCoverage            string `json:"fak_owned_cache_coverage"`
}

// AuditDenominatorRow is the denominator for one source parser. RecordTypes makes
// schema drift observable without treating harmless metadata records as usage;
// FixtureFilesExcluded makes narrowly classified fixture exclusions observable.
type AuditDenominatorRow struct {
	Schema                string         `json:"schema"`
	Kind                  string         `json:"kind"`
	Source                string         `json:"source"`
	Root                  string         `json:"root"`
	RootPresent           bool           `json:"root_present"`
	FilesDiscovered       int            `json:"files_discovered"`
	FilesScanned          int            `json:"files_scanned"`
	FilesMatched          int            `json:"files_matched,omitempty"`
	FixtureFilesExcluded  int            `json:"fixture_files_excluded"`
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
	Schema               string                        `json:"schema"`
	Kind                 string                        `json:"kind"`
	Source               string                        `json:"source"`
	TranscriptID         string                        `json:"session_id"`
	SourcePath           string                        `json:"source_path"`
	Models               []string                      `json:"models"`
	BuildIdentities      []AuditBuildIdentity          `json:"build_identities"`
	Tokens               AuditTokens                   `json:"tokens"`
	CodexCache           *AuditCodexCacheObservation   `json:"codex_cache,omitempty"`
	ToolCalls            int                           `json:"tool_calls"`
	ToolErrors           int                           `json:"tool_errors"`
	Distribution         []AuditDistributionRow        `json:"distribution,omitempty"`
	ToolDistribution     []AuditDistributionRow        `json:"tool_distribution,omitempty"`
	ToolResults          []AuditToolResultRow          `json:"tool_results,omitempty"`
	StorageDistribution  []AuditStorageRow             `json:"storage_distribution,omitempty"`
	UnknownExemplars     AuditUnknownExemplarReservoir `json:"unknown_exemplars"`
	RepeatedFailures     int                           `json:"repeated_failures"`
	ExpectedWaitTimeouts int                           `json:"expected_wait_timeouts"`
	MutationChurn        int                           `json:"mutation_churn"`
	MutationChurnEvents  []QwenMutationChurn           `json:"mutation_churn_events,omitempty"`
	HookP95MS            *int64                        `json:"hook_p95_ms"`
	UsageRecords         int                           `json:"usage_records"`
	SourcePaths          []string                      `json:"source_paths,omitempty"`
	usageByID            map[string]AuditTokens
	failureCounts        map[string]int
	schemaShapes         map[string]auditShapeSet
	fragmentDigest       string
	fragmentDigests      map[string]struct{}
}

// AuditBottleneckRow ranks sessions by exact accounted tokens. DominantBucket
// explains the largest token component without fabricating a cross-vendor price.
type AuditBottleneckRow struct {
	Schema          string   `json:"schema"`
	Kind            string   `json:"kind"`
	Rank            int      `json:"rank"`
	Source          string   `json:"source"`
	TranscriptID    string   `json:"session_id"`
	SourcePath      string   `json:"source_path"`
	SourcePaths     []string `json:"source_paths,omitempty"`
	AccountedTokens int64    `json:"accounted_tokens"`
	DominantBucket  string   `json:"dominant_bucket"`
	DominantTokens  int64    `json:"dominant_tokens"`
}

// AuditSummaryRow is the machine-wide exact rollup and baseline input.
type AuditSummaryRow struct {
	Schema                              string                        `json:"schema"`
	Kind                                string                        `json:"kind"`
	Sources                             int                           `json:"sources"`
	Transcripts                         int                           `json:"sessions"`
	RawFragments                        int                           `json:"raw_fragments"`
	CanonicalTranscripts                int                           `json:"canonical_transcripts"`
	FilesDiscovered                     int                           `json:"files_discovered"`
	FilesScanned                        int                           `json:"files_scanned"`
	FixtureFilesExcluded                int                           `json:"fixture_files_excluded"`
	FilesMatched                        int                           `json:"files_matched,omitempty"`
	Records                             int                           `json:"records"`
	UsageRecordsExact                   int                           `json:"usage_records_exact"`
	RefusedRecords                      int                           `json:"refused_records"`
	Tokens                              AuditTokens                   `json:"tokens"`
	InputOutputRatio                    *float64                      `json:"input_output_ratio"`
	PromptWriteFraction                 *float64                      `json:"prompt_write_fraction"`
	RepeatedFailures                    int                           `json:"repeated_failures"`
	RepeatedFailuresPerSession          *float64                      `json:"repeated_failures_per_session"`
	RepeatedFailureSemantics            string                        `json:"repeated_failure_semantics"`
	RepeatedFailureNormalization        string                        `json:"repeated_failure_normalization"`
	ExpectedWaitTimeouts                int                           `json:"expected_wait_timeouts"`
	MutationChurn                       int                           `json:"mutation_churn"`
	HookP95MS                           *int64                        `json:"hook_p95_ms"`
	DistinctTranscripts                 int                           `json:"distinct_transcripts"`
	DuplicateFragments                  int                           `json:"duplicate_fragments"`
	EmptyUsageFiles                     int                           `json:"empty_usage_files"`
	ToolCalls                           int                           `json:"tool_calls"`
	ToolErrors                          int                           `json:"tool_errors"`
	DistributionUnit                    string                        `json:"distribution_unit"`
	DistributionProvenance              string                        `json:"distribution_provenance"`
	Distribution                        []AuditDistributionRow        `json:"distribution,omitempty"`
	ToolDistribution                    []AuditDistributionRow        `json:"tool_distribution,omitempty"`
	ToolResults                         []AuditToolResultRow          `json:"tool_results,omitempty"`
	StorageUnit                         string                        `json:"storage_unit"`
	StorageDistribution                 []AuditStorageRow             `json:"storage_distribution,omitempty"`
	UnknownExemplars                    AuditUnknownExemplarReservoir `json:"unknown_exemplars"`
	ToolErrorFraction                   *float64                      `json:"tool_error_fraction"`
	TopTenTokenFraction                 *float64                      `json:"top_ten_token_fraction"`
	QwenTopContributor                  *string                       `json:"qwen_top_contributor"`
	QwenTopContributorTokens            *int64                        `json:"qwen_top_contributor_tokens"`
	QwenTotalTokens                     *int64                        `json:"qwen_total_tokens"`
	QwenTopContributorTokenFraction     *float64                      `json:"qwen_top_contributor_token_fraction"`
	QwenTokenConcentrationThreshold     float64                       `json:"qwen_token_concentration_threshold"`
	QwenTopContributorTokenConcentrated *bool                         `json:"qwen_top_contributor_token_concentrated"`
}

// AuditDeltaRow compares one higher-is-worse metric with a prior summary.
type AuditDeltaRow struct {
	Schema               string   `json:"schema"`
	Kind                 string   `json:"kind"`
	Metric               string   `json:"metric"`
	Current              *float64 `json:"current"`
	Baseline             *float64 `json:"baseline"`
	Delta                *float64 `json:"delta"`
	RawCurrent           *int     `json:"raw_current,omitempty"`
	RawBaseline          *int     `json:"raw_baseline,omitempty"`
	CurrentExposure      *int     `json:"current_exposure,omitempty"`
	BaselineExposure     *int     `json:"baseline_exposure,omitempty"`
	Normalization        string   `json:"normalization,omitempty"`
	RawComparable        *bool    `json:"raw_comparable,omitempty"`
	NormalizedCurrent    *float64 `json:"normalized_current,omitempty"`
	NormalizedBaseline   *float64 `json:"normalized_baseline,omitempty"`
	NormalizedDelta      *float64 `json:"normalized_delta,omitempty"`
	NormalizedComparable *bool    `json:"normalized_comparable,omitempty"`
	NormalizedRegression *bool    `json:"normalized_regression,omitempty"`
	Comparable           bool     `json:"comparable"`
	ComparabilityStatus  string   `json:"comparability_status"`
	ComparabilityReason  string   `json:"comparability_reason,omitempty"`
	Regression           bool     `json:"regression"`
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

// AuditCorpusRow binds snapshot-backed output to the verified, content-addressed
// input corpus. It is absent for ordinary live audits, preserving their bytes.
type AuditCorpusRow struct {
	Schema       string `json:"schema"`
	Kind         string `json:"kind"`
	CorpusSchema string `json:"corpus_schema"`
	Digest       string `json:"corpus_digest"`
	Verified     bool   `json:"verified"`
	Files        int    `json:"files"`
	Bytes        int64  `json:"bytes"`
}

// AuditResult is the complete versioned artifact before rendering.
type AuditConclusionStatus struct {
	BroadEfficiencySupported bool `json:"broad_efficiency_supported"`
	RefusalCount             int  `json:"refusal_count"`
	SchemaDriftCount         int  `json:"schema_drift_count"`
	BreakingSchemaDrift      int  `json:"breaking_schema_drift"`
}

type AuditResult struct {
	Corpus            *AuditCorpusRow       `json:"corpus,omitempty"`
	ConclusionStatus  AuditConclusionStatus `json:"conclusion_status"`
	Summary           AuditSummaryRow
	Denominators      []AuditDenominatorRow
	Transcripts       []AuditTranscriptRow
	Bottlenecks       []AuditBottleneckRow
	Baseline          []AuditDeltaRow
	EventSchemas      []AuditEventSchemaRow `json:"event_schemas,omitempty"`
	SchemaDrift       []AuditSchemaDriftRow `json:"schema_drift,omitempty"`
	Refusals          []AuditRefusalRow     `json:"refusals,omitempty"`
	ToolErrorFamilies []QwenToolErrorFamily `json:"tool_error_families,omitempty"`
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

// RunAudit returns a deterministic rollup over the selected transcript roots.
func RunAudit(opts AuditOptions) (AuditResult, error) {
	if len(opts.Sources) == 0 {
		opts.Sources = DefaultAuditSources()
	}
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}

	result := AuditResult{}
	allHookDurations := []int64{}
	allToolErrorEvents := []QwenToolErrorEvent{}
	for _, source := range opts.Sources {
		denominator, transcripts, refusals, hookDurations, toolErrorEvents, err := auditSource(source, opts)
		if err != nil {
			return AuditResult{}, err
		}
		result.Denominators = append(result.Denominators, denominator)
		result.Transcripts = append(result.Transcripts, transcripts...)
		result.Refusals = append(result.Refusals, refusals...)
		allToolErrorEvents = append(allToolErrorEvents, toolErrorEvents...)
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
	canonical, canonicalRefusals := canonicalAuditTranscripts(result.Transcripts)
	result.Refusals = append(result.Refusals, canonicalRefusals...)
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
	addAuditCanonicalRefusalDenominators(result.Denominators, canonicalRefusals)
	result.Summary = summarizeAudit(result.Denominators, canonical, allHookDurations)
	result.Summary.RefusedRecords = len(result.Refusals)
	result.Summary.RawFragments = len(result.Transcripts)
	result.Summary.CanonicalTranscripts = len(canonical)
	result.Bottlenecks = rankAuditBottlenecks(canonical)
	result.ToolErrorFamilies = rankQwenToolErrorFamilies(allToolErrorEvents)
	if opts.Baseline != nil {
		result.Baseline = auditBaselineDeltas(result.Summary, *opts.Baseline)
	}
	currentSchema := auditCurrentSchemaEvents(result.Transcripts)
	for _, event := range currentSchema {
		result.EventSchemas = append(result.EventSchemas, AuditEventSchemaRow{Schema: AuditSchema, Kind: "event_schema", AuditSchemaEvent: event})
	}
	schemaBaseline := opts.SchemaBaseline
	if schemaBaseline == nil {
		loaded, err := DefaultAuditSchemaBaseline()
		if err != nil {
			return AuditResult{}, err
		}
		schemaBaseline = &loaded
	}
	result.SchemaDrift = compareAuditSchema(currentSchema, auditSchemaBaselineForPresentSources(*schemaBaseline, result.Denominators))
	breakingSchemaDrift := 0
	for _, row := range result.SchemaDrift {
		if row.Compatibility == "breaking" {
			breakingSchemaDrift++
		}
	}
	result.ConclusionStatus = AuditConclusionStatus{
		BroadEfficiencySupported: len(result.Refusals) == 0,
		RefusalCount:             len(result.Refusals),
		SchemaDriftCount:         len(result.SchemaDrift),
		BreakingSchemaDrift:      breakingSchemaDrift,
	}
	return result, nil
}

func addAuditCanonicalRefusalDenominators(denominators []AuditDenominatorRow, refusals []AuditRefusalRow) {
	for _, refusal := range refusals {
		for i := range denominators {
			if denominators[i].Source == refusal.Source {
				denominators[i].RefusedRecords++
				break
			}
		}
	}
}

func auditSource(source AuditSource, opts AuditOptions) (AuditDenominatorRow, []AuditTranscriptRow, []AuditRefusalRow, []int64, []QwenToolErrorEvent, error) {
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
		return denominator, nil, nil, nil, nil, fmt.Errorf("trajectory audit: source %q has no parser", source.Name)
	}

	info, err := os.Stat(source.Root)
	if err != nil {
		if os.IsNotExist(err) {
			return denominator, nil, nil, nil, nil, nil
		}
		return denominator, nil, nil, nil, nil, fmt.Errorf("trajectory audit: stat %s root: %w", source.Name, err)
	}
	if !info.IsDir() {
		return denominator, nil, nil, nil, nil, fmt.Errorf("trajectory audit: %s root is not a directory", source.Name)
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
		return denominator, nil, nil, nil, nil, fmt.Errorf("trajectory audit: discover %s transcripts: %w", source.Name, err)
	}
	sort.Strings(files)
	denominator.FilesDiscovered = len(files)

	var transcripts []AuditTranscriptRow
	var refusals []AuditRefusalRow
	var hookDurations []int64
	var toolErrorEvents []QwenToolErrorEvent
	// Canonical merging still receives every row so duplicate-usage conflicts
	// remain visible. These file-derived side channels have no canonical row
	// representation, so deduplicate them here by the same transcript identity
	// and exact fragment digest used by canonicalAuditTranscripts.
	sideChannelFragments := map[string]struct{}{}
	for _, path := range files {
		if opts.Since > 0 {
			stat, statErr := os.Stat(path)
			if statErr != nil {
				return denominator, nil, nil, nil, nil, fmt.Errorf("trajectory audit: stat transcript: %w", statErr)
			}
			if stat.ModTime().Before(opts.Now.Add(-opts.Since)) {
				continue
			}
		}
		rel, relErr := filepath.Rel(source.Root, path)
		if relErr != nil {
			return denominator, nil, nil, nil, nil, fmt.Errorf("trajectory audit: relativize transcript: %w", relErr)
		}
		rel = filepath.ToSlash(rel)
		if opts.UserContains != "" {
			matched, matchErr := auditFileUserContains(path, opts.UserContains)
			if matchErr != nil {
				return denominator, nil, nil, nil, nil, matchErr
			}
			if !matched {
				continue
			}
			denominator.FilesMatched++
		}
		if source.Name == AuditSourceClaude && auditIsClaudePytestFixture(path, rel) {
			denominator.FixtureFilesExcluded++
			continue
		}
		row, fileRefusals, fileHooks, fileToolErrors, parseErr := parseAuditFile(source.Name, path, rel, &denominator)
		if parseErr != nil {
			return denominator, nil, nil, nil, nil, parseErr
		}
		denominator.FilesScanned++
		transcripts = append(transcripts, row)
		refusals = append(refusals, fileRefusals...)
		sideChannelKey := row.TranscriptID + "\x00" + row.fragmentDigest
		_, duplicateFragment := sideChannelFragments[sideChannelKey]
		duplicateFragment = duplicateFragment && row.fragmentDigest != ""
		if !duplicateFragment {
			hookDurations = append(hookDurations, fileHooks...)
			toolErrorEvents = append(toolErrorEvents, fileToolErrors...)
			if row.fragmentDigest != "" {
				sideChannelFragments[sideChannelKey] = struct{}{}
			}
		}
	}
	return denominator, transcripts, refusals, hookDurations, toolErrorEvents, nil
}

func auditFileUserContains(path, needle string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("trajectory audit: open transcript for --user-contains: %w", err)
	}
	defer file.Close()

	needle = strings.ToLower(needle)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		var record map[string]any
		decoder := json.NewDecoder(strings.NewReader(scanner.Text()))
		if decoder.Decode(&record) != nil {
			continue
		}
		if strings.Contains(strings.ToLower(auditUserText(record)), needle) {
			return true, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return false, fmt.Errorf("trajectory audit: scan transcript for --user-contains: %w", err)
	}
	return false, nil
}

func auditUserText(record map[string]any) string {
	kind, _ := record["type"].(string)
	switch kind {
	case "user": // Claude transcript row.
		message, _ := record["message"].(map[string]any)
		if role, _ := message["role"].(string); role == "user" {
			return auditText(message["content"])
		}
	case "response_item": // Codex transcript row.
		payload, _ := record["payload"].(map[string]any)
		if itemType, _ := payload["type"].(string); itemType == "message" {
			if role, _ := payload["role"].(string); role == "user" {
				return auditCodexUserText(payload["content"])
			}
		}
	}
	return ""
}

func auditCodexUserText(content any) string {
	parts, ok := content.([]any)
	if !ok {
		text := auditText(content)
		if auditCodexInjectedUserEnvelope(text) {
			return ""
		}
		return text
	}
	var builder strings.Builder
	for _, part := range parts {
		text := auditText(part)
		if auditCodexInjectedUserEnvelope(text) {
			continue
		}
		builder.WriteString(text)
		builder.WriteByte(' ')
	}
	return builder.String()
}

// Codex records repository instructions and launch environment as content
// parts in a role=user message even though the harness supplied both. Each has
// a complete tagged envelope, unlike operator task text in the same record.
func auditCodexInjectedUserEnvelope(text string) bool {
	text = strings.TrimSpace(text)
	if strings.HasPrefix(text, "<environment_context>") && strings.HasSuffix(text, "</environment_context>") {
		return true
	}
	if !strings.HasPrefix(text, "# AGENTS.md instructions for ") || !strings.HasSuffix(text, "</INSTRUCTIONS>") {
		return false
	}
	return strings.Contains(text, "\n<INSTRUCTIONS>")
}

func summarizeAudit(denominators []AuditDenominatorRow, transcripts []AuditTranscriptRow, hookDurations []int64) AuditSummaryRow {
	summary := AuditSummaryRow{
		Schema: AuditSchema, Kind: "summary", Sources: len(denominators), Transcripts: len(transcripts),
		RepeatedFailureSemantics: auditRepeatedFailureSemantics, RepeatedFailureNormalization: auditRepeatedFailureNormalization,
	}
	for _, row := range denominators {
		summary.FilesDiscovered += row.FilesDiscovered
		summary.FilesScanned += row.FilesScanned
		summary.FilesMatched += row.FilesMatched
		summary.FixtureFilesExcluded += row.FixtureFilesExcluded
		summary.Records += row.Records
		summary.UsageRecordsExact += row.UsageRecordsExact
		summary.RefusedRecords += row.RefusedRecords
	}
	transcriptIDs := make(map[string]struct{}, len(transcripts))
	accountedTokens := make([]int64, 0, len(transcripts))
	categoryTotals := map[string]int64{}
	toolTotals := map[string]*AuditDistributionRow{}
	var toolResultGroups [][]AuditToolResultRow
	storageTotals := map[string]*AuditStorageRow{}
	exemplars := newDefaultAuditUnknownExemplarReservoir()
	for _, transcript := range transcripts {
		summary.Tokens.add(transcript.Tokens)
		summary.RepeatedFailures += transcript.RepeatedFailures
		summary.ExpectedWaitTimeouts += transcript.ExpectedWaitTimeouts
		summary.MutationChurn += transcript.MutationChurn
		summary.ToolCalls += transcript.ToolCalls
		summary.ToolErrors += transcript.ToolErrors
		exemplars.merge(transcript.UnknownExemplars)
		for _, r := range transcript.Distribution {
			categoryTotals[r.Name] += r.Bytes
		}
		for _, r := range transcript.StorageDistribution {
			k := r.Source + "\x00" + r.Subtype
			t := storageTotals[k]
			if t == nil {
				t = &AuditStorageRow{Source: r.Source, Subtype: r.Subtype}
				storageTotals[k] = t
			}
			t.Bytes += r.Bytes
			t.Records += r.Records
		}
		for _, r := range transcript.ToolDistribution {
			t := toolTotals[r.Name]
			if t == nil {
				t = &AuditDistributionRow{Name: r.Name}
				toolTotals[r.Name] = t
			}
			t.Bytes += r.Bytes
			t.Calls += r.Calls
		}
		toolResultGroups = append(toolResultGroups, transcript.ToolResults)
		if transcript.Tokens.accountedTotal() == 0 {
			summary.EmptyUsageFiles++
		}
		transcriptIDs[transcript.Source+"\x00"+transcript.TranscriptID] = struct{}{}
		accountedTokens = append(accountedTokens, transcript.Tokens.accountedTotal())
	}
	qwenTotals := make(map[string]int64)
	for _, transcript := range transcripts {
		isQwen := false
		for _, model := range transcript.Models {
			if strings.Contains(strings.ToLower(model), "qwen") {
				isQwen = true
				break
			}
		}
		if isQwen {
			contributor := transcript.Source + ":" + transcript.TranscriptID
			qwenTotals[contributor] += transcript.Tokens.accountedTotal()
		}
	}
	summary.QwenTokenConcentrationThreshold = qwenTopContributorConcentrationThreshold
	if len(qwenTotals) > 0 {
		contributors := make([]string, 0, len(qwenTotals))
		var total int64
		for contributor, tokens := range qwenTotals {
			contributors = append(contributors, contributor)
			total += tokens
		}
		sort.Strings(contributors)
		top := contributors[0]
		for _, contributor := range contributors[1:] {
			if qwenTotals[contributor] > qwenTotals[top] {
				top = contributor
			}
		}
		topTokens := qwenTotals[top]
		summary.QwenTopContributor = &top
		summary.QwenTopContributorTokens = &topTokens
		summary.QwenTotalTokens = &total
		if total > 0 {
			fraction := float64(topTokens) / float64(total)
			concentrated := fraction > qwenTopContributorConcentrationThreshold
			summary.QwenTopContributorTokenFraction = &fraction
			summary.QwenTopContributorTokenConcentrated = &concentrated
		}
	}
	summary.DistributionUnit = AuditDistributionUnit
	summary.DistributionProvenance = "deterministic model-visible content UTF-8 bytes; storage/event envelopes reported separately; not provider-billed per-block tokens"
	summary.Distribution = distributionRows(categoryTotals)
	summary.ToolDistribution = toolDistributionRows(toolTotals)
	summary.ToolResults = mergeAuditToolResultRows(toolResultGroups...)
	summary.StorageUnit = AuditStorageUnit
	summary.StorageDistribution = storageDistributionRows(storageTotals)
	summary.UnknownExemplars = exemplars.snapshot()
	linkAuditUnknownExemplars(summary.Distribution, summary.StorageDistribution, summary.UnknownExemplars.Exemplars)
	summary.DistinctTranscripts = len(transcriptIDs)
	summary.DuplicateFragments = summary.Transcripts - summary.DistinctTranscripts
	if summary.ToolCalls > 0 {
		v := float64(summary.ToolErrors) / float64(summary.ToolCalls)
		summary.ToolErrorFraction = &v
	}
	if summary.Transcripts > 0 {
		repeated := float64(summary.RepeatedFailures) / float64(summary.Transcripts)
		summary.RepeatedFailuresPerSession = &repeated
	}
	sort.Slice(accountedTokens, func(i, j int) bool { return accountedTokens[i] > accountedTokens[j] })
	if total := summary.Tokens.accountedTotal(); total > 0 {
		var top int64
		for i := 0; i < len(accountedTokens) && i < 10; i++ {
			top += accountedTokens[i]
		}
		v := float64(top) / float64(total)
		summary.TopTenTokenFraction = &v
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
		newAuditRepeatedFailureDelta(current, baseline),
		newAuditDelta("hook_p95_ms", floatOfInt64(current.HookP95MS), floatOfInt64(baseline.HookP95MS)),
	}
}

func newAuditDelta(metric string, current, baseline *float64) AuditDeltaRow {
	row := AuditDeltaRow{
		Schema: AuditSchema, Kind: "baseline_delta", Metric: metric, Current: current, Baseline: baseline,
		ComparabilityStatus: "missing_value",
	}
	if current == nil || baseline == nil {
		return row
	}
	delta := *current - *baseline
	row.Delta = &delta
	row.Comparable = true
	row.ComparabilityStatus = "comparable"
	row.Regression = delta > 0
	return row
}

func newAuditRepeatedFailureDelta(current, baseline AuditSummaryRow) AuditDeltaRow {
	rawCurrent, rawBaseline := current.RepeatedFailures, baseline.RepeatedFailures
	currentExposure, baselineExposure := current.Transcripts, baseline.Transcripts
	rawComparable := false
	normalizedComparable, normalizedRegression := false, false
	currentRawValue, baselineRawValue := float64(rawCurrent), float64(rawBaseline)
	row := AuditDeltaRow{
		Schema: AuditSchema, Kind: "baseline_delta", Metric: "repeated_failures",
		Current: &currentRawValue, Baseline: &baselineRawValue,
		RawCurrent: &rawCurrent, RawBaseline: &rawBaseline,
		CurrentExposure: &currentExposure, BaselineExposure: &baselineExposure,
		Normalization: current.RepeatedFailureNormalization, RawComparable: &rawComparable,
		NormalizedComparable: &normalizedComparable, NormalizedRegression: &normalizedRegression,
		ComparabilityStatus: "semantics_mismatch",
		ComparabilityReason: "baseline must explicitly carry the same repeated-failure classification and normalization semantics",
	}
	if current.Transcripts > 0 {
		value := float64(current.RepeatedFailures) / float64(current.Transcripts)
		row.NormalizedCurrent = &value
	}
	if baseline.Transcripts > 0 {
		value := float64(baseline.RepeatedFailures) / float64(baseline.Transcripts)
		row.NormalizedBaseline = &value
	}
	if current.RepeatedFailureSemantics != auditRepeatedFailureSemantics ||
		baseline.RepeatedFailureSemantics != current.RepeatedFailureSemantics ||
		current.RepeatedFailureNormalization != auditRepeatedFailureNormalization ||
		baseline.RepeatedFailureNormalization != current.RepeatedFailureNormalization {
		return row
	}
	if current.Transcripts <= 0 || baseline.Transcripts <= 0 {
		row.ComparabilityStatus = "missing_exposure"
		row.ComparabilityReason = "per-session comparison requires non-zero current and baseline session exposure"
		return row
	}
	currentRate := float64(current.RepeatedFailures) / float64(current.Transcripts)
	baselineRate := float64(baseline.RepeatedFailures) / float64(baseline.Transcripts)
	normalizedDelta := currentRate - baselineRate
	row.NormalizedCurrent = &currentRate
	row.NormalizedBaseline = &baselineRate
	row.NormalizedDelta = &normalizedDelta
	normalizedComparable = true
	normalizedRegression = normalizedDelta > 0
	row.NormalizedComparable = &normalizedComparable
	row.NormalizedRegression = &normalizedRegression
	rawComparable = current.Transcripts == baseline.Transcripts
	row.RawComparable = &rawComparable
	if rawComparable {
		rawDelta := currentRawValue - baselineRawValue
		row.Delta = &rawDelta
		row.Comparable = true
		row.Regression = rawDelta > 0
		row.ComparabilityStatus = "raw_and_normalized"
		row.ComparabilityReason = "raw counts share equal session exposure; normalized_regression uses the per-session rate"
	} else {
		row.ComparabilityStatus = "normalized_only"
		row.ComparabilityReason = "raw counts have unequal session exposure; only the per-session rate is comparable"
	}
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
	if result.Corpus != nil {
		rows = append(rows, *result.Corpus)
	}
	for _, row := range result.Denominators {
		rows = append(rows, row)
	}
	for _, row := range result.Transcripts {
		rows = append(rows, row)
	}
	for _, transcript := range result.Transcripts {
		for _, churn := range transcript.MutationChurnEvents {
			rows = append(rows, struct {
				Schema string `json:"schema"`
				Kind   string `json:"kind"`
				QwenMutationChurn
			}{AuditSchema, "mutation_churn", churn})
		}
	}
	for _, row := range result.Bottlenecks {
		rows = append(rows, row)
	}
	for _, row := range result.Baseline {
		rows = append(rows, row)
	}
	for _, row := range result.EventSchemas {
		rows = append(rows, row)
	}
	for _, row := range result.SchemaDrift {
		rows = append(rows, row)
	}
	for _, row := range result.Refusals {
		rows = append(rows, row)
	}
	for _, row := range result.ToolErrorFamilies {
		rows = append(rows, struct {
			Schema string `json:"schema"`
			Kind   string `json:"kind"`
			QwenToolErrorFamily
		}{AuditSchema, "tool_error_family", row})
	}
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			return fmt.Errorf("trajectory audit: encode JSONL: %w", err)
		}
	}
	return nil
}
