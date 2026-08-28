package trajectory

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	AttributionBudgetSchema  = "fak-trajectory-attribution-budget/1"
	AttributionReceiptSchema = "fak-trajectory-attribution-receipt/1"

	AttributionStatusPass              = "pass"
	AttributionStatusBudgetFailed      = "budget_failed"
	AttributionStatusNoData            = "no_data"
	AttributionStatusCollectionFailed  = "collection_failed"
	AttributionStatusPublicationFailed = "publication_failed"
)

// AttributionBudget is the versioned operating envelope for the source-side
// nightly audit. The byte and file ceilings bound how much recent private
// transcript data is read; only AttributionReceipt leaves the source host.
type AttributionBudget struct {
	Schema                    string              `json:"schema"`
	Version                   string              `json:"version"`
	Window                    string              `json:"window"`
	MaxFilesPerSource         int                 `json:"max_files_per_source"`
	MaxWalkEntriesPerSource   int                 `json:"max_walk_entries_per_source"`
	MaxSubtypesPerSource      int                 `json:"max_subtypes_per_source"`
	MaxBytesPerFile           int64               `json:"max_bytes_per_file"`
	MaxBytesPerSource         int64               `json:"max_bytes_per_source"`
	MaxSampleCoordinates      int                 `json:"max_sample_coordinates"`
	MaxUnknownShare           float64             `json:"max_unknown_share"`
	MaxDuplicateEvents        int                 `json:"max_duplicate_events"`
	MaxMalformedRows          int                 `json:"max_malformed_rows"`
	MaxUnmatchedToolEvents    int                 `json:"max_unmatched_tool_events"`
	MaxSchemaDriftSignals     int                 `json:"max_schema_drift_signals"`
	MaxBoundedFilesOmitted    int                 `json:"max_bounded_files_omitted"`
	MaxBoundedSubtypesOmitted int                 `json:"max_bounded_subtypes_omitted"`
	KnownSourceRecordTypes    map[string][]string `json:"known_source_record_types"`
}

func (b AttributionBudget) Validate() error {
	if b.Schema != AttributionBudgetSchema {
		return fmt.Errorf("trajectory nightly: budget schema %q has no reader", b.Schema)
	}
	if strings.TrimSpace(b.Version) == "" {
		return errors.New("trajectory nightly: budget version is required")
	}
	window, err := time.ParseDuration(b.Window)
	if err != nil || window <= 0 {
		return fmt.Errorf("trajectory nightly: budget window %q must be a positive duration", b.Window)
	}
	if b.MaxFilesPerSource <= 0 || b.MaxWalkEntriesPerSource <= 0 || b.MaxSubtypesPerSource <= 0 || b.MaxSubtypesPerSource > 4096 || b.MaxBytesPerFile <= 0 || b.MaxBytesPerSource < b.MaxBytesPerFile {
		return errors.New("trajectory nightly: walk, file, subtype, and byte scan bounds are invalid")
	}
	if b.MaxSampleCoordinates < 7 || b.MaxSampleCoordinates > 100 {
		return errors.New("trajectory nightly: max_sample_coordinates must be in [7,100]")
	}
	if b.MaxUnknownShare < 0 || b.MaxUnknownShare > 1 {
		return errors.New("trajectory nightly: max_unknown_share must be in [0,1]")
	}
	for name, value := range map[string]int{
		"max_duplicate_events":         b.MaxDuplicateEvents,
		"max_malformed_rows":           b.MaxMalformedRows,
		"max_unmatched_tool_events":    b.MaxUnmatchedToolEvents,
		"max_schema_drift_signals":     b.MaxSchemaDriftSignals,
		"max_bounded_files_omitted":    b.MaxBoundedFilesOmitted,
		"max_bounded_subtypes_omitted": b.MaxBoundedSubtypesOmitted,
	} {
		if value < 0 {
			return fmt.Errorf("trajectory nightly: %s must be non-negative", name)
		}
	}
	for _, source := range []string{AuditSourceClaude, AuditSourceCodex} {
		if len(b.KnownSourceRecordTypes[source]) == 0 {
			return fmt.Errorf("trajectory nightly: known_source_record_types.%s must not be empty", source)
		}
	}
	return nil
}

func (b AttributionBudget) duration() time.Duration {
	d, _ := time.ParseDuration(b.Window)
	return d
}

// ReadAttributionBudget rejects unknown fields so a misspelled threshold cannot
// silently turn an enforced budget into an unlimited one.
func ReadAttributionBudget(path string) (AttributionBudget, error) {
	file, err := os.Open(path)
	if err != nil {
		return AttributionBudget{}, fmt.Errorf("trajectory nightly: open budget: %w", err)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var budget AttributionBudget
	if err := decoder.Decode(&budget); err != nil {
		return AttributionBudget{}, fmt.Errorf("trajectory nightly: decode budget: %w", err)
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		return AttributionBudget{}, errors.New("trajectory nightly: budget must contain exactly one JSON object")
	}
	if err := budget.Validate(); err != nil {
		return AttributionBudget{}, err
	}
	return budget, nil
}

type AttributionSubtypeCoverage struct {
	Subtype string `json:"subtype"`
	Records int    `json:"records"`
}

type AttributionSourceCoverage struct {
	Source               string                       `json:"source"`
	Root                 string                       `json:"root"`
	RootFingerprint      string                       `json:"root_fingerprint"`
	RootPresent          bool                         `json:"root_present"`
	FilesDiscovered      int                          `json:"files_discovered"`
	FilesScanned         int                          `json:"files_scanned"`
	FixtureFilesExcluded int                          `json:"fixture_files_excluded"`
	Records              int                          `json:"records"`
	SubtypesObserved     int                          `json:"subtypes_observed"`
	SubtypesOmitted      int                          `json:"subtypes_omitted"`
	RecordTypes          []AttributionSubtypeCoverage `json:"record_types"`
}

type AttributionMetrics struct {
	VisibleBytes           int64    `json:"visible_bytes"`
	UnknownVisibleBytes    int64    `json:"unknown_visible_bytes"`
	UnknownShare           *float64 `json:"unknown_share"`
	DuplicateEvents        int      `json:"duplicate_events"`
	MalformedRows          int      `json:"malformed_rows"`
	UnmatchedToolEvents    int      `json:"unmatched_tool_events"`
	SchemaDriftSignals     int      `json:"schema_drift_signals"`
	BoundedFilesOmitted    int      `json:"bounded_files_omitted"`
	BoundedSubtypesOmitted int      `json:"bounded_subtypes_omitted"`
}

type AttributionBudgetBreach struct {
	Metric  string  `json:"metric"`
	Actual  float64 `json:"actual"`
	Maximum float64 `json:"maximum"`
	Unit    string  `json:"unit"`
}

// AttributionSampleCoordinate is deliberately content-free. SourcePath is a
// stable hash of the root-relative path, never private path structure.
type AttributionSampleCoordinate struct {
	Metric     string `json:"metric"`
	Source     string `json:"source"`
	SourcePath string `json:"source_path"`
	Line       int    `json:"line,omitempty"`
	Subtype    string `json:"subtype,omitempty"`
}

type AttributionTrend struct {
	Comparable         bool               `json:"comparable"`
	PreviousObservedAt string             `json:"previous_observed_at,omitempty"`
	Reason             string             `json:"reason,omitempty"`
	Deltas             map[string]float64 `json:"deltas,omitempty"`
}

type AttributionReceipt struct {
	Schema           string                        `json:"schema"`
	BudgetSchema     string                        `json:"budget_schema"`
	BudgetVersion    string                        `json:"budget_version"`
	ObservedAt       string                        `json:"observed_at"`
	Corpus           string                        `json:"corpus"`
	Window           string                        `json:"window"`
	Status           string                        `json:"status"`
	CollectionError  string                        `json:"collection_error,omitempty"`
	PublicationError string                        `json:"publication_error,omitempty"`
	Coverage         []AttributionSourceCoverage   `json:"coverage"`
	Metrics          AttributionMetrics            `json:"metrics"`
	Breaches         []AttributionBudgetBreach     `json:"breaches"`
	Samples          []AttributionSampleCoordinate `json:"samples"`
	Trend            AttributionTrend              `json:"trend"`
}

type AttributionNightlyOptions struct {
	Sources []AuditSource
	Budget  AttributionBudget
	Now     time.Time
	Corpus  string
}

type attributionNightlyEvidence struct {
	samples          []AttributionSampleCoordinate
	unmatched        int
	omitted          int
	recordSamples    map[string]AttributionSampleCoordinate
	rootFingerprints map[string]string
}

type attributionAuditFile struct {
	path    string
	rel     string
	modTime time.Time
	size    int64
}

var (
	errAttributionDiscoveryLimit = errors.New("trajectory nightly: source discovery limit reached")
	errAttributionSourceChanged  = errors.New("trajectory nightly: source file changed during audit")
)

// RunAttributionNightly performs the same exact parser fold as RunAudit, but
// selects the newest in-window files under versioned file/byte ceilings and
// reduces the result to a scrubbed operational receipt.
func RunAttributionNightly(opts AttributionNightlyOptions) AttributionReceipt {
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}
	receipt := AttributionReceipt{
		Schema: AttributionReceiptSchema, BudgetSchema: opts.Budget.Schema,
		BudgetVersion: opts.Budget.Version, ObservedAt: now.UTC().Format(time.RFC3339Nano),
		Corpus: scrubAttributionToken(opts.Corpus), Window: opts.Budget.Window,
		Status: AttributionStatusCollectionFailed, Breaches: []AttributionBudgetBreach{},
		Samples: []AttributionSampleCoordinate{}, Trend: AttributionTrend{Comparable: false, Reason: "no_previous_receipt"},
	}
	if receipt.Corpus == "" {
		receipt.Corpus = "unspecified"
	}
	if err := opts.Budget.Validate(); err != nil {
		receipt.CollectionError = "invalid_budget"
		return receipt
	}
	if len(opts.Sources) == 0 {
		opts.Sources = DefaultAuditSources()
	}
	result, evidence, err := runBoundedAttributionAudit(opts.Sources, opts.Budget, now)
	receipt = foldAttributionReceipt(receipt, result, evidence, opts.Budget)
	if err != nil {
		receipt.Status = AttributionStatusCollectionFailed
		receipt.CollectionError = attributionCollectionErrorCode(err)
	}
	return receipt
}

func runBoundedAttributionAudit(sources []AuditSource, budget AttributionBudget, now time.Time) (AuditResult, attributionNightlyEvidence, error) {
	result := AuditResult{}
	evidence := attributionNightlyEvidence{
		recordSamples:    map[string]AttributionSampleCoordinate{},
		rootFingerprints: map[string]string{},
	}
	var allHookDurations []int64
	var allToolErrorEvents []QwenToolErrorEvent
	for _, source := range sources {
		denominator := AuditDenominatorRow{
			Schema: AuditSchema, Kind: "source_denominator", Source: source.Name,
			Root: attributionSourceRootLabel(source.Name), RecordTypes: map[string]int{},
		}
		evidence.rootFingerprints[source.Name] = scrubAttributionRoot(source.Root)
		switch source.Name {
		case AuditSourceClaude:
			denominator.TokenSemantics = "message usage buckets are disjoint; duplicate message ids are counted once"
		case AuditSourceCodex:
			denominator.TokenSemantics = "final cumulative input per segment; only a versioned task_started boundary may begin a segment after a decrease; cached/cache-write subsets remain exact subtraction"
		default:
			return result, evidence, fmt.Errorf("unsupported source")
		}

		files, present, err := attributionAuditFiles(source, budget.duration(), now, budget.MaxWalkEntriesPerSource)
		if err != nil {
			result.Denominators = append(result.Denominators, denominator)
			return result, evidence, err
		}
		denominator.RootPresent = present
		denominator.FilesDiscovered = len(files)
		var sourceBytes int64
		for index, candidate := range files {
			if index >= budget.MaxFilesPerSource || candidate.size > budget.MaxBytesPerFile || candidate.size > budget.MaxBytesPerSource-sourceBytes {
				evidence.omitted++
				evidence.samples = append(evidence.samples, AttributionSampleCoordinate{Metric: "bounded_files_omitted", Source: source.Name, SourcePath: candidate.rel, Subtype: "scan_budget"})
				continue
			}
			remaining := budget.MaxBytesPerSource - sourceBytes
			readLimit := min(budget.MaxBytesPerFile, remaining)
			contents, readErr := readAttributionCandidate(candidate, readLimit)
			if readErr != nil {
				result.Denominators = append(result.Denominators, denominator)
				return result, evidence, readErr
			}
			sourceBytes += int64(len(contents))
			if source.Name == AuditSourceClaude && attributionIsClaudePytestFixture(contents, candidate.rel) {
				denominator.FixtureFilesExcluded++
				continue
			}
			fileDenominator := AuditDenominatorRow{RecordTypes: map[string]int{}}
			row, refusals, hooks, toolErrors, parseErr := parseAttributionAuditBytes(source.Name, contents, candidate.path, candidate.rel, &fileDenominator)
			if parseErr != nil {
				result.Denominators = append(result.Denominators, denominator)
				return result, evidence, parseErr
			}
			denominator.FilesScanned++
			mergeAttributionDenominator(&denominator, fileDenominator)
			for subtype := range fileDenominator.RecordTypes {
				key := source.Name + "\x00" + subtype
				if _, exists := evidence.recordSamples[key]; !exists {
					evidence.recordSamples[key] = AttributionSampleCoordinate{Metric: "schema_drift_signals", Source: source.Name, SourcePath: candidate.rel, Subtype: scrubAttributionRecordType(budget, source.Name, subtype)}
				}
			}
			if fileDenominator.DuplicateUsageRecords > 0 {
				evidence.samples = append(evidence.samples, AttributionSampleCoordinate{Metric: "duplicate_events", Source: source.Name, SourcePath: candidate.rel, Subtype: "usage_record"})
			}
			unmatched, unmatchedSamples, signalErr := attributionUnmatchedTools(source.Name, contents, candidate.rel)
			if signalErr != nil {
				result.Denominators = append(result.Denominators, denominator)
				return result, evidence, signalErr
			}
			evidence.unmatched += unmatched
			evidence.samples = append(evidence.samples, unmatchedSamples...)
			result.Transcripts = append(result.Transcripts, row)
			result.Refusals = append(result.Refusals, refusals...)
			allHookDurations = append(allHookDurations, hooks...)
			allToolErrorEvents = append(allToolErrorEvents, toolErrors...)
		}
		result.Denominators = append(result.Denominators, denominator)
	}

	sort.Slice(result.Denominators, func(i, j int) bool { return result.Denominators[i].Source < result.Denominators[j].Source })
	sort.Slice(result.Transcripts, func(i, j int) bool {
		if result.Transcripts[i].Source != result.Transcripts[j].Source {
			return result.Transcripts[i].Source < result.Transcripts[j].Source
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
	canonical, canonicalRefusals := canonicalAuditTranscripts(result.Transcripts)
	result.Refusals = append(result.Refusals, canonicalRefusals...)
	result.Summary = summarizeAudit(result.Denominators, canonical, allHookDurations)
	result.Summary.RawFragments = len(result.Transcripts)
	result.Summary.CanonicalTranscripts = len(canonical)
	result.Bottlenecks = rankAuditBottlenecks(canonical)
	result.ToolErrorFamilies = rankQwenToolErrorFamilies(allToolErrorEvents)
	result.ConclusionStatus = AuditConclusionStatus{
		BroadEfficiencySupported: len(result.Refusals) == 0,
		RefusalCount:             len(result.Refusals),
	}
	return result, evidence, nil
}

func attributionAuditFiles(source AuditSource, window time.Duration, now time.Time, maxWalkEntries int) ([]attributionAuditFile, bool, error) {
	info, err := os.Stat(source.Root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("stat source root: %w", err)
	}
	if !info.IsDir() {
		return nil, false, errors.New("source root is not a directory")
	}
	var files []attributionAuditFile
	visited := 0
	err = filepath.WalkDir(source.Root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		visited++
		if visited > maxWalkEntries {
			return errAttributionDiscoveryLimit
		}
		if entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".jsonl") {
			return nil
		}
		stat, statErr := entry.Info()
		if statErr != nil {
			return statErr
		}
		if stat.ModTime().Before(now.Add(-window)) {
			return nil
		}
		rel, relErr := filepath.Rel(source.Root, path)
		if relErr != nil {
			return relErr
		}
		files = append(files, attributionAuditFile{path: path, rel: filepath.ToSlash(rel), modTime: stat.ModTime(), size: stat.Size()})
		return nil
	})
	if err != nil {
		return nil, true, fmt.Errorf("discover source transcripts: %w", err)
	}
	sort.Slice(files, func(i, j int) bool {
		if !files[i].modTime.Equal(files[j].modTime) {
			return files[i].modTime.After(files[j].modTime)
		}
		return files[i].rel < files[j].rel
	})
	return files, true, nil
}

func readAttributionCandidate(candidate attributionAuditFile, limit int64) ([]byte, error) {
	file, err := os.Open(candidate.path)
	if err != nil {
		return nil, fmt.Errorf("trajectory nightly: open source transcript: %w", err)
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("trajectory nightly: stat source transcript: %w", err)
	}
	if before.Size() != candidate.size || !before.ModTime().Equal(candidate.modTime) || before.Size() > limit {
		return nil, errAttributionSourceChanged
	}
	contents, err := readAttributionBounded(file, limit)
	if err != nil {
		return nil, fmt.Errorf("trajectory nightly: read source transcript: %w", err)
	}
	after, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("trajectory nightly: restat source transcript: %w", err)
	}
	if after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) || int64(len(contents)) != after.Size() {
		return nil, errAttributionSourceChanged
	}
	return contents, nil
}

func readAttributionBounded(reader io.Reader, limit int64) ([]byte, error) {
	return io.ReadAll(io.LimitReader(reader, limit))
}

func attributionIsClaudePytestFixture(contents []byte, rel string) bool {
	rel = filepath.ToSlash(rel)
	if !auditClaudePytestFixturePath.MatchString(rel) {
		return false
	}
	parts := strings.Split(rel, "/")
	if len(parts) != 2 {
		return false
	}
	var records []map[string]any
	scanner := bufio.NewScanner(bytes.NewReader(contents))
	scanner.Buffer(make([]byte, 4096), 64*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record map[string]any
		if json.Unmarshal([]byte(line), &record) != nil {
			return false
		}
		records = append(records, record)
	}
	if scanner.Err() != nil || len(records) != 5 || !auditClaudePytestUser(records[0], parts[0]) {
		return false
	}
	for index, timestamp := range []string{
		"2026-06-01T14:00:00.000Z", "2026-06-01T14:01:00.000Z",
		"2026-06-01T14:02:00.000Z", "2026-06-01T14:03:00.000Z",
	} {
		if !auditClaudePytestAssistant(records[index+1], timestamp) {
			return false
		}
	}
	return true
}

// parseAttributionAuditBytes is the nightly, already-bounded reader seam for
// the canonical parser state. Keeping it here lets the source file be opened
// exactly once without changing the general audit parser.
func parseAttributionAuditBytes(source string, contents []byte, transcriptName, rel string, denominator *AuditDenominatorRow) (AuditTranscriptRow, []AuditRefusalRow, []int64, []QwenToolErrorEvent, error) {
	state := auditParseState{
		distribution: newAuditDistribution(),
		row: AuditTranscriptRow{
			Schema: AuditSchema, Kind: "session", Source: source,
			TranscriptID: strings.TrimSuffix(filepath.Base(transcriptName), filepath.Ext(transcriptName)), SourcePath: rel,
		},
		models:          map[string]struct{}{},
		calls:           map[string]auditToolCall{},
		seenCalls:       map[string]struct{}{},
		failureCounts:   map[string]int{},
		mutationCounts:  map[string]int{},
		claudeUsageByID: map[string]AuditTokens{},
	}
	var refusals []AuditRefusalRow
	scanner := bufio.NewScanner(bytes.NewReader(contents))
	scanner.Buffer(make([]byte, 64*1024), auditMaxLineBytes)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		denominator.Records++
		state.distribution.observe(source, line)
		var record map[string]any
		decoder := json.NewDecoder(bytes.NewReader(line))
		decoder.UseNumber()
		if err := decoder.Decode(&record); err != nil {
			refusals = append(refusals, newAuditRefusal(source, rel, lineNumber, "malformed_json", err.Error()))
			denominator.RefusedRecords++
			continue
		}
		denominator.RecordTypes[auditRecordType(record)]++
		before := len(refusals)
		switch source {
		case AuditSourceClaude:
			parseClaudeAuditRecord(record, lineNumber, rel, &state, &refusals)
		case AuditSourceCodex:
			parseCodexAuditRecord(record, lineNumber, rel, &state, &refusals)
		}
		if len(refusals) > before {
			denominator.RefusedRecords += len(refusals) - before
		}
	}
	if err := scanner.Err(); err != nil {
		if !errors.Is(err, bufio.ErrTooLong) {
			return AuditTranscriptRow{}, nil, nil, nil, fmt.Errorf("trajectory audit: scan transcript: %w", err)
		}
		lineNumber++
		denominator.Records++
		denominator.RefusedRecords++
		refusals = append(refusals, newAuditRefusal(source, rel, lineNumber, "line_too_large", fmt.Sprintf("line exceeds %d-byte limit", auditMaxLineBytes)))
	}
	denominator.UsageRecordsSeen += state.usageSeen
	denominator.UsageRecordsExact += state.usageExact
	denominator.DuplicateUsageRecords += state.usageDuplicates
	denominator.UsageRecordsApplied += state.row.UsageRecords
	state.row.usageByID = make(map[string]AuditTokens, len(state.claudeUsageByID))
	for id, usage := range state.claudeUsageByID {
		state.row.usageByID[id] = usage
	}
	state.row.Models = make([]string, 0, len(state.models))
	for model := range state.models {
		state.row.Models = append(state.row.Models, model)
	}
	sort.Strings(state.row.Models)
	state.row.failureCounts = cloneAuditFailureCounts(state.failureCounts)
	state.row.RepeatedFailures = auditRepeatedFailureCount(state.failureCounts)
	state.row.MutationChurnEvents = DetectQwenMutationChurn(state.mutationEvents)
	for _, churn := range state.row.MutationChurnEvents {
		state.row.MutationChurn += churn.Count - 1
	}
	applyQwenToolErrorAttribution(state.toolErrorEvents, state.toolErrorAttributions, state.mutationCounts)
	state.row.HookP95MS = auditPercentile(state.hookDurations, 95)
	state.row.Distribution = distributionRows(state.distribution.categories)
	state.row.ToolDistribution = toolDistributionRows(state.distribution.tools)
	state.row.ToolResults = state.distribution.toolResultRows()
	state.row.StorageDistribution = storageDistributionRows(state.distribution.storage)
	if source == AuditSourceCodex && state.codexRawTotal != nil {
		state.row.UsageRecords++
		denominator.UsageRecordsApplied++
	}
	return state.row, refusals, append([]int64(nil), state.hookDurations...), append([]QwenToolErrorEvent(nil), state.toolErrorEvents...), nil
}

func mergeAttributionDenominator(dst *AuditDenominatorRow, src AuditDenominatorRow) {
	dst.Records += src.Records
	dst.UsageRecordsSeen += src.UsageRecordsSeen
	dst.UsageRecordsExact += src.UsageRecordsExact
	dst.UsageRecordsApplied += src.UsageRecordsApplied
	dst.DuplicateUsageRecords += src.DuplicateUsageRecords
	dst.RefusedRecords += src.RefusedRecords
	for subtype, count := range src.RecordTypes {
		dst.RecordTypes[subtype] += count
	}
}

func foldAttributionReceipt(receipt AttributionReceipt, result AuditResult, evidence attributionNightlyEvidence, budget AttributionBudget) AttributionReceipt {
	boundedSubtypesOmitted := 0
	for _, denominator := range result.Denominators {
		coverage := AttributionSourceCoverage{
			Source: denominator.Source, Root: denominator.Root, RootPresent: denominator.RootPresent,
			RootFingerprint: evidence.rootFingerprints[denominator.Source],
			FilesDiscovered: denominator.FilesDiscovered, FilesScanned: denominator.FilesScanned,
			FixtureFilesExcluded: denominator.FixtureFilesExcluded, Records: denominator.Records,
			RecordTypes: []AttributionSubtypeCoverage{},
		}
		type subtypeCoverage struct {
			raw string
			row AttributionSubtypeCoverage
		}
		rows := make([]subtypeCoverage, 0, len(denominator.RecordTypes))
		for subtype, records := range denominator.RecordTypes {
			rows = append(rows, subtypeCoverage{raw: subtype, row: AttributionSubtypeCoverage{Subtype: scrubAttributionRecordType(budget, denominator.Source, subtype), Records: records}})
		}
		sort.Slice(rows, func(i, j int) bool {
			if rows[i].row.Records != rows[j].row.Records {
				return rows[i].row.Records > rows[j].row.Records
			}
			return rows[i].row.Subtype < rows[j].row.Subtype
		})
		coverage.SubtypesObserved = len(rows)
		if len(rows) > budget.MaxSubtypesPerSource {
			coverage.SubtypesOmitted = len(rows) - budget.MaxSubtypesPerSource
			boundedSubtypesOmitted += coverage.SubtypesOmitted
			omitted := rows[budget.MaxSubtypesPerSource]
			if sample, ok := evidence.recordSamples[denominator.Source+"\x00"+omitted.raw]; ok {
				sample.Metric = "bounded_subtypes_omitted"
				receipt.Samples = append(receipt.Samples, sample)
			}
			rows = rows[:budget.MaxSubtypesPerSource]
		}
		for _, row := range rows {
			coverage.RecordTypes = append(coverage.RecordTypes, row.row)
		}
		sort.Slice(coverage.RecordTypes, func(i, j int) bool { return coverage.RecordTypes[i].Subtype < coverage.RecordTypes[j].Subtype })
		receipt.Coverage = append(receipt.Coverage, coverage)
	}
	metrics := AttributionMetrics{
		UnmatchedToolEvents: evidence.unmatched, BoundedFilesOmitted: evidence.omitted,
		BoundedSubtypesOmitted: boundedSubtypesOmitted,
	}
	for _, row := range result.Summary.Distribution {
		metrics.VisibleBytes += row.Bytes
		if row.Name == "visible_unknown" {
			metrics.UnknownVisibleBytes += row.Bytes
		}
	}
	for _, transcript := range result.Transcripts {
		for _, distribution := range transcript.Distribution {
			if distribution.Name == "visible_unknown" && distribution.Bytes > 0 {
				receipt.Samples = append(receipt.Samples, AttributionSampleCoordinate{
					Metric: "unknown_share", Source: transcript.Source,
					SourcePath: filepath.ToSlash(transcript.SourcePath), Subtype: "visible_unknown",
				})
				break
			}
		}
	}
	if metrics.VisibleBytes > 0 {
		share := float64(metrics.UnknownVisibleBytes) / float64(metrics.VisibleBytes)
		metrics.UnknownShare = &share
	}
	for _, denominator := range result.Denominators {
		metrics.DuplicateEvents += denominator.DuplicateUsageRecords
	}
	fragmentKeys := map[string]bool{}
	for _, row := range result.Transcripts {
		key := row.Source + "\x00" + row.TranscriptID
		if fragmentKeys[key] {
			metrics.DuplicateEvents++
		}
		fragmentKeys[key] = true
	}

	known := map[string]map[string]bool{}
	for source, subtypes := range budget.KnownSourceRecordTypes {
		known[source] = map[string]bool{}
		for _, subtype := range subtypes {
			known[source][subtype] = true
		}
	}
	drift := map[string]bool{}
	for _, denominator := range result.Denominators {
		for subtype := range denominator.RecordTypes {
			if !known[denominator.Source][subtype] {
				key := denominator.Source + "\x00" + subtype
				drift[key] = true
				if sample, ok := evidence.recordSamples[key]; ok {
					receipt.Samples = append(receipt.Samples, sample)
				}
			}
		}
	}
	for _, refusal := range result.Refusals {
		sample := AttributionSampleCoordinate{Source: refusal.Source, SourcePath: filepath.ToSlash(refusal.SourcePath), Line: refusal.Line, Subtype: scrubAttributionToken(refusal.Code)}
		switch refusal.Code {
		case "malformed_json", "line_too_large":
			metrics.MalformedRows++
			sample.Metric = "malformed_rows"
		default:
			key := refusal.Source + "\x00refusal:" + refusal.Code
			drift[key] = true
			sample.Metric = "schema_drift_signals"
		}
		receipt.Samples = append(receipt.Samples, sample)
	}
	metrics.SchemaDriftSignals = len(drift)
	receipt.Metrics = metrics
	receipt.Samples = append(receipt.Samples, evidence.samples...)
	if len(fragmentKeys) < len(result.Transcripts) {
		seen := map[string]string{}
		for _, row := range result.Transcripts {
			key := row.Source + "\x00" + row.TranscriptID
			if _, exists := seen[key]; exists {
				receipt.Samples = append(receipt.Samples, AttributionSampleCoordinate{Metric: "duplicate_events", Source: row.Source, SourcePath: filepath.ToSlash(row.SourcePath), Subtype: "transcript_fragment"})
				continue
			}
			seen[key] = row.SourcePath
		}
	}
	receipt.Samples = normalizeAttributionSamples(receipt.Samples, budget.MaxSampleCoordinates)

	if result.Summary.Records == 0 && metrics.BoundedFilesOmitted == 0 {
		receipt.Status = AttributionStatusNoData
		return receipt
	}
	receipt.Breaches = attributionBreaches(metrics, budget)
	if len(receipt.Breaches) > 0 {
		receipt.Status = AttributionStatusBudgetFailed
	} else {
		receipt.Status = AttributionStatusPass
	}
	return receipt
}

func attributionBreaches(metrics AttributionMetrics, budget AttributionBudget) []AttributionBudgetBreach {
	var rows []AttributionBudgetBreach
	add := func(metric string, actual, maximum float64, unit string) {
		if actual > maximum {
			rows = append(rows, AttributionBudgetBreach{Metric: metric, Actual: actual, Maximum: maximum, Unit: unit})
		}
	}
	if metrics.UnknownShare != nil {
		add("unknown_share", *metrics.UnknownShare, budget.MaxUnknownShare, "fraction")
	}
	add("duplicate_events", float64(metrics.DuplicateEvents), float64(budget.MaxDuplicateEvents), "events")
	add("malformed_rows", float64(metrics.MalformedRows), float64(budget.MaxMalformedRows), "rows")
	add("unmatched_tool_events", float64(metrics.UnmatchedToolEvents), float64(budget.MaxUnmatchedToolEvents), "events")
	add("schema_drift_signals", float64(metrics.SchemaDriftSignals), float64(budget.MaxSchemaDriftSignals), "signals")
	add("bounded_files_omitted", float64(metrics.BoundedFilesOmitted), float64(budget.MaxBoundedFilesOmitted), "files")
	add("bounded_subtypes_omitted", float64(metrics.BoundedSubtypesOmitted), float64(budget.MaxBoundedSubtypesOmitted), "subtypes")
	sort.Slice(rows, func(i, j int) bool { return rows[i].Metric < rows[j].Metric })
	if rows == nil {
		return []AttributionBudgetBreach{}
	}
	return rows
}

func normalizeAttributionSamples(samples []AttributionSampleCoordinate, limit int) []AttributionSampleCoordinate {
	for i := range samples {
		samples[i].Source = scrubAttributionToken(samples[i].Source)
		samples[i].SourcePath = scrubAttributionPath(samples[i].SourcePath)
		samples[i].Subtype = scrubAttributionToken(samples[i].Subtype)
	}
	sort.Slice(samples, func(i, j int) bool {
		a, b := samples[i], samples[j]
		if a.Metric != b.Metric {
			return a.Metric < b.Metric
		}
		if a.Source != b.Source {
			return a.Source < b.Source
		}
		if a.SourcePath != b.SourcePath {
			return a.SourcePath < b.SourcePath
		}
		if a.Line != b.Line {
			return a.Line < b.Line
		}
		return a.Subtype < b.Subtype
	})
	deduped := make([]AttributionSampleCoordinate, 0, len(samples))
	var previous AttributionSampleCoordinate
	for _, sample := range samples {
		if len(deduped) > 0 && sample == previous {
			continue
		}
		deduped = append(deduped, sample)
		previous = sample
	}
	if len(deduped) <= limit {
		return deduped
	}
	// Reserve the first slots for one coordinate per signal family. With seven
	// budgeted families and a minimum configured cap of seven, this keeps every
	// breached family actionable before filling the remaining slots in stable
	// coordinate order.
	out := make([]AttributionSampleCoordinate, 0, limit)
	selected := make(map[int]bool)
	seenMetric := map[string]bool{}
	for index, sample := range deduped {
		if seenMetric[sample.Metric] {
			continue
		}
		seenMetric[sample.Metric] = true
		selected[index] = true
		out = append(out, sample)
		if len(out) == limit {
			return out
		}
	}
	for index, sample := range deduped {
		if selected[index] {
			continue
		}
		out = append(out, sample)
		if len(out) == limit {
			return out
		}
	}
	return out
}

func scrubAttributionPath(value string) string {
	value = filepath.ToSlash(filepath.Clean(strings.TrimSpace(value)))
	if value == "" || value == "." {
		return ""
	}
	digest := sha256.Sum256([]byte("trajectory-coordinate-path\x00" + value))
	return "sha256:" + hex.EncodeToString(digest[:6])
}

func scrubAttributionRoot(value string) string {
	if absolute, err := filepath.Abs(value); err == nil {
		value = absolute
	}
	value = filepath.ToSlash(filepath.Clean(value))
	digest := sha256.Sum256([]byte("trajectory-source-root\x00" + value))
	return "sha256:" + hex.EncodeToString(digest[:6])
}

func attributionSourceRootLabel(source string) string {
	switch source {
	case AuditSourceClaude:
		return "claude/projects"
	case AuditSourceCodex:
		return "codex/sessions"
	default:
		return "unsupported"
	}
}

func scrubAttributionRecordType(budget AttributionBudget, source, value string) string {
	for _, trusted := range budget.KnownSourceRecordTypes[source] {
		if value == trusted {
			return scrubAttributionToken(value)
		}
	}
	digest := sha256.Sum256([]byte("trajectory-record-subtype\x00" + value))
	return "sha256:" + hex.EncodeToString(digest[:6])
}

func scrubAttributionToken(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if len(value) <= 80 {
		safe := true
		for _, r := range value {
			if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("._:/-", r)) {
				safe = false
				break
			}
		}
		if safe {
			return value
		}
	}
	digest := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(digest[:6])
}

func attributionCollectionErrorCode(err error) string {
	if errors.Is(err, errAttributionDiscoveryLimit) {
		return "source_discovery_limit"
	}
	if errors.Is(err, errAttributionSourceChanged) {
		return "source_file_changed"
	}
	if errors.Is(err, fs.ErrPermission) {
		return "source_permission_denied"
	}
	return "source_collection_failed"
}

func attributionUnmatchedTools(source string, contents []byte, rel string) (int, []AttributionSampleCoordinate, error) {
	calls := map[string]AttributionSampleCoordinate{}
	results := map[string]AttributionSampleCoordinate{}
	scanner := bufio.NewScanner(bytes.NewReader(contents))
	scanner.Buffer(make([]byte, 64*1024), auditMaxLineBytes)
	line := 0
	for scanner.Scan() {
		line++
		var record map[string]any
		decoder := json.NewDecoder(bytes.NewReader(scanner.Bytes()))
		decoder.UseNumber()
		if decoder.Decode(&record) != nil {
			continue
		}
		switch source {
		case AuditSourceClaude:
			message, _ := record["message"].(map[string]any)
			blocks, _ := message["content"].([]any)
			for index, raw := range blocks {
				block, _ := raw.(map[string]any)
				kind, _ := block["type"].(string)
				id := ""
				subtype := ""
				switch kind {
				case "tool_use":
					id, _ = block["id"].(string)
					subtype = "tool_call_without_result"
				case "tool_result":
					id, _ = block["tool_use_id"].(string)
					subtype = "tool_result_without_call"
				}
				if subtype == "" {
					continue
				}
				if id == "" {
					id = fmt.Sprintf("line:%d:block:%d", line, index)
				}
				coordinate := AttributionSampleCoordinate{Metric: "unmatched_tool_events", Source: source, SourcePath: rel, Line: line, Subtype: subtype}
				if kind == "tool_use" {
					calls[id] = coordinate
				} else {
					results[id] = coordinate
				}
			}
		case AuditSourceCodex:
			if record["type"] != "response_item" {
				continue
			}
			payload, _ := record["payload"].(map[string]any)
			kind, _ := payload["type"].(string)
			id, _ := payload["call_id"].(string)
			if id == "" {
				id = fmt.Sprintf("line:%d", line)
			}
			coordinate := AttributionSampleCoordinate{Metric: "unmatched_tool_events", Source: source, SourcePath: rel, Line: line}
			switch kind {
			case "function_call", "custom_tool_call":
				coordinate.Subtype = "tool_call_without_result"
				calls[id] = coordinate
			case "function_call_output", "custom_tool_call_output":
				coordinate.Subtype = "tool_result_without_call"
				results[id] = coordinate
			}
		}
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return 0, nil, nil
		}
		return 0, nil, err
	}
	var samples []AttributionSampleCoordinate
	for id, coordinate := range calls {
		if _, matched := results[id]; !matched {
			samples = append(samples, coordinate)
		}
	}
	for id, coordinate := range results {
		if _, matched := calls[id]; !matched {
			samples = append(samples, coordinate)
		}
	}
	return len(samples), samples, nil
}

// DefaultAttributionHistoryPath keeps live receipt history out of the tracked
// tree. A reviewed receipt can be published explicitly without publishing raw
// transcripts or relying on an auto-committer.
func DefaultAttributionHistoryPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".fak", "nightrun", "trajectory-attribution.jsonl")
}

// AppendAttributionReceipt adds one comparable row after folding the newest
// prior row for the same corpus into receipt.Trend.
func AppendAttributionReceipt(path string, receipt *AttributionReceipt) error {
	if err := PrepareAttributionReceiptTrend(path, receipt); err != nil {
		return err
	}
	return AppendPreparedAttributionReceipt(path, receipt)
}

// PrepareAttributionReceiptTrend reads but does not mutate history. Callers
// that also publish a latest receipt can stage both outputs before appending.
func PrepareAttributionReceiptTrend(path string, receipt *AttributionReceipt) error {
	previous, found, err := readPreviousAttributionReceipt(path, receipt.Corpus)
	if err != nil {
		return err
	}
	receipt.Trend = compareAttributionReceipts(receipt, previous, found)
	return nil
}

// AppendPreparedAttributionReceipt appends a receipt whose trend was already
// prepared. A partial write is truncated back to the original history size.
func AppendPreparedAttributionReceipt(path string, receipt *AttributionReceipt) error {
	_, err := AppendPreparedAttributionReceiptWithRollback(path, receipt)
	return err
}

// AppendPreparedAttributionReceiptWithRollback returns a rollback for callers
// that must commit another output after the append. The rollback refuses to
// truncate if another writer extended the history in the meantime.
func AppendPreparedAttributionReceiptWithRollback(path string, receipt *AttributionReceipt) (func() error, error) {
	encoded, err := json.Marshal(receipt)
	if err != nil {
		return nil, fmt.Errorf("trajectory nightly: encode receipt: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("trajectory nightly: create history directory: %w", err)
	}
	_, statErr := os.Stat(path)
	existed := statErr == nil
	if statErr != nil && !os.IsNotExist(statErr) {
		return nil, fmt.Errorf("trajectory nightly: stat history: %w", statErr)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("trajectory nightly: open history: %w", err)
	}
	originalSize, err := file.Seek(0, io.SeekEnd)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("trajectory nightly: seek history: %w", err)
	}
	payload := append(encoded, '\n')
	rollback := func() error {
		info, err := os.Stat(path)
		if err != nil {
			return fmt.Errorf("trajectory nightly: stat history rollback: %w", err)
		}
		if info.Size() != originalSize+int64(len(payload)) {
			return errors.New("trajectory nightly: history changed before rollback")
		}
		if !existed {
			if err := os.Remove(path); err != nil {
				return fmt.Errorf("trajectory nightly: remove new history during rollback: %w", err)
			}
			return nil
		}
		if err := os.Truncate(path, originalSize); err != nil {
			return fmt.Errorf("trajectory nightly: truncate history rollback: %w", err)
		}
		return nil
	}
	written, writeErr := file.Write(payload)
	if writeErr != nil || written != len(payload) {
		_ = file.Truncate(originalSize)
		file.Close()
		if writeErr == nil {
			writeErr = io.ErrShortWrite
		}
		return nil, fmt.Errorf("trajectory nightly: append history: %w", writeErr)
	}
	if err := file.Sync(); err != nil {
		_ = file.Truncate(originalSize)
		file.Close()
		return nil, fmt.Errorf("trajectory nightly: sync history: %w", err)
	}
	if err := file.Close(); err != nil {
		if rollbackErr := rollback(); rollbackErr != nil {
			return nil, fmt.Errorf("trajectory nightly: close history: %v; rollback: %w", err, rollbackErr)
		}
		return nil, fmt.Errorf("trajectory nightly: close history: %w", err)
	}
	return rollback, nil
}

func readPreviousAttributionReceipt(path, corpus string) (AttributionReceipt, bool, error) {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return AttributionReceipt{}, false, nil
		}
		return AttributionReceipt{}, false, fmt.Errorf("trajectory nightly: open history: %w", err)
	}
	defer file.Close()
	var previous AttributionReceipt
	found := false
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		var row AttributionReceipt
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return AttributionReceipt{}, false, fmt.Errorf("trajectory nightly: decode history row: %w", err)
		}
		if row.Schema != AttributionReceiptSchema || row.Corpus != corpus {
			continue
		}
		previous, found = row, true
	}
	if err := scanner.Err(); err != nil {
		return AttributionReceipt{}, false, fmt.Errorf("trajectory nightly: scan history: %w", err)
	}
	return previous, found, nil
}

func compareAttributionReceipts(current *AttributionReceipt, previous AttributionReceipt, found bool) AttributionTrend {
	if !found {
		return AttributionTrend{Comparable: false, Reason: "no_previous_receipt"}
	}
	trend := AttributionTrend{PreviousObservedAt: previous.ObservedAt}
	if current.Schema != previous.Schema {
		trend.Reason = "receipt_schema_changed"
		return trend
	}
	if current.BudgetSchema != previous.BudgetSchema {
		trend.Reason = "budget_schema_changed"
		return trend
	}
	if current.BudgetVersion != previous.BudgetVersion {
		trend.Reason = "budget_version_changed"
		return trend
	}
	if current.Window != previous.Window {
		trend.Reason = "window_changed"
		return trend
	}
	if current.Status == AttributionStatusNoData || previous.Status == AttributionStatusNoData {
		trend.Reason = "no_data"
		return trend
	}
	if current.Status == AttributionStatusCollectionFailed || previous.Status == AttributionStatusCollectionFailed {
		trend.Reason = "collection_failed"
		return trend
	}
	if current.Status == AttributionStatusPublicationFailed || previous.Status == AttributionStatusPublicationFailed {
		trend.Reason = "publication_failed"
		return trend
	}
	if !attributionCoverageEqual(current.Coverage, previous.Coverage, false) {
		trend.Reason = "source_coverage_changed"
		return trend
	}
	if !attributionCoverageEqual(current.Coverage, previous.Coverage, true) {
		trend.Reason = "exposure_changed"
		return trend
	}
	trend.Comparable = true
	trend.Deltas = map[string]float64{
		"duplicate_events":         float64(current.Metrics.DuplicateEvents - previous.Metrics.DuplicateEvents),
		"malformed_rows":           float64(current.Metrics.MalformedRows - previous.Metrics.MalformedRows),
		"unmatched_tool_events":    float64(current.Metrics.UnmatchedToolEvents - previous.Metrics.UnmatchedToolEvents),
		"schema_drift_signals":     float64(current.Metrics.SchemaDriftSignals - previous.Metrics.SchemaDriftSignals),
		"bounded_files_omitted":    float64(current.Metrics.BoundedFilesOmitted - previous.Metrics.BoundedFilesOmitted),
		"bounded_subtypes_omitted": float64(current.Metrics.BoundedSubtypesOmitted - previous.Metrics.BoundedSubtypesOmitted),
	}
	if current.Metrics.UnknownShare != nil && previous.Metrics.UnknownShare != nil {
		trend.Deltas["unknown_share"] = *current.Metrics.UnknownShare - *previous.Metrics.UnknownShare
	}
	return trend
}

func attributionCoverageEqual(current, previous []AttributionSourceCoverage, includeExposure bool) bool {
	if len(current) != len(previous) {
		return false
	}
	currentBySource := make(map[string]AttributionSourceCoverage, len(current))
	for _, row := range current {
		if _, duplicate := currentBySource[row.Source]; duplicate {
			return false
		}
		currentBySource[row.Source] = row
	}
	for _, prior := range previous {
		row, ok := currentBySource[prior.Source]
		if !ok || row.Root != prior.Root || row.RootFingerprint != prior.RootFingerprint || row.RootPresent != prior.RootPresent {
			return false
		}
		if includeExposure && (row.FilesDiscovered != prior.FilesDiscovered ||
			row.FilesScanned != prior.FilesScanned ||
			row.FixtureFilesExcluded != prior.FixtureFilesExcluded ||
			row.Records != prior.Records || row.SubtypesObserved != prior.SubtypesObserved ||
			row.SubtypesOmitted != prior.SubtypesOmitted) {
			return false
		}
	}
	return true
}

func WriteAttributionReceipt(w io.Writer, receipt AttributionReceipt) error {
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(receipt); err != nil {
		return fmt.Errorf("trajectory nightly: write receipt: %w", err)
	}
	return nil
}
