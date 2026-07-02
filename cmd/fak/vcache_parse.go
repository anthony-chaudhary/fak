package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

type vcacheAnchorInput struct {
	Key              string  `json:"key"`
	Anchor           string  `json:"anchor"`
	ID               string  `json:"id"`
	PrefixDigest     string  `json:"prefix_digest"`
	Frequency        float64 `json:"frequency"`
	Freq             float64 `json:"freq"`
	Count            float64 `json:"count"`
	AccessRatePerSec float64 `json:"access_rate_per_sec"`
	Size             float64 `json:"size"`
	Tokens           float64 `json:"tokens"`
	PrefixTokens     float64 `json:"prefix_tokens"`
	ReuseDensity     float64 `json:"reuse_density"`
	Reuse            float64 `json:"reuse"`
	Reuses           float64 `json:"reuses"`
	ExpectedReuse    float64 `json:"expected_reuse"`
	Weight           float64 `json:"weight"`
}

func readVCacheAnchors(path string, stdin io.Reader) ([]vcachecal.RankedVBlock, error) {
	var data []byte
	var err error
	if path == "-" {
		if stdin == nil {
			return nil, errors.New("stdin is not available")
		}
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, err
	}
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, errors.New("anchor workload is empty")
	}

	var rows []vcacheAnchorInput
	switch trimmed[0] {
	case '[':
		if err := json.Unmarshal(trimmed, &rows); err != nil {
			return nil, err
		}
	case '{':
		rows, err = readVCacheAnchorJSONL(trimmed)
	default:
		rows, err = readVCacheAnchorCSV(trimmed)
	}
	if err != nil {
		return nil, err
	}
	ranked := vcachescore.NormalizeRanked(anchorInputsToRanked(rows))
	if len(ranked) == 0 {
		return nil, errors.New("anchor workload has no positive-weight rows")
	}
	return ranked, nil
}

func readVCacheAnchorJSONL(raw []byte) ([]vcacheAnchorInput, error) {
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var rows []vcacheAnchorInput
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row vcacheAnchorInput
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("anchor line %d: %w", lineNo, err)
		}
		rows = append(rows, row)
	}
	return rows, sc.Err()
}

func readVCacheAnchorCSV(raw []byte) ([]vcacheAnchorInput, error) {
	cr := csv.NewReader(bytes.NewReader(raw))
	cr.TrimLeadingSpace = true
	cr.FieldsPerRecord = -1
	records, err := cr.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, errors.New("anchor CSV is empty")
	}
	header := map[string]int{}
	for i, h := range records[0] {
		header[normalizeVCacheAnchorField(h)] = i
	}
	rows := make([]vcacheAnchorInput, 0, len(records)-1)
	for i, rec := range records[1:] {
		row, err := parseVCacheAnchorCSVRecord(header, rec)
		if err != nil {
			return nil, fmt.Errorf("anchor CSV row %d: %w", i+2, err)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func parseVCacheAnchorCSVRecord(header map[string]int, rec []string) (vcacheAnchorInput, error) {
	var row vcacheAnchorInput
	row.Key = csvString(header, rec, "key", "anchor", "id", "prefix_digest")
	var err error
	if row.Frequency, err = csvFloat(header, rec, "frequency", "freq", "count", "access_rate_per_sec"); err != nil {
		return row, err
	}
	if row.Size, err = csvFloat(header, rec, "size", "tokens", "prefix_tokens"); err != nil {
		return row, err
	}
	if row.ReuseDensity, err = csvFloat(header, rec, "reuse_density", "reuse", "reuses", "expected_reuse"); err != nil {
		return row, err
	}
	if row.Weight, err = csvFloat(header, rec, "weight"); err != nil {
		return row, err
	}
	return row, nil
}

func normalizeVCacheAnchorField(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, "-", "_")
	s = strings.ReplaceAll(s, " ", "_")
	return s
}

func csvString(header map[string]int, rec []string, names ...string) string {
	for _, name := range names {
		if idx, ok := header[name]; ok && idx < len(rec) {
			return strings.TrimSpace(rec[idx])
		}
	}
	return ""
}

func csvFloat(header map[string]int, rec []string, names ...string) (float64, error) {
	for _, name := range names {
		idx, ok := header[name]
		if !ok || idx >= len(rec) {
			continue
		}
		s := strings.TrimSpace(rec[idx])
		if s == "" {
			return 0, nil
		}
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, fmt.Errorf("%s=%q: %w", name, s, err)
		}
		return v, nil
	}
	return 0, nil
}

func anchorInputsToRanked(rows []vcacheAnchorInput) []vcachecal.RankedVBlock {
	out := make([]vcachecal.RankedVBlock, 0, len(rows))
	for _, row := range rows {
		v := vcachecal.RankedVBlock{
			Key:          firstAnchorString(row.Key, row.Anchor, row.ID, row.PrefixDigest),
			Frequency:    firstAnchorFloat(row.Frequency, row.Freq, row.Count, row.AccessRatePerSec),
			Size:         firstAnchorFloat(row.Size, row.Tokens, row.PrefixTokens),
			ReuseDensity: firstAnchorFloat(row.ReuseDensity, row.Reuse, row.Reuses, row.ExpectedReuse),
		}
		if row.Weight != 0 {
			v.Frequency = row.Weight
			v.Size = 1
			v.ReuseDensity = 1
		}
		out = append(out, v)
	}
	return out
}

func firstAnchorString(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func firstAnchorFloat(values ...float64) float64 {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}

type vcacheTelemetryJSONLRow struct {
	InputTokens              float64             `json:"input_tokens"`
	PromptTokens             float64             `json:"prompt_tokens"`
	CachedTokens             float64             `json:"cached_tokens"`
	CacheCreationInputTokens float64             `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     float64             `json:"cache_read_input_tokens"`
	Ephemeral1hInputTokens   float64             `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens   float64             `json:"ephemeral_5m_input_tokens"`
	Usage                    *vcacheOpenAIUsage  `json:"usage"`
	Type                     string              `json:"type"`
	Payload                  *vcacheCodexPayload `json:"payload"`
}

type vcacheOpenAIUsage struct {
	InputTokens         float64                   `json:"input_tokens"`
	PromptTokens        float64                   `json:"prompt_tokens"`
	CachedInputTokens   float64                   `json:"cached_input_tokens"`
	InputTokensDetails  vcacheCachedTokensDetails `json:"input_tokens_details"`
	PromptTokensDetails vcacheCachedTokensDetails `json:"prompt_tokens_details"`
}

type vcacheCachedTokensDetails struct {
	CachedTokens float64 `json:"cached_tokens"`
}

type vcacheCodexPayload struct {
	Type string               `json:"type"`
	Info vcacheCodexTokenInfo `json:"info"`
}

type vcacheCodexTokenInfo struct {
	LastTokenUsage vcacheCodexTokenUsage `json:"last_token_usage"`
}

type vcacheCodexTokenUsage struct {
	InputTokens       float64 `json:"input_tokens"`
	CachedInputTokens float64 `json:"cached_input_tokens"`
}

// openInputOrStdin opens path for streaming, or returns stdin when path is "-". The
// returned closer MUST be deferred by the caller (it is a no-op on the stdin path); it
// keeps the file open for the lifetime of the caller's read, matching an inline
// `defer f.Close()`.
func openInputOrStdin(path string, stdin io.Reader) (io.Reader, func() error, error) {
	if path == "-" {
		return stdin, func() error { return nil }, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	return f, f.Close, nil
}

func readVCacheTelemetry(path string, stdin io.Reader) ([]vcachegov.TelemetryRow, error) {
	r, closeInput, err := openInputOrStdin(path, stdin)
	if err != nil {
		return nil, err
	}
	defer closeInput()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var rows []vcachegov.TelemetryRow
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var raw vcacheTelemetryJSONLRow
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		row, ok := raw.telemetryRow()
		if ok {
			rows = append(rows, row)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func (r vcacheTelemetryJSONLRow) telemetryRow() (vcachegov.TelemetryRow, bool) {
	if r.hasClaudeCounters() {
		return vcachegov.TelemetryRow{
			InputTokens:              r.InputTokens,
			CacheCreationInputTokens: r.CacheCreationInputTokens,
			CacheReadInputTokens:     r.CacheReadInputTokens,
			Ephemeral1hInputTokens:   r.Ephemeral1hInputTokens,
			Ephemeral5mInputTokens:   r.Ephemeral5mInputTokens,
		}, true
	}
	if r.Usage != nil {
		total, cached := r.Usage.openAITokens()
		return openAITelemetryRow(total, cached), true
	}
	if r.Payload != nil && r.Type == "event_msg" && r.Payload.Type == "token_count" {
		usage := r.Payload.Info.LastTokenUsage
		if usage.InputTokens != 0 || usage.CachedInputTokens != 0 {
			return openAITelemetryRow(usage.InputTokens, usage.CachedInputTokens), true
		}
	}
	if r.InputTokens != 0 || r.PromptTokens != 0 || r.CachedTokens != 0 {
		return openAITelemetryRow(firstNonZero(r.InputTokens, r.PromptTokens), r.CachedTokens), true
	}
	return vcachegov.TelemetryRow{}, false
}

func (r vcacheTelemetryJSONLRow) hasClaudeCounters() bool {
	return r.CacheCreationInputTokens != 0 ||
		r.CacheReadInputTokens != 0 ||
		r.Ephemeral1hInputTokens != 0 ||
		r.Ephemeral5mInputTokens != 0
}

func (u vcacheOpenAIUsage) openAITokens() (float64, float64) {
	total := firstNonZero(u.InputTokens, u.PromptTokens)
	cached := firstNonZero(u.InputTokensDetails.CachedTokens, u.PromptTokensDetails.CachedTokens, u.CachedInputTokens)
	return total, cached
}

func openAITelemetryRow(total, cached float64) vcachegov.TelemetryRow {
	if total < 0 {
		total = 0
	}
	if cached < 0 {
		cached = 0
	}
	if cached > total {
		cached = total
	}
	return vcachegov.TelemetryRow{
		InputTokens:          total - cached,
		CacheReadInputTokens: cached,
	}
}

func firstNonZero(values ...float64) float64 {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}
