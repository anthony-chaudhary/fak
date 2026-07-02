package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
)

func runVCacheCalibrate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache calibrate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	samplesPath := fs.String("samples", "", "provider-cache probe samples JSON/JSONL ('-' for stdin)")
	filePath := fs.String("file", "", "alias for --samples")
	asJSON := fs.Bool("json", false, "emit the fitted calibration JSON")
	out := fs.String("out", "", "write the fitted calibration JSON to this file")
	defaults := vcachecal.DefaultHypothesis()
	ttlMillis := fs.Int64("ttl-ms", defaults.TTLMillis, "fallback TTL hypothesis when probes do not measure it")
	minPrefix := fs.Int64("min-prefix-tokens", defaults.MinPrefixTokens, "fallback provider minimum prefix hypothesis")
	readMult := fs.Float64("read-mult", defaults.ReadMult, "fallback cached-read multiplier hypothesis")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	path := strings.TrimSpace(*samplesPath)
	alias := strings.TrimSpace(*filePath)
	if path != "" && alias != "" && path != alias {
		fmt.Fprintln(stderr, "fak vcache calibrate: use either --samples or --file, not both")
		return 2
	}
	if path == "" {
		path = alias
	}
	if path == "" {
		fmt.Fprintln(stderr, "fak vcache calibrate: --samples is required")
		return 2
	}
	if *ttlMillis <= 0 || *minPrefix <= 0 || *readMult <= 0 {
		fmt.Fprintln(stderr, "fak vcache calibrate: --ttl-ms, --min-prefix-tokens, and --read-mult must be positive")
		return 2
	}

	samples, err := readVCacheProbeSamples(path, os.Stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache calibrate: %v\n", err)
		return 2
	}
	cal := vcachecal.FitCalibration(samples, vcachecal.Hypothesis{
		TTLMillis:       *ttlMillis,
		MinPrefixTokens: *minPrefix,
		ReadMult:        *readMult,
	})
	if strings.TrimSpace(*out) != "" {
		if err := writeJSONFile(*out, cal); err != nil {
			fmt.Fprintf(stderr, "fak vcache calibrate: write %q: %v\n", *out, err)
			return 2
		}
	}
	if *asJSON {
		return writeJSON(stdout, cal)
	}
	renderVCacheCalibration(stdout, cal, samples, strings.TrimSpace(*out))
	return 0
}

type vcacheProbeSamplesFile struct {
	Samples []vcacheProbeSampleInput `json:"samples"`
}

type vcacheProbeSampleInput struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	ModelID  string `json:"model_id"`
	Endpoint string `json:"endpoint"`

	ModelIDCamel *string `json:"ModelID"`

	DelayMillis      *int64   `json:"delay_millis"`
	DelayMS          *int64   `json:"delay_ms"`
	DelayMillisCamel *int64   `json:"DelayMillis"`
	DelayMillisJSON  *int64   `json:"delayMillis"`
	DelaySeconds     *float64 `json:"delay_seconds"`

	PrefixTokens      *int64 `json:"prefix_tokens"`
	PrefixTokensCamel *int64 `json:"PrefixTokens"`
	PrefixTokensJSON  *int64 `json:"prefixTokens"`

	CachedTokens          *int64 `json:"cached_tokens"`
	CachedTokensCamel     *int64 `json:"CachedTokens"`
	CachedTokensJSON      *int64 `json:"cachedTokens"`
	CacheReadInputTokens  *int64 `json:"cache_read_input_tokens"`
	CacheReadInputTokens2 *int64 `json:"cache_read_tokens"`

	ReadCostEquiv      *float64 `json:"read_cost_equiv"`
	ReadCostEquivCamel *float64 `json:"ReadCostEquiv"`
	ReadCostEquivJSON  *float64 `json:"readCostEquiv"`
	CachedReadCost     *float64 `json:"cached_read_cost_equiv"`
}

func readVCacheProbeSamples(path string, stdin io.Reader) ([]vcachecal.ProbeSample, error) {
	r, closeInput, err := openInputOrStdin(path, stdin)
	if err != nil {
		return nil, err
	}
	defer closeInput()
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return parseVCacheProbeSamples(raw)
}

func parseVCacheProbeSamples(raw []byte) ([]vcachecal.ProbeSample, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil, errors.New("probe sample input is empty")
	}
	if trimmed[0] == '[' {
		var rows []vcacheProbeSampleInput
		if err := json.Unmarshal(trimmed, &rows); err != nil {
			return nil, err
		}
		return normalizeVCacheProbeSamples(rows)
	}
	if trimmed[0] == '{' {
		var wrapped vcacheProbeSamplesFile
		if err := json.Unmarshal(trimmed, &wrapped); err == nil && wrapped.Samples != nil {
			return normalizeVCacheProbeSamples(wrapped.Samples)
		}
		var one vcacheProbeSampleInput
		if err := json.Unmarshal(trimmed, &one); err == nil {
			return normalizeVCacheProbeSamples([]vcacheProbeSampleInput{one})
		}
	}
	return parseVCacheProbeSampleJSONL(trimmed)
}

func parseVCacheProbeSampleJSONL(raw []byte) ([]vcachecal.ProbeSample, error) {
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var rows []vcacheProbeSampleInput
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var row vcacheProbeSampleInput
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return normalizeVCacheProbeSamples(rows)
}

func normalizeVCacheProbeSamples(rows []vcacheProbeSampleInput) ([]vcachecal.ProbeSample, error) {
	if len(rows) == 0 {
		return nil, errors.New("probe sample input has no samples")
	}
	out := make([]vcachecal.ProbeSample, 0, len(rows))
	for i, row := range rows {
		s, err := normalizeVCacheProbeSample(row)
		if err != nil {
			return nil, fmt.Errorf("sample %d: %w", i+1, err)
		}
		out = append(out, s)
	}
	return out, nil
}

func normalizeVCacheProbeSample(row vcacheProbeSampleInput) (vcachecal.ProbeSample, error) {
	delay, ok := firstInt64Ptr(row.DelayMillis, row.DelayMS, row.DelayMillisCamel, row.DelayMillisJSON)
	if !ok && row.DelaySeconds != nil {
		delay = int64(*row.DelaySeconds * 1000)
		ok = true
	}
	if !ok {
		return vcachecal.ProbeSample{}, errors.New("delay_millis is required")
	}
	prefix, ok := firstInt64Ptr(row.PrefixTokens, row.PrefixTokensCamel, row.PrefixTokensJSON)
	if !ok {
		return vcachecal.ProbeSample{}, errors.New("prefix_tokens is required")
	}
	cached, ok := firstInt64Ptr(row.CachedTokens, row.CachedTokensCamel, row.CachedTokensJSON, row.CacheReadInputTokens, row.CacheReadInputTokens2)
	if !ok {
		return vcachecal.ProbeSample{}, errors.New("cached_tokens is required")
	}
	readCost, _ := firstFloat64Ptr(row.ReadCostEquiv, row.ReadCostEquivCamel, row.ReadCostEquivJSON, row.CachedReadCost)
	if delay < 0 {
		return vcachecal.ProbeSample{}, errors.New("delay_millis must be non-negative")
	}
	if prefix <= 0 {
		return vcachecal.ProbeSample{}, errors.New("prefix_tokens must be positive")
	}
	if cached < 0 {
		return vcachecal.ProbeSample{}, errors.New("cached_tokens must be non-negative")
	}
	if readCost < 0 {
		return vcachecal.ProbeSample{}, errors.New("read_cost_equiv must be non-negative")
	}
	return vcachecal.ProbeSample{
		Provider:      strings.TrimSpace(row.Provider),
		ModelID:       firstNonEmpty(row.ModelID, derefString(row.ModelIDCamel), row.Model),
		Endpoint:      strings.TrimSpace(row.Endpoint),
		DelayMillis:   delay,
		PrefixTokens:  prefix,
		CachedTokens:  cached,
		ReadCostEquiv: readCost,
	}, nil
}

func firstInt64Ptr(values ...*int64) (int64, bool) {
	for _, v := range values {
		if v != nil {
			return *v, true
		}
	}
	return 0, false
}

func firstFloat64Ptr(values ...*float64) (float64, bool) {
	for _, v := range values {
		if v != nil {
			return *v, true
		}
	}
	return 0, false
}

func derefString(v *string) string {
	if v == nil {
		return ""
	}
	return *v
}

func renderVCacheCalibration(w io.Writer, cal vcachecal.Calibration, samples []vcachecal.ProbeSample, out string) {
	hits, misses := vcacheProbeHitMiss(samples)
	fmt.Fprintf(w, "vCache calibration: %s / %s / %s\n",
		vcacheEmptyDash(cal.Provider), vcacheEmptyDash(cal.ModelID), vcacheEmptyDash(cal.Endpoint))
	fmt.Fprintf(w, "samples: %d (hits %d, misses %d)\n", len(samples), hits, misses)
	fmt.Fprintf(w, "ttl: %d ms (%s)\n", cal.TTLMillis, measuredLabel(cal.TTLMeasured))
	fmt.Fprintf(w, "min prefix: %d tokens (%s)\n", cal.MinPrefixTokens, measuredLabel(cal.MinPrefixMeasured))
	fmt.Fprintf(w, "cached-read multiplier: %.6g (%s)\n", cal.ReadMult, measuredLabel(cal.ReadMultMeasured))
	if out != "" {
		fmt.Fprintf(w, "calibration: %s\n", out)
	}
}

func vcacheProbeHitMiss(samples []vcachecal.ProbeSample) (hits, misses int) {
	for _, s := range samples {
		if s.CachedTokens > 0 {
			hits++
		} else {
			misses++
		}
	}
	return hits, misses
}

func measuredLabel(ok bool) string {
	if ok {
		return "measured"
	}
	return "assumed"
}

func vcacheEmptyDash(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "-"
	}
	return s
}
