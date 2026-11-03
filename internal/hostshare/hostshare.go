// Package hostshare reads host-share evidence out of committed performance
// receipts and profile notes, turning one-off prose measurements (the PHL-051
// "host_share 0.130" style) into a systematic, gateable reading.
//
// host_share is the host-phase share of decode step time: the fraction of each
// decode step's wall clock spent outside device work — kernel launch latency,
// per-layer host setup, synchronous host<->device round-trips. At small batch
// sizes decode is often launch-bound rather than compute-bound, and host_share
// is the single number that separates those regimes.
//
// The reader derives host_share ONLY from recorded evidence, in priority order:
//
//   - a directly recorded host_share / host_share_p50 fraction (ad-hoc measured),
//   - the native v3 host-overhead decomposition on perf receipts:
//     repetitions[].host_phase_us over repetitions[].decode_milliseconds,
//     averaged over the repetitions that record both,
//   - a recorded device-busy fraction (device_busy / device_busy_p50): host_share = 1 - device_busy,
//   - a recorded GPU busy percent (gpu_busy_percent / gpu_busy_p50): host_share = 1 - pct/100,
//   - a recorded ms/token split (gpu_ms_per_token over wall_ms_per_token): host_share = 1 - gpu/wall.
//
// A receipt carrying none of these yields an UNKNOWN reading — never a
// fabricated zero, so an absent measurement can never masquerade as a healthy
// regime. Evidence fields are read leniently (any JSON record shape), while
// perf-receipt files (schema fak-native-performance-receipt/v*) are always
// admitted as decode-step readings even when they record no device-busy data,
// because their repetitions ARE the decode window.
package hostshare

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Source kinds name which recorded evidence produced a reading's HostShare.
// An empty source means UNKNOWN (no device-busy evidence in the record).
const (
	SourceHostShareRecorded      = "host-share-recorded"
	SourceHostPhaseDecomposition = "host-phase-decomposition"
	SourceDeviceBusyFraction     = "device-busy-fraction"
	SourceGPUBusyPercent         = "gpu-busy-percent"
	SourceMSPerTokenSplit        = "ms-per-token-split"
)

// nativeReceiptSchemaPrefix matches the portable A/B performance receipts
// (fak-native-performance-receipt/v1|v2) whose repetitions bound the decode
// window even though the schema records no device-busy fraction.
const nativeReceiptSchemaPrefix = "fak-native-performance-receipt/"

// maxFileBytes bounds one scanned file; receipts and profile notes are KiB-scale.
const maxFileBytes = 16 << 20

// Reading is one host-share observation extracted from a receipt or profile
// note. HostShare is nil when the record carries no device-busy evidence; it
// is never defaulted to zero.
type Reading struct {
	Path      string    `json:"path"`
	Model     string    `json:"model,omitempty"`
	Hardware  string    `json:"hardware,omitempty"`
	HostShare *float64  `json:"host_share"`
	Source    string    `json:"source,omitempty"`
	Timestamp time.Time `json:"timestamp,omitempty"`
}

// Report is the outcome of scanning one or more receipt directories.
type Report struct {
	Readings       []Reading
	FilesScanned   int // JSON/JSONL files inspected under the scan roots
	EvidenceFree   int // parsed files that yielded no decode-step reading at all
	MalformedFiles []string
}

// Scan roots recursively for *.json / *.jsonl files (a root may also be a
// single receipt file) and returns the decode-step readings, oldest first
// (most recent last). Malformed JSON content is skipped and listed in
// Report.MalformedFiles — a shared receipts tree routinely holds peers'
// half-written files — while I/O errors fail the scan.
func Scan(roots []string) ([]Reading, error) {
	rep, err := ScanReport(roots)
	return rep.Readings, err
}

// ScanReport is Scan with the census counters callers render in summaries.
func ScanReport(roots []string) (Report, error) {
	var rep Report
	for _, root := range roots {
		info, err := os.Stat(root)
		if err != nil {
			return rep, err
		}
		if !info.IsDir() {
			if err := rep.scanFile(root); err != nil {
				return rep, err
			}
			continue
		}
		walkErr := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			ext := strings.ToLower(filepath.Ext(p))
			if ext != ".json" && ext != ".jsonl" {
				return nil
			}
			return rep.scanFile(p)
		})
		if walkErr != nil {
			return rep, walkErr
		}
	}
	sortReadings(rep.Readings)
	return rep, nil
}

// scanFile folds one parsed file into the report census.
func (rep *Report) scanFile(p string) error {
	readings, malformed, err := readPath(p)
	if err != nil {
		return err
	}
	rep.FilesScanned++
	if len(readings) == 0 {
		if malformed {
			rep.MalformedFiles = append(rep.MalformedFiles, filepath.ToSlash(p))
		} else {
			rep.EvidenceFree++
		}
		return nil
	}
	rep.Readings = append(rep.Readings, readings...)
	return nil
}

// readPath parses one file into zero or more readings. malformed reports that
// the file was neither one JSON document nor line-delimited JSON objects.
func readPath(path string) ([]Reading, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false, err
	}
	if len(data) > maxFileBytes {
		return nil, false, fmt.Errorf("hostshare: %s exceeds %d bytes", path, maxFileBytes)
	}
	rel := filepath.ToSlash(path)
	var records []map[string]any
	if rec, ok := parseObject(data); ok {
		records = []map[string]any{rec}
	} else if lines := splitLines(data); lines != nil {
		for _, line := range lines {
			if rec, ok := parseObject([]byte(line)); ok {
				records = append(records, rec)
			}
		}
	}
	if records == nil {
		return nil, true, nil
	}
	var out []Reading
	for _, rec := range records {
		if !isDecodeStepRecord(rec) {
			continue
		}
		out = append(out, newReading(rel, rec))
	}
	return out, false, nil
}

// parseObject decodes one JSON document that must be an object.
func parseObject(data []byte) (map[string]any, bool) {
	var rec map[string]any
	if err := json.Unmarshal(data, &rec); err != nil || rec == nil {
		return nil, false
	}
	return rec, true
}

// splitLines returns non-blank lines of a JSONL document, or nil when the
// input has no usable line.
func splitLines(data []byte) []string {
	var lines []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return nil
	}
	return lines
}

// isDecodeStepRecord reports whether a parsed JSON record is a decode-step
// reading: a nativeperf receipt (its repetitions bound the decode window), a
// record carrying host-share evidence, or a record shaped like decode timing
// (a ledger summary, a kernel window, a gateway perf record).
func isDecodeStepRecord(rec map[string]any) bool {
	if schema, _ := rec["schema"].(string); strings.HasPrefix(schema, nativeReceiptSchemaPrefix) {
		return true
	}
	if _, src := deriveHostShare(rec); src != "" {
		return true
	}
	for _, key := range []string{"decode_ms", "decode_tps", "decode_steps", "wall_ms_per_token", "native_timing"} {
		if _, ok := rec[key]; ok {
			return true
		}
	}
	return false
}

// newReading builds one Reading from a parsed record.
func newReading(path string, rec map[string]any) Reading {
	share, source := deriveHostShare(rec)
	return Reading{
		Path:      path,
		Model:     extractModel(rec),
		Hardware:  extractHardware(rec),
		HostShare: share,
		Source:    source,
		Timestamp: recordTimestamp(rec, path),
	}
}

// deriveHostShare applies the evidence priority rules. It returns nil when no
// admissible evidence is recorded — the caller must render that as UNKNOWN.
func deriveHostShare(rec map[string]any) (*float64, string) {
	if v, ok := finiteInRange(rec, "host_share", 0, 1); ok {
		return &v, SourceHostShareRecorded
	}
	if v, ok := finiteInRange(rec, "host_share_p50", 0, 1); ok {
		return &v, SourceHostShareRecorded
	}
	if v, ok := nativeHostPhaseShare(rec); ok {
		return &v, SourceHostPhaseDecomposition
	}
	if v, ok := finitePositiveInRange(rec, "device_busy", 0, 1); ok {
		hs := clamp01(1 - v)
		return &hs, SourceDeviceBusyFraction
	}
	if v, ok := finitePositiveInRange(rec, "device_busy_p50", 0, 1); ok {
		hs := clamp01(1 - v)
		return &hs, SourceDeviceBusyFraction
	}
	if v, ok := finitePositiveInRange(rec, "gpu_busy_percent", 0, 100); ok {
		hs := clamp01(1 - v/100)
		return &hs, SourceGPUBusyPercent
	}
	if v, ok := finitePositiveInRange(rec, "gpu_busy_p50", 0, 100); ok {
		hs := clamp01(1 - v/100)
		return &hs, SourceGPUBusyPercent
	}
	if g, ok := numField(rec, "gpu_ms_per_token"); ok && g > 0 {
		if w, wok := numField(rec, "wall_ms_per_token"); wok && w >= g*(1-1e-9) {
			hs := clamp01(1 - g/w)
			return &hs, SourceMSPerTokenSplit
		}
	}
	return nil, ""
}

// nativeHostPhaseShare reads the native v3 host-overhead decomposition out of
// a performance receipt: repetitions[].host_phase_us over
// repetitions[].decode_milliseconds, averaged over the repetitions that
// record both. It returns false when no repetition carries the pair — the
// caller must treat that as UNKNOWN, never a zero. The share may exceed 1 on
// multi-threaded hosts, matching the process-CPU-over-wall semantics of the
// collector that records host_phase_us.
func nativeHostPhaseShare(rec map[string]any) (float64, bool) {
	schema, _ := rec["schema"].(string)
	if !strings.HasPrefix(schema, nativeReceiptSchemaPrefix) {
		return 0, false
	}
	reps, _ := rec["repetitions"].([]any)
	var sum, n float64
	for _, rv := range reps {
		rm, ok := rv.(map[string]any)
		if !ok {
			continue
		}
		hostUs, ok1 := numField(rm, "host_phase_us")
		decodeMs, ok2 := numField(rm, "decode_milliseconds")
		if !ok1 || !ok2 || decodeMs <= 0 {
			continue
		}
		share := hostUs / (decodeMs * 1e3)
		if math.IsNaN(share) || math.IsInf(share, 0) || share < 0 {
			continue
		}
		sum += share
		n++
	}
	if n == 0 {
		return 0, false
	}
	return sum / n, true
}

// extractModel best-efforts the model identity from common record shapes.
func extractModel(rec map[string]any) string {
	if s := strField(rec, "model"); s != "" {
		return s
	}
	for _, parent := range []string{"controls", "served_by"} {
		if m, ok := rec[parent].(map[string]any); ok {
			if s := strField(m, "model"); s != "" {
				return s
			}
		}
	}
	// Nativeperf receipts carry the model only inside the envelope id
	// (e.g. "qwen38-apple-metal-serving"); use it verbatim as the identity.
	if schema, _ := rec["schema"].(string); strings.HasPrefix(schema, nativeReceiptSchemaPrefix) {
		if s := strField(rec, "envelope_id"); s != "" {
			return s
		}
	}
	return ""
}

// extractHardware best-efforts the hardware identity from common record shapes.
func extractHardware(rec map[string]any) string {
	if m, ok := rec["machine"].(map[string]any); ok {
		for _, key := range []string{"gpu_name", "cpu_model"} {
			if s := strField(m, key); s != "" {
				return s
			}
		}
		platform, backend := strField(m, "platform"), strField(m, "backend")
		switch {
		case platform != "" && backend != "":
			return platform + "/" + backend
		case platform != "":
			return platform
		case backend != "":
			return backend
		}
	}
	for _, key := range []string{"device_name", "hardware", "gpu_name", "cpu_model"} {
		if s := strField(rec, key); s != "" {
			return s
		}
	}
	return ""
}

var timestampFields = []string{"generated_at", "observed_utc", "observed_at", "captured_at", "timestamp", "date"}

// recordTimestamp resolves a record's capture time from its fields, falling
// back to a YYYYMMDD / YYYY-MM-DD token in the file name. The zero time means
// "unrecorded"; unrecorded readings sort before dated ones.
func recordTimestamp(rec map[string]any, path string) time.Time {
	for _, field := range timestampFields {
		s, ok := rec[field].(string)
		if !ok {
			continue
		}
		for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05", "2006-01-02"} {
			if t, err := time.Parse(layout, s); err == nil {
				return t
			}
		}
	}
	if v, ok := numField(rec, "unix_ms"); ok && v > 0 {
		return time.UnixMilli(int64(v)).UTC()
	}
	return filenameDate(filepath.Base(path))
}

var filenameDateRe = regexp.MustCompile(`(?:^|[^0-9])([0-9]{4})[-_]?([0-9]{2})[-_]?([0-9]{2})(?:$|[^0-9])`)

func filenameDate(base string) time.Time {
	m := filenameDateRe.FindStringSubmatch(base)
	if m == nil {
		return time.Time{}
	}
	y, mo, d := atoi(m[1]), atoi(m[2]), atoi(m[3])
	if y < 2000 || mo < 1 || mo > 12 || d < 1 || d > 31 {
		return time.Time{}
	}
	return time.Date(y, time.Month(mo), d, 0, 0, 0, 0, time.UTC)
}

// sortReadings orders readings oldest-first by (timestamp, path); undated
// readings (zero timestamp) come first, ties break lexicographically on path.
func sortReadings(rs []Reading) {
	sort.SliceStable(rs, func(i, j int) bool {
		ti, tj := rs[i].Timestamp, rs[j].Timestamp
		if !ti.Equal(tj) {
			return ti.Before(tj)
		}
		return rs[i].Path < rs[j].Path
	})
}

// Latest returns the most recent reading.
func Latest(rs []Reading) (Reading, bool) {
	if len(rs) == 0 {
		return Reading{}, false
	}
	return rs[len(rs)-1], true
}

// Verdict classifies one reading against the threshold.
func Verdict(share *float64, threshold float64) string {
	if share == nil {
		return "UNKNOWN"
	}
	if *share > threshold {
		return "LAUNCH-BOUND"
	}
	return "OK"
}

// Evaluate applies the regression gate: the most recent reading decides. An
// UNKNOWN most-recent reading never fails the gate — absence of evidence is
// not evidence of health, but it is not evidence of regression either.
func Evaluate(rs []Reading, threshold float64) (gated Reading, exceeded bool, ok bool) {
	gated, ok = Latest(rs)
	if !ok || gated.HostShare == nil {
		return gated, false, ok
	}
	return gated, *gated.HostShare > threshold, true
}

// --- field access helpers ---------------------------------------------------

func strField(rec map[string]any, key string) string {
	s, _ := rec[key].(string)
	return strings.TrimSpace(s)
}

func numField(rec map[string]any, key string) (float64, bool) {
	v, ok := rec[key].(float64)
	if !ok || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return v, true
}

func finiteInRange(rec map[string]any, key string, lo, hi float64) (float64, bool) {
	v, ok := numField(rec, key)
	if !ok || v < lo || v > hi {
		return 0, false
	}
	return v, true
}

// finitePositiveInRange is finiteInRange plus v > 0: a recorded busy fraction
// of exactly zero means the meter was absent, not that the device idled for
// the whole window, so it is treated as unrecorded.
func finitePositiveInRange(rec map[string]any, key string, lo, hi float64) (float64, bool) {
	v, ok := finiteInRange(rec, key, lo, hi)
	if !ok || v <= 0 {
		return 0, false
	}
	return v, true
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func atoi(s string) int {
	n := 0
	for _, r := range s {
		n = n*10 + int(r-'0')
	}
	return n
}
