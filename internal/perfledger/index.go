// Adapted from ai-dynamo/aiperf
// modified from the original): compressed run-index store using stdlib
// compress/gzip + indexed summary columns keyed by (namespace, run_id, epoch),
// with fail-safe disk-scan fallback when the index file is absent or corrupt.
package perfledger

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// RunIndexSchemaVersion identifies the on-disk compressed index format.
const RunIndexSchemaVersion = 1

// IndexReadSource reports whether a query was answered by the compressed index
// file or degraded to a disk scan.
type IndexReadSource string

const (
	IndexSourceFile         IndexReadSource = "index"
	IndexSourceDiskFallback IndexReadSource = "disk_fallback"
)

// RunIndexRow stores narrow indexed summary columns alongside the gzipped
// full JSON run payload, keyed uniquely by (Namespace, RunID, Epoch).
type RunIndexRow struct {
	Namespace     string  `json:"namespace"`
	RunID         string  `json:"run_id"`
	Epoch         int64   `json:"epoch"`
	Model         string  `json:"model,omitempty"`
	Status        string  `json:"status,omitempty"`
	CreatedUnixMS int64   `json:"created_unix_ms"`
	TotalRequests int64   `json:"total_requests,omitempty"`
	ErrorRate     float64 `json:"error_rate,omitempty"`
	ThroughputTPS float64 `json:"throughput_tps,omitempty"`
	TTFTP50MS     float64 `json:"ttft_p50_ms,omitempty"`
	TTFTP99MS     float64 `json:"ttft_p99_ms,omitempty"`
	E2EP95MS      float64 `json:"e2e_p95_ms,omitempty"`
	ArtifactDir   string  `json:"artifact_dir,omitempty"`
	PayloadGzip   []byte  `json:"payload_gzip,omitempty"`
}

// RunIndexFilter constrains and orders List results over indexed columns.
type RunIndexFilter struct {
	Namespace  string
	Model      string
	Status     string
	SortBy     string
	Descending bool
	Limit      int
}

// RunIndexOptions configures a RunIndexStore.
type RunIndexOptions struct {
	IndexPath       string
	ResultsDir      string
	FallbackScanner func(ctx context.Context, resultsDir string) ([]RunIndexRow, error)
}

type runIndexFilePayload struct {
	SchemaVersion int           `json:"schema_version"`
	Rows          []RunIndexRow `json:"rows"`
}

// RunIndexStore persists run summary rows and gzip-compressed payloads in an
// atomic gzipped JSON index file, degrading transparently to a disk scan when
// the index file is missing or corrupt.
type RunIndexStore struct {
	mu      sync.Mutex
	opts    RunIndexOptions
	scanner func(ctx context.Context, resultsDir string) ([]RunIndexRow, error)
}

// NewRunIndexStore validates options and returns a RunIndexStore.
func NewRunIndexStore(opts RunIndexOptions) (*RunIndexStore, error) {
	if opts.IndexPath == "" && opts.ResultsDir == "" {
		return nil, errors.New("perfledger: RunIndexStore requires IndexPath or ResultsDir")
	}
	scanner := opts.FallbackScanner
	if scanner == nil {
		scanner = DefaultDiskScanRuns
	}
	return &RunIndexStore{
		opts:    opts,
		scanner: scanner,
	}, nil
}

// Upsert inserts or replaces a row keyed by (Namespace, RunID, Epoch), gzipping
// payload (when non-nil) after scrubbing non-finite floats, and persists the
// index atomically.
func (s *RunIndexStore) Upsert(ctx context.Context, row RunIndexRow, payload any) error {
	if s == nil {
		return errors.New("perfledger: RunIndexStore.Upsert: nil store")
	}
	if strings.TrimSpace(row.RunID) == "" {
		return errors.New("perfledger: RunIndexStore.Upsert: RunID must be non-empty")
	}
	if err := validateRowFloats(row); err != nil {
		return err
	}
	if payload != nil {
		gz, err := CompressRunPayload(payload)
		if err != nil {
			return err
		}
		row.PayloadGzip = gz
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	rows, err := s.readIndexFile()
	if err != nil {
		// If the index file is missing or corrupt at upsert time, recover any
		// existing runs from disk first so valid entries are not silently lost.
		if s.opts.ResultsDir != "" {
			if scanned, scanErr := s.scanner(ctx, s.opts.ResultsDir); scanErr == nil {
				rows = scanned
			}
		}
	}

	rows = upsertRowSlice(rows, row)
	if s.opts.IndexPath == "" {
		return nil
	}
	return s.writeIndexFile(rows)
}

// List queries indexed rows using filter. When the index file is absent or
// corrupt, List falls back to scanning ResultsDir so queries never fail on a
// broken index.
func (s *RunIndexStore) List(ctx context.Context, filter RunIndexFilter) ([]RunIndexRow, IndexReadSource, error) {
	if s == nil {
		return nil, "", errors.New("perfledger: RunIndexStore.List: nil store")
	}
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	source := IndexSourceFile
	rows, err := s.readIndexFile()
	if err != nil {
		source = IndexSourceDiskFallback
		rows, err = s.scanner(ctx, s.opts.ResultsDir)
		if err != nil {
			return nil, source, fmt.Errorf("perfledger: RunIndexStore disk fallback failed: %w", err)
		}
	}

	filtered := filterAndSortRows(rows, filter)
	return filtered, source, nil
}

// RebuildFromDisk scans ResultsDir and writes a fresh compressed index file.
func (s *RunIndexStore) RebuildFromDisk(ctx context.Context) (int, error) {
	if s == nil {
		return 0, errors.New("perfledger: RunIndexStore.RebuildFromDisk: nil store")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	rows, err := s.scanner(ctx, s.opts.ResultsDir)
	if err != nil {
		return 0, err
	}
	if s.opts.IndexPath != "" {
		if err := s.writeIndexFile(rows); err != nil {
			return 0, err
		}
	}
	return len(rows), nil
}

// CompressRunPayload scrubs non-finite floats and gzip-compresses payload JSON.
func CompressRunPayload(payload any) ([]byte, error) {
	raw, err := MarshalScrubbedJSON(payload)
	if err != nil {
		return nil, fmt.Errorf("perfledger: compress run payload marshal: %w", err)
	}
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := gz.Write(raw); err != nil {
		_ = gz.Close()
		return nil, fmt.Errorf("perfledger: compress run payload gzip: %w", err)
	}
	if err := gz.Close(); err != nil {
		return nil, fmt.Errorf("perfledger: compress run payload close: %w", err)
	}
	return buf.Bytes(), nil
}

// DecompressRunPayload unzips row.PayloadGzip and unmarshals it into dst.
func DecompressRunPayload(row RunIndexRow, dst any) error {
	if len(row.PayloadGzip) == 0 {
		return errors.New("perfledger: DecompressRunPayload: empty payload")
	}
	gr, err := gzip.NewReader(bytes.NewReader(row.PayloadGzip))
	if err != nil {
		return fmt.Errorf("perfledger: DecompressRunPayload gunzip: %w", err)
	}
	defer gr.Close()
	raw, err := io.ReadAll(gr)
	if err != nil {
		return fmt.Errorf("perfledger: DecompressRunPayload read: %w", err)
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("perfledger: DecompressRunPayload json: %w", err)
	}
	return nil
}

func (s *RunIndexStore) readIndexFile() ([]RunIndexRow, error) {
	if s.opts.IndexPath == "" {
		return nil, fs.ErrNotExist
	}
	data, err := os.ReadFile(s.opts.IndexPath)
	if err != nil {
		return nil, err
	}
	gr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("perfledger: corrupt index gzip header: %w", err)
	}
	defer gr.Close()
	raw, err := io.ReadAll(gr)
	if err != nil {
		return nil, fmt.Errorf("perfledger: corrupt index gzip stream: %w", err)
	}
	var file runIndexFilePayload
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("perfledger: corrupt index json: %w", err)
	}
	if file.SchemaVersion != RunIndexSchemaVersion {
		return nil, fmt.Errorf("perfledger: unsupported index schema version %d", file.SchemaVersion)
	}
	return file.Rows, nil
}

func (s *RunIndexStore) writeIndexFile(rows []RunIndexRow) error {
	dir := filepath.Dir(s.opts.IndexPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	payload := runIndexFilePayload{
		SchemaVersion: RunIndexSchemaVersion,
		Rows:          rows,
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := gz.Write(raw); err != nil {
		_ = gz.Close()
		return err
	}
	if err := gz.Close(); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".runs_index_*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.Remove(tmpPath)
		}
	}()
	if _, err := tmp.Write(buf.Bytes()); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, s.opts.IndexPath); err != nil {
		return err
	}
	cleanup = false
	return nil
}

// DefaultDiskScanRuns walks resultsDir to recover RunIndexRows when the index
// file is missing or unreadable. Directories with an active processing marker
// and no ready marker are skipped fail-closed; corrupt JSON files are skipped
// without failing the scan.
func DefaultDiskScanRuns(ctx context.Context, resultsDir string) ([]RunIndexRow, error) {
	if resultsDir == "" {
		return nil, nil
	}
	info, err := os.Stat(resultsDir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("perfledger: DefaultDiskScanRuns: %q is not a directory", resultsDir)
	}

	var rows []RunIndexRow
	walkErr := filepath.WalkDir(resultsDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // tolerate unreadable subtree without aborting scan
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if d.IsDir() {
			if path != resultsDir && ResultsProcessing(path) && !ResultsReady(path) {
				return filepath.SkipDir
			}
			return nil
		}
		name := d.Name()
		if strings.HasPrefix(name, ".") || !strings.HasSuffix(name, ".json") {
			return nil
		}
		row, ok := parseDiskRunJSON(path)
		if ok {
			rows = upsertRowSlice(rows, row)
		}
		return nil
	})
	if walkErr != nil {
		return nil, walkErr
	}
	return rows, nil
}

func parseDiskRunJSON(path string) (RunIndexRow, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return RunIndexRow{}, false
	}
	var raw map[string]any
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil || len(raw) == 0 {
		return RunIndexRow{}, false
	}

	// Only treat files that declare a run_id (or contain summary metrics) as run rows.
	runID, _ := raw["run_id"].(string)
	if runID == "" {
		return RunIndexRow{}, false
	}

	row := RunIndexRow{
		Namespace:     stringField(raw, "namespace"),
		RunID:         runID,
		Epoch:         int64Field(raw, "epoch"),
		Model:         stringField(raw, "model"),
		Status:        stringField(raw, "status"),
		CreatedUnixMS: int64Field(raw, "created_unix_ms"),
		TotalRequests: int64Field(raw, "total_requests"),
		ErrorRate:     floatField(raw, "error_rate"),
		ThroughputTPS: floatField(raw, "throughput_tps"),
		TTFTP50MS:     floatField(raw, "ttft_p50_ms"),
		TTFTP99MS:     floatField(raw, "ttft_p99_ms"),
		E2EP95MS:      floatField(raw, "e2e_p95_ms"),
		ArtifactDir:   filepath.Dir(path),
	}
	if row.ArtifactDir == "" {
		row.ArtifactDir = stringField(raw, "artifact_dir")
	}

	if len(row.PayloadGzip) == 0 {
		var payloadTarget any = raw
		if nested, ok := raw["payload"]; ok && nested != nil {
			payloadTarget = nested
		}
		if gz, err := CompressRunPayload(payloadTarget); err == nil {
			row.PayloadGzip = gz
		}
	}
	if err := validateRowFloats(row); err != nil {
		return RunIndexRow{}, false
	}
	return row, true
}

func stringField(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
}

func int64Field(m map[string]any, key string) int64 {
	v, ok := m[key]
	if !ok || v == nil {
		return 0
	}
	if f, ok := summaryFinite(v); ok {
		return summaryCount(f)
	}
	return 0
}

func floatField(m map[string]any, key string) float64 {
	v, ok := m[key]
	if !ok || v == nil {
		return 0
	}
	if f, ok := summaryFinite(v); ok {
		return f
	}
	return 0
}

func validateRowFloats(row RunIndexRow) error {
	for _, f := range []struct {
		name string
		val  float64
	}{
		{"error_rate", row.ErrorRate},
		{"throughput_tps", row.ThroughputTPS},
		{"ttft_p50_ms", row.TTFTP50MS},
		{"ttft_p99_ms", row.TTFTP99MS},
		{"e2e_p95_ms", row.E2EP95MS},
	} {
		if math.IsNaN(f.val) || math.IsInf(f.val, 0) {
			return fmt.Errorf("perfledger: RunIndexRow.%s must be finite, got %v", f.name, f.val)
		}
	}
	return nil
}

func upsertRowSlice(rows []RunIndexRow, row RunIndexRow) []RunIndexRow {
	for i := range rows {
		if rows[i].Namespace == row.Namespace && rows[i].RunID == row.RunID && rows[i].Epoch == row.Epoch {
			rows[i] = row
			return rows
		}
	}
	return append(rows, row)
}

func filterAndSortRows(rows []RunIndexRow, filter RunIndexFilter) []RunIndexRow {
	var out []RunIndexRow
	for _, r := range rows {
		if filter.Namespace != "" && r.Namespace != filter.Namespace {
			continue
		}
		if filter.Model != "" && r.Model != filter.Model {
			continue
		}
		if filter.Status != "" && r.Status != filter.Status {
			continue
		}
		// Copy PayloadGzip to prevent callers mutating backing memory.
		if len(r.PayloadGzip) > 0 {
			r.PayloadGzip = append([]byte(nil), r.PayloadGzip...)
		}
		out = append(out, r)
	}

	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		cmp := compareRows(a, b, filter.SortBy)
		if cmp == 0 {
			if a.Namespace != b.Namespace {
				cmp = strings.Compare(a.Namespace, b.Namespace)
			} else if a.RunID != b.RunID {
				cmp = strings.Compare(a.RunID, b.RunID)
			} else if a.Epoch < b.Epoch {
				cmp = -1
			} else if a.Epoch > b.Epoch {
				cmp = 1
			}
		}
		if filter.Descending {
			return cmp > 0
		}
		return cmp < 0
	})

	if filter.Limit > 0 && len(out) > filter.Limit {
		out = out[:filter.Limit]
	}
	return out
}

func compareRows(a, b RunIndexRow, sortBy string) int {
	switch sortBy {
	case "created_unix_ms":
		return cmpInt64(a.CreatedUnixMS, b.CreatedUnixMS)
	case "throughput_tps":
		return cmpFloat64(a.ThroughputTPS, b.ThroughputTPS)
	case "ttft_p50_ms":
		return cmpFloat64(a.TTFTP50MS, b.TTFTP50MS)
	case "ttft_p99_ms":
		return cmpFloat64(a.TTFTP99MS, b.TTFTP99MS)
	case "e2e_p95_ms":
		return cmpFloat64(a.E2EP95MS, b.E2EP95MS)
	case "error_rate":
		return cmpFloat64(a.ErrorRate, b.ErrorRate)
	default:
		return cmpInt64(a.Epoch, b.Epoch)
	}
}

func cmpInt64(a, b int64) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

func cmpFloat64(a, b float64) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}
