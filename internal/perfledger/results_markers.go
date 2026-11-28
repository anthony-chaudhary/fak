// results_markers.go ports aiperf's fail-closed result-directory markers to Go.
//
// NOTICE: ported from ai-dynamo/aiperf, src/aiperf/common/results_markers.py,
// readable only once its ready marker has been installed, atomically and
// last; an export brackets itself with a processing marker so a crash leaves
// the directory reading as not-ready instead of partially ready. Checkpoint
// artifacts are written incrementally and bypass the gate.
package perfledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Marker and directory names shared by every reader and writer of a results
// directory.
const (
	ReadyMarkerName       = ".aiperf_results_ready.json"
	ProcessingMarkerName  = ".aiperf_results_processing.json"
	ResultCheckpointsName = "checkpoints"
)

// rmarkRunKeyRE accepts bare epoch seconds (9 or 10 digits) optionally
// followed by exactly one six-digit suffix; 11-14 digit keys would decode to
// a near-1970 instant, so they are rejected rather than misread.
var rmarkRunKeyRE = regexp.MustCompile(`\A\d{9,10}(\d{6})?\z`)

// ValidResultRunKey reports whether key is a well-formed run-directory key.
func ValidResultRunKey(key string) bool { return rmarkRunKeyRE.MatchString(key) }

// ReadyMarker is the payload of the ready marker. Field order matches the
// upstream JSON so the encoded bytes are interchangeable; the terminal fields
// are pointers because an empty string is recorded while nil is omitted.
type ReadyMarker struct {
	Ready           bool     `json:"ready"`
	WasCancelled    bool     `json:"was_cancelled"`
	Partial         bool     `json:"partial"`
	TerminalPhase   *string  `json:"terminal_phase,omitempty"`
	TerminalError   *string  `json:"terminal_error,omitempty"`
	FailedExporters []string `json:"failed_exporters,omitempty"`
}

// ReadyMarkerPath returns the ready-marker path under dir.
func ReadyMarkerPath(dir string) string { return filepath.Join(dir, ReadyMarkerName) }

// ProcessingMarkerPath returns the processing-marker path under dir.
func ProcessingMarkerPath(dir string) string { return filepath.Join(dir, ProcessingMarkerName) }

// ResultsReady reports whether dir's ready marker is present as a regular
// file. Anything else, including a stat error, reads as not ready.
func ResultsReady(dir string) bool { return rmarkIsFile(ReadyMarkerPath(dir)) }

// ResultsProcessing reports whether an export into dir is in flight.
func ResultsProcessing(dir string) bool { return rmarkIsFile(ProcessingMarkerPath(dir)) }

// WriteProcessingMarker begins an export transaction: any earlier ready
// marker is removed before the processing marker is installed, so a crash at
// any later point leaves top-level results hidden.
func WriteProcessingMarker(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	switch err := os.Remove(ReadyMarkerPath(dir)); {
	case err == nil:
		rmarkSyncDir(dir)
	case !errors.Is(err, fs.ErrNotExist):
		return fmt.Errorf("perfledger: remove stale ready marker: %w", err)
	}
	return rmarkAtomicWrite(ProcessingMarkerPath(dir), rmarkProcessingPayload)
}

// ClearProcessingMarker removes the processing marker; absence is not an error.
func ClearProcessingMarker(dir string) error {
	err := os.Remove(ProcessingMarkerPath(dir))
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err == nil {
		rmarkSyncDir(dir)
	}
	return err
}

// WriteReadyMarker publishes readiness once every artifact is stable. m.Ready
// is forced true. On any failure readiness is rolled back and the processing
// marker restored, so the directory stays closed.
func WriteReadyMarker(dir string, m ReadyMarker) error {
	m.Ready = true
	if len(m.FailedExporters) == 0 {
		m.FailedExporters = nil
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(m); err != nil {
		return err
	}
	payload := rmarkUnescapeLineSeps(bytes.TrimSuffix(buf.Bytes(), []byte{'\n'}))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := rmarkAtomicWrite(ReadyMarkerPath(dir), payload); err != nil {
		rmarkRollback(dir)
		return err
	}
	if err := ClearProcessingMarker(dir); err != nil {
		rmarkRollback(dir)
		return fmt.Errorf("perfledger: clear processing marker: %w", err)
	}
	return nil
}

// rmarkUnescapeLineSeps undoes encoding/json's U+2028/U+2029 escapes, which
// orjson leaves raw, so marker bytes stay identical to upstream's.
func rmarkUnescapeLineSeps(b []byte) []byte {
	out := make([]byte, 0, len(b))
	for i := 0; i < len(b); i++ {
		if b[i] != '\\' || i+1 >= len(b) {
			out = append(out, b[i])
			continue
		}
		if b[i+1] == 'u' && i+6 <= len(b) {
			switch string(b[i+2 : i+6]) {
			case "2028":
				out = append(out, " "...)
				i += 5
				continue
			case "2029":
				out = append(out, " "...)
				i += 5
				continue
			}
		}
		out = append(out, b[i], b[i+1])
		i++
	}
	return out
}

// ReadReadyMarkerInfo returns the partial flag and failed exporters recorded
// on the ready marker. A missing or malformed marker reports a clean result:
// the marker's existence, not its optional fields, gates readiness.
func ReadReadyMarkerInfo(dir string) (partial bool, failedExporters []string) {
	if !ResultsReady(dir) {
		return false, nil
	}
	raw, err := os.ReadFile(ReadyMarkerPath(dir))
	if err != nil {
		return false, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var top any
	if dec.Decode(&top) != nil || dec.Decode(new(any)) != io.EOF {
		return false, nil
	}
	doc, ok := top.(map[string]any)
	if !ok {
		return false, nil
	}
	partial = rmarkTruthy(doc["partial"])
	if !rmarkTruthy(doc["failed_exporters"]) {
		return partial, nil
	}
	items, ok := doc["failed_exporters"].([]any)
	if !ok {
		return partial, nil
	}
	for _, it := range items {
		failedExporters = append(failedExporters, rmarkPyStr(it))
	}
	return partial, failedExporters
}

// rmarkTruthy mirrors Python truthiness over decoded JSON, which is how the
// upstream reader interprets the optional fields.
func rmarkTruthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case json.Number:
		f, err := strconv.ParseFloat(x.String(), 64)
		return err != nil || f != 0
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	return true
}

// rmarkPyStr renders a decoded JSON value the way Python's str() does for
// scalars, so non-string exporter names match upstream byte for byte.
func rmarkPyStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case json.Number:
		lit := x.String()
		if !strings.ContainsAny(lit, ".eE") {
			if strings.Trim(lit, "-0") == "" {
				return "0"
			}
			return lit
		}
		f, err := strconv.ParseFloat(lit, 64)
		if err != nil && !math.IsInf(f, 0) {
			return lit
		}
		return rmarkPyFloat(f)
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// rmarkPyFloat matches Python float repr: shortest round-trip digits,
// positional within [1e-4, 1e16) with a mandatory ".0", exponent otherwise.
func rmarkPyFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64)
	exp, _ := strconv.Atoi(e[strings.IndexByte(e, 'e')+1:])
	if f == 0 || (exp >= -4 && exp < 16) {
		s := strconv.FormatFloat(f, 'f', -1, 64)
		if !strings.Contains(s, ".") {
			s += ".0"
		}
		return s
	}
	return e
}

// ResultFileUnavailableError explains why a result file cannot be served.
// Status is HTTP-shaped (400 invalid name, 404 not ready) so transport
// adapters can forward it verbatim.
type ResultFileUnavailableError struct {
	Status int
	Detail string
}

func (e *ResultFileUnavailableError) Error() string { return e.Detail }

// ResultsNotReadyDetail is the detail served while dir is not ready.
func ResultsNotReadyDetail(dir string) string {
	processing := ""
	if ResultsProcessing(dir) {
		processing = " export still processing;"
	}
	return fmt.Sprintf("Results not ready for %s;%s marker file %s not present — retry after completion",
		rmarkDirName(dir), processing, ReadyMarkerName)
}

// ResolveResultFile resolves name under dir for download. Traversal and
// marker names are refused with 400; top-level artifacts are refused with
// 404 until the ready marker commits; checkpoint artifacts are always served.
func ResolveResultFile(dir, name string) (string, error) {
	rel, real, ok := rmarkSafeRel(dir, name)
	if !ok || rmarkReserved(path.Base(rel)) || rmarkReserved(path.Base(real)) {
		return "", &ResultFileUnavailableError{
			Status: 400,
			Detail: fmt.Sprintf("Invalid filename %q: path traversal or reserved marker name", name),
		}
	}
	// The gate judges the symlink-resolved target, so a checkpoint link cannot
	// expose a top-level artifact before readiness.
	if !ResultsReady(dir) && !rmarkUnderCheckpoints(real) {
		return "", &ResultFileUnavailableError{Status: 404, Detail: ResultsNotReadyDetail(dir)}
	}
	return filepath.Join(dir, filepath.FromSlash(rel)), nil
}

// ResultArtifact is one downloadable file: its slash-separated name relative
// to the results directory and its size in bytes.
type ResultArtifact struct {
	Name string
	Size int64
}

// CollectReadyArtifacts lists downloadable artifacts sorted by name.
// Top-level artifacts appear only once dir is ready; checkpoint artifacts are
// always listed. Markers are never listed and files that vanish mid-walk are
// skipped. recursive extends the ready listing below the top level.
func CollectReadyArtifacts(dir string, recursive bool) ([]ResultArtifact, error) {
	var out []ResultArtifact
	add := func(p string) {
		info, err := os.Stat(p)
		if err != nil || !info.Mode().IsRegular() {
			return
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return
		}
		out = append(out, ResultArtifact{Name: filepath.ToSlash(rel), Size: info.Size()})
	}
	cpDir := filepath.Join(dir, ResultCheckpointsName)
	if ResultsReady(dir) {
		err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				if p == dir {
					return err
				}
				return nil
			}
			if d.IsDir() {
				if p != dir && (!recursive || p == cpDir) {
					return filepath.SkipDir
				}
				return nil
			}
			if !rmarkReserved(d.Name()) {
				add(p)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	if rmarkIsDir(cpDir) {
		_ = filepath.WalkDir(cpDir, func(p string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				add(p)
			}
			return nil
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

var rmarkProcessingPayload = []byte(`{"processing":true}`)

// rmarkDirName matches pathlib's name, which is empty for "." and roots.
func rmarkDirName(dir string) string {
	b := filepath.Base(dir)
	if b == "." || strings.HasSuffix(b, string(filepath.Separator)) {
		return ""
	}
	return b
}

func rmarkIsFile(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.Mode().IsRegular()
}

func rmarkIsDir(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
}

// rmarkReserved covers the markers and their in-flight temporaries, which
// must never be served even though they live beside result files.
func rmarkReserved(name string) bool {
	if name == ReadyMarkerName || name == ProcessingMarkerName {
		return true
	}
	return strings.HasSuffix(name, ".tmp") &&
		(strings.HasPrefix(name, "."+ReadyMarkerName+".") || strings.HasPrefix(name, "."+ProcessingMarkerName+"."))
}

func rmarkUnderCheckpoints(rel string) bool {
	first, _, _ := strings.Cut(rel, "/")
	return first == ResultCheckpointsName
}

// rmarkSafeRel returns name as a clean slash-separated path local to dir plus
// its symlink-resolved twin, or false. Both POSIX and Windows absolute forms
// are refused on every OS so the policy does not depend on the host. A symlink
// that escapes dir is refused.
func rmarkSafeRel(dir, name string) (rel, real string, ok bool) {
	if name == "" || strings.ContainsRune(name, 0) {
		return "", "", false
	}
	slashed := strings.ReplaceAll(name, `\`, "/")
	if strings.HasPrefix(slashed, "/") || rmarkHasDrive(slashed) {
		return "", "", false
	}
	rel = path.Clean(slashed)
	if rel == "." || rel == ".." || strings.HasPrefix(rel, "../") {
		return "", "", false
	}
	if !filepath.IsLocal(filepath.FromSlash(rel)) {
		return "", "", false
	}
	r, err := filepath.Rel(rmarkResolve(dir), rmarkResolve(filepath.Join(dir, filepath.FromSlash(rel))))
	if err != nil || !filepath.IsLocal(r) {
		return "", "", false
	}
	return rel, filepath.ToSlash(r), true
}

// rmarkResolve resolves symlinks along the longest existing prefix of p, so a
// symlinked parent that escapes is caught even when the leaf does not exist.
func rmarkResolve(p string) string {
	p, _ = filepath.Abs(p)
	if real, err := filepath.EvalSymlinks(p); err == nil {
		return real
	}
	parent := filepath.Dir(p)
	if parent == p {
		return p
	}
	return filepath.Join(rmarkResolve(parent), filepath.Base(p))
}

func rmarkHasDrive(p string) bool {
	if len(p) < 2 || p[1] != ':' {
		return false
	}
	c := p[0] | 0x20
	return c >= 'a' && c <= 'z'
}

// rmarkAtomicWrite installs a fully written, fsynced file with one rename so
// readers never observe a torn marker. On failure neither the temporary nor
// a possibly half-installed target is left behind.
func rmarkAtomicWrite(target string, payload []byte) (err error) {
	dir := filepath.Dir(target)
	tmpName := ""
	defer func() {
		if err != nil {
			if tmpName != "" {
				_ = os.Remove(tmpName)
			}
			_ = os.Remove(target)
			rmarkSyncDir(dir)
		}
	}()
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(target)+".*.tmp")
	if err != nil {
		return err
	}
	tmpName = tmp.Name()
	if _, err = tmp.Write(payload); err != nil {
		_ = tmp.Close()
		return err
	}
	if err = tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	if err = os.Rename(tmpName, target); err != nil {
		return err
	}
	rmarkSyncDir(dir)
	return nil
}

// rmarkRollback restores the fail-closed state after a failed ready commit.
// Errors are dropped: once readiness is gone the directory is already closed.
func rmarkRollback(dir string) {
	_ = os.Remove(ReadyMarkerPath(dir))
	rmarkSyncDir(dir)
	if !ResultsProcessing(dir) {
		_ = rmarkAtomicWrite(ProcessingMarkerPath(dir), rmarkProcessingPayload)
	}
}

// rmarkSyncDir persists directory-entry changes where the platform supports
// fsync on a directory; elsewhere it is a no-op.
func rmarkSyncDir(dir string) {
	f, err := os.Open(dir)
	if err != nil {
		return
	}
	_ = f.Sync()
	_ = f.Close()
}
