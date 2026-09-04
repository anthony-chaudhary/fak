package dispatchcache

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Invariant: backlog snapshot reconciliation is fail-closed and deterministic; mismatched keys or corrupt state abort without modifying cache.

// BacklogSchema defines the schema identifier for full backlog snapshots.
const BacklogSchema = "fak.dispatch-backlog.v1"

// BacklogWatermarkSchema tags the sidecar that carries the watermark of a tick whose delta
// changed nothing. It is deliberately a separate file from the snapshot: the snapshot's issue
// array is multi-megabyte, the watermark is 28 bytes, and a quiet tick only moves the latter.
const BacklogWatermarkSchema = "fak.dispatch-backlog-watermark.v1"

// BacklogIssue represents a single issue number and its raw JSON content payload.
type BacklogIssue struct {
	Number int             `json:"number"`
	Data   json.RawMessage `json:"data"`
}

// BacklogSnapshot represents an on-disk snapshot of the backlog at a specific watermark time.
type BacklogSnapshot struct {
	Schema    string         `json:"schema"`
	Key       string         `json:"key"`
	Watermark time.Time      `json:"watermark"`
	Issues    []BacklogIssue `json:"issues"`
}

// backlogWatermark is the on-disk shape of the sidecar. It repeats the key so a snapshot written
// under a different key can never inherit an unrelated tick's watermark.
type backlogWatermark struct {
	Schema    string    `json:"schema"`
	Key       string    `json:"key"`
	Watermark time.Time `json:"watermark"`
}

// MergeBacklog merges baseline issues with updated issues and purges closed issue numbers.
func MergeBacklog(base []BacklogIssue, changed []BacklogIssue, closed []int) []BacklogIssue {
	byNumber := make(map[int]json.RawMessage, len(base)+len(changed))
	for _, row := range base {
		if row.Number > 0 {
			byNumber[row.Number] = append(json.RawMessage(nil), row.Data...)
		}
	}
	for _, number := range closed {
		delete(byNumber, number)
	}
	for _, row := range changed {
		if row.Number > 0 {
			byNumber[row.Number] = append(json.RawMessage(nil), row.Data...)
		}
	}
	out := make([]BacklogIssue, 0, len(byNumber))
	for number, data := range byNumber {
		out = append(out, BacklogIssue{Number: number, Data: data})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Number < out[j].Number })
	return out
}

// SyncBacklog merges a delta into the cached snapshot and persists the result, returning the
// merged rows. A tick that learns nothing -- no changed row, no closure, so the merge is
// byte-identical to the base -- writes only the watermark sidecar, leaving the snapshot's bytes
// and mtime untouched; a tick that learns anything rewrites the whole snapshot exactly as before.
// Callers must not write the snapshot themselves on the delta path.
func SyncBacklog(path, key string, watermark time.Time, base, changed []BacklogIssue, closed []int) ([]BacklogIssue, error) {
	merged := MergeBacklog(base, changed, closed)
	if sameBacklog(base, merged) {
		return merged, WriteBacklogWatermark(path, key, watermark)
	}
	return merged, WriteBacklog(path, key, watermark, merged)
}

// sameBacklog reports whether two merged row sets carry the same issues with the same bytes.
// Comparing the merge result (rather than just len(changed)==0 && len(closed)==0) also catches a
// delta that re-sends an unchanged issue or closes one the snapshot never held.
func sameBacklog(a, b []BacklogIssue) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Number != b[i].Number || !bytes.Equal(a[i].Data, b[i].Data) {
			return false
		}
	}
	return true
}

// WriteBacklog atomically serializes and writes a full backlog snapshot to disk.
func WriteBacklog(path, key string, watermark time.Time, issues []BacklogIssue) error {
	if path == "" || key == "" {
		return errors.New("dispatchcache: backlog path and key are required")
	}
	b, err := json.Marshal(BacklogSnapshot{Schema: BacklogSchema, Key: key, Watermark: watermark.UTC(), Issues: issues})
	if err != nil {
		return err
	}
	if err = writeAtomic(path, b); err != nil {
		return err
	}
	// The snapshot now carries the newest watermark, so any sidecar left by earlier quiet ticks
	// is redundant. Removal is best-effort: ReadBacklog only honours a sidecar that is strictly
	// ahead of the snapshot, so a survivor is inert.
	_ = os.Remove(backlogWatermarkPath(path))
	return nil
}

// WriteBacklogWatermark advances the watermark without touching the snapshot. The next delta
// search still starts where this tick finished, so a long run of quiet ticks does not widen the
// window back to the last time an issue actually changed.
func WriteBacklogWatermark(path, key string, watermark time.Time) error {
	if path == "" || key == "" {
		return errors.New("dispatchcache: backlog path and key are required")
	}
	b, err := json.Marshal(backlogWatermark{Schema: BacklogWatermarkSchema, Key: key, Watermark: watermark.UTC()})
	if err != nil {
		return err
	}
	return writeAtomic(backlogWatermarkPath(path), b)
}

func backlogWatermarkPath(path string) string {
	return strings.TrimSuffix(path, filepath.Ext(path)) + ".watermark.json"
}

func writeAtomic(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".backlog-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(append(b, '\n')); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}

// ReadBacklog loads and unmarshals a backlog snapshot matching the specified key.
func ReadBacklog(path, key string) (BacklogSnapshot, bool) {
	var s BacklogSnapshot
	b, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(b, &s) != nil || s.Schema != BacklogSchema || s.Key != key || s.Watermark.IsZero() {
		return BacklogSnapshot{}, false
	}
	if w, ok := readBacklogWatermark(path, key); ok && w.After(s.Watermark) {
		s.Watermark = w
	}
	return s, true
}

// readBacklogWatermark returns the sidecar watermark for this key. A missing, corrupt, or
// foreign-key sidecar is simply ignored -- the snapshot's own watermark stays authoritative.
func readBacklogWatermark(path, key string) (time.Time, bool) {
	var w backlogWatermark
	b, err := os.ReadFile(backlogWatermarkPath(path))
	if err != nil || json.Unmarshal(b, &w) != nil || w.Schema != BacklogWatermarkSchema || w.Key != key || w.Watermark.IsZero() {
		return time.Time{}, false
	}
	return w.Watermark, true
}
