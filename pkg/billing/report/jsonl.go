package report

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// ErrStoreClosed is returned when a Store is used after Close, or through
// a nil receiver: the store fails closed rather than silently re-opening.
var ErrStoreClosed = errors.New("report: store is closed")

// ErrMalformedLine marks a JSONL line that is not exactly one decodable
// v1 record. It wraps billing.ErrMalformedPayload, so errors.Is against
// either sentinel identifies the fault.
var ErrMalformedLine = errors.New("report: malformed JSONL line")

// dayFilePrefix and dayFileSuffix frame the file name for one UTC day; the
// day itself ("2006-01-02") sits between them.
const (
	dayFilePrefix = "usage-"
	dayFileSuffix = ".jsonl"
	dayFormat     = "2006-01-02"
)

// newline terminates one JSONL line.
var newline = []byte{'\n'}

// DayFile renders the file name for one UTC day string ("2006-01-02").
func DayFile(day string) string {
	return dayFilePrefix + day + dayFileSuffix
}

// Store persists usage records as daily JSONL files under a directory:
// one file per UTC day, one strictly-encoded v1 record per line. It is the
// append-safe, replayable arm of the reporting plumbing: a writer appends
// settled records as they happen, and a reader — possibly after a restart,
// possibly in another process — replays them back into UsageRecord values.
//
// Replay is faithful: the reader decodes every line and applies no dedup,
// no pricing, and no policy. Dedup-by-event-id, if wanted, is the
// consumer's job (see billing.Seen).
//
// One writer per directory at a time is the supported configuration; each
// line is a single write to an O_APPEND handle, which keeps lines whole
// under that configuration. Readers may run concurrently with the writer.
type Store struct {
	dir    string
	mu     sync.Mutex
	files  map[string]*os.File
	closed bool
}

// OpenStore opens (creating if needed) the directory a Store persists
// into.
func OpenStore(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &Store{dir: dir, files: make(map[string]*os.File)}, nil
}

// Dir reports the store's directory. A nil receiver reports "".
func (s *Store) Dir() string {
	if s == nil {
		return ""
	}
	return s.dir
}

// Append writes one record to the JSONL file for its UTC day, creating the
// file if needed. The record must pass billing.Validate before anything is
// written. Appending after Close (or through a nil receiver) fails with
// ErrStoreClosed.
func (s *Store) Append(rec billing.UsageRecord) error {
	if s == nil {
		return ErrStoreClosed
	}
	if err := rec.Validate(); err != nil {
		return err
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	day := rec.RequestedAt.UTC().Format(dayFormat)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrStoreClosed
	}
	f, ok := s.files[day]
	if !ok {
		path := filepath.Join(s.dir, DayFile(day))
		f, err = os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return err
		}
		s.files[day] = f
	}
	if _, err := f.Write(line); err != nil {
		return err
	}
	return nil
}

// ReadOptions bounds a replay by UTC day; zero times select all days.
// Bounds are inclusive.
type ReadOptions struct {
	Since time.Time
	Until time.Time
}

// Read replays the records persisted for the option window, in
// day-then-line order, decoding each line strictly (unknown fields are
// refused, exactly as on the wire). Replay stops at the first malformed
// line, which is returned wrapped in ErrMalformedLine with its file and
// line number; records decoded before it are not returned. Reading works
// after Close and through concurrent writers, because it opens its own
// handles. A nil receiver fails with ErrStoreClosed.
func (s *Store) Read(opts ReadOptions) ([]billing.UsageRecord, error) {
	if s == nil {
		return nil, ErrStoreClosed
	}
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}

	var sinceDay, untilDay string
	if !opts.Since.IsZero() {
		sinceDay = opts.Since.UTC().Format(dayFormat)
	}
	if !opts.Until.IsZero() {
		untilDay = opts.Until.UTC().Format(dayFormat)
	}

	var days []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, dayFilePrefix) || !strings.HasSuffix(name, dayFileSuffix) {
			continue
		}
		day := strings.TrimSuffix(strings.TrimPrefix(name, dayFilePrefix), dayFileSuffix)
		if _, perr := time.Parse(dayFormat, day); perr != nil {
			continue // not a day file this store owns
		}
		if sinceDay != "" && day < sinceDay {
			continue
		}
		if untilDay != "" && day > untilDay {
			continue
		}
		days = append(days, day)
	}
	sort.Strings(days)

	out := []billing.UsageRecord{}
	for _, day := range days {
		data, err := os.ReadFile(filepath.Join(s.dir, DayFile(day)))
		if err != nil {
			return nil, err
		}
		lines := bytes.Split(bytes.TrimRight(data, "\n"), newline)
		for i, line := range lines {
			if len(line) == 0 {
				continue
			}
			rec, err := billing.DecodeUsageRecord(line)
			if err != nil {
				return nil, fmt.Errorf("%w: %s line %d: %w", ErrMalformedLine, DayFile(day), i+1, err)
			}
			out = append(out, rec)
		}
	}
	return out, nil
}

// Close releases the store's open day files. Subsequent Appends fail with
// ErrStoreClosed; Reads still work, because they open their own handles.
// Closing an already-closed (or nil) store is a no-op.
func (s *Store) Close() error {
	if s == nil {
		return ErrStoreClosed
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	var first error
	for day, f := range s.files {
		if err := f.Close(); err != nil && first == nil {
			first = err
		}
		delete(s.files, day)
	}
	return first
}
