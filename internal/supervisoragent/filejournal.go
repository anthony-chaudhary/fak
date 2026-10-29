package supervisoragent

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// FileJournal is the durable backend behind the same Journal interface
// MemoryJournal owns the golden behavior for: one compact JSON line per
// DecisionRecord in an append-only file. The sequence survives restarts —
// Append derives the next Seq from the existing records, so a restart resumes
// the series instead of restarting it (design-note D1: a restart is a replay,
// not a re-decision). A malformed or torn line is a hard error: the journal
// never guesses over a record it cannot read.
//
// Single writer per file (the doctrine's no-unshared-writers rule): two
// processes appending to one file can race the derived sequence. The seat is
// one process; the census reader only Replays.

// FileJournalMaxLine caps one journal line; a DecisionRecord is small and
// fixed-shape, so a line past the cap is corrupt by construction.
const FileJournalMaxLine = 64 * 1024

type FileJournal struct {
	path string
	mu   sync.Mutex
}

// NewFileJournal opens (creating when missing) the append-only decision
// journal at path, creating parent directories as needed.
func NewFileJournal(path string) (*FileJournal, error) {
	if path == "" {
		return nil, errors.New("supervisoragent: file journal requires a non-empty path")
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("supervisoragent: open journal dir %s: %w", dir, err)
		}
	}
	return &FileJournal{path: path}, nil
}

// Path reports the journal file's location.
func (j *FileJournal) Path() string { return j.path }

// readRecords loads every record in file order. A missing file is the empty
// journal (not an error); any malformed line is.
func (j *FileJournal) readRecords() ([]DecisionRecord, error) {
	f, err := os.Open(j.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []DecisionRecord{}, nil
		}
		return nil, fmt.Errorf("supervisoragent: open journal %s: %w", j.path, err)
	}
	defer f.Close()

	var out []DecisionRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), FileJournalMaxLine)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r DecisionRecord
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, fmt.Errorf("supervisoragent: journal %s line %d is not a valid DecisionRecord: %w", j.path, lineNo, err)
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("supervisoragent: read journal %s: %w", j.path, err)
	}
	return out, nil
}

// Append stamps r with the next monotonic Seq (max existing + 1, so the first
// record on a fresh file is Seq 1 and a restart continues the series) and
// writes exactly one fsync'd JSON line. The single write + sync pair is what
// makes a crash leave a complete prefix Replay can resume from.
func (j *FileJournal) Append(r DecisionRecord) error {
	j.mu.Lock()
	defer j.mu.Unlock()

	recs, err := j.readRecords()
	if err != nil {
		return err
	}
	var maxSeq uint64
	for _, rec := range recs {
		if rec.Seq > maxSeq {
			maxSeq = rec.Seq
		}
	}
	r.Seq = maxSeq + 1

	b, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("supervisoragent: encode record: %w", err)
	}
	b = append(b, '\n')

	f, err := os.OpenFile(j.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("supervisoragent: open journal %s: %w", j.path, err)
	}
	defer f.Close()
	if _, err := f.Write(b); err != nil {
		return fmt.Errorf("supervisoragent: write journal %s: %w", j.path, err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("supervisoragent: sync journal %s: %w", j.path, err)
	}
	return nil
}

// Replay returns every record with Seq >= fromSeq in sequence order, as
// copies. A cursor past the tail yields an empty (non-nil) slice — the
// restart-resume shape MemoryJournal pins.
func (j *FileJournal) Replay(fromSeq uint64) ([]DecisionRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()

	recs, err := j.readRecords()
	if err != nil {
		return nil, err
	}
	out := make([]DecisionRecord, 0, len(recs))
	for _, r := range recs {
		if r.Seq >= fromSeq {
			out = append(out, r)
		}
	}
	return out, nil
}

// Last returns the most recently appended record and true, or the zero record
// and false for an empty (or missing) journal.
func (j *FileJournal) Last() (DecisionRecord, bool) {
	recs, err := j.Replay(0)
	if err != nil {
		return DecisionRecord{}, false
	}
	if len(recs) == 0 {
		return DecisionRecord{}, false
	}
	return recs[len(recs)-1], true
}
