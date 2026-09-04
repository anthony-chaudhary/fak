package sessionjournal

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

const (
	// DefaultActiveSegmentName is the filename of the active rolling journal segment.
	DefaultActiveSegmentName = "session-journal.active.jsonl"

	// DefaultCheckpointName is the filename of the compacted snapshot.
	DefaultCheckpointName = "checkpoint.json"

	// CheckpointSchema tags checkpoint JSON snapshots.
	CheckpointSchema = "fak.sessionjournal.checkpoint.v1"

	// DefaultMaxSegmentBytes is the default size threshold (10MB) to rotate the active segment.
	DefaultMaxSegmentBytes int64 = 10 * 1024 * 1024

	// DefaultMaxSegmentEntries is the default row count threshold (10,000 entries) to rotate.
	DefaultMaxSegmentEntries int = 10000
)

// Checkpoint is a point-in-time folded snapshot of historical sessions.
type Checkpoint struct {
	Schema       string    `json:"schema"`
	CompactedSeq uint64    `json:"compacted_seq"`
	UpdatedAt    time.Time `json:"updated_at"`
	Sessions     []Session `json:"sessions"`
}

// SegmentedJournal provides generational rolling segmented journals with background compaction
// and sub-50ms fast recovery from checkpoints.
type SegmentedJournal struct {
	Dir             string
	MaxSizeBytes    int64
	MaxEntries      int
	RetainCompacted bool
	SyncOnAppend    bool

	mu            sync.RWMutex
	activeEntries int
	lastSeq       uint64
}

// SegmentedOption configures a SegmentedJournal.
type SegmentedOption func(*SegmentedJournal)

// WithMaxSizeBytes sets the byte-size limit for rotating the active segment.
func WithMaxSizeBytes(bytes int64) SegmentedOption {
	return func(sj *SegmentedJournal) {
		sj.MaxSizeBytes = bytes
	}
}

// WithMaxEntries sets the entry-count limit for rotating the active segment.
func WithMaxEntries(entries int) SegmentedOption {
	return func(sj *SegmentedJournal) {
		sj.MaxEntries = entries
	}
}

// WithRetainCompacted configures whether compacted sealed segments remain on disk.
func WithRetainCompacted(retain bool) SegmentedOption {
	return func(sj *SegmentedJournal) {
		sj.RetainCompacted = retain
	}
}

// WithSyncOnAppend configures whether fsync is called after each append.
func WithSyncOnAppend(sync bool) SegmentedOption {
	return func(sj *SegmentedJournal) {
		sj.SyncOnAppend = sync
	}
}

// NewSegmentedJournal creates a new SegmentedJournal in dir with applied options.
func NewSegmentedJournal(dir string, opts ...SegmentedOption) *SegmentedJournal {
	if dir == "" {
		dir = filepath.Dir(DefaultPath())
	}
	sj := &SegmentedJournal{
		Dir:          dir,
		MaxSizeBytes: DefaultMaxSegmentBytes,
		MaxEntries:   DefaultMaxSegmentEntries,
		SyncOnAppend: true,
	}
	for _, opt := range opts {
		opt(sj)
	}
	return sj
}

func (sj *SegmentedJournal) initDefaults() {
	if sj.Dir == "" {
		sj.Dir = filepath.Dir(DefaultPath())
	}
	if sj.MaxSizeBytes <= 0 {
		sj.MaxSizeBytes = DefaultMaxSegmentBytes
	}
	if sj.MaxEntries <= 0 {
		sj.MaxEntries = DefaultMaxSegmentEntries
	}
}

// ActivePath returns the full path to the active segment.
func (sj *SegmentedJournal) ActivePath() string {
	sj.initDefaults()
	return filepath.Join(sj.Dir, DefaultActiveSegmentName)
}

// CheckpointPath returns the full path to the checkpoint snapshot.
func (sj *SegmentedJournal) CheckpointPath() string {
	sj.initDefaults()
	return filepath.Join(sj.Dir, DefaultCheckpointName)
}

// AppendEvent writes one event to the active segment, rotating first if limits are reached.
func (sj *SegmentedJournal) AppendEvent(ev Event) error {
	if ev.Schema == "" {
		ev.Schema = Schema
	}
	if match := pathutil.CheckCaptureSource(ev.CWD); match.Refused {
		refusal := Event{
			Schema:       Schema,
			Kind:         KindRefuse,
			ID:           ev.ID,
			TS:           ev.TS,
			Boot:         ev.Boot,
			Reason:       match.Reason,
			SourceDigest: match.SourceDigest,
		}
		if err := sj.appendRaw(refusal); err != nil {
			return fmt.Errorf("%w (refusal audit unavailable: %v)", ErrSourceDenied, err)
		}
		return ErrSourceDenied
	}
	return sj.appendRaw(ev)
}

// AppendEvents appends multiple events sequentially.
func (sj *SegmentedJournal) AppendEvents(events []Event) error {
	for _, ev := range events {
		if err := sj.AppendEvent(ev); err != nil {
			if errors.Is(err, ErrSourceDenied) {
				continue
			}
			return err
		}
	}
	return nil
}

func (sj *SegmentedJournal) appendRaw(ev Event) error {
	sj.initDefaults()
	sj.mu.Lock()
	defer sj.mu.Unlock()

	lockBase := filepath.Join(sj.Dir, "session-journal")
	return withJournalLock(lockBase, func() error {
		if err := os.MkdirAll(sj.Dir, 0o755); err != nil {
			return err
		}

		activePath := filepath.Join(sj.Dir, DefaultActiveSegmentName)
		fi, err := os.Stat(activePath)
		if err == nil {
			if sj.activeEntries == 0 && fi.Size() > 0 {
				sj.activeEntries = countLines(activePath)
			}
			shouldRotate := false
			if sj.MaxEntries > 0 && sj.activeEntries >= sj.MaxEntries {
				shouldRotate = true
			}
			if sj.MaxSizeBytes > 0 && fi.Size() >= sj.MaxSizeBytes {
				shouldRotate = true
			}
			if shouldRotate {
				if err := sj.rotateActiveLocked(); err != nil {
					return err
				}
			}
		}

		f, err := os.OpenFile(activePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer f.Close()

		b, err := json.Marshal(ev)
		if err != nil {
			return err
		}
		if _, err = f.Write(append(b, '\n')); err != nil {
			return err
		}
		if sj.SyncOnAppend {
			if err = f.Sync(); err != nil {
				return err
			}
		}
		sj.activeEntries++
		return nil
	})
}

// RotateActive manually forces rotation of the active segment if non-empty.
func (sj *SegmentedJournal) RotateActive() error {
	sj.initDefaults()
	sj.mu.Lock()
	defer sj.mu.Unlock()

	lockBase := filepath.Join(sj.Dir, "session-journal")
	return withJournalLock(lockBase, func() error {
		return sj.rotateActiveLocked()
	})
}

func (sj *SegmentedJournal) rotateActiveLocked() error {
	activePath := filepath.Join(sj.Dir, DefaultActiveSegmentName)
	fi, err := os.Stat(activePath)
	if os.IsNotExist(err) || (err == nil && fi.Size() == 0) {
		sj.activeEntries = 0
		return nil
	}
	if err != nil {
		return err
	}

	nextSeq := sj.findNextSeqLocked()
	rotatedName := fmt.Sprintf("session-journal.%06d.jsonl", nextSeq)
	rotatedPath := filepath.Join(sj.Dir, rotatedName)

	_ = os.Remove(rotatedPath)
	if err := os.Rename(activePath, rotatedPath); err != nil {
		return fmt.Errorf("sessionjournal: rotate active segment: %w", err)
	}

	sj.activeEntries = 0
	sj.lastSeq = nextSeq
	return nil
}

func (sj *SegmentedJournal) findNextSeqLocked() uint64 {
	entries, err := os.ReadDir(sj.Dir)
	var maxSeq uint64
	if err == nil {
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			if seq, ok := parseSegmentSeq(entry.Name()); ok {
				if seq > maxSeq {
					maxSeq = seq
				}
			}
		}
	}
	cpPath := filepath.Join(sj.Dir, DefaultCheckpointName)
	if cp, err := LoadCheckpoint(cpPath); err == nil && cp != nil {
		if cp.CompactedSeq > maxSeq {
			maxSeq = cp.CompactedSeq
		}
	}
	if sj.lastSeq > maxSeq {
		maxSeq = sj.lastSeq
	}
	return maxSeq + 1
}

// CompactCheckpoint folds all sealed rotated segments into checkpoint.json.
func (sj *SegmentedJournal) CompactCheckpoint() error {
	sj.initDefaults()
	sj.mu.Lock()
	defer sj.mu.Unlock()

	lockBase := filepath.Join(sj.Dir, "session-journal")
	return withJournalLock(lockBase, func() error {
		return sj.compactCheckpointLocked()
	})
}

func (sj *SegmentedJournal) compactCheckpointLocked() error {
	cpPath := filepath.Join(sj.Dir, DefaultCheckpointName)
	var baseSessions []Session
	var compactedSeq uint64

	if cp, err := LoadCheckpoint(cpPath); err == nil && cp != nil {
		baseSessions = cp.Sessions
		compactedSeq = cp.CompactedSeq
	}

	entries, err := os.ReadDir(sj.Dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	type segInfo struct {
		seq  uint64
		path string
	}
	var toCompact []segInfo
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if seq, ok := parseSegmentSeq(entry.Name()); ok {
			if seq > compactedSeq {
				toCompact = append(toCompact, segInfo{
					seq:  seq,
					path: filepath.Join(sj.Dir, entry.Name()),
				})
			}
		}
	}

	if len(toCompact) == 0 {
		return nil
	}

	sort.Slice(toCompact, func(i, j int) bool {
		return toCompact[i].seq < toCompact[j].seq
	})

	var newEvents []Event
	maxSeq := compactedSeq
	for _, seg := range toCompact {
		events := LoadFile(seg.path)
		newEvents = append(newEvents, events...)
		if seg.seq > maxSeq {
			maxSeq = seg.seq
		}
	}

	updatedSessions := FoldEventsOnSessions(baseSessions, newEvents)

	newCp := &Checkpoint{
		Schema:       CheckpointSchema,
		CompactedSeq: maxSeq,
		UpdatedAt:    time.Now().UTC(),
		Sessions:     updatedSessions,
	}

	if err := WriteCheckpoint(cpPath, newCp); err != nil {
		return fmt.Errorf("sessionjournal: write checkpoint: %w", err)
	}

	if !sj.RetainCompacted {
		for _, seg := range toCompact {
			_ = os.Remove(seg.path)
		}
	}

	if maxSeq > sj.lastSeq {
		sj.lastSeq = maxSeq
	}

	return nil
}

// FastFoldRecovery reads checkpoint.json + uncompacted segments + active segment
// without scanning older historical segments, achieving sub-50ms recovery even over
// hundreds of thousands of historical lifecycle events.
func (sj *SegmentedJournal) FastFoldRecovery() ([]Session, error) {
	sj.initDefaults()
	sj.mu.RLock()
	defer sj.mu.RUnlock()

	lockBase := filepath.Join(sj.Dir, "session-journal")
	var sessions []Session
	err := withJournalLock(lockBase, func() error {
		var err error
		sessions, err = sj.fastFoldRecoveryLocked()
		return err
	})
	return sessions, err
}

func (sj *SegmentedJournal) fastFoldRecoveryLocked() ([]Session, error) {
	cpPath := filepath.Join(sj.Dir, DefaultCheckpointName)
	var baseSessions []Session
	var compactedSeq uint64

	if cp, err := LoadCheckpoint(cpPath); err == nil && cp != nil {
		baseSessions = cp.Sessions
		compactedSeq = cp.CompactedSeq
	}

	entries, err := os.ReadDir(sj.Dir)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	type segInfo struct {
		seq  uint64
		path string
	}
	var uncompacted []segInfo
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if seq, ok := parseSegmentSeq(entry.Name()); ok {
			// Skip all historical segments that are already folded in the checkpoint.
			if seq > compactedSeq {
				uncompacted = append(uncompacted, segInfo{
					seq:  seq,
					path: filepath.Join(sj.Dir, entry.Name()),
				})
			}
		}
	}

	sort.Slice(uncompacted, func(i, j int) bool {
		return uncompacted[i].seq < uncompacted[j].seq
	})

	var recentEvents []Event
	for _, seg := range uncompacted {
		events := LoadFile(seg.path)
		recentEvents = append(recentEvents, events...)
	}

	activePath := filepath.Join(sj.Dir, DefaultActiveSegmentName)
	activeEvents := LoadFile(activePath)
	recentEvents = append(recentEvents, activeEvents...)

	return FoldEventsOnSessions(baseSessions, recentEvents), nil
}

// SealedSegments returns the paths of all sealed segment files on disk,
// sorted by sequence number ascending.
func (sj *SegmentedJournal) SealedSegments() ([]string, error) {
	sj.initDefaults()
	entries, err := os.ReadDir(sj.Dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	type seg struct {
		seq  uint64
		path string
	}
	var segs []seg
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if seq, ok := parseSegmentSeq(entry.Name()); ok {
			segs = append(segs, seg{
				seq:  seq,
				path: filepath.Join(sj.Dir, entry.Name()),
			})
		}
	}
	sort.Slice(segs, func(i, j int) bool {
		return segs[i].seq < segs[j].seq
	})
	out := make([]string, len(segs))
	for i, s := range segs {
		out[i] = s.path
	}
	return out, nil
}

// CurrentCheckpoint loads the active checkpoint snapshot if one exists.
func (sj *SegmentedJournal) CurrentCheckpoint() (*Checkpoint, error) {
	sj.initDefaults()
	cpPath := filepath.Join(sj.Dir, DefaultCheckpointName)
	return LoadCheckpoint(cpPath)
}

// StartBackgroundCompactor launches a background goroutine that periodically executes
// CompactCheckpoint at the given interval until the returned stop function is called.
func (sj *SegmentedJournal) StartBackgroundCompactor(interval time.Duration) func() {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				_ = sj.CompactCheckpoint()
			}
		}
	}()
	return func() {
		close(done)
		wg.Wait()
	}
}

// LoadCheckpoint reads and parses a checkpoint file.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cp Checkpoint
	if err := json.Unmarshal(b, &cp); err == nil && cp.Schema != "" {
		return &cp, nil
	}
	// Fallback to bare []Session if checkpoint was written without envelope
	var sessions []Session
	if err := json.Unmarshal(b, &sessions); err == nil {
		return &Checkpoint{
			Schema:   CheckpointSchema,
			Sessions: sessions,
		}, nil
	}
	return nil, fmt.Errorf("sessionjournal: invalid checkpoint data at %s", path)
}

// WriteCheckpoint writes a checkpoint atomically via temporary file and replace.
func WriteCheckpoint(path string, cp *Checkpoint) error {
	if cp.Schema == "" {
		cp.Schema = CheckpointSchema
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".checkpoint-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(cp); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	_ = os.Remove(path)
	return os.Rename(tmpName, path)
}

// FoldEventsOnSessions applies events on top of an existing base set of folded sessions.
// Semantics match FoldEvents: events are sorted by timestamp and applied in order,
// and the resulting sessions are sorted newest-started first.
func FoldEventsOnSessions(base []Session, events []Event) []Session {
	byID := make(map[string]*Session, len(base))
	order := make([]string, 0, len(base))

	for _, s := range base {
		cloned := cloneSession(s)
		byID[cloned.ID] = cloned
		order = append(order, cloned.ID)
	}

	if len(events) == 0 {
		out := make([]Session, 0, len(order))
		for _, id := range order {
			out = append(out, *byID[id])
		}
		sort.SliceStable(out, func(i, j int) bool { return out[i].StartedAt.After(out[j].StartedAt) })
		return out
	}

	sorted := append([]Event(nil), events...)
	sort.SliceStable(sorted, func(i, j int) bool { return eventTime(sorted[i]).Before(eventTime(sorted[j])) })

	for _, ev := range sorted {
		if ev.Kind == KindRefuse {
			continue // an audit refusal is not a session lifecycle transition
		}
		t := eventTime(ev)
		s, ok := byID[ev.ID]
		if !ok {
			s = &Session{ID: ev.ID}
			byID[ev.ID] = s
			order = append(order, ev.ID)
		}
		switch ev.Kind {
		case KindOpen:
			s.StartedAt = t
			s.Closed = false
			s.CloseReason = ""
			applyProvenance(s, ev)
		case KindBeat:
			applyProvenance(s, ev)
		case KindClose:
			applyProvenance(s, ev)
			s.Closed = true
			s.CloseReason = ev.Reason
		}
		if t.After(s.LastSeen) {
			s.LastSeen = t
		}
	}

	out := make([]Session, 0, len(order))
	for _, id := range order {
		out = append(out, *byID[id])
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].StartedAt.After(out[j].StartedAt) })
	return out
}

func cloneSession(s Session) *Session {
	cp := s
	if len(s.Argv) > 0 {
		cp.Argv = append([]string(nil), s.Argv...)
	}
	if s.Drive != nil {
		d := *s.Drive
		cp.Drive = &d
	}
	if s.Registration != nil {
		r := *s.Registration
		if len(s.Registration.Scope) > 0 {
			r.Scope = append([]string(nil), s.Registration.Scope...)
		}
		cp.Registration = &r
	}
	return &cp
}

func parseSegmentSeq(name string) (uint64, bool) {
	if !strings.HasPrefix(name, "session-journal.") || !strings.HasSuffix(name, ".jsonl") {
		return 0, false
	}
	mid := strings.TrimPrefix(name, "session-journal.")
	mid = strings.TrimSuffix(mid, ".jsonl")
	if mid == "active" {
		return 0, false
	}
	seq, err := strconv.ParseUint(mid, 10, 64)
	if err != nil {
		return 0, false
	}
	return seq, true
}

func countLines(path string) int {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	count := 0
	for sc.Scan() {
		if len(bytes.TrimSpace(sc.Bytes())) > 0 {
			count++
		}
	}
	return count
}
