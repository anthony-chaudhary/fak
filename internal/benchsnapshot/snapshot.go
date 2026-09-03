package benchsnapshot

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	// ErrSequenceCollision is returned when an attempt is made to write a snapshot
	// with a sequence number that already exists on disk.
	ErrSequenceCollision = errors.New("benchsnapshot: sequence already exists")

	snapshotFileRegex = regexp.MustCompile(`^([0-9]{6,})\.json$`)
	genRegex          = regexp.MustCompile(`^(.+)-gen(\d+)$`)
)

// RestartMode defines how a SnapshotWriter behaves when existing snapshots
// or collisions are encountered upon start or during writes.
type RestartMode int

const (
	// RestartRejectCollision rejects writing if a snapshot with the allocated
	// sequence already exists on disk, preventing silent overwrites.
	RestartRejectCollision RestartMode = iota

	// RestartResumeNext inspects existing committed snapshots and resumes
	// allocation at the next contiguous sequence number (maxSeq + 1).
	RestartResumeNext

	// RestartFreshGeneration allocates a new generation for the phase if committed
	// snapshots already exist in the target directory (e.g., <phase>-gen2).
	RestartFreshGeneration
)

func (p RestartMode) String() string {
	switch p {
	case RestartRejectCollision:
		return "RestartRejectCollision"
	case RestartResumeNext:
		return "RestartResumeNext"
	case RestartFreshGeneration:
		return "RestartFreshGeneration"
	default:
		return fmt.Sprintf("RestartMode(%d)", p)
	}
}

// SnapshotReceipt records the verified result of an atomically committed snapshot.
type SnapshotReceipt struct {
	RunID       string `json:"run_id"`
	Phase       string `json:"phase"`
	Sequence    int    `json:"sequence"`
	Path        string `json:"path"`
	ByteCount   int64  `json:"byte_count"`
	SHA256      string `json:"sha256"`
	CommittedAt string `json:"committed_at"`
}

// SnapshotEntry represents one committed snapshot discovered by SnapshotWatcher.
type SnapshotEntry struct {
	RunID       string    `json:"run_id"`
	Phase       string    `json:"phase"`
	Sequence    int       `json:"sequence"`
	Path        string    `json:"path"`
	ByteCount   int64     `json:"byte_count"`
	SHA256      string    `json:"sha256"`
	CommittedAt string    `json:"committed_at,omitempty"`
	ModTime     time.Time `json:"mod_time,omitempty"`
}

// ReadPayload reads and returns the full byte payload of this committed snapshot.
func (e SnapshotEntry) ReadPayload() ([]byte, error) {
	return os.ReadFile(e.Path)
}

// SnapshotWriter manages atomic, monotonic snapshot publication under <base>/<run-id>/<phase>/.
type SnapshotWriter struct {
	mu          sync.Mutex
	baseDir     string
	runID       string
	phase       string
	dir         string
	nextSeq     int
	initialized bool
	RestartMode RestartMode
}

// WriterOption configures a SnapshotWriter.
type WriterOption func(*SnapshotWriter)

// WithRestartMode sets the writer's restart policy.
func WithRestartMode(policy RestartMode) WriterOption {
	return func(w *SnapshotWriter) {
		w.RestartMode = policy
	}
}

// NewWriter creates a new SnapshotWriter for the specified layout:
// <baseDir>/<runID>/<phase>/<zero-padded-seq>.json.
func NewWriter(baseDir, runID, phase string, opts ...WriterOption) (*SnapshotWriter, error) {
	if baseDir == "" {
		return nil, errors.New("benchsnapshot: baseDir cannot be empty")
	}
	if runID == "" {
		return nil, errors.New("benchsnapshot: runID cannot be empty")
	}
	if phase == "" {
		return nil, errors.New("benchsnapshot: phase cannot be empty")
	}

	w := &SnapshotWriter{
		baseDir:     baseDir,
		runID:       runID,
		phase:       phase,
		dir:         filepath.Join(baseDir, runID, phase),
		nextSeq:     1,
		RestartMode: RestartRejectCollision,
	}

	for _, opt := range opts {
		if opt != nil {
			opt(w)
		}
	}

	return w, nil
}

// NewWriterWithMode creates a SnapshotWriter with an explicit restart mode.
func NewWriterWithMode(baseDir, runID, phase string, mode RestartMode) (*SnapshotWriter, error) {
	return NewWriter(baseDir, runID, phase, WithRestartMode(mode))
}

// SetRestartMode updates the restart mode on the writer before first write.
func (w *SnapshotWriter) SetRestartMode(mode RestartMode) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.RestartMode = mode
}

// RunID returns the current run identity.
func (w *SnapshotWriter) RunID() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.runID
}

// Phase returns the current phase (which may have advanced under RestartFreshGeneration).
func (w *SnapshotWriter) Phase() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.phase
}

// BaseDir returns the base directory.
func (w *SnapshotWriter) BaseDir() string {
	return w.baseDir
}

// Dir returns the current target directory.
func (w *SnapshotWriter) Dir() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.dir
}

// NextSequence returns the sequence number that will be assigned next.
func (w *SnapshotWriter) NextSequence() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.nextSeq
}

func (w *SnapshotWriter) initRestartModeLocked() error {
	switch w.RestartMode {
	case RestartRejectCollision:
		w.nextSeq = 1
		return nil

	case RestartResumeNext:
		maxSeq, err := findMaxCommittedSequence(w.dir)
		if err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("benchsnapshot: scan existing snapshots for resume: %w", err)
		}
		w.nextSeq = maxSeq + 1
		return nil

	case RestartFreshGeneration:
		hasSnapshots, err := hasCommittedSnapshots(w.dir)
		if err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("benchsnapshot: check existing snapshots for fresh generation: %w", err)
		}
		if hasSnapshots {
			w.phase = nextGenerationPhase(w.baseDir, w.runID, w.phase)
			w.dir = filepath.Join(w.baseDir, w.runID, w.phase)
		}
		w.nextSeq = 1
		return nil

	default:
		return fmt.Errorf("benchsnapshot: unknown restart policy: %v", w.RestartMode)
	}
}

// WriteSnapshot allocates the next contiguous sequence (1-based, monotonic),
// writes the payload to a staging file, flushes/syncs it, and atomically commits it
// to <base>/<run-id>/<phase>/<zero-padded-seq>.json, returning a verified receipt.
func (w *SnapshotWriter) WriteSnapshot(payload []byte) (SnapshotReceipt, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.initialized {
		if err := w.initRestartModeLocked(); err != nil {
			return SnapshotReceipt{}, err
		}
		w.initialized = true
	}

	seq := w.nextSeq
	targetName := fmt.Sprintf("%06d.json", seq)
	targetPath := filepath.Join(w.dir, targetName)

	// Pre-check collision: reject overwrite when sequence already exists.
	if _, err := os.Stat(targetPath); err == nil {
		return SnapshotReceipt{}, fmt.Errorf("%w: sequence %d already exists at %s", ErrSequenceCollision, seq, targetPath)
	} else if !os.IsNotExist(err) {
		return SnapshotReceipt{}, fmt.Errorf("benchsnapshot: check target collision: %w", err)
	}

	// Ensure destination directory exists.
	if err := os.MkdirAll(w.dir, 0755); err != nil {
		return SnapshotReceipt{}, fmt.Errorf("benchsnapshot: create target directory: %w", err)
	}

	// Create temporary staging file in the same directory: <seq>.tmp.<rand>.
	randBytes := make([]byte, 8)
	if _, err := rand.Read(randBytes); err != nil {
		randBytes = []byte(fmt.Sprintf("%08x", time.Now().UnixNano()))
	}
	stageName := fmt.Sprintf("%06d.tmp.%s", seq, hex.EncodeToString(randBytes))
	stagePath := filepath.Join(w.dir, stageName)

	stageFile, err := os.OpenFile(stagePath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return SnapshotReceipt{}, fmt.Errorf("benchsnapshot: create staging file: %w", err)
	}

	committed := false
	defer func() {
		if !committed {
			_ = stageFile.Close()
			_ = os.Remove(stagePath)
		}
	}()

	if len(payload) > 0 {
		if _, err := stageFile.Write(payload); err != nil {
			return SnapshotReceipt{}, fmt.Errorf("benchsnapshot: write staging file: %w", err)
		}
	}

	// Fully flush and sync to disk.
	if err := stageFile.Sync(); err != nil {
		return SnapshotReceipt{}, fmt.Errorf("benchsnapshot: sync staging file: %w", err)
	}

	if err := stageFile.Close(); err != nil {
		return SnapshotReceipt{}, fmt.Errorf("benchsnapshot: close staging file: %w", err)
	}

	// Re-check target file right before rename.
	if _, err := os.Stat(targetPath); err == nil {
		return SnapshotReceipt{}, fmt.Errorf("%w: sequence %d already exists at %s", ErrSequenceCollision, seq, targetPath)
	}

	// Atomically rename staging file to final destination.
	if err := os.Rename(stagePath, targetPath); err != nil {
		return SnapshotReceipt{}, fmt.Errorf("benchsnapshot: commit snapshot rename: %w", err)
	}
	committed = true

	// Advance sequence.
	w.nextSeq++

	h := sha256.Sum256(payload)
	shaHex := hex.EncodeToString(h[:])
	committedAt := time.Now().UTC().Format(time.RFC3339Nano)

	receipt := SnapshotReceipt{
		RunID:       w.runID,
		Phase:       w.phase,
		Sequence:    seq,
		Path:        targetPath,
		ByteCount:   int64(len(payload)),
		SHA256:      shaHex,
		CommittedAt: committedAt,
	}
	return receipt, nil
}

// SnapshotWatcher scans for committed evidence snapshots under <baseDir>/<runID>/<phase>/.
type SnapshotWatcher struct {
	BaseDir string
	RunID   string
	Phase   string
}

// NewWatcher creates a new SnapshotWatcher for the given run and phase.
func NewWatcher(baseDir, runID, phase string) *SnapshotWatcher {
	return &SnapshotWatcher{
		BaseDir: baseDir,
		RunID:   runID,
		Phase:   phase,
	}
}

// ScanCommitted returns sorted, committed snapshots in strictly increasing lexicographic sequence order.
// Partial or staging temporary files are ignored.
func (w *SnapshotWatcher) ScanCommitted() ([]SnapshotEntry, error) {
	dir := filepath.Join(w.BaseDir, w.RunID, w.Phase)
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return []SnapshotEntry{}, nil
		}
		return nil, fmt.Errorf("benchsnapshot: read phase directory %s: %w", dir, err)
	}

	entries := make([]SnapshotEntry, 0, len(dirEntries))
	for _, de := range dirEntries {
		if de.IsDir() {
			continue
		}
		name := de.Name()
		seq, ok := parseSnapshotSequence(name)
		if !ok {
			continue
		}

		fullPath := filepath.Join(dir, name)
		data, err := os.ReadFile(fullPath)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("benchsnapshot: read snapshot file %s: %w", fullPath, err)
		}

		h := sha256.Sum256(data)
		shaHex := hex.EncodeToString(h[:])

		var modTime time.Time
		var committedAt string
		if info, err := de.Info(); err == nil && info != nil {
			modTime = info.ModTime()
			committedAt = modTime.UTC().Format(time.RFC3339Nano)
		}

		entry := SnapshotEntry{
			RunID:       w.RunID,
			Phase:       w.Phase,
			Sequence:    seq,
			Path:        fullPath,
			ByteCount:   int64(len(data)),
			SHA256:      shaHex,
			CommittedAt: committedAt,
			ModTime:     modTime,
		}
		entries = append(entries, entry)
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Sequence != entries[j].Sequence {
			return entries[i].Sequence < entries[j].Sequence
		}
		return entries[i].Path < entries[j].Path
	})

	return entries, nil
}

func parseSnapshotSequence(name string) (int, bool) {
	if strings.Contains(name, ".tmp") || !strings.HasSuffix(name, ".json") {
		return 0, false
	}
	matches := snapshotFileRegex.FindStringSubmatch(name)
	if len(matches) != 2 {
		return 0, false
	}
	seq, err := strconv.Atoi(matches[1])
	if err != nil || seq <= 0 {
		return 0, false
	}
	return seq, true
}

func isCommittedSnapshotFilename(name string) bool {
	_, ok := parseSnapshotSequence(name)
	return ok
}

func hasCommittedSnapshots(dir string) (bool, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if isCommittedSnapshotFilename(e.Name()) {
			return true, nil
		}
	}
	return false, nil
}

func findMaxCommittedSequence(dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	maxSeq := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if seq, ok := parseSnapshotSequence(e.Name()); ok {
			if seq > maxSeq {
				maxSeq = seq
			}
		}
	}
	return maxSeq, nil
}

func nextGenerationPhase(baseDir, runID, currentPhase string) string {
	baseName := currentPhase
	startGen := 2
	if matches := genRegex.FindStringSubmatch(currentPhase); len(matches) == 3 {
		baseName = matches[1]
		if g, err := strconv.Atoi(matches[2]); err == nil {
			startGen = g + 1
		}
	}

	for gen := startGen; ; gen++ {
		nextPhase := fmt.Sprintf("%s-gen%d", baseName, gen)
		nextPhaseDir := filepath.Join(baseDir, runID, nextPhase)
		has, err := hasCommittedSnapshots(nextPhaseDir)
		if err != nil && os.IsNotExist(err) {
			return nextPhase
		}
		if !has {
			return nextPhase
		}
	}
}
