package microagent

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// SpawnBudgetRecordKind classifies one durable spawn budget journal entry.
type SpawnBudgetRecordKind string

const (
	RecordRoot      SpawnBudgetRecordKind = "root"
	RecordAdmit     SpawnBudgetRecordKind = "admit"
	RecordReconcile SpawnBudgetRecordKind = "reconcile"
	RecordRelease   SpawnBudgetRecordKind = "release"
)

// SpawnBudgetRecord is one persisted event record written to disk.
type SpawnBudgetRecord struct {
	Kind      SpawnBudgetRecordKind `json:"kind"`
	Timestamp int64                 `json:"timestamp,omitempty"`

	RootID   string `json:"root_id,omitempty"`
	RootGoal string `json:"root_goal,omitempty"`

	ParentID string        `json:"parent_id,omitempty"`
	ChildID  string        `json:"child_id,omitempty"`
	Goal     string        `json:"goal,omitempty"`
	Depth    int           `json:"depth,omitempty"`
	Budget   LineageBudget `json:"budget,omitempty"`

	Actual LineageBudget `json:"actual,omitempty"`
}

// SpawnBudgetStore manages durable persistence of lineage records on disk.
type SpawnBudgetStore struct {
	mu      sync.Mutex
	dir     string
	path    string
	hasRoot bool
}

// DurableSpawnBudgetStore is an alias for SpawnBudgetStore.
type DurableSpawnBudgetStore = SpawnBudgetStore

// NewSpawnBudgetStore creates or opens a spawn budget store at pathOrDir.
func NewSpawnBudgetStore(pathOrDir string) (*SpawnBudgetStore, error) {
	return OpenSpawnBudgetStore(pathOrDir)
}

// OpenSpawnBudgetStore opens a durable store in a specified directory or file.
func OpenSpawnBudgetStore(pathOrDir string) (*SpawnBudgetStore, error) {
	dir, path, err := resolveStorePath(pathOrDir)
	if err != nil {
		return nil, err
	}
	return &SpawnBudgetStore{
		dir:  dir,
		path: path,
	}, nil
}

// Path returns the canonical file path of the underlying journal.
func (s *SpawnBudgetStore) Path() string {
	if s == nil {
		return ""
	}
	return s.path
}

// Dir returns the root directory containing the store journal.
func (s *SpawnBudgetStore) Dir() string {
	if s == nil {
		return ""
	}
	return s.dir
}

// LogRoot writes the root identity record if not already recorded.
func (s *SpawnBudgetStore) LogRoot(rootID, rootGoal string) error {
	if s == nil {
		return fmt.Errorf("%w: nil store", ErrSpawnBudget)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hasRoot {
		return nil
	}
	records, err := s.recordsLocked()
	if err == nil {
		for _, rec := range records {
			if rec.Kind == RecordRoot {
				s.hasRoot = true
				return nil
			}
		}
	}
	rec := SpawnBudgetRecord{
		Kind:      RecordRoot,
		Timestamp: time.Now().UnixNano(),
		RootID:    rootID,
		RootGoal:  rootGoal,
	}
	if err := s.appendRecordLocked(rec); err != nil {
		return err
	}
	s.hasRoot = true
	return nil
}

// LogAdmit appends a child admission record atomically to disk.
func (s *SpawnBudgetStore) LogAdmit(request SpawnRequest) error {
	if s == nil {
		return fmt.Errorf("%w: nil store", ErrSpawnBudget)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rec := SpawnBudgetRecord{
		Kind:      RecordAdmit,
		Timestamp: time.Now().UnixNano(),
		ParentID:  request.ParentID,
		ChildID:   request.ChildID,
		Goal:      request.Goal,
		Depth:     request.Depth,
		Budget:    request.Budget,
	}
	return s.appendRecordLocked(rec)
}

// LogReconcile appends a child consumption reconciliation record to disk.
func (s *SpawnBudgetStore) LogReconcile(childID string, actual LineageBudget) error {
	if s == nil {
		return fmt.Errorf("%w: nil store", ErrSpawnBudget)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rec := SpawnBudgetRecord{
		Kind:      RecordReconcile,
		Timestamp: time.Now().UnixNano(),
		ChildID:   childID,
		Actual:    actual,
	}
	return s.appendRecordLocked(rec)
}

// LogRelease appends a child admission release record to disk.
func (s *SpawnBudgetStore) LogRelease(request SpawnRequest) error {
	if s == nil {
		return fmt.Errorf("%w: nil store", ErrSpawnBudget)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rec := SpawnBudgetRecord{
		Kind:      RecordRelease,
		Timestamp: time.Now().UnixNano(),
		ParentID:  request.ParentID,
		ChildID:   request.ChildID,
	}
	return s.appendRecordLocked(rec)
}

// Admit logs a child admission to the store.
func (s *SpawnBudgetStore) Admit(request SpawnRequest) error {
	return s.LogAdmit(request)
}

// Reconcile logs a child consumption reconciliation to the store.
func (s *SpawnBudgetStore) Reconcile(childID string, actual LineageBudget) error {
	return s.LogReconcile(childID, actual)
}

// Settle is an alias for Reconcile.
func (s *SpawnBudgetStore) Settle(childID string, actual LineageBudget) error {
	return s.LogReconcile(childID, actual)
}

// Records reads all durable records from the journal.
func (s *SpawnBudgetStore) Records() ([]SpawnBudgetRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recordsLocked()
}

func (s *SpawnBudgetStore) recordsLocked() ([]SpawnBudgetRecord, error) {
	f, err := os.Open(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("microagent: open store file: %w", err)
	}
	defer f.Close()

	var records []SpawnBudgetRecord
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec SpawnBudgetRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			return nil, fmt.Errorf("microagent: corrupt store record in %s: %w", s.path, err)
		}
		records = append(records, rec)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("microagent: read store file: %w", err)
	}
	return records, nil
}

// Close is a no-op for file-backed append store.
func (s *SpawnBudgetStore) Close() error {
	return nil
}

func (s *SpawnBudgetStore) appendRecordLocked(rec SpawnBudgetRecord) error {
	b, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("microagent: marshal store record: %w", err)
	}
	f, err := os.OpenFile(s.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("microagent: open store file %s: %w", s.path, err)
	}
	defer f.Close()
	line := append(b, '\n')
	if _, err := f.Write(line); err != nil {
		return fmt.Errorf("microagent: write store file: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("microagent: sync store file: %w", err)
	}
	return nil
}

// DurableSpawnBudget wraps SpawnBudget with durable on-disk journaling.
type DurableSpawnBudget struct {
	*SpawnBudget
	store *SpawnBudgetStore
	dir   string
	path  string
}

// NewDurableSpawnBudget creates or reopens a durable spawn budget against pathOrDir.
func NewDurableSpawnBudget(pathOrDir string, config any) (*DurableSpawnBudget, error) {
	return OpenDurableSpawnBudget(pathOrDir, config)
}

// OpenDurableSpawnBudget opens a persistent budget from a store directory or file.
// On open/reopen, it reconstructs the root's ancestry, active reservations, and
// spent capacity. Duplicate child admission cannot reserve twice, and unknown
// consumption stays reserved until reconciled.
func OpenDurableSpawnBudget(pathOrDir string, config any) (*DurableSpawnBudget, error) {
	var cfg *SpawnBudget
	switch c := config.(type) {
	case *SpawnBudget:
		if c != nil {
			cfg = c
		} else {
			cfg = &SpawnBudget{}
		}
	case SpawnBudget:
		cfg = &c
	case nil:
		cfg = &SpawnBudget{}
	default:
		return nil, fmt.Errorf("%w: invalid config type %T", ErrSpawnBudget, config)
	}

	store, err := OpenSpawnBudgetStore(pathOrDir)
	if err != nil {
		return nil, err
	}
	records, err := store.Records()
	if err != nil {
		return nil, err
	}

	b := &SpawnBudget{
		RootID:           cfg.RootID,
		RootGoal:         cfg.RootGoal,
		MaxDepth:         cfg.MaxDepth,
		MaxChildren:      cfg.MaxChildren,
		MaxDescendants:   cfg.MaxDescendants,
		MaxTokens:        cfg.MaxTokens,
		MaxOutputTokens:  cfg.MaxOutputTokens,
		MaxCostMicrosUSD: cfg.MaxCostMicrosUSD,
		store:            store,
		children:         make(map[string]int),
		reservations:     make(map[string]LineageBudget),
		goals:            make(map[string]goalLineage),
		goalOwners:       make(map[string]string),
	}

	for _, rec := range records {
		switch rec.Kind {
		case RecordRoot:
			if b.RootID != "" && b.RootID != rec.RootID {
				return nil, fmt.Errorf("%w: configured root %q does not match stored root %q", ErrInvalidAncestry, b.RootID, rec.RootID)
			}
			b.RootID = rec.RootID
			b.RootGoal = rec.RootGoal
			b.rootID = rec.RootID
			rootFP, err := fingerprintGoal(rec.RootGoal)
			if err != nil {
				return nil, err
			}
			b.rootGoal = rootFP
			b.goals[rec.RootID] = goalLineage{
				goalFingerprint: rootFP,
				pathFingerprint: rootFP,
				ancestors:       map[string]struct{}{},
				depth:           0,
			}
			b.goalOwners[rootFP] = rec.RootID
			store.hasRoot = true

		case RecordAdmit:
			if b.rootID == "" {
				if err := b.ensureGoalRoot(); err != nil {
					return nil, err
				}
			}
			goalFingerprint, err := fingerprintGoal(rec.Goal)
			if err != nil {
				return nil, err
			}
			parent, ok := b.goals[rec.ParentID]
			if !ok {
				return nil, fmt.Errorf("%w: parent %q not found for child %q during replay", ErrInvalidAncestry, rec.ParentID, rec.ChildID)
			}
			ancestors := cloneFingerprints(parent.ancestors)
			ancestors[parent.goalFingerprint] = struct{}{}
			b.goals[rec.ChildID] = goalLineage{
				goalFingerprint: goalFingerprint,
				pathFingerprint: fingerprintGoalPath(parent.pathFingerprint, goalFingerprint),
				ancestors:       ancestors,
				depth:           rec.Depth,
			}
			b.goalOwners[goalFingerprint] = rec.ChildID
			b.children[rec.ParentID]++
			b.descendants++
			b.reserved = addLineageBudget(b.reserved, rec.Budget)
			b.reservations[rec.ChildID] = rec.Budget

		case RecordReconcile:
			if res, ok := b.reservations[rec.ChildID]; ok {
				b.reserved = subtractLineageBudget(b.reserved, res)
				b.spent = addLineageBudget(b.spent, rec.Actual)
				delete(b.reservations, rec.ChildID)
			}

		case RecordRelease:
			if res, ok := b.reservations[rec.ChildID]; ok {
				b.reserved = subtractLineageBudget(b.reserved, res)
				delete(b.reservations, rec.ChildID)
			}
			if g, ok := b.goals[rec.ChildID]; ok {
				delete(b.goals, rec.ChildID)
				delete(b.goalOwners, g.goalFingerprint)
			}
			if b.children[rec.ParentID] > 0 {
				b.children[rec.ParentID]--
				b.descendants--
			}
		}
	}

	if b.rootID == "" && b.RootID != "" && b.RootGoal != "" {
		if err := b.ensureGoalRoot(); err != nil {
			return nil, err
		}
	}

	return &DurableSpawnBudget{
		SpawnBudget: b,
		store:       store,
		dir:         store.dir,
		path:        store.path,
	}, nil
}

// Admit reserves a child slot and writes through to disk before returning success.
func (d *DurableSpawnBudget) Admit(request SpawnRequest) error {
	if d == nil || d.SpawnBudget == nil {
		return fmt.Errorf("%w: missing durable spawn budget", ErrSpawnBudget)
	}
	return d.SpawnBudget.Admit(request)
}

// Reconcile settles a completed child's usage, updating spent and releasing reserved capacity.
func (d *DurableSpawnBudget) Reconcile(childID string, actual LineageBudget) error {
	if d == nil || d.SpawnBudget == nil {
		return fmt.Errorf("%w: missing durable spawn budget", ErrSpawnBudget)
	}
	return d.SpawnBudget.reconcile(childID, actual)
}

// Settle is an alias for Reconcile.
func (d *DurableSpawnBudget) Settle(childID string, actual LineageBudget) error {
	return d.Reconcile(childID, actual)
}

// Descendants reports host-admitted children across the entire lineage.
func (d *DurableSpawnBudget) Descendants() int {
	if d == nil || d.SpawnBudget == nil {
		return 0
	}
	return d.SpawnBudget.Descendants()
}

// Reserved reports active conservative reservations across all unreconciled children.
func (d *DurableSpawnBudget) Reserved() LineageBudget {
	if d == nil || d.SpawnBudget == nil {
		return LineageBudget{}
	}
	return d.SpawnBudget.Reserved()
}

// Spent reports settled consumption across all completed children.
func (d *DurableSpawnBudget) Spent() LineageBudget {
	if d == nil || d.SpawnBudget == nil {
		return LineageBudget{}
	}
	return d.SpawnBudget.Spent()
}

// Remaining reports remaining unreserved capacity for the lineage.
func (d *DurableSpawnBudget) Remaining() LineageBudget {
	if d == nil || d.SpawnBudget == nil {
		return LineageBudget{}
	}
	return d.SpawnBudget.Remaining()
}

// Allowance is an alias for Remaining.
func (d *DurableSpawnBudget) Allowance() LineageBudget {
	return d.Remaining()
}

// Reservation reports the active reservation for a child, if any.
func (d *DurableSpawnBudget) Reservation(childID string) (LineageBudget, bool) {
	if d == nil || d.SpawnBudget == nil {
		return LineageBudget{}, false
	}
	return d.SpawnBudget.Reservation(childID)
}

// Reservations returns a snapshot of active reservations.
func (d *DurableSpawnBudget) Reservations() map[string]LineageBudget {
	if d == nil || d.SpawnBudget == nil {
		return nil
	}
	return d.SpawnBudget.Reservations()
}

// Path returns the file path of the durable store.
func (d *DurableSpawnBudget) Path() string {
	if d == nil {
		return ""
	}
	return d.path
}

// Dir returns the directory path of the durable store.
func (d *DurableSpawnBudget) Dir() string {
	if d == nil {
		return ""
	}
	return d.dir
}

// Store returns the underlying SpawnBudgetStore.
func (d *DurableSpawnBudget) Store() *SpawnBudgetStore {
	if d == nil {
		return nil
	}
	return d.store
}

// Budget returns the underlying *SpawnBudget.
func (d *DurableSpawnBudget) Budget() *SpawnBudget {
	if d == nil {
		return nil
	}
	return d.SpawnBudget
}

// Close closes the underlying store.
func (d *DurableSpawnBudget) Close() error {
	if d == nil || d.store == nil {
		return nil
	}
	return d.store.Close()
}

func resolveStorePath(pathOrDir string) (dir string, file string, err error) {
	clean := filepath.Clean(strings.TrimSpace(pathOrDir))
	if clean == "" || clean == "." {
		if strings.TrimSpace(pathOrDir) == "" {
			return "", "", fmt.Errorf("%w: store path cannot be empty", ErrSpawnBudget)
		}
	}
	fi, statErr := os.Stat(clean)
	if statErr == nil && fi.IsDir() {
		return clean, filepath.Join(clean, "spawn_budget.jsonl"), nil
	}
	ext := strings.ToLower(filepath.Ext(clean))
	if ext == ".json" || ext == ".jsonl" {
		parent := filepath.Dir(clean)
		if err := os.MkdirAll(parent, 0o755); err != nil {
			return "", "", fmt.Errorf("microagent: mkdir store parent %s: %w", parent, err)
		}
		return parent, clean, nil
	}
	if err := os.MkdirAll(clean, 0o755); err != nil {
		return "", "", fmt.Errorf("microagent: mkdir store dir %s: %w", clean, err)
	}
	return clean, filepath.Join(clean, "spawn_budget.jsonl"), nil
}
