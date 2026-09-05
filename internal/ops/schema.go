// Package ops implements the autonomous operations daemon and machine maintenance subsystem (#11156, #11158).
// It defines the configuration schema, watermarks, and event ledger (fak-ops-event/1) for host self-healing.
package ops

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// EventSchemaV1 is the schema identifier for operations audit ledger entries.
const EventSchemaV1 = "fak-ops-event/1"

// ActionType is the closed vocabulary for autonomous maintenance actions.
type ActionType string

const (
	ActionStorageReclaim ActionType = "STORAGE_RECLAIM"
	ActionProcessReap    ActionType = "PROCESS_REAP"
	ActionLockEvict      ActionType = "LOCK_EVICT"
	ActionWorktreePrune  ActionType = "WORKTREE_PRUNE"
	ActionAssetSync      ActionType = "ASSET_SYNC"
)

// Event records a single witnessed autonomous maintenance action in fak-ops-event/1.
type Event struct {
	Schema         string     `json:"schema"`
	Timestamp      time.Time  `json:"timestamp"`
	ActionType     ActionType `json:"action_type"`
	Details        string     `json:"details"`
	BytesReclaimed int64      `json:"bytes_reclaimed,omitempty"`
	PIDsAffected   []int      `json:"pids_affected,omitempty"`
	DurationMS     int64      `json:"duration_ms"`
	Error          string     `json:"error,omitempty"`
}

// Config declares the operations thresholds, watermarks, and intervals.
type Config struct {
	WarningFreeBytes  uint64        `json:"warning_free_bytes"`
	RefuseFreeBytes   uint64        `json:"refuse_free_bytes"`
	GoCacheHighBytes  uint64        `json:"go_cache_high_bytes"`
	GoCacheLowBytes   uint64        `json:"go_cache_low_bytes"`
	GoCacheMinAge     time.Duration `json:"go_cache_min_age"`
	ScratchTTL        time.Duration `json:"scratch_ttl"`
	MaxThreads        int           `json:"max_threads"`
	CPUPinWindow      time.Duration `json:"cpu_pin_window"`
	OrphanReapEnabled bool          `json:"orphan_reap_enabled"`
}

// DefaultConfig returns standard operations defaults.
func DefaultConfig() Config {
	return Config{
		WarningFreeBytes:  4 * 1024 * 1024 * 1024,  // 4 GB
		RefuseFreeBytes:   2 * 1024 * 1024 * 1024,  // 2 GB
		GoCacheHighBytes:  32 * 1024 * 1024 * 1024, // 32 GB
		GoCacheLowBytes:   24 * 1024 * 1024 * 1024, // 24 GB
		GoCacheMinAge:     7 * 24 * time.Hour,      // 7 days
		ScratchTTL:        24 * time.Hour,          // 24 hours
		MaxThreads:        2000,
		CPUPinWindow:      15 * time.Second,
		OrphanReapEnabled: true,
	}
}

// Ledger provides thread-safe append-only persistence for ops events.
type Ledger struct {
	mu   sync.Mutex
	path string
}

// OpenLedger opens or creates an operations ledger file.
func OpenLedger(path string) (*Ledger, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return &Ledger{path: path}, nil
}

// DefaultLedgerPath returns the standard ops event log path under the workspace or user cache.
func DefaultLedgerPath(root string) string {
	if root != "" {
		return filepath.Join(root, ".fak", "ops-events.jsonl")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "fak-ops-events.jsonl")
	}
	return filepath.Join(home, ".fak", "ops-events.jsonl")
}

// Record appends an Event to the ledger file.
func (l *Ledger) Record(ev Event) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if ev.Schema == "" {
		ev.Schema = EventSchemaV1
	}
	if ev.Timestamp.IsZero() {
		ev.Timestamp = time.Now().UTC()
	}

	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = f.Write(data)
	return err
}

// QueryEvents reads all events from the ledger matching the given since time window.
func (l *Ledger) QueryEvents(since time.Duration) ([]Event, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	f, err := os.Open(l.path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var cutoff time.Time
	if since > 0 {
		cutoff = time.Now().Add(-since)
	}

	dec := json.NewDecoder(f)
	var events []Event
	for {
		var ev Event
		if err := dec.Decode(&ev); err != nil {
			if err == io.EOF {
				break
			}
			return events, err
		}
		if cutoff.IsZero() || ev.Timestamp.After(cutoff) {
			events = append(events, ev)
		}
	}
	return events, nil
}
