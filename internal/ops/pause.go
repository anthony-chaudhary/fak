package ops

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// PauseSchemaV1 is the schema identifier for the durable ops pause record.
const PauseSchemaV1 = "fak-ops-load-pause/1"

// DefaultPauseStateRel is the durable pause-state path relative to the repo root,
// stored as a sibling of the fak-ops-event/1 and fak-ops-load/1 ledgers.
const DefaultPauseStateRel = ".fak/ops-load-pause.json"

// PauseReasonClass is the closed vocabulary for why the ops engine paused (#14476).
type PauseReasonClass string

const (
	PauseReasonHostPressure      PauseReasonClass = "HOST_PRESSURE"
	PauseReasonDutyCycleCap      PauseReasonClass = "DUTY_CYCLE_CAP"
	PauseReasonIdleNoWork        PauseReasonClass = "IDLE_NO_WORK"
	PauseReasonProbePending      PauseReasonClass = "PROBE_PENDING"
	PauseReasonOperatorRequested PauseReasonClass = "OPERATOR_REQUESTED"
	PauseReasonUnknownHostState  PauseReasonClass = "UNKNOWN_HOST_STATE"
	PauseReasonMaxPauseReached   PauseReasonClass = "MAX_PAUSE_REACHED"
	PauseReasonConditionCleared  PauseReasonClass = "CONDITION_CLEARED"
)

// ValidPauseReasonClass reports whether r is a member of the closed pause reason vocabulary.
func ValidPauseReasonClass(r PauseReasonClass) bool {
	switch r {
	case PauseReasonHostPressure,
		PauseReasonDutyCycleCap,
		PauseReasonIdleNoWork,
		PauseReasonProbePending,
		PauseReasonOperatorRequested,
		PauseReasonUnknownHostState,
		PauseReasonMaxPauseReached,
		PauseReasonConditionCleared:
		return true
	}
	return false
}

// ReactivateCondition declares as data what condition will un-pause the ops engine.
type ReactivateCondition struct {
	PressureRatioBelow float64 `json:"pressure_ratio_below,omitempty"`
	ConsecutiveSamples int     `json:"consecutive_samples,omitempty"`
	MaxPause           string  `json:"max_pause,omitempty"`
	OperatorAction     string  `json:"operator_action,omitempty"`
}

// IsEmpty reports whether no re-activation condition has been declared.
func (c ReactivateCondition) IsEmpty() bool {
	return c.PressureRatioBelow == 0 &&
		c.ConsecutiveSamples == 0 &&
		c.MaxPause == "" &&
		c.OperatorAction == ""
}

// PauseState is the durable fak-ops-load-pause/1 record persisted next to the ops ledger.
type PauseState struct {
	Schema                  string              `json:"schema"`
	Paused                  bool                `json:"paused"`
	ReasonClass             PauseReasonClass    `json:"reason_class,omitempty"`
	ReasonDetail            string              `json:"reason_detail,omitempty"`
	PausedAt                time.Time           `json:"paused_at,omitempty"`
	PausedTicks             uint64              `json:"paused_ticks"`
	ReactivateWhen          ReactivateCondition `json:"reactivate_when,omitempty"`
	LastProbeAt             time.Time           `json:"last_probe_at,omitempty"`
	LastProbeResult         string              `json:"last_probe_result,omitempty"`
	ProbeAttempts           uint64              `json:"probe_attempts,omitempty"`
	ConsecutiveClearSamples int                 `json:"consecutive_clear_samples,omitempty"`
	ResumedAt               time.Time           `json:"resumed_at,omitempty"`
	ResumeReasonClass       PauseReasonClass    `json:"resume_reason_class,omitempty"`
}

// Validate enforces the fak-ops-load-pause/1 schema and closed reason vocabulary.
func (s PauseState) Validate() error {
	if s.Schema != PauseSchemaV1 {
		return fmt.Errorf("ops pause: unexpected schema %q", s.Schema)
	}
	if s.ReasonClass != "" && !ValidPauseReasonClass(s.ReasonClass) {
		return fmt.Errorf("ops pause: reason_class %q is outside the closed vocabulary", s.ReasonClass)
	}
	if s.Paused {
		if !ValidPauseReasonClass(s.ReasonClass) {
			return fmt.Errorf("ops pause: paused state requires a valid reason_class, got %q", s.ReasonClass)
		}
		if s.ReactivateWhen.IsEmpty() {
			return fmt.Errorf("ops pause: paused state requires a non-empty reactivate_when condition")
		}
	}
	return nil
}

// DefaultPauseStatePath returns the durable pause-state file path for a repo root.
func DefaultPauseStatePath(root string) string {
	if root != "" {
		return filepath.Join(root, filepath.FromSlash(DefaultPauseStateRel))
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "fak-ops-load-pause.json")
	}
	return filepath.Join(home, ".fak", "ops-load-pause.json")
}

// PauseStore provides thread-safe atomic persistence for fak-ops-load-pause/1.
type PauseStore struct {
	mu   sync.Mutex
	path string
}

// OpenPauseStore opens or initializes a PauseStore at path.
func OpenPauseStore(path string) (*PauseStore, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return &PauseStore{path: path}, nil
}

// Read loads and validates the on-disk pause record. A missing file returns an
// unpaused default record with nil error.
func (s *PauseStore) Read() (PauseState, error) {
	if s == nil {
		return PauseState{Schema: PauseSchemaV1}, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	raw, err := os.ReadFile(s.path)
	if os.IsNotExist(err) {
		return PauseState{Schema: PauseSchemaV1}, nil
	}
	if err != nil {
		return PauseState{}, err
	}
	var st PauseState
	if err := json.Unmarshal(raw, &st); err != nil {
		return PauseState{}, err
	}
	if err := st.Validate(); err != nil {
		return PauseState{}, err
	}
	return st, nil
}

// ReadFailSafe loads the pause record and degrades any read, parse, or
// validation error to "not paused" so a corrupt file can never wedge the daemon.
func (s *PauseStore) ReadFailSafe() PauseState {
	st, err := s.Read()
	if err != nil {
		return PauseState{Schema: PauseSchemaV1, Paused: false}
	}
	return st
}

// Write validates and atomically persists the pause record to disk.
func (s *PauseStore) Write(st PauseState) error {
	if s == nil {
		return nil
	}
	if st.Schema == "" {
		st.Schema = PauseSchemaV1
	}
	if err := st.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.path), "ops-load-pause-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, s.path); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return nil
}
