package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Process exit codes for `fak up` (sysexits.h): a supervisor restarts a transient stop
// and must not restart a structural misconfiguration into a loop.
const (
	upExitTransient  = 75 // EX_TEMPFAIL: the memory guard stopped a healthy-config process
	upExitStructural = 78 // EX_CONFIG: the configuration cannot serve (e.g. --max-rss too low)
)

// ErrMaxRSSBelowResident matches a boot refusal where the measured resident footprint
// plus one session's KV already meets or exceeds the --max-rss ceiling, so every request
// would be declined.
var ErrMaxRSSBelowResident = errors.New("fak up: --max-rss ceiling is at or below the resident footprint")

// MaxRSSBelowResidentError names both sides of the boot-time ceiling check.
type MaxRSSBelowResidentError struct {
	Ceiling       uint64
	ResidentBytes uint64
	SessionKV     uint64
	// Suggested is the derived ceiling (footprint + KV + headroom) that would admit a
	// session; 0 when not computed.
	Suggested uint64
}

func (e *MaxRSSBelowResidentError) Error() string {
	msg := fmt.Sprintf("%v: ceiling %d bytes (%s) <= resident %d bytes (%s) + per-session KV %d bytes (%s); raise --max-rss or reduce --context",
		ErrMaxRSSBelowResident, e.Ceiling, formatBytes(e.Ceiling), e.ResidentBytes, formatBytes(e.ResidentBytes), e.SessionKV, formatBytes(e.SessionKV))
	if e.Suggested > 0 {
		msg += fmt.Sprintf("; set --max-rss auto or at least %s (%d bytes)", formatBytes(e.Suggested), e.Suggested)
	}
	return msg
}

func (e *MaxRSSBelowResidentError) Is(target error) bool { return target == ErrMaxRSSBelowResident }

// upResidentFootprint is the boot-check probe seam; 0 means unknown and skips the check.
var upResidentFootprint = platformCurrentRSS

// checkMaxRSSAgainstResident refuses a ceiling that cannot admit one session at idle.
// A zero ceiling (guard disabled) or an unknown footprint passes.
func checkMaxRSSAgainstResident(ceiling, resident, sessionKV uint64) error {
	if ceiling == 0 || resident == 0 {
		return nil
	}
	if resident+sessionKV >= ceiling {
		return &MaxRSSBelowResidentError{Ceiling: ceiling, ResidentBytes: resident, SessionKV: sessionKV}
	}
	return nil
}

// checkMaxRSS runs the boot check against the server's live footprint and plan.
func (s *turnkeyServer) checkMaxRSS(ceiling uint64) error {
	if s == nil || ceiling == 0 {
		return nil
	}
	return checkMaxRSSAgainstResident(ceiling, upResidentFootprint(), s.capacity().PerSessionKVBytes)
}

// readinessAdmissionStarved is the closed readiness state and reason when the armed host
// budget has no room even with nothing in flight.
const readinessAdmissionStarved = "admission_starved"

type hostBudgetReporter interface {
	HostMemoryBudgetStats() agent.HostMemoryBudgetStats
}

func (s *turnkeyServer) hostBudgetStats() (agent.HostMemoryBudgetStats, bool) {
	if s == nil {
		return agent.HostMemoryBudgetStats{}, false
	}
	r, ok := s.planner.(hostBudgetReporter)
	if !ok {
		return agent.HostMemoryBudgetStats{}, false
	}
	st := r.HostMemoryBudgetStats()
	return st, st.Armed
}

func (s *turnkeyServer) admissionStarved() bool {
	st, ok := s.hostBudgetStats()
	return ok && st.UsedKnown && st.Structural
}

// turnkeyHostBudgetBlock is the /healthz host_budget projection. AvailSigned is not
// clamped; LastDeclineAt is null until the first decline.
type turnkeyHostBudgetBlock struct {
	Ceiling       int64      `json:"ceiling"`
	Used          int64      `json:"used"`
	AvailSigned   int64      `json:"avail_signed"`
	DeclinedTotal int64      `json:"declined_total"`
	LastDeclineAt *time.Time `json:"last_decline_at"`
	Structural    bool       `json:"structural"`
}

// hostBudgetBlock returns nil (key absent) when no host budget is armed.
func (s *turnkeyServer) hostBudgetBlock() *turnkeyHostBudgetBlock {
	st, ok := s.hostBudgetStats()
	if !ok {
		return nil
	}
	b := &turnkeyHostBudgetBlock{
		Ceiling:       st.Ceiling,
		Used:          st.Used,
		AvailSigned:   st.AvailSigned,
		DeclinedTotal: st.DeclinedTotal,
		Structural:    st.Structural,
	}
	if !st.LastDeclineAt.IsZero() {
		at := st.LastDeclineAt.UTC()
		b.LastDeclineAt = &at
	}
	return b
}

// writeInferenceError counts a capacity refusal as a shed before writing it.
func (s *turnkeyServer) writeInferenceError(w http.ResponseWriter, err error) {
	var capErr *agent.InKernelCapacityError
	var oomErr *agent.InKernelOOMError
	if s != nil && (errors.As(err, &capErr) || errors.As(err, &oomErr)) {
		s.mu.Lock()
		s.shedTotal++
		s.mu.Unlock()
	}
	writeTurnkeyInferenceError(w, err)
}

// upLifecycleSchema versions the guard-stop record appended to the lifecycle log.
const upLifecycleSchema = "fak.up.lifecycle.v1"

// upLifecycleReasonMaxRSS is the reason token for a memory-guard stop.
const upLifecycleReasonMaxRSS = "max_rss_guard"

type upLifecycleRecord struct {
	Schema  string    `json:"schema"`
	At      time.Time `json:"at"`
	Reason  string    `json:"reason"`
	RSS     uint64    `json:"rss"`
	Ceiling uint64    `json:"ceiling"`
	PID     int       `json:"pid"`
}

// upLifecycleLogPath is the seam for the lifecycle log location; the default honors HOME.
var upLifecycleLogPath = func() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".fak", "up", "lifecycle.jsonl"), nil
}

// appendUpLifecycleRecord appends one JSON line, creating the directory 0700.
func appendUpLifecycleRecord(path string, rec upLifecycleRecord) error {
	if rec.Schema == "" {
		rec.Schema = upLifecycleSchema
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(line, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// recordMemGuardStop appends the guard-stop record; failures are logged, never fatal.
func recordMemGuardStop(ev memGuardStopEvent, logf func(format string, args ...any)) {
	path, err := upLifecycleLogPath()
	if err == nil {
		err = appendUpLifecycleRecord(path, upLifecycleRecord{
			Schema:  upLifecycleSchema,
			At:      ev.At.UTC(),
			Reason:  upLifecycleReasonMaxRSS,
			RSS:     ev.RSS,
			Ceiling: ev.Ceiling,
			PID:     os.Getpid(),
		})
	}
	if err != nil && logf != nil {
		logf("fak up: lifecycle record not written: %v", err)
	}
}

// memGuardFired reports whether the memory guard (not a signal or idle exit) stopped the server.
func (s *turnkeyServer) memGuardFired() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	g := s.memGuard
	s.mu.Unlock()
	return g.didFire()
}
