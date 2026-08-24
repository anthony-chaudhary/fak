package hostresurrect

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

const CohortSchema = "fak.host-resurrection-cohort.v1"

// CohortFileName is the last known-good pre-crash liveness snapshot. The
// control loop refreshes it on every clean poll and consumes the previous file
// when a crash signal appears.
const CohortFileName = "host-resurrection-cohort.json"

type cohortEnvelope struct {
	Schema string `json:"schema"`
	Cohort
}

func CohortPath(regDir string) string { return filepath.Join(regDir, CohortFileName) }

// Capture keeps only rows whose current PID is independently alive. The caller
// supplies the probe so fixture tests never depend on host process state.
func Capture(rows []guardsessions.Row, capturedAt time.Time, alive func(int) bool, started func(int) (time.Time, bool)) Cohort {
	if alive == nil || started == nil {
		return Cohort{}
	}
	live := guardsessions.LiveInteractive(rows)
	out := Cohort{CapturedAt: capturedAt.UTC().Format(time.RFC3339Nano)}
	for _, row := range live {
		if row.PID <= 0 || !alive(row.PID) {
			continue
		}
		processStarted, ok := started(row.PID)
		rowStarted, err := time.Parse(time.RFC3339Nano, row.StartedAt)
		if err != nil {
			rowStarted, err = time.Parse(time.RFC3339, row.StartedAt)
		}
		if !ok || err != nil || processStarted.Before(rowStarted.Add(-time.Second)) {
			continue
		}
		out.Sessions = append(out.Sessions, CohortEntry{Handle: row.Handle, PID: row.PID, StartedAt: row.StartedAt})
	}
	return out
}

func LoadCohort(path string) (Cohort, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Cohort{}, nil
	}
	if err != nil {
		return Cohort{}, err
	}
	var envelope cohortEnvelope
	if json.Unmarshal(b, &envelope) != nil || envelope.Schema != CohortSchema || strings.TrimSpace(envelope.CapturedAt) == "" {
		return Cohort{}, errors.New("invalid host-resurrection cohort")
	}
	return envelope.Cohort, nil
}

func StoreCohort(path string, cohort Cohort) error {
	if strings.TrimSpace(cohort.CapturedAt) == "" {
		return errors.New("host-resurrection cohort missing capture time")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(cohortEnvelope{Schema: CohortSchema, Cohort: cohort})
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".host-resurrection-cohort-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
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
	return os.Rename(tmpPath, path)
}
