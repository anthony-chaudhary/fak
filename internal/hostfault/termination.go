package hostfault

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const HostTerminationSchema = "fak.host-termination.v1"

// HostTerminationMarker is the observable subset Windows supplies before a
// console-control teardown. It deliberately does not name a killer: forced
// TerminateProcess exits have no handler callback and remain EXTERNAL_UNKNOWN.
type HostTerminationMarker struct {
	Schema         string `json:"schema"`
	ControlType    string `json:"control_type"`
	GuardPID       int    `json:"guard_pid"`
	ConsoleSession uint32 `json:"console_session"`
	ObservedAt     string `json:"observed_at"`
}

func (m HostTerminationMarker) Valid() bool {
	switch m.ControlType {
	case "CTRL_CLOSE_EVENT", "CTRL_LOGOFF_EVENT", "CTRL_SHUTDOWN_EVENT":
		return m.Schema == HostTerminationSchema && m.GuardPID > 0 && m.ObservedAt != ""
	default:
		return false
	}
}

// AppendHostTermination writes to the same host evidence ledger as Event 1000
// crash signals. One ledger and one reader boundary prevents duplicate joins.
func AppendHostTermination(path string, m HostTerminationMarker) error {
	if !m.Valid() {
		return fmt.Errorf("invalid host termination marker")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if _, err = f.Write(append(b, '\n')); err != nil {
		return err
	}
	return f.Sync()
}

func ReadHostTerminations(path string) ([]HostTerminationMarker, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []HostTerminationMarker
	s := bufio.NewScanner(f)
	for s.Scan() {
		var m HostTerminationMarker
		if json.Unmarshal(s.Bytes(), &m) == nil && m.Valid() {
			out = append(out, m)
		}
	}
	return out, s.Err()
}

// CorrelateHostTermination returns the nearest observed control teardown in
// the bounded crash-wave window. No marker is evidence only for UNKNOWN.
func CorrelateHostTermination(markers []HostTerminationMarker, wave time.Time, window time.Duration) (HostTerminationMarker, bool) {
	var best HostTerminationMarker
	bestDelta := window + time.Nanosecond
	for _, m := range markers {
		at, err := time.Parse(time.RFC3339Nano, m.ObservedAt)
		if err != nil {
			continue
		}
		d := at.Sub(wave)
		if d < 0 {
			d = -d
		}
		if d <= window && d < bestDelta {
			best, bestDelta = m, d
		}
	}
	return best, best.Valid()
}
