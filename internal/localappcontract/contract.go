package localappcontract

import (
	"encoding/json"
	"errors"
	"strings"
)

// Schema defines the canonical local app contract schema version.
const Schema = "fak.local-app-contract/1"

// Terminal represents the terminal lifecycle outcome of a local app task.
type Terminal string

// Terminal lifecycle outcomes for local app tasks.
const (
	Completed Terminal = "completed"
	Cancelled Terminal = "cancelled"
	Refused   Terminal = "refused"
	Failed    Terminal = "failed"
	HandedOff Terminal = "handed_off"
)

// Manifest declares the signed execution contract and task definitions for an engine.
type Manifest struct {
	Schema    string   `json:"schema"`
	Revision  string   `json:"revision"`
	Signature string   `json:"signature"`
	Engine    string   `json:"engine"`
	Tasks     []string `json:"tasks"`
}

// Event represents an ordered lifecycle occurrence within a task execution stream.
type Event struct {
	Schema   string `json:"schema"`
	Sequence uint64 `json:"sequence"`
	TaskID   string `json:"task_id"`
	Kind     string `json:"kind"`
	Reason   string `json:"reason,omitempty"`
}

// Receipt captures the audited terminal execution record of a task.
type Receipt struct {
	Schema           string           `json:"schema"`
	TaskID           string           `json:"task_id"`
	Engine           string           `json:"engine"`
	Location         string           `json:"location"`
	Revision         string           `json:"revision"`
	AdmittedEnvelope map[string]int64 `json:"admitted_envelope"`
	ObservedEnvelope map[string]int64 `json:"observed_envelope"`
	Quality          string           `json:"quality"`
	Attempts         int              `json:"attempts"`
	Authority        string           `json:"authority"`
	Terminal         Terminal         `json:"terminal"`
	Reason           string           `json:"reason,omitempty"`
}

// Validate checks that required fields are present and the terminal state is recognized.
func (r Receipt) Validate() error {
	if r.Schema != Schema || r.TaskID == "" || r.Engine == "" || r.Attempts < 1 {
		return errors.New("invalid receipt")
	}
	switch r.Terminal {
	case Completed, Cancelled, Refused, Failed, HandedOff:
		return nil
	}
	return errors.New("unknown terminal state")
}

// Marshal validates the receipt and serializes it to JSON.
func (r Receipt) Marshal() ([]byte, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(r)
}

// CheckAdditiveCompatibility verifies that no fields from oldRaw were removed in newRaw.
func CheckAdditiveCompatibility(oldRaw, newRaw []byte) error {
	var a, b map[string]any
	if json.Unmarshal(oldRaw, &a) != nil || json.Unmarshal(newRaw, &b) != nil {
		return errors.New("invalid json")
	}
	for k := range a {
		if _, ok := b[k]; !ok {
			return errors.New("field removed: " + k)
		}
	}
	return nil
}

// ContainsSensitiveField reports whether raw JSON contains sensitive payload fields.
func ContainsSensitiveField(raw []byte) bool {
	s := strings.ToLower(string(raw))
	for _, x := range []string{"prompt", "output_text", "file_path", "username", "device_id"} {
		if strings.Contains(s, "\""+x+"\"") {
			return true
		}
	}
	return false
}

// Replay checks that events are schema-compliant and strictly monotonic in sequence.
func Replay(events []Event) error {
	var last uint64
	for i, e := range events {
		if e.Schema != Schema || e.Sequence == 0 || (i > 0 && e.Sequence <= last) {
			return errors.New("events must be schema-valid and strictly ordered")
		}
		last = e.Sequence
	}
	return nil
}
