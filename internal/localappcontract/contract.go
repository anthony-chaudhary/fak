package localappcontract

import (
	"encoding/json"
	"errors"
	"strings"
)

const Schema = "fak.local-app-contract/1"

type Terminal string

const (
	Completed Terminal = "completed"
	Cancelled Terminal = "cancelled"
	Refused   Terminal = "refused"
	Failed    Terminal = "failed"
	HandedOff Terminal = "handed_off"
)

type Manifest struct {
	Schema    string   `json:"schema"`
	Revision  string   `json:"revision"`
	Signature string   `json:"signature"`
	Engine    string   `json:"engine"`
	Tasks     []string `json:"tasks"`
}
type Event struct {
	Schema   string `json:"schema"`
	Sequence uint64 `json:"sequence"`
	TaskID   string `json:"task_id"`
	Kind     string `json:"kind"`
	Reason   string `json:"reason,omitempty"`
}
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
func (r Receipt) Marshal() ([]byte, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(r)
}
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
func ContainsSensitiveField(raw []byte) bool {
	s := strings.ToLower(string(raw))
	for _, x := range []string{"prompt", "output_text", "file_path", "username", "device_id"} {
		if strings.Contains(s, "\""+x+"\"") {
			return true
		}
	}
	return false
}
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
