package model

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

// PrefixProfileEvent attributes prefix-cache ownership and transfer work. Events
// are emitted only when FAK_PREFIX_PROFILE names a JSONL file, keeping the normal
// decode path free of clocks and I/O.
type PrefixProfileEvent struct {
	Schema        string    `json:"schema"`
	At            time.Time `json:"at"`
	Operation     string    `json:"operation"`
	Component     string    `json:"component"`
	DurationNS    int64     `json:"duration_ns"`
	HostBytes     int64     `json:"host_bytes"`
	DeviceBytes   int64     `json:"device_bytes"`
	TransferBytes int64     `json:"transfer_bytes"`
	Tokens        int       `json:"tokens"`
	Backend       string    `json:"backend,omitempty"`
}

var prefixProfile = struct {
	sync.Mutex
	path string
}{path: os.Getenv("FAK_PREFIX_PROFILE")}

func prefixProfileStart() time.Time {
	if prefixProfile.path == "" {
		return time.Time{}
	}
	return time.Now()
}

func emitPrefixProfile(start time.Time, operation, component string, p *PrefixSnapshot, h *HostPrefixSnapshot) {
	if start.IsZero() {
		return
	}
	e := PrefixProfileEvent{Schema: "fak.prefix-profile/1", At: time.Now().UTC(), Operation: operation, Component: component, DurationNS: time.Since(start).Nanoseconds()}
	if p != nil {
		e.HostBytes, e.DeviceBytes = p.ResidencyBytes()
		e.Tokens = p.Tokens
		if p.Backend != nil {
			e.Backend = p.Backend.Name()
		}
	}
	if h != nil {
		e.HostBytes = h.ResidentBytes()
		e.TransferBytes = h.TransferBytes()
		e.Tokens = h.Tokens()
		if h.backend != nil {
			e.Backend = h.backend.Name()
		}
	}
	b, err := json.Marshal(e)
	if err != nil {
		return
	}
	prefixProfile.Lock()
	defer prefixProfile.Unlock()
	f, err := os.OpenFile(prefixProfile.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	_, _ = f.Write(append(b, '\n'))
	_ = f.Close()
}
